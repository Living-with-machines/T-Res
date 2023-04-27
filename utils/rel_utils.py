import os
import sys
import json
import sqlite3
from array import array
import numpy as np
from ast import literal_eval

sys.path.insert(0, os.path.abspath(os.path.pardir))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def get_db_emb(path_db, mentions, embtype):
    """
    This function returns the wikipedi2vec embedding for a given
    entity or word. If it is an entity, the prefix "ENTITY/" is
    preappended. If it is a word, the string is lowercased. This
    function opens a connection to the DB, performs the query,
    and closes the connection.

    Arguments:
        path_db: The path to the wikipedia2vec db.
        mentions: The list of words or entitys whose embeddings to extract.
        embtype: A string, either "word", "entity" or "snd". If "word", we
            use wikipedia2vec word embeddings; if "entity, we use wikipedia2vec
            entity embeddings; if "glove", we use glove word embeddings.

    Returns:
        obtained_embedding: A list of arrays (or None) with the embedding.
    """

    results = []
    with sqlite3.connect(path_db) as conn:
        c = conn.cursor()
        for mention in mentions:
            result = None
            # Preprocess the mention depending on which embedding to obtain:
            if embtype == "entity":
                if not mention == "#ENTITY/UNK#":
                    mention = "ENTITY/" + mention
                result = c.execute(
                    "SELECT emb FROM entity_embeddings WHERE word=?", (mention,)
                ).fetchone()
            if embtype == "word" or embtype == "snd":
                if mention in ["#WORD/UNK#", "#SND/UNK#"]:
                    mention = "#WORD/UNK#"
                else:
                    mention = mention.lower()
                result = c.execute(
                    "SELECT emb FROM entity_embeddings WHERE word=?", (mention,)
                ).fetchone()
            results.append(result if result is None else array("f", result[0]).tolist())

    return results


def eval_with_exception(str2parse, in_case=""):
    """
    Given a string in the form or a list or dictionary, parse it
    to read it as such.

    Arguments:
        str2parse (str): the string to parse.
        in_case (str): what should be returned in case of error.
    """
    try:
        return literal_eval(str2parse)
    except ValueError:
        return in_case


def prepare_initial_data(df, context_len=100):
    """
    This function takes a dataframe (as processed by the
    experiments/prepare_data.py script) and generates the
    equivalent json needed to train a REL model.

    Args:
        df: The dataframe containing the linking training data.
        context_len: The maximum number of words in the left
            and right contexts of the sentence where the target
            mention is.

    Returns:
        dict_mentions: a dictionary in which the article id is the
            key, and whose values are a list of dictionaries, each
            containing information about a mention. At this point,
            the mention dictionaries do not yet have the "gold"
            field (the gold standard Wikipedia title) or the selected
            candidates.
    """
    dict_mentions = dict()
    for i, row in df.iterrows():
        article_id = str(row["article_id"])
        dict_sentences = dict()
        for s in eval_with_exception(row["sentences"]):
            dict_sentences[int(s["sentence_pos"])] = s["sentence_text"]

        # Build a mention dictionary per mention:
        for df_mention in eval_with_exception(row["annotations"]):
            dict_mention = dict()
            mention = df_mention["mention"]
            sent_idx = int(df_mention["sent_pos"])
            sentence_id = article_id + "_" + str(sent_idx)

            # Generate left-hand context:
            left_context = ""
            if sent_idx - 1 in dict_sentences:
                left_context = dict_sentences[sent_idx - 1]

            # Generate right-hand context:
            right_context = ""
            if sent_idx + 1 in dict_sentences:
                right_context = dict_sentences[sent_idx + 1]

            dict_mention["mention"] = df_mention["mention"]
            dict_mention["sent_idx"] = sent_idx
            dict_mention["sentence"] = dict_sentences[sent_idx]
            dict_mention["ngram"] = mention
            dict_mention["context"] = [left_context, right_context]
            dict_mention["pos"] = df_mention["mention_start"]
            dict_mention["end_pos"] = df_mention["mention_end"]
            dict_mention["place"] = row["place"]
            dict_mention["place_wqid"] = row["place_wqid"]
            dict_mention["candidates"] = []
            dict_mention["ner_label"] = df_mention["entity_type"]

            # Check this:
            dict_mention["gold"] = [df_mention["wkdt_qid"]]
            if not df_mention["wkdt_qid"].startswith("Q"):
                dict_mention["gold"] = "NIL"

            if sentence_id in dict_mentions:
                dict_mentions[sentence_id].append(dict_mention)
            else:
                dict_mentions[sentence_id] = [dict_mention]

    return dict_mentions


def rank_candidates(rel_json, wk_cands, mentions_to_wikidata):
    new_json = dict()
    for article in rel_json:
        new_json[article] = []
        for mention_dict in rel_json[article]:
            cands = []
            tmp_cands = []
            max_cand_freq = 0
            ranker_cands = wk_cands.get(mention_dict["mention"], dict())
            for c in ranker_cands:
                # DeezyMatch confidence score (cosine similarity):
                cand_selection_score = ranker_cands[c]["Score"]
                # For each Wikidata candidate:
                for qc in ranker_cands[c]["Candidates"]:
                    # Mention-to-wikidata absolute relevance:
                    qcrlv_score = mentions_to_wikidata[c][qc]
                    if qcrlv_score > max_cand_freq:
                        max_cand_freq = qcrlv_score
                    qcm2w_score = ranker_cands[c]["Candidates"][qc]
                    # Average of CS conf score and mention2wiki norm relv:
                    if cand_selection_score:
                        qcm2w_score = (qcm2w_score + cand_selection_score) / 2
                    tmp_cands.append((qc, qcrlv_score, qcm2w_score))
            # Append candidate and normalized score weighted by candidate selection conf:
            for cand in tmp_cands:
                qc_id = cand[0]
                # Normalize absolute mention-to-wikidata relevance by entity:
                qc_score_1 = round(cand[1] / max_cand_freq, 3)
                # Candidate selection confidence:
                qc_score_2 = round(cand[2], 3)
                # Averaged relevances and normalize between 0 and 0.9:
                qc_score = ((qc_score_1 + qc_score_2) / 2) * 0.9
                cands.append([qc_id, qc_score])
            # Sort candidates and normalize between 0 and 1, and so they add up to 1.
            cands = sorted(cands, key=lambda x: (x[1], x[0]), reverse=True)

            mention_dict["candidates"] = cands
            new_json[article].append(mention_dict)
    return new_json


def add_publication(rel_json, publname="", publwqid=""):
    """
    TO DO.
    """
    new_json = rel_json.copy()
    for article in rel_json:
        place = publname
        place_wqid = publwqid
        if article != "linking":
            place = rel_json[article][0].get("place", publname)
            place_wqid = rel_json[article][0].get("place_wqid", publwqid)
        preffix_sentence = "This article is published in "
        sentence = preffix_sentence + place + "."
        dict_publ = {
            "mention": place,
            "sent_idx": 0,
            "sentence": sentence,
            "gold": [place_wqid],
            "ngram": place,
            "context": ["", ""],
            "pos": len(preffix_sentence),
            "end_pos": len(preffix_sentence + sentence),
            "candidates": [[place_wqid, 1.0]],
            "place": place,
            "place_wqid": place_wqid,
            "ner_label": "LOC",
        }
        new_json[article].append(dict_publ)
    return new_json


def prepare_rel_trainset(df, mylinker, myranker, dsplit):
    """
    This function takes as input the data prepared for linking, plus
    a Linking and Ranking objects. It prepares the data in the format
    that is required to train and test a REL disambiguation model,
    using the candidates from our ranker.

    Arguments:
        df: a pandas dataframe containing the dataset generated in
            the `experiments/prepare_data.py` script.
        mylinker: a Linking object.
        myranker: a Ranking object.

    This function stores the dataset formatted as a json.
    """
    rel_json = prepare_initial_data(df, mylinker.rel_params["context_length"])

    # Get unique mentions, to run them through the ranker:
    all_mentions = []
    for article in rel_json:
        if mylinker.rel_params["without_microtoponyms"]:
            all_mentions += [
                y["mention"] for y in rel_json[article] if y["ner_label"] == "LOC"
            ]
        else:
            all_mentions += [y["mention"] for y in rel_json[article]]
    all_mentions = list(set(all_mentions))
    # Format the mentions are required by the ranker:
    all_mentions = [{"mention": mention} for mention in all_mentions]
    # Use the ranker to find candidates:
    wk_cands, myranker.already_collected_cands = myranker.find_candidates(all_mentions)
    # Rank the candidates:
    rel_json = rank_candidates(
        rel_json,
        wk_cands,
        mylinker.linking_resources["mentions_to_wikidata"],
    )
    # If "publ" is taken into account for the disambiguation, add the place
    # of publication as an additional already disambiguated entity per row:
    if mylinker.rel_params["with_publication"] == True:
        rel_json = add_publication(
            rel_json,
            mylinker.rel_params["default_publname"],
            mylinker.rel_params["default_publwqid"],
        )

    ## TO DO
    with open(
        os.path.join(mylinker.rel_params["data_path"], "rel_{}.json").format(dsplit),
        "w",
    ) as f:
        json.dump(rel_json, f)

    return rel_json
