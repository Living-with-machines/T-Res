import os
import sys
import json
import sqlite3
from array import array
import pandas as pd
from ast import literal_eval

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_wikipedia


def get_wiki2vec_emb(path_db, mentions, is_entity):
    """
    This function returns the wikipedi2vec embedding for a given
    entity or word. If it is an entity, the prefix "ENTITY/" is
    preappended. If it is a word, the string is lowercased. This
    function opens a connection to the DB, performs the query,
    and closes the connection.

    Arguments:
        path_db: The path to the wikipedia2vec db.
        mentions: The list of words or entitys whose embeddings to extract.
        is_entity: Boolean, whether the string is an entity or a word.

    Returns:
        obtained_embedding: A list of arrays (or None) with the embedding.
    """

    results = []
    with sqlite3.connect(path_db) as conn:
        c = conn.cursor()
        for mention in mentions:
            # Preprocess the mention depending on whether it's an entity or a word:
            mention = "ENTITY/" + mention if is_entity else mention.lower()
            # Get the embeddings from the db:
            result = c.execute(
                "SELECT emb FROM embeddings WHERE word=?", (mention,)
            ).fetchone()
        results.append(result if result is None else array("f", result[0]).tolist())

    return results


def get_glove_emb(path_db, words):
    """
    This function returns the wikipedi2vec embedding for a given
    entity or word. If it is an entity, the prefix "ENTITY/" is
    preappended. If it is a word, the string is lowercased. This
    function opens a connection to the DB, performs the query,
    and closes the connection.

    Arguments:
        path_db: The path to the wikipedia2vec db.
        words: The list of words whose embeddings to extract.

    Returns:
        obtained_embedding: A list of arrays (or None) with the embedding.
    """
    results = []
    with sqlite3.connect(path_db) as conn:
        c = conn.cursor()
        for word in words:
            result = c.execute(
                "SELECT emb FROM embeddings WHERE word=?", (word,)
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


def prepare_initial_data(df, wikimapper_path, context_len=100):
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

            # Generate left-hand context:
            left_context = ""
            if sent_idx - 1 in dict_sentences:
                left_context = dict_sentences[sent_idx - 1]
            for i in range(sent_idx - 2, 0, -1):
                tmp_context = dict_sentences[i] + " " + left_context
                if len(tmp_context.split(" ")) < context_len:
                    left_context = tmp_context
                else:
                    break

            # Generate right-hand context:
            right_context = ""
            if sent_idx + 1 in dict_sentences:
                right_context = dict_sentences[sent_idx + 1]
            for i in range(sent_idx + 2, len(dict_sentences) + 1):
                tmp_context = right_context + " " + dict_sentences[i]
                if len(tmp_context.split(" ")) < context_len:
                    right_context = tmp_context
                else:
                    break

            dict_mention["mention"] = df_mention["mention"]
            dict_mention["sent_idx"] = sent_idx
            dict_mention["sentence"] = dict_sentences[sent_idx]
            dict_mention["wkdt_gold"] = df_mention["wkdt_qid"]
            dict_mention["ngram"] = mention
            dict_mention["context"] = [left_context, right_context]
            dict_mention["pos"] = df_mention["mention_start"]
            dict_mention["end_pos"] = df_mention["mention_end"]
            dict_mention["place"] = row["place"]
            dict_mention["place_wqid"] = row["place_wqid"]
            dict_mention["candidates"] = []

            dict_mention["gold"] = "NIL"
            gold_ids = process_wikipedia.id_to_title(
                df_mention["wkdt_qid"], wikimapper_path
            )
            # Get the first of the wikipedia titles returned (they're sorted
            # by their autoincrement id):
            if gold_ids:
                dict_mention["gold"] = [gold_ids[0]]

            if article_id in dict_mentions:
                dict_mentions[article_id].append(dict_mention)
            else:
                dict_mentions[article_id] = [dict_mention]

    return dict_mentions


def rank_candidates(rel_json, wk_cands, mentions_to_wikidata, wikimapper_db, topn=10):
    new_json = dict()
    for article in rel_json:
        for mention_dict in rel_json[article]:
            cands = []
            tmp_cands = []
            max_cand_freq = 0
            ranker_cands = wk_cands[mention_dict["mention"]]
            for c in ranker_cands:
                cand_selection_score = ranker_cands[c]["Score"]
                for qc in ranker_cands[c]["Candidates"]:
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
                qc_score_1 = round(cand[1] / max_cand_freq, 3)
                qc_score_2 = round(cand[2], 3)
                qc_score = (qc_score_1 + qc_score_2) / 2
                cands.append([qc_id, qc_score])
            # Sort candidates and normalize between 0 and 1, and so they add up to 1.
            cands = sorted(cands, key=lambda x: (x[1], x[0]), reverse=True)
            # Limit to the most likely candidates:
            cands = cands[:topn]
            # Wikidata entity to Wikipedia
            wkpd_cands = []
            for cand in cands:
                qc_wikipedia = ""
                gold_ids = process_wikipedia.id_to_title(cand[0], wikimapper_db)
                if gold_ids:
                    qc_wikipedia = gold_ids[0]
                    wkpd_cands.append((qc_wikipedia, cand[1]))

            mention_dict["wikidata_candidates"] = cands
            mention_dict["candidates"] = wkpd_cands
            if article in new_json:
                new_json[article].append(mention_dict)
            else:
                new_json[article] = [mention_dict]
    return new_json


def add_publication(rel_json, wikimapper_path):
    """
    TO DO.
    """
    return rel_json


def prepare_rel_trainset(df, mylinker, myranker, full_data_path):
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
    wikimapper_db = mylinker.wikimapper_path
    rel_json = prepare_initial_data(
        df, wikimapper_db, mylinker.rel_params["context_length"]
    )

    # Get unique mentions, to run them through the ranker:
    all_mentions = []
    for article in rel_json:
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
        mylinker.wikimapper_path,
        mylinker.rel_params["topn_candidates"],
    )
    # # If "publ" is taken into account for the disambiguation, add the place
    # # of publication as an additional already disambiguated entity per row:
    # if mylinker.rel_params["with_publication"] == True:
    #     rel_json = add_publication(rel_json, wikimapper_db)

    # Store the dataset:
    with open(full_data_path, "w", encoding="utf-8") as f:
        json.dump(rel_json, f, ensure_ascii=False)


def generate_data_path(mylinker, split):
    # Generate the full data path based on the different variables:
    data_filename = mylinker.rel_params["training_split"] + "_" + split
    if mylinker.rel_params["with_publication"] == True:
        data_filename += "_withPublication"
    else:
        data_filename += "_noPublication"
    data_filename += ".json"
    full_data_path = os.path.join(mylinker.rel_params["data_path"], data_filename)
    return full_data_path
