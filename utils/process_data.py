import os
import sys
import hashlib
import json
import pathlib
import re
import urllib
from ast import literal_eval
from pathlib import Path
from regex import D
from tqdm import tqdm

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import utils, ner


# Load wikipedia2wikidata mapper:
path = "/resources/wikipedia/extractedResources/"
wikipedia2wikidata = dict()
if Path(path + "wikipedia2wikidata.json").exists():
    with open(path + "wikipedia2wikidata.json", "r") as f:
        wikipedia2wikidata = json.load(f)
else:
    print("Warning: wikipedia2wikidata.json does not exist.")


# ------------------------------
# Get the Wikidata ID from a Wikipedia title
def turn_wikipedia2wikidata(wikipedia_title):
    """
    Get wikidata ID from wikipedia URL
    """
    if not wikipedia_title == "NIL" and not wikipedia_title == "*":
        wikipedia_title = wikipedia_title.split("/wiki/")[-1]
        wikipedia_title = urllib.parse.unquote(wikipedia_title)
        wikipedia_title = wikipedia_title.replace("_", " ")
        wikipedia_title = urllib.parse.quote(wikipedia_title)
        if "/" in wikipedia_title or len(wikipedia_title) > 200:
            wikipedia_title = hashlib.sha224(
                wikipedia_title.encode("utf-8")
            ).hexdigest()
        if not wikipedia_title in wikipedia2wikidata:
            print(
                "    >>> Warning: "
                + wikipedia_title
                + " is not in wikipedia2wikidata, the wkdt_qid will be NIL."
            )
        return wikipedia2wikidata.get(wikipedia_title)
    return "NIL"


# ----------------------------------------------
def load_tagset(filtering_labels):
    """
    Selects entities that will be considered for entity linking
    at inference time: if "all" is selected, we will link all
    detections regardless of their entity type; if "loc" is
    selected, we will link detections of the "loc" type.
    """
    tagset = dict()
    tagset["loc"] = ["loc", "b-loc", "i-loc"]
    tagset["all"] = [
        "loc",
        "b-loc",
        "i-loc",
        "street",
        "b-street",
        "i-street",
        "building",
        "b-building",
        "i-building",
        "other",
        "b-other",
        "i-other",
    ]
    return tagset[filtering_labels]


# ----------------------------------------------------
def prepare_sents(df):
    """
    Prepares annotated data and metadata on a sentence basis.

    Returns:
        dSentences (dict): dictionary in which we keep, for each article/sentence
            (expressed as e.g. "10732214_1", where "10732214" is the article_id
            and "1" is the order of the sentence in the article), the full original
            unprocessed sentence.
        dAnnotated (dict): dictionary in which we keep, for each article/sentence,
            an inner dictionary mapping the position of an annotated named entity (i.e.
            its start and end character, as a tuple, as the key) and another tuple as
            its value, which consists of: the type of named entity (such as LOC
            or BUILDING, the mention, and its annotated link), all extracted from
            the gold standard.
        dMetadata (dict): dictionary in which we keep, for each article/sentence,
            its metadata: place (of publication), year, ocr_quality_mean, ocr_quality_sd,
            publication_title, publication_code, and place_wqid (Wikidata ID of the
            place of publication).
    """

    dAnnotated = dict()
    dSentences = dict()
    dMetadata = dict()
    for i, row in df.iterrows():

        sentences = utils.eval_with_exception(row["sentences"], [])
        annotations = utils.eval_with_exception(row["annotations"], [])

        for s in sentences:
            # Sentence position:
            s_pos = s["sentence_pos"]
            # Article-sentence pair unique identifier:
            artsent_id = str(row["article_id"]) + "_" + str(s_pos)
            # Sentence text:
            dSentences[artsent_id] = s["sentence_text"]
            # Annotations in NER-required format:
            for a in annotations:
                if a["sent_pos"] == s_pos:
                    position = (int(a["mention_start"]), int(a["mention_end"]))
                    wqlink = a["wkdt_qid"]
                    if not isinstance(wqlink, str):
                        wqlink = "NIL"
                    elif wqlink == "*":
                        wqlink = "NIL"
                    if artsent_id in dAnnotated:
                        dAnnotated[artsent_id][position] = (
                            a["entity_type"],
                            a["mention"],
                            wqlink,
                        )
                    else:
                        dAnnotated[artsent_id] = {
                            position: (a["entity_type"], a["mention"], wqlink)
                        }

            # Keep metadata:
            dMetadata[artsent_id] = dict()
            dMetadata[artsent_id]["place"] = row["place"]
            dMetadata[artsent_id]["year"] = row["year"]
            dMetadata[artsent_id]["ocr_quality_mean"] = row["ocr_quality_mean"]
            dMetadata[artsent_id]["ocr_quality_sd"] = row["ocr_quality_sd"]
            dMetadata[artsent_id]["publication_title"] = row["publication_title"]
            dMetadata[artsent_id]["publication_code"] = row["publication_code"]
            dMetadata[artsent_id]["place_wqid"] = row["place_wqid"]

    # Now add also an empty annotations dictionary where no mentions have been annotated:
    for artsent_id in dSentences:
        if not artsent_id in dAnnotated:
            dAnnotated[artsent_id] = dict()

    # Now add also the metadata of sentences where no mentions have been annotated:
    for artsent_id in dSentences:
        if not artsent_id in dMetadata:
            dMetadata[artsent_id] = dict()

    return dAnnotated, dSentences, dMetadata


# ----------------------------------------------
def align_gold(predictions, annotations):
    """
    The gold standard tokenisation is not aligned with the tokenization
    produced through the BERT model (as it uses its own tokenizer). To
    be able to assess the performance of the entity recogniser, we must
    align the two tokenisations. This function aligns the output of BERT
    NER and the gold standard labels. It does so based on the start and
    end position of each predicted token. By default, a predicted token
    is assigned the "O" label, unless its position overlaps with the
    position an annotated entity, in which case we relabel it according
    to the label found in this position.

    Arguments:
        predictions (list): list of dictionaries, each corresponding to
            a mention.
        annotations (dict): a dictionary, in which the key is a tuple
            containing the first and last character position of a gold
            standard detection in a sentence and the value is a tuple
            with the type of label (e.g. "LOC"), the mention (e.g.
            "Point Petre", and the link (e.g. "Q335322")).

    Returns:
        gold_standard ():
    """

    gold_standard = []
    for pred_ent in predictions:
        gs_for_eval = pred_ent.copy()
        # This has been manually annotated, so perfect score
        gs_for_eval["score"] = 1.0
        # We instantiate the entity class as "O" ("outside", i.e. not a NE)
        gs_for_eval["entity"] = "O"
        gs_for_eval["link"] = "O"
        # It's prefixed as "B-" if the token is the first in a sequence,
        # otherwise it's prefixed as "I-"
        for gse in annotations:
            if pred_ent["start"] == gse[0] and pred_ent["end"] <= gse[1]:
                gs_for_eval["entity"] = "B-" + annotations[gse][0].upper()
                gs_for_eval["link"] = "B-" + annotations[gse][2]
            elif pred_ent["start"] > gse[0] and pred_ent["end"] <= gse[1]:
                gs_for_eval["entity"] = "I-" + annotations[gse][0].upper()
                gs_for_eval["link"] = "I-" + annotations[gse][2]
        gold_standard.append(gs_for_eval)

    return gold_standard


# ----------------------------------------------------
def postprocess_predictions(predictions, gold_positions, accepted_labels):
    """
    Postprocess predictions to be used later in the pipeline.
    """
    postprocessed_sentence = dict()
    sentence_preds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], float(x["score"])]
        for x in predictions
    ]
    sentence_trues = [
        [x["word"], x["entity"], x["link"], x["start"], x["end"]]
        for x in gold_positions
    ]
    sentence_skys = [
        [x["word"], x["entity"], "O", x["start"], x["end"]] for x in gold_positions
    ]

    # Filter by accepted labels:
    sentence_trues = [
        [x[0], x[1], "NIL", x[3], x[4]]
        if x[1] != "O" and x[1].lower() not in accepted_labels
        else x
        for x in sentence_trues
    ]

    postprocessed_sentence["sentence_preds"] = sentence_preds
    postprocessed_sentence["sentence_trues"] = sentence_trues
    postprocessed_sentence["sentence_skys"] = sentence_skys

    return postprocessed_sentence


def ner_and_process(dSentences, dAnnotated, myner, accepted_labels):
    """
    Perform NER in the LwM way, and postprocess the output.
    """
    gold_tokenization = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()
    dMentionsPred = dict()  # Dictionary of detected mentions
    dMentionsGold = dict()  # Dictionary of goldstandard mentions
    for sent_id in tqdm(list(dSentences.keys())):
        sent = dSentences[sent_id]
        annotations = dAnnotated[sent_id]
        predictions = myner.ner_predict(sent)
        gold_positions = align_gold(predictions, annotations)
        sentence_postprocessing = postprocess_predictions(
            predictions, gold_positions, accepted_labels
        )
        dPreds[sent_id] = sentence_postprocessing["sentence_preds"]
        dTrues[sent_id] = sentence_postprocessing["sentence_trues"]
        dSkys[sent_id] = sentence_postprocessing["sentence_skys"]
        gold_tokenization[sent_id] = gold_positions
        dMentionsPred[sent_id] = ner.aggregate_mentions(
            sentence_postprocessing["sentence_preds"], accepted_labels, "pred"
        )
        dMentionsGold[sent_id] = ner.aggregate_mentions(
            sentence_postprocessing["sentence_trues"], accepted_labels, "gold"
        )
    return dPreds, dTrues, dSkys, gold_tokenization, dMentionsPred, dMentionsGold


def load_processed_data(mydata):

    output_path = (
        mydata.data_path
        + mydata.dataset
        + "/"
        + mydata.myner.model_name
        + "_"
        + mydata.myner.filtering_labels
    )

    cand_approach = mydata.myranker.method
    if mydata.myranker.method == "deezymatch":
        cand_approach += "+" + str(mydata.myranker.deezy_parameters["num_candidates"])
        cand_approach += "+" + str(
            mydata.myranker.deezy_parameters["selection_threshold"]
        )

    print("* Prefix of data to load:", output_path)
    output_processed_data = dict()
    try:
        with open(output_path + "_ner_predictions.json") as fr:
            output_processed_data["preds"] = json.load(fr)
        with open(output_path + "_gold_standard.json") as fr:
            output_processed_data["trues"] = json.load(fr)
        with open(output_path + "_ner_skyline.json") as fr:
            output_processed_data["skys"] = json.load(fr)
        with open(output_path + "_gold_positions.json") as fr:
            output_processed_data["gold_tok"] = json.load(fr)
        with open(output_path + "_dict_sentences.json") as fr:
            output_processed_data["dSentences"] = json.load(fr)
        with open(output_path + "_dict_metadata.json") as fr:
            output_processed_data["dMetadata"] = json.load(fr)
        with open(output_path + "_dict_REL.json") as fr:
            output_processed_data["dREL"] = json.load(fr)
        with open(output_path + "_pred_mentions.json") as fr:
            output_processed_data["dMentionsPred"] = json.load(fr)
        with open(output_path + "_gold_mentions.json") as fr:
            output_processed_data["dMentionsGold"] = json.load(fr)
        with open(output_path + "_candidates_" + cand_approach + ".json") as fr:
            output_processed_data["dCandidates"] = json.load(fr)

        return output_processed_data
    except FileNotFoundError:
        print("File not found, process data.")
        return dict()


# ----------------------------------------------------
def store_processed_data(
    mydata,
    preds,
    trues,
    skys,
    gold_tok,
    dSentences,
    dMetadata,
    dREL,
    dMentionsPred,
    dMentionsGold,
    dCandidates,
):
    """
    This function stores all the postprocessed data as jsons.
    """
    data_path = mydata.data_path
    dataset = mydata.dataset
    model_name = mydata.myner.model_name
    filtering_labels = mydata.myner.filtering_labels
    output_path = data_path + dataset + "/" + model_name + "_" + filtering_labels

    cand_approach = mydata.myranker.method
    if mydata.myranker.method == "deezymatch":
        cand_approach += "+" + str(mydata.myranker.deezy_parameters["num_candidates"])
        cand_approach += "+" + str(
            mydata.myranker.deezy_parameters["selection_threshold"]
        )

    # Store NER predictions using a specific NER model:
    with open(output_path + "_ner_predictions.json", "w") as fw:
        json.dump(preds, fw)

    # Store gold standard:
    with open(output_path + "_gold_standard.json", "w") as fw:
        json.dump(trues, fw)

    # Store NER skyline:
    with open(output_path + "_ner_skyline.json", "w") as fw:
        json.dump(skys, fw)

    # Store gold tokenisation positions:
    with open(output_path + "_gold_positions.json", "w") as fw:
        json.dump(gold_tok, fw)

    # Store the dictionary of sentences:
    with open(output_path + "_dict_sentences.json", "w") as fw:
        json.dump(dSentences, fw)

    # Store the dictionary of metadata per sentence:
    with open(output_path + "_dict_metadata.json", "w") as fw:
        json.dump(dMetadata, fw)

    # Store the dictionary of REL results:
    with open(output_path + "_dict_REL.json", "w") as fw:
        json.dump(dREL, fw)

    # Store the dictionary of predicted results:
    with open(output_path + "_pred_mentions.json", "w") as fw:
        json.dump(dMentionsPred, fw)

    # Store the dictionary of gold standard:
    with open(output_path + "_gold_mentions.json", "w") as fw:
        json.dump(dMentionsGold, fw)

    # Store the dictionary of gold standard:
    with open(output_path + "_candidates_" + cand_approach + ".json", "w") as fw:
        json.dump(dCandidates, fw)

    dict_processed_data = dict()
    dict_processed_data["preds"] = preds
    dict_processed_data["trues"] = trues
    dict_processed_data["skys"] = skys
    dict_processed_data["gold_tok"] = gold_tok
    dict_processed_data["dSentences"] = dSentences
    dict_processed_data["dMetadata"] = dMetadata
    dict_processed_data["dREL"] = dREL
    dict_processed_data["dMentionsPred"] = dMentionsPred
    dict_processed_data["dMentionsGold"] = dMentionsGold
    dict_processed_data["dCandidates"] = dCandidates
    return dict_processed_data


def find_candidates(dMentionsPred, myranker):
    myranker.mentions_to_wikidata = myranker.load_resources()
    dCandidates = dict()
    for sentence_id in tqdm(dMentionsPred):
        pred_mentions_sent = dMentionsPred[sentence_id]
        mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
        cands, myranker.already_collected_cands = myranker.run(mentions)

        wk_cands = dict()
        for found_mention in cands:
            # Find Wikidata ID and relv.
            found_cands = myranker.mentions_to_wikidata.get(found_mention, dict())
            wk_cands[found_mention] = {"Matches": cands[found_mention]}
            wk_cands[found_mention]["Candidates"] = found_cands

        dCandidates[sentence_id] = wk_cands
    return dCandidates


# ----------------------------------------------------
def rel_end_to_end(sent):
    """REL end-to-end using the API."""
    API_URL = "https://rel.cs.ru.nl/api"
    el_result = requests.post(API_URL, json={"text": sent, "spans": []}).json()
    return el_result


# ----------------------------------------------------
def get_rel_from_api(dSentences, rel_end2end_path):
    """
    Uses the REL API to do end-to-end entity linking.

    Arguments:
        dSentences (dict): dictionary of sentences, where the
            key is the article-sent identifier and the value
            is the full text of the sentence.
        rel_end2end_path (str): the path of the file where the
            REL results will be stored.

    Returns:
        A JSON file with the REL results.
    """
    # Dictionary to store REL predictions:
    rel_preds = dict()
    if Path(rel_end2end_path).exists():
        with open(rel_end2end_path) as f:
            rel_preds = json.load(f)
    print("\nObtain REL linking from API (unless already stored):")
    for s in tqdm(dSentences):
        if not s in rel_preds:
            rel_preds[s] = rel_end_to_end(dSentences[s])
            # Append per processed sentence in case of API limit:
            with open(rel_end2end_path, "w") as fp:
                json.dump(rel_preds, fp)
            with open(rel_end2end_path) as f:
                rel_preds = json.load(f)


def match_ent(pred_ents, start, end, prev_ann, accepted_labels):
    for ent in pred_ents:
        if ent[-1].lower() in accepted_labels:
            st_ent = ent[0]
            len_ent = ent[1]
            if start >= st_ent and end <= (st_ent + len_ent):
                if prev_ann == ent[-1]:
                    ent_pos = "I-"
                else:
                    ent_pos = "B-"
                    prev_ann = ent[-1]

                n = ent_pos + ent[-1]
                try:
                    el = ent_pos + turn_wikipedia2wikidata(ent[3])
                except Exception as e:
                    # to be checked but it seems some Wikipedia pages are not in our Wikidata
                    # see for instance Zante%2C%20California
                    return n, ent_pos + "NIL", prev_ann
                return n, el, prev_ann
    return "O", "O", ""


def postprocess_rel(rel_end2end_path, dSentences, gold_tokenization, accepted_labels):
    with open(rel_end2end_path) as f:
        rel_preds = json.load(f)

    dREL = dict()
    for sent_id in tqdm(list(dSentences.keys())):
        sentence_preds = []
        prev_ann = ""
        for token in gold_tokenization[sent_id]:
            start = token["start"]
            end = token["end"]
            word = token["word"]
            n, el, prev_ann = match_ent(
                rel_preds[sent_id], start, end, prev_ann, accepted_labels
            )
            sentence_preds.append([word, n, el])

        dREL[sent_id] = sentence_preds
    return dREL


def process_for_linking(df):
    """
    Process data for linking, resulting in a dataframe with one mention
    per row.
    """

    # Create dataframe by mention:
    rows = []
    for i, row in df.iterrows():
        article_id = row["article_id"]
        place = row["place"]
        year = row["year"]
        ocr_quality_mean = row["ocr_quality_mean"]
        ocr_quality_sd = row["ocr_quality_sd"]
        publication_title = row["publication_title"]
        publication_code = str(row["publication_code"]).zfill(7)
        sentences = literal_eval(row["sentences"])
        annotations = literal_eval(row["annotations"])
        for s in sentences:
            for a in annotations:
                if s["sentence_pos"] == a["sent_pos"]:
                    rows.append(
                        (
                            article_id,
                            s["sentence_pos"],
                            s["sentence_text"],
                            a["mention_pos"],
                            a["mention"],
                            a["wkdt_qid"],
                            a["mention_start"],
                            a["mention_end"],
                            a["entity_type"],
                            year,
                            place,
                            ocr_quality_mean,
                            ocr_quality_sd,
                            publication_title,
                            publication_code,
                        )
                    )

    training_df = pd.DataFrame(
        columns=[
            "article_id",
            "sentence_pos",
            "sentence",
            "mention_pos",
            "mention",
            "wkdt_qid",
            "mention_start",
            "mention_end",
            "entity_type",
            "year",
            "place",
            "ocr_quality_mean",
            "ocr_quality_sd",
            "publication_title",
            "publication_code",
        ],
        data=rows,
    )

    return training_df


def create_mentions_df(mydata):
    """
    Create a dataframe for the linking experiment, with one
    mention per row.
    """
    dMentions = mydata.processed_data["dMentionsPred"]
    dGoldSt = mydata.processed_data["dMentionsGold"]
    dSentences = mydata.processed_data["dSentences"]
    dMetadata = mydata.processed_data["dMetadata"]
    dCandidates = mydata.processed_data["dCandidates"]

    cand_approach = mydata.myranker.method
    if mydata.myranker.method == "deezymatch":
        cand_approach += "+" + str(mydata.myranker.deezy_parameters["num_candidates"])
        cand_approach += "+" + str(
            mydata.myranker.deezy_parameters["selection_threshold"]
        )

    rows = []
    for sentence_id in dMentions:
        for mention in dMentions[sentence_id]:
            if mention:
                article_id = sentence_id.split("_")[0]
                sentence_pos = sentence_id.split("_")[1]
                sentence = dSentences[sentence_id]
                token_start = mention["start_offset"]
                token_end = mention["end_offset"]
                char_start = mention["start_char"]
                char_end = mention["end_char"]
                ner_score = round(mention["ner_score"], 3)
                pred_mention = mention["mention"]
                entity_type = mention["ner_label"]
                place = dMetadata[sentence_id]["place"]
                year = dMetadata[sentence_id]["year"]
                publication = dMetadata[sentence_id]["publication_code"]
                place_wqid = dMetadata[sentence_id]["place_wqid"]
                # Match predicted mention with gold standard mention (will just be used for training):
                max_tok_overlap = 0
                gold_standard_link = "NIL"
                gold_standard_ner = "O"
                gold_mention = ""
                for gs in dGoldSt[sentence_id]:
                    pred_token_range = range(token_start, token_end + 1)
                    gs_token_range = range(gs["start_offset"], gs["end_offset"] + 1)
                    overlap = len(list(set(pred_token_range) & set(gs_token_range)))
                    if overlap > max_tok_overlap:
                        max_tok_overlap = overlap
                        gold_mention = gs["mention"]
                        gold_standard_link = gs["entity_link"]
                        gold_standard_ner = gs["ner_label"]
                candidates = dCandidates[sentence_id][mention["mention"]]

                rows.append(
                    [
                        sentence_id,
                        article_id,
                        sentence_pos,
                        sentence,
                        token_start,
                        token_end,
                        char_start,
                        char_end,
                        ner_score,
                        pred_mention,
                        entity_type,
                        place,
                        year,
                        publication,
                        place_wqid,
                        gold_mention,
                        gold_standard_link,
                        gold_standard_ner,
                        candidates,
                    ]
                )

    processed_df = pd.DataFrame(
        columns=[
            "sentence_id",
            "article_id",
            "sentence_pos",
            "sentence",
            "token_start",
            "token_end",
            "char_start",
            "char_end",
            "ner_score",
            "pred_mention",
            "pred_ner_label",
            "place",
            "year",
            "publication",
            "place_wqid",
            "gold_mention",
            "gold_entity_link",
            "gold_ner_label",
            "candidates",
        ],
        data=rows,
    )

    output_path = (
        mydata.data_path
        + mydata.dataset
        + "/"
        + mydata.myner.model_name
        + "_"
        + mydata.myner.filtering_labels
        + "_"
        + cand_approach
    )

    # List of columns to merge (i.e. columns where we have indicated
    # out data splits), and "article_id", the columns on which we
    # will merge the data:
    keep_columns = [
        "article_id",
        "originalsplit",
        "traindevtest",
        "Ashton1860",
        "Dorchester1820",
        "Dorchester1830",
        "Dorchester1860",
        "Manchester1780",
        "Manchester1800",
        "Manchester1820",
        "Manchester1830",
        "Manchester1860",
        "Poole1860",
    ]

    # Add data splits from original dataframe:
    df = mydata.dataset_df[[c for c in keep_columns if c in mydata.dataset_df.columns]]

    # Convert article_id to string (it's read as an int):
    df = df.assign(article_id=lambda d: d["article_id"].astype(str))
    processed_df = processed_df.assign(article_id=lambda d: d["article_id"].astype(str))
    processed_df = pd.merge(processed_df, df, on=["article_id"], how="left")

    # Store mentions dataframe:
    processed_df.to_csv(output_path + "_linking_df_mentions.tsv", sep="\t")

    return processed_df


# Storing results for evaluation using the CLEF-HIPE scorer
def store_for_scorer(hipe_scorer_results_path, scenario_name, dresults, articles_test):
    """
    Store results in the right format to be used by the CLEF-HIPE
    scorer: https://github.com/impresso/CLEF-HIPE-2020-scorer.

    Assuming the CLEF-HIPE scorer is stored in ../CLEF-HIPE-2020-scorer/,
    run scorer as follows:
    For NER:
    > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nerc_coarse --outdir outputs/results/
    For EL:
    > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nel --outdir outputs/results/
    """
    # Bundle 2 associated tasks: NERC-coarse and NEL
    with open(
        hipe_scorer_results_path + "/" + scenario_name + ".tsv",
        "w",
    ) as fw:
        fw.write(
            "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n"
        )
        for sent_id in dresults:
            # Filter by article in test:
            if sent_id.split("_")[0] in articles_test:
                fw.write("# sentence_id = " + sent_id + "\n")
                for t in dresults[sent_id]:
                    elink = t[2]
                    if t[2].startswith("B-"):
                        elink = t[2].replace("B-", "")
                    elif t[2].startswith("I-"):
                        elink = t[2].replace("I-", "")
                    elif t[1] != "O":
                        elink = "NIL"
                    fw.write(
                        t[0]
                        + "\t"
                        + t[1]
                        + "\t"
                        + t[1]
                        + "\tO\tO\tO\tO\t"
                        + elink
                        + "\t"
                        + elink
                        + "\tO\n"
                    )
                fw.write("\n")


def store_results(mydata):
    hipe_scorer_results_path = mydata.results_path + mydata.dataset + "/"
    scenario_name = (
        "ner_" + mydata.myner.model_name + "_" + mydata.myner.filtering_labels + "_"
    )

    # Find article ids of the test set (original split, for NER):
    ner_all = mydata.dataset_df
    ner_test_articles = list(
        ner_all[ner_all["originalsplit"] == "test"].article_id.unique()
    )
    ner_test_articles = [str(art) for art in ner_test_articles]

    # Store predictions results formatted for CLEF-HIPE scorer:
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + "preds",
        mydata.processed_data["preds"],
        ner_test_articles,
    )

    # Store gold standard results formatted for CLEF-HIPE scorer:
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + "trues",
        mydata.processed_data["trues"],
        ner_test_articles,
    )

    # Store REL results formatted for CLEF-HIPE scorer:
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + "rel",
        mydata.processed_data["dREL"],
        ner_test_articles,
    )


##################################################
##################################################
#############       NOT USED YET       ###########
##################################################
##################################################


"""
def crate_training_for_el(df):

    # Create dataframe by mention:
    rows = []
    for i, row in df.iterrows():
        article_id = row["article_id"]
        place = row["place"]
        year = row["year"]
        ocr_quality_mean = row["ocr_quality_mean"]
        ocr_quality_sd = row["ocr_quality_sd"]
        publication_title = row["publication_title"]
        publication_code = str(row["publication_code"]).zfill(7)
        sentences = literal_eval(row["sentences"])
        annotations = literal_eval(row["annotations"])
        for s in sentences:
            for a in annotations:
                if s["sentence_pos"] == a["sent_pos"]:
                    rows.append(
                        (
                            article_id,
                            s["sentence_pos"],
                            s["sentence_text"],
                            a["mention_pos"],
                            a["mention"],
                            a["wkdt_qid"],
                            a["mention_start"],
                            a["mention_end"],
                            a["entity_type"],
                            year,
                            place,
                            ocr_quality_mean,
                            ocr_quality_sd,
                            publication_title,
                            publication_code,
                        )
                    )

    training_df = pd.DataFrame(
        columns=[
            "article_id",
            "sentence_pos",
            "sentence",
            "mention_pos",
            "mention",
            "wkdt_qid",
            "mention_start",
            "mention_end",
            "entity_type",
            "year",
            "place",
            "ocr_quality_mean",
            "ocr_quality_sd",
            "publication_title",
            "publication_code",
        ],
        data=rows,
    )

    return training_df
"""

# # ------------------------------
# # EVALUATION WITH CLEF SCORER
# # ------------------------------

# # Skyline
# def store_resolution_skyline(dataset, approach, value):
#     pathlib.Path("outputs/results/" + dataset + "/").mkdir(parents=True, exist_ok=True)
#     skyline = open("outputs/results/" + dataset + "/" + approach + ".skyline", "w")
#     skyline.write(str(value))
#     skyline.close()


# # Storing results for evaluation using the CLEF-HIPE scorer
# def store_results_hipe(dataset, dataresults, dresults):
#     """
#     Store results in the right format to be used by the CLEF-HIPE
#     scorer: https://github.com/impresso/CLEF-HIPE-2020-scorer.

#     Assuming the CLEF-HIPE scorer is stored in ../CLEF-HIPE-2020-scorer/,
#     run scorer as follows:
#     For NER:
#     > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nerc_coarse --outdir outputs/results/
#     For EL:
#     > python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nel --outdir outputs/results/
#     """
#     pathlib.Path("outputs/results/" + dataset + "/").mkdir(parents=True, exist_ok=True)
#     # Bundle 2 associated tasks: NERC-coarse and NEL
#     with open(
#         "outputs/results/" + dataset + "/" + dataresults + "_bundle2_en_1.tsv", "w"
#     ) as fw:
#         fw.write(
#             "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n"
#         )
#         for sent_id in dresults:
#             fw.write("# sentence_id = " + sent_id + "\n")
#             for t in dresults[sent_id]:
#                 elink = t[2]
#                 if t[2].startswith("B-"):
#                     elink = t[2].replace("B-", "")
#                 elif t[2].startswith("I-"):
#                     elink = t[2].replace("I-", "")
#                 elif t[1] != "O":
#                     elink = "NIL"
#                 fw.write(
#                     t[0]
#                     + "\t"
#                     + t[1]
#                     + "\t"
#                     + t[1]
#                     + "\tO\tO\tO\tO\t"
#                     + elink
#                     + "\t"
#                     + elink
#                     + "\tO\n"
#                 )
#             fw.write("\n")
