import os
import sys
import json
import urllib
from ast import literal_eval
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import ner


# Load wikipedia2wikidata mapper:
path = "/resources/wikipedia/extractedResources/"
wikipedia2wikidata = dict()
if Path(path + "wikipedia2wikidata.json").exists():
    with open(path + "wikipedia2wikidata.json", "r") as f:
        wikipedia2wikidata = json.load(f)
else:
    print("Warning: wikipedia2wikidata.json does not exist.")


# Load gazetteer (our knowledge base):
gazetteer_ids = set(
    list(
        pd.read_csv("/resources/wikidata/wikidata_gazetteer.csv", low_memory=False)[
            "wikidata_id"
        ].unique()
    )
)


# ----------------------------------------------------
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


def get_wikidata_instance_ids(mylinker):
    """helper to map wikidata entitiess to class ids
    if there is more than one, we take the most general class
    i.e. the one with lowest number
    """
    mylinker.linking_resources["wikidata_id2inst_id"] = {}
    for i, row in tqdm(mylinker.linking_resources["gazetteer"].iterrows()):
        instances = row["instance_of"]
        if instances:
            if len(instances) > 1:
                instance = instances[0]
                for i in instances[1:]:
                    if int(i[1:]) < int(instance[1:]):
                        instance = i
                mylinker.linking_resources["wikidata_id2inst_id"][
                    row.wikidata_id
                ] = instance
            else:
                mylinker.linking_resources["wikidata_id2inst_id"][
                    row.wikidata_id
                ] = instances[0]
    return mylinker.linking_resources["wikidata_id2inst_id"]


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

        sentences = eval_with_exception(row["sentences"], [])
        annotations = eval_with_exception(row["annotations"], [])

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


# ----------------------------------------------------
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
def postprocess_predictions(predictions, gold_positions):
    """
    Postprocess predictions to be used later in the pipeline.

    Arguments:
        predictions (list): the output of the recogniser.ner_predict()
            method, where, given a sentence, a list of dictionaries is
            returned, where each dictionary corresponds to a recognised
            token, e.g.: {'entity': 'O', 'score': 0.99975187, 'word':
            'From', 'start': 0, 'end': 4}
        gold_positions(list): the output of the align_gold() function,
            which aligns the gold standard text to the tokenisation
            performed by the named entity recogniser, to enable
            assessing the performance of the NER and linking steps.

    Returns:
        postprocessed_sentence (dict): a dictionary with three key-value
            pairs: (1) sentence_preds is mapped to the list of lists
            representation of 'predictions', (2) sentence_trues is
            mapped to the list of lists representation of 'gold_positions',
            and (3) sentence_skys is the same as sentence_trues, but
            with empty link.
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
        [x["word"], x["entity"], "O", x["start"], x["end"]] for x in predictions
    ]

    postprocessed_sentence["sentence_preds"] = sentence_preds
    postprocessed_sentence["sentence_trues"] = sentence_trues
    postprocessed_sentence["sentence_skys"] = sentence_skys

    return postprocessed_sentence


# ----------------------------------------------------
def ner_and_process(dSentences, dAnnotated, myner):
    """
    Perform named entity recognition in the LwM way, and postprocess the
    output to prepare it for the experiments.

    Arguments:
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
        myner (recogniser.Recogniser): a Recogniser object, for NER.

    Returns:
        dPreds (dict): dictionary where the NER predictions are stored, where the key
            is the sentence_id (i.e. article_id + "_" + sentence_pos) and the value is
            a list of lists, where each element corresponds to one token in a sentence,
            for example:
                ["From", "O", "O", 0, 4, 0.999826967716217]
            ...where the the elements are: (1) the token, (2) the NER tag, (3), the link
            to wikidata, set to "O" for now because we haven't performed linking yet, (4)
            the starting character of the token, (5) the end character of the token, and
            (6) the NER prediction score. This dictionary is stored in the 'outputs/data'
            folder, with prefix "_ner_predictions.json".
        dTrues (dict): a dictionary where the gold standard named entities are stored,
            which has the same format as dPreds, but with the manually annotated data
            instead of the predictions. This dictionary is stored in the 'outputs/data'
            folder, with prefix "_gold_standard.json".
        dSkys (dict): a dictionary where the skyline will be stored, for the linking
            experiments. At this point, it will be the same as dPreds, without the
            NER prediction score. During linking, it will be filled with the gold standard
            entities when these have been retrieved using candidates. This dictionary
            is stored in the 'outputs/data' folder, with prefix "_ner_skyline.json".
        gold_tokenization (dict): a dictionary where the gold standard entities are
            stored, wehre the key is the sentence_id (i.e. article_id + "_" + sentence_pos)
            and the value is a list of dictionaries, each looking like this:
                {"entity": "B-LOC", "score": 1.0, "word": "Unitec", "start": 193,
                "end": 199, "link": "B-Q30"}
            ...this dictionary is stored in the 'outputs/data' folder, with prefix
            "_gold_positions.json".
        dMentionsPred (dict): dictionary of detected mentions but not yet linked mentions,
            for example:
                "sn83030483-1790-03-03-a-i0001_9": [
                    {
                        "mention": "Unitec ? States",
                        "start_offset": 38,
                        "end_offset": 40,
                        "start_char": 193,
                        "end_char": 206,
                        "ner_score": 0.79,
                        "ner_label": "LOC",
                        "entity_link": "O"
                    }
                ],
            ...this dictionary is stored in the 'outputs/data' folder, with prefix" _pred_mentions.json"
        dMentionsGold (dict): dictionary of gold standard mentions, analogous to the dictionary
            of detected mentions, but with the gold standard ner_label and entity_link.
    """
    gold_tokenization = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()
    dMentionsPred = dict()  # Dictionary of detected mentions
    dMentionsGold = dict()  # Dictionary of gold standard mentions
    for sent_id in tqdm(list(dSentences.keys())):
        sent = dSentences[sent_id]
        annotations = dAnnotated[sent_id]
        predictions = myner.ner_predict(sent)
        gold_positions = align_gold(predictions, annotations)
        sentence_postprocessing = postprocess_predictions(predictions, gold_positions)
        dPreds[sent_id] = sentence_postprocessing["sentence_preds"]
        dTrues[sent_id] = sentence_postprocessing["sentence_trues"]
        dSkys[sent_id] = sentence_postprocessing["sentence_skys"]
        gold_tokenization[sent_id] = gold_positions
        dMentionsPred[sent_id] = ner.aggregate_mentions(
            sentence_postprocessing["sentence_preds"], "pred"
        )
        dMentionsGold[sent_id] = ner.aggregate_mentions(
            sentence_postprocessing["sentence_trues"], "gold"
        )

    return dPreds, dTrues, dSkys, gold_tokenization, dMentionsPred, dMentionsGold


# ----------------------------------------------------
def update_with_linking(ner_predictions, link_predictions):
    """
    Updates the NER predictions with linking results.

    Arguments:
        ner_predictions (dict): dictionary with NER predictions (token-per-token)
            for a given sentence.
        link_predictions (pd.Series): a pandas series, corresponding to one
            row of the test_df, corresponding to one mention.

    Returns:
        resulting_preds (dict): a dictionary like ner_predictions, only with
            the added link to wikidata.
    """
    resulting_preds = ner_predictions
    link_predictions = link_predictions.to_dict(orient="index")
    for lp in link_predictions:
        for x in range(
            link_predictions[lp]["token_start"], link_predictions[lp]["token_end"] + 1
        ):
            position_ner = resulting_preds[x][1][:2]
            resulting_preds[x][2] = position_ner + link_predictions[lp]["pred_wqid"]
    return resulting_preds


# ----------------------------------------------------
def update_with_skyline(ner_predictions, link_predictions):
    """
    Update NER predictions with linking results.

    Arguments:
        ner_predictions (dict): dictionary with NER predictions (token-per-token)
            for a given sentence.
        link_predictions (pd.Series): a pandas series, corresponding to one
            row of the test_df, corresponding to one mention.

    Returns:
        resulting_preds (dict): a dictionary like ner_predictions, only with
            the added skyline link to wikidata (i.e. the gold standard candidate
            if the candidate has been retrieved via candidate ranking).
    """
    resulting_preds = ner_predictions
    link_predictions = link_predictions.to_dict(orient="index")
    for lp in link_predictions:
        all_candidates = [
            list(link_predictions[lp]["candidates"][x]["Candidates"].keys())
            for x in link_predictions[lp]["candidates"]
        ]
        all_candidates = [item for sublist in all_candidates for item in sublist]
        for x in range(
            link_predictions[lp]["token_start"], link_predictions[lp]["token_end"] + 1
        ):
            position_ner = resulting_preds[x][1][:2]
            if link_predictions[lp]["gold_entity_link"] in all_candidates:
                resulting_preds[x][2] = (
                    position_ner + link_predictions[lp]["gold_entity_link"]
                )
            else:
                resulting_preds[x][2] = "O"
    return resulting_preds


# ----------------------------------------------------
def prepare_storing_links(processed_data, all_test, test_df):
    """
    Updates the processed data dictionaries (preds and skys) with the
    predicted links for "preds" and with the skyline for "skys" (where
    the skyline is "if the gold standard entity is among the candidates,
    choose that one", providing the skyline of the maximum we can possibly
    achieve with linking).

    Arguments:
        processed_data (dict): dictionary of all processed data.
        all_test (list): ids of articles in current data split used for testing.
        test_df (pd.DataFrame): dataframe with one-mention-per-row that will
            be used for testing in this current experiment.
    """
    for sent_id in processed_data["preds"]:
        article_id = sent_id.split("_")[0]
        # First: is sentence in the current dev/test set?
        if article_id in all_test:
            # Update predictions with linking:
            # >> Step 1. If there is no mention in the sentence, it will
            #            not be in the per-mention dataframe. However,
            #            it should still be in the results file.
            if not sent_id in test_df["sentence_id"].unique():
                processed_data["preds"][sent_id] = processed_data["preds"][sent_id]
            if not sent_id in test_df["sentence_id"].unique():
                processed_data["skys"][sent_id] = processed_data["skys"][sent_id]
            # >> Step 2: If there is a mention in the sentence, update the link:
            else:
                processed_data["preds"][sent_id] = update_with_linking(
                    processed_data["preds"][sent_id],  # NER predictions
                    test_df[test_df["sentence_id"] == sent_id],  # Processed df
                )
                processed_data["skys"][sent_id] = update_with_skyline(
                    processed_data["skys"][sent_id],  # NER predictions
                    test_df[test_df["sentence_id"] == sent_id],  # Processed df
                )
    return processed_data


# ----------------------------------------------------
def load_processed_data(experiment):
    """
    Loads the data already processed in a previous run of the code, using
    the same parameters.

    Arguments:
        experiment (preparation.Experiment): an Experiment object.

    Returns:
        output_processed_data (dict): a dictionary where the already
            processed data is stored.
    """

    output_path = os.path.join(
        experiment.data_path, experiment.dataset, experiment.myner.model_name
    )

    # Add the candidate experiment info to the path:
    cand_approach = experiment.myranker.method
    if experiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["selection_threshold"]
        )

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
    experiment,
    preds,
    trues,
    skys,
    gold_tok,
    dSentences,
    dMetadata,
    dMentionsPred,
    dMentionsGold,
    dCandidates,
):
    """
    This function stores all the postprocessed data as jsons.

    Arguments:
        experiment (preparation.Experiment): the experiment object.
        preds (dict): dictionary of tokens with predictions, per sentence.
        trues (dict): dictionary of tokens with gold standard annotations, per sentence.
        skys (dict): dictionary of tokens which will keep the skyline, per sentence.
        gold_tok (dict): dictionary of tokens with gold standard annotations
            as dictionaries, per sentence.
        dSentences (dict): dictionary that maps a sentence id with the text.
        dMetadata (dict): dictionary that maps a sentence id with the associated metadata.
        dMentionsPred (dict): dictionary of predicted mentions, per sentence.
        dMentionsGold (dict): dictionary of gold standard mentions, per sentence.
        dCandidates (dict): dictionary of candidates, per mention in sentence.

    Returns:
        dict_processed_data (dict): dictionary of dictionaries, keeping
            all processed data (predictions, REL, gold standard, candidates)
            in one place.
        Also returns one .json file per dictionary, stored in 'outputs/data'.
    """
    data_path = experiment.data_path
    dataset = experiment.dataset
    model_name = experiment.myner.model_name
    output_path = data_path + dataset + "/" + model_name

    cand_approach = experiment.myranker.method
    if experiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["selection_threshold"]
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
    dict_processed_data["dMentionsPred"] = dMentionsPred
    dict_processed_data["dMentionsGold"] = dMentionsGold
    dict_processed_data["dCandidates"] = dCandidates
    return dict_processed_data


# ----------------------------------------------------
def rel_end_to_end(sent):
    """
    REL end-to-end using the API.

    Arguments:
        sent (str): a sentence in plain text.

    Returns:
        el_result (dict): the output from REL end-to-end API
            for the input sentence.
    """
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


# ----------------------------------------------------
def rel_process_results(
    mentions_dataset,
    predictions,
    processed,
    include_offset=False,
):
    """
    Function that can be used to process the End-to-End results.
    :return: dictionary with results and document as key.
    """
    res = {}
    for doc in mentions_dataset:
        if doc not in predictions:
            # No mentions found, we return empty list.
            continue
        pred_doc = predictions[doc]
        ment_doc = mentions_dataset[doc]
        text = processed[doc][0]
        res_doc = []

        for pred, ment in zip(pred_doc, ment_doc):
            sent = ment["sentence"]
            idx = ment["sent_idx"]
            start_pos = ment["pos"]
            mention_length = int(ment["end_pos"] - ment["pos"])

            if pred["prediction"] != "NIL":
                temp = (
                    start_pos,
                    mention_length,
                    ment["ngram"],
                    pred["prediction"],
                    pred["conf_ed"],
                    ment["conf_md"] if "conf_md" in ment else 0.0,
                    ment["tag"] if "tag" in ment else "NULL",
                )
                res_doc.append(temp)
        res[doc] = res_doc
    return res


# ----------------------------------------------------
def get_rel_locally(dSentences, mention_detection, tagger_ner, linking_model):
    """
    Uses the REL API to do end-to-end entity linking.

    Arguments:
        dSentences (dict): dictionary of sentences, where the
            key is the article-sent identifier and the value
            is the full text of the sentence.
        XXXX
        XXXX

    Returns:
        A JSON file with the REL results.
    """
    # Dictionary to store REL predictions:
    print("\nObtain REL linking locally\n")

    def rel_sentence_preprocessing(s, sentence):
        text = sentence
        processed = {s: [text, []]}
        return processed

    dREL = dict()
    for s in tqdm(dSentences):
        input_text = rel_sentence_preprocessing(s, dSentences[s])
        mentions_dataset, n_mentions = mention_detection.find_mentions(
            input_text, tagger_ner
        )
        predictions, timing = linking_model.predict(mentions_dataset)
        result = rel_process_results(mentions_dataset, predictions, input_text)
        for k in result:
            dREL[k] = result[k]
    return dREL


# ----------------------------------------------------
def match_wikipedia_to_wikidata(wiki_title):
    """
    Get the Wikidata ID from a Wikipedia title.

    Arguments:
        wiki_title (str): a Wikipedia title, underscore-separated.

    Returns:
        a string, either the Wikidata QID corresponding entity, or NIL.
    """
    el = urllib.parse.quote(wiki_title.replace("_", " "))
    try:
        el = wikipedia2wikidata[el]
        return el
    except Exception:
        # to be checked but it seems some Wikipedia pages are not in our Wikidata
        # see for instance Zante%2C%20California
        return "NIL"


# ----------------------------------------------------
def match_ent(pred_ents, start, end, prev_ann):
    """
    Function that, given the position in a sentence of a
    specific gold standard token, finds the corresponding
    string and prediction information returned by REL.

    Arguments:
        pred_ents (list): a list of lists, each inner list
            corresponds to a token.
        start (int): start character of a token in the gold standard.
        end (int): end character of a token in the gold standard.
        prev_ann (str): entity type of the previous token.

    Returns:
        A tuple with three elements: (1) the entity type, (2) the
        entity link and (3) the entity type of the previous token.
    """
    for ent in pred_ents:
        wqid = match_wikipedia_to_wikidata(ent[3])
        # If entity is a LOC or linked entity is in our KB:
        if ent[-1] == "LOC" or wqid in gazetteer_ids:
            # Any place with coordinates is considered a location
            # throughout our experiments:
            ent_type = "LOC"
            st_ent = ent[0]
            len_ent = ent[1]
            if start >= st_ent and end <= (st_ent + len_ent):
                if prev_ann == ent_type:
                    ent_pos = "I-"
                else:
                    ent_pos = "B-"
                    prev_ann = ent_type

                n = ent_pos + ent_type
                try:
                    el = ent_pos + match_wikipedia_to_wikidata(ent[3])
                except Exception as e:
                    print(e)
                    # to be checked but it seems some Wikipedia pages are not in our Wikidata
                    # see for instance Zante%2C%20California
                    return n, "O", ""
                return n, el, prev_ann
    return "O", "O", ""


# ----------------------------------------------------
def postprocess_rel(rel_preds, dSentences, gold_tokenization):
    """
    For each sentence, retokenizes the REL output to match the gold
    standard tokenization.

    Arguments:
        XXX
        dSentences (dict): dictionary that maps a sentence id to the text.
        gold_tokenization (dict): dictionary that contains the tokenized
            sentence with gold standard annotations of entity type and
            link, per sentence.

    Returns:
        dREL (dict): dictionary that maps a sentence id with the REL predictions,
            retokenized as in the gold standard.
    """
    dREL = dict()
    for sent_id in tqdm(list(dSentences.keys())):
        sentence_preds = []
        prev_ann = ""
        for token in gold_tokenization[sent_id]:
            start = token["start"]
            end = token["end"]
            word = token["word"]
            current_preds = rel_preds.get(sent_id, [])
            n, el, prev_ann = match_ent(current_preds, start, end, prev_ann)
            sentence_preds.append([word, n, el])
        dREL[sent_id] = sentence_preds
    return dREL


# ----------------------------------------------------
def create_mentions_df(experiment):
    """
    Create a dataframe for the linking experiment, with one
    mention per row.

    Arguments:
        experiment (preparation.Experiment): the current experiment.

    Returns:
        processed_df (pd.DataFrame): a dataframe with one mention
            per row, and containing all relevant information for subsequent
            steps (i.e. for linking).
        It also returns a .tsv file in the "outputs/data/[dataset]/" folder.
    """
    dMentions = experiment.processed_data["dMentionsPred"]
    dGoldSt = experiment.processed_data["dMentionsGold"]
    dSentences = experiment.processed_data["dSentences"]
    dMetadata = experiment.processed_data["dMetadata"]
    dCandidates = experiment.processed_data["dCandidates"]

    cand_approach = experiment.myranker.method
    if experiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["selection_threshold"]
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
        experiment.data_path
        + experiment.dataset
        + "/"
        + experiment.myner.model_name
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
    df = experiment.dataset_df[
        [c for c in keep_columns if c in experiment.dataset_df.columns]
    ]

    # Convert article_id to string (it's read as an int):
    df = df.assign(article_id=lambda d: d["article_id"].astype(str))
    processed_df = processed_df.assign(article_id=lambda d: d["article_id"].astype(str))
    processed_df = pd.merge(processed_df, df, on=["article_id"], how="left")

    # Store mentions dataframe:
    processed_df.to_csv(output_path + "_mentions.tsv", sep="\t")

    return processed_df


# ----------------------------------------------------
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

    Argument:
        hipe_scorer_results_path (str): first part of the path of the output path.
        scenario_name (str): second part of the path of the output file.
        dresults (dict): dictionary with the results.
        articles_test (list): list of sentences that are part of the
            split we're using for evaluating the performance on this
            particular experiment.

    Returns:
        A tsv with the results in the Conll format required by the scorer.
    """
    # Bundle 2 associated tasks: NERC-coarse and NEL
    with open(
        os.path.join(hipe_scorer_results_path, scenario_name + ".tsv"),
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


# ----------------------------------------------------
def store_results(experiment, task, how_split, which_split):
    """
    Function which stores the results of an experiment in the format
    required by the HIPE 2020 evaluation scorer.

    Arguments:
        experiment (preparation.Experiment): the current experiment.
        task (str): either "ner" or "linking". Store the results for just
            ner or with links as well.
        how_split (str): which way of splitting the data are we using?
            It could be the "originalsplit" or "Ashton1860", for example,
            which would mean that "Ashton1860" is left out for test only.
        which_split (str): on which split are we testing our experiments?
            It can be "dev" while we're developing the code, or "test"
            when we run it in the final experiments.
    """
    hipe_scorer_results_path = os.path.join(experiment.results_path, experiment.dataset)
    Path(hipe_scorer_results_path).mkdir(parents=True, exist_ok=True)
    scenario_name = task + "_" + experiment.myner.model_name + "_"
    if task == "linking":
        cand_approach = experiment.myranker.method
        if experiment.myranker.method == "deezymatch":
            cand_approach += "+" + str(
                experiment.myranker.deezy_parameters["num_candidates"]
            )
            cand_approach += "+" + str(
                experiment.myranker.deezy_parameters["selection_threshold"]
            )
        scenario_name += cand_approach + "_" + how_split + "-" + which_split + "_"

    # Find article ids of the corresponding test set (e.g. 'dev' of the original split,
    # 'test' of the Ashton1860 split, etc):
    all = experiment.dataset_df
    test_articles = list(all[all[how_split] == which_split].article_id.unique())
    test_articles = [str(art) for art in test_articles]

    # Store predictions results formatted for CLEF-HIPE scorer:
    preds_name = experiment.mylinker.method if task == "linking" else "preds"
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + preds_name,
        experiment.processed_data["preds"],
        test_articles,
    )

    # Store gold standard results formatted for CLEF-HIPE scorer:
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + "trues",
        experiment.processed_data["trues"],
        test_articles,
    )

    if task == "linking":
        # If task is "linking", store the skyline results:
        store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + "skys",
            experiment.processed_data["skys"],
            test_articles,
        )


def store_rel(experiment, dREL, approach, how_split, which_split):
    hipe_scorer_results_path = os.path.join(experiment.results_path, experiment.dataset)
    scenario_name = (
        approach
        + "_"
        + experiment.myner.model_name  # The model name is needed due to tokenization
        + "_"
        + how_split
        + "-"
        + which_split
    )

    # Find article ids of the corresponding test set (e.g. 'dev' of the original split,
    # 'test' of the Ashton1860 split, etc):
    all = experiment.dataset_df
    test_articles = list(all[all[how_split] == which_split].article_id.unique())
    test_articles = [str(art) for art in test_articles]

    # Store REL results formatted for CLEF-HIPE scorer:
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name,
        dREL,
        test_articles,
    )
