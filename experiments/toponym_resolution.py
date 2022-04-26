import json
import os
import sys
from pathlib import Path

from pandarallel import pandarallel

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))

import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils import candidate_selection, linking, ner, process_data, training

# ---------------------------------------------------
# End-to-end toponym resolution parameters
# ---------------------------------------------------

# Datasets:
datasets = ["lwm", "hipe"]

# Approach:
ner_model_id = "lwm"  # lwm or rel
cand_select_method = "deezymatch"  # either perfectmatch, partialmatch, levenshtein or deezymatch
top_res_method = "featclassifier"  # either mostpopular, mostpopularnormalised, or featclassifier
do_training = True  # some resolution methods will need training
accepted_labels_str = "loc"  # entities considered for linking: all or loc

if ner_model_id == "rel" or top_res_method in ["mostpopular", "mostpopularnormalised"]:
    do_training = False

if cand_select_method in ["partialmatch", "levenshtein"]:
    pandarallel.initialize(nb_workers=10)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Entity types considered for linking (lower-cased):
accepted_labels = dict()
accepted_labels["all"] = [
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
accepted_labels["loc"] = ["loc", "b-loc", "i-loc"]

# Ranking parameters for DeezyMatch:
myranker = dict()
if ner_model_id == "lwm" and cand_select_method == "deezymatch":
    myranker["ranking_metric"] = "faiss"
    myranker["selection_threshold"] = 10
    myranker["num_candidates"] = 3
    myranker["search_size"] = 3
    # Path to DeezyMatch model and combined candidate vectors:
    myranker["dm_path"] = "outputs/deezymatch/"
    myranker["dm_cands"] = "wkdtalts"
    myranker["dm_model"] = "ocr_avgpool"
    myranker["dm_output"] = "deezymatch_on_the_fly"

# Instantiate a dictionary to keep linking parameters:
mylinker = dict()

# Instantiate a dictionary to collect candidates:
already_collected_cands = {}  # to speed up candidate selection methods


# ---------------------------------------------------
# Create entity linking training data
# ---------------------------------------------------
training_df = pd.DataFrame()
if do_training:
    training_path = "outputs/data/lwm/linking_df_train.tsv"
    # Create a dataset for entity linking, with candidates:
    training_df = training.create_trainset(
        training_path, cand_select_method, myranker, already_collected_cands
    )
    # Add linking columns (geotype and geoscope)
    training_df = training.add_linking_columns(training_path, training_df)
    # Train a mention to geotype classifier:
    model2type = training.mention2type_classifier(training_df)
    mylinker["model2type"] = model2type
    # Train a mention to geoscope classifier:
    model2scope = training.mention2scope_classifier(training_df)
    mylinker["model2scope"] = model2scope


# ---------------------------------------------------
# End-to-end toponym resolution
# ---------------------------------------------------

for dataset in datasets:

    # Path to dev dataframe:
    dev = pd.read_csv(
        "outputs/data/" + dataset + "/linking_df_dev.tsv",
        sep="\t",
    )

    # Path where to store gold tokenization:
    gold_path = (
        "outputs/results/" + dataset + "/lwm_gold_tokenisation_" + accepted_labels_str + ".json"
    )

    # Path where to store REL API output:
    rel_end_to_end = "outputs/results/" + dataset + "/rel_end_to_end.json"

    if ner_model_id == "lwm":
        # Path to NER Model:
        ner_model = "outputs/models/" + ner_model_id + "-ner.model"
        ner_pipe = pipeline("ner", model=ner_model)
        gold_tokenisation = {}

    if ner_model_id == "rel":
        if Path(rel_end_to_end).is_file():
            with open(rel_end_to_end) as f:
                rel_preds = json.load(f)
        else:
            rel_preds = {}

        gold_standard = process_data.read_gold_standard(gold_path)
        # currently it's all based on REL
        # but we could for instance use our pipeline and rely on REL only for disambiguation
        cand_select_method = "rel"
        top_res_method = "rel"

    dAnnotated, dSentences, dMetadata = ner.format_for_ner(dev)

    true_mentions_sents = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()

    for sent_id in tqdm.tqdm(dSentences.keys()):

        if ner_model_id == "rel":
            if Path(rel_end_to_end).is_file():
                pred_ents = rel_preds[sent_id]
            else:
                pred_ents = linking.rel_end_to_end(dSentences[sent_id])
                rel_preds[sent_id] = pred_ents

            sentence_preds = []
            prev_ann = ""
            for token in gold_standard[sent_id]:
                start = token["start"]
                end = token["end"]
                word = token["word"]
                n, el, prev_ann = process_data.match_ent(pred_ents, start, end, prev_ann)
                sentence_preds.append([word, n, el])

            dPreds[sent_id] = sentence_preds

        else:
            # Toponym recognition
            gold_standard, predictions = ner.ner_predict(
                dSentences[sent_id], dAnnotated[sent_id], ner_pipe, dataset
            )
            gold_tokenisation[sent_id] = gold_standard
            sentence_preds = [
                [x["word"], x["entity"], "O", x["start"], x["end"]] for x in predictions
            ]
            sentence_trues = [
                [x["word"], x["entity"], x["link"], x["start"], x["end"]] for x in gold_standard
            ]
            sentence_skys = [
                [x["word"], x["entity"], "O", x["start"], x["end"]] for x in gold_standard
            ]

            # Filter by accepted labels:
            sentence_trues = [
                [x[0], x[1], "NIL", x[3], x[4]]
                if x[1] != "O" and x[1].lower() not in accepted_labels[accepted_labels_str]
                else x
                for x in sentence_trues
            ]

            pred_mentions_sent = ner.aggregate_mentions(
                sentence_preds, accepted_labels[accepted_labels_str]
            )
            true_mentions_sent = ner.aggregate_mentions(
                sentence_trues, accepted_labels[accepted_labels_str]
            )
            # Candidate selection
            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands, already_collected_cands = candidate_selection.select(
                mentions, cand_select_method, myranker, already_collected_cands
            )

            # # Toponym resolution
            for mention in pred_mentions_sent:
                text_mention = mention["mention"]
                start_offset = mention["start_offset"]
                end_offset = mention["end_offset"]
                start_char = mention["start_char"]
                end_char = mention["end_char"]

                mention_context = dict()
                mention_context["sentence"] = dSentences[sent_id]
                mention_context["mention"] = text_mention
                mention_context["mention_start"] = start_char
                mention_context["mention_end"] = end_char
                mylinker["mention_context"] = mention_context
                mylinker["metadata"] = dMetadata[sent_id]

                # TO DO: FIND CORRECT PLACE OF PUBLICATION FOR HIPE:
                if dataset == "hipe":
                    mylinker["metadata"]["place"] = "New York"

                # to be extended so that it can include multiple features
                res = linking.select(cands[text_mention], top_res_method, mylinker)
                if res:
                    link, score, other_cands = res
                    for x in range(start_offset, end_offset + 1):
                        position_ner = sentence_preds[x][1][:2]
                        sentence_preds[x][2] = position_ner + link
                        sentence_preds[x].append(other_cands)
                        true_label = sentence_trues[x][2].split("-")[-1]
                        if true_label in other_cands:
                            sentence_skys[x][2] = sentence_trues[x][2]

            dPreds[sent_id] = sentence_preds
            dTrues[sent_id] = sentence_trues
            dSkys[sent_id] = sentence_skys

    if ner_model_id == "lwm":
        process_data.store_results_hipe(dataset, "true_" + accepted_labels_str, dTrues)
        process_data.store_results_hipe(
            dataset,
            "skyline:"
            + ner_model_id
            + "+"
            + cand_select_method
            + str(myranker.get("num_candidates", ""))
            + "+"
            + accepted_labels_str,
            dSkys,
        )
        with open(gold_path, "w") as fp:
            json.dump(gold_tokenisation, fp)

    if ner_model_id == "rel":
        if not Path(rel_end_to_end).is_file():
            with open(rel_end_to_end, "w") as fp:
                json.dump(rel_preds, fp)

    process_data.store_results_hipe(
        dataset,
        ner_model_id
        + "+"
        + cand_select_method
        + str(myranker.get("num_candidates", ""))
        + "+"
        + top_res_method
        + "+"
        + accepted_labels_str,
        dPreds,
    )
