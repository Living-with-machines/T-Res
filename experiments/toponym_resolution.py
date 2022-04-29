import json
import os
import sys
from pathlib import Path

from pandarallel import pandarallel

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import pandas as pd
import tqdm
from pipeline import ELPipeline
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils import candidate_selection, linking, ner, process_data, training

# ---------------------------------------------------
# End-to-end toponym resolution parameters
# ---------------------------------------------------

# Datasets:
datasets = ["lwm", "hipe"]

# Approach:
ner_model_id = "rel"  # lwm or rel
cand_select_method = "rel"  # either perfectmatch, partialmatch, levenshtein or deezymatch
top_res_method = "rel"  # either mostpopular, mostpopularnormalised, or featclassifier
do_training = True  # some resolution methods will need training
accepted_labels_str = "all"  # entities considered for linking: all or loc

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
        ner_pipe = None
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

    end_to_end = ELPipeline(
        ner_model_id=ner_model_id,
        cand_select_method=cand_select_method,
        top_res_method=top_res_method,
        myranker=myranker,
        mylinker=mylinker,
        accepted_labels=accepted_labels[accepted_labels_str],
        ner_pipe=ner_pipe,
    )

    for sent_id in tqdm.tqdm(dSentences.keys()):

        if ner_model_id == "rel":
            if Path(rel_end_to_end).is_file():
                preds = rel_preds[sent_id]
            else:
                preds = end_to_end.run(dSentences[sent_id], gold_positions=gold_standard[sent_id])[
                    "sentence_preds"
                ]
                rel_preds[sent_id] = preds
        else:
            output = end_to_end.run(
                dSentences[sent_id],
                dataset=dataset,
                annotations=dAnnotated[sent_id],
                metadata=dMetadata[sent_id],
            )
            dPreds[sent_id] = output["sentence_preds"]
            dTrues[sent_id] = output["sentence_trues"]
            gold_tokenisation[sent_id] = output["gold_positions"]
            dSkys[sent_id] = output["skyline"]

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
