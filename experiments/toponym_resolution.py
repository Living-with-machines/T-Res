import json
import os
import sys
from pathlib import Path

from pandarallel import pandarallel

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import pandas as pd
import tqdm
from resolution_pipeline import ELPipeline
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils import process_data, ner, ranking, linking


# -----------------------------------------------------
# BEGINNING OF USER INPUT

# Datasets:
datasets = ["lwm", "hipe"]

# Named entity recognition approach, options are:
# * rel
# * lwm
ner_model_id = "lwm"

# Candidate selection approach, options are:
# * perfectmatch
# * partialmatch
# * levenshtein
# * deezymatch
cand_select_method = "deezymatch"

# Toponym resolution approach, options are:
# * mostpopular
# * mostpopularnormalised
top_res_method = "mostpopular"

# Perform training if needed:
do_training = False

# Entities considered for linking, options are:
# * all
# * loc
accepted_labels_str = "all"

# Initiate the ranker object:
myranker = ranking.Ranker(
    method=cand_select_method,
    resources_path="/resources/wikidata/",
    mentions_to_wikidata=dict(),
    deezy_parameters={
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/deezymatch/",
        "dm_cands": "wkdtalts",
        "dm_model": "ocr_avgpool",
        "dm_output": "deezymatch_on_the_fly",
        # Ranking measures:
        "ranking_metric": "faiss",
        "selection_threshold": 10,
        "num_candidates": 3,
        "search_size": 3,
        "use_predict": False,
        "verbose": False,
    },
)

# Initiate the linker object:
mylinker = linking.Linker(
    method=top_res_method,
    accepted_labels=accepted_labels_str,
    do_training=do_training,
    training_csv="/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/data/lwm/linking_df_train.tsv",
    resources_path="/resources/wikidata/",
    linking_resources=dict(),
    myranker=myranker,
)

# END OF USER INPUT
# -----------------------------------------------------


# Load the ranker and linker resources:
print("*** Loading the resources...")
myranker.mentions_to_wikidata = myranker.load_resources()
mylinker.linking_resources = mylinker.load_resources()
print("*** Resources loaded!\n")

# Parallelize if ranking method is one of the following:
if myranker.method in ["partialmatch", "levenshtein"]:
    pandarallel.initialize(nb_workers=10)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Some methods are unsupervised, set the do_training flag to False:
if mylinker.method in ["rel", "mostpopular", "mostpopularnormalised"]:
    mylinker.do_training = False

# Instantiate gold tokenization dictionary:
gold_tokenisation = {}


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
        "outputs/results/"
        + dataset
        + "/lwm_gold_tokenisation_"
        + accepted_labels_str
        + ".json"
    )

    # Path where to store REL API output:
    rel_end_to_end = "outputs/results/" + dataset + "/rel_end_to_end.json"

    if ner_model_id == "lwm":
        # Path to NER Model:
        ner_model = "outputs/models/" + ner_model_id + "-ner.model"
        ner_pipe = pipeline("ner", model=ner_model)

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
        myranker.method = "rel"
        mylinker.method = "rel"

    dAnnotated, dSentences, dMetadata = ner.format_for_ner(dev)

    true_mentions_sents = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()

    # Print the contents fo the ranker and linker objects:
    print(myranker)
    print(mylinker)

    # Instantiate the entity linking pipeline:
    end_to_end = ELPipeline(
        ner_model_id=ner_model_id,
        myranker=myranker,
        mylinker=mylinker,
        ner_pipe=ner_pipe,
        dataset=dataset,
    )

    for sent_id in tqdm.tqdm(dSentences.keys()):

        if end_to_end.ner_model_id == "rel":
            if Path(rel_end_to_end).is_file():
                preds = rel_preds[sent_id]
            else:
                pred_ents = linking.rel_end_to_end(dSentences[sent_id])
                rel_preds[sent_id] = pred_ents
                with open(rel_end_to_end, "w") as fp:
                    json.dump(rel_preds, fp)
            output = end_to_end.run(
                dSentences[sent_id],
                dataset=dataset,
                annotations=rel_preds[sent_id],
                gold_positions=gold_standard[sent_id],
            )

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

    if end_to_end.ner_model_id == "lwm":
        process_data.store_results_hipe(dataset, "true_" + accepted_labels_str, dTrues)
        process_data.store_results_hipe(
            dataset,
            "skyline:"
            + end_to_end.ner_model_id
            + "+"
            + myranker.method
            + "+"
            + accepted_labels_str,
            dSkys,
        )
        with open(gold_path, "w") as fp:
            json.dump(gold_tokenisation, fp)

    process_data.store_results_hipe(
        dataset,
        end_to_end.ner_model_id
        + "+"
        + myranker.method
        + "+"
        + mylinker.method
        + "+"
        + accepted_labels_str,
        dPreds,
    )
