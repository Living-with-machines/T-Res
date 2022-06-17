import os
import sys
import pandas as pd

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import preparation, recogniser, ranking, linking

dataset = "lwm"  # "hipe" or "lwm"

# Candidate selection approach, options are:
# * perfectmatch
# * partialmatch
# * levenshtein
# * deezymatch
cand_select_method = "deezymatch"

# Toponym resolution approach, options are:
# * mostpopular
# * contextualized
# * reldisamb:relcs # REL disambiguation with their candidates (our mentions)
# * reldisamb:lwmcs # REL disambiguation with our candidates (our mentions)
# top_res_method = "mostpopular"
# top_res_method = "reldisamb:lwmcs:relv"
# top_res_method = "reldisamb:relcs"
# top_res_method = "reldisamb:lwmcs:dist"
# top_res_method = "reldisamb:lwmcs:relvdist"
top_res_method = "gnn"


# --------------------------------------
# Instantiate the recogniser:
myner = recogniser.Recogniser(
    method="lwm",  # NER method
    model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
    model=None,  # We'll store the NER model here
    pipe=None,  # We'll store the NER pipeline here
    base_model="/resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
    train_dataset="outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
    test_dataset="outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
    output_model_path="outputs/models/",  # Path where the NER model is or will be stored
    training_args={
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
    },
    overwrite_training=False,  # Set to True if you want to overwrite model if existing
    do_test=False,  # Set to True if you want to train on test mode
    training_tagset="fine",  # Options are: "coarse" or "fine"
)


# --------------------------------------
# Instantiate the ranker:
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
        "num_candidates": 2,
        "search_size": 2,
        "use_predict": False,
        "verbose": False,
    },
)


# --------------------------------------
# Instantiate the linker:
mylinker = linking.Linker(
    method=top_res_method,
    resources_path="/resources/wikidata/",
    linking_resources=dict(),
    base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
    rel_params={"base_path": "/resources/rel_db/", "wiki_version": "wiki_2019/"},
    gnn_params={
        "level": "sentence_id",
        "max_distance": 200,
        "similarity_threshold": 0.7,
        "model_path": "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/gnn_models/",
    },
    overwrite_training=False,
)


# --------------------------------------
# Instantiate the experiment:
experiment = preparation.Experiment(
    dataset=dataset,
    data_path="outputs/data/",
    dataset_df=pd.DataFrame(),
    results_path="outputs/results/",
    myner=myner,
    myranker=myranker,
    mylinker=mylinker,
    overwrite_processing=False,  # If True, do data processing, else load existing processing, if exists.
    processed_data=dict(),  # Dictionary where we'll keep the processed data for the experiments.
    test_split="test",  # "dev" while experimenting, "test" when running final experiments.
    rel_experiments=False,  # False if we're not interested in running the different experiments with REL, True otherwise.
)

# Print experiment information:
print(experiment)
print(myner)
print(myranker)
print(mylinker)

# Load processed data if existing:
experiment.processed_data = experiment.load_data()

# Perform data postprocessing:
experiment.processed_data = experiment.prepare_data()

# Linker load resources:
print("\n* Load linking resources...")
mylinker.linking_resources = mylinker.load_resources()
print("... resources loaded, linking in progress!\n")

# Do the linking experiments:
experiment.linking_experiments()
