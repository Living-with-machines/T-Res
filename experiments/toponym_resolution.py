import os
import sys
import pandas as pd
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import experiment, recogniser, ranking, linking

# List of experiments:
experiments = [
    ["lwm", "perfectmatch", "mostpopular", "fine", "", ""],
    ["lwm", "perfectmatch", "mostpopular", "coarse", "", ""],
    ["hipe", "perfectmatch", "mostpopular", "coarse", "", ""],
    ["lwm", "deezymatch", "mostpopular", "fine", "", ""],
    ["hipe", "deezymatch", "mostpopular", "coarse", "", ""],
    ["lwm", "deezymatch", "bydistance", "fine", "", ""],
    ["hipe", "deezymatch", "bydistance", "coarse", "", ""],
    ["lwm", "relcs", "reldisamb", "fine", "relv", ""],
    ["lwm", "relcs", "reldisamb", "coarse", "relv", ""],
    ["hipe", "relcs", "reldisamb", "coarse", "relv", ""],
    ["lwm", "deezymatch", "reldisamb", "fine", "relv", ""],
    ["hipe", "deezymatch", "reldisamb", "coarse", "relv", ""],
    ["lwm", "deezymatch", "reldisamb", "fine", "publ", False],
    ["hipe", "deezymatch", "reldisamb", "coarse", "publ", False],
    ["lwm", "deezymatch", "reldisamb", "fine", "publ", True],
    ["hipe", "deezymatch", "reldisamb", "coarse", "publ", True],
]

# Mapping experiment parameters:
for exp_param in experiments:
    print("============")
    print(exp_param)
    print("============")
    dataset = exp_param[0]
    cand_select_method = exp_param[1]
    top_res_method = exp_param[2]
    training_tagset = exp_param[3]
    link_rank = exp_param[4]
    two_step = exp_param[5]

    # --------------------------------------
    # Instantiate the recogniser:
    myner = recogniser.Recogniser(
        model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
        model=None,  # We'll store the NER model here
        pipe=None,  # We'll store the NER pipeline here
        base_model="/resources/models/bert/bert_1760_1900/",  # Path to the base model to fine-tune
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
        training_tagset=training_tagset,  # Options are: "coarse" or "fine"
    )

    # --------------------------------------
    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method=cand_select_method,
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "minimum_relv": 0.005,  # Filter mentions with more than X relv
        },
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("outputs/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 25,
            "num_candidates": 3,
            "search_size": 3,
            "use_predict": False,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "w2v_ocr_path": str(Path("outputs/models/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "do_test": False,
        },
    )

    # --------------------------------------
    # Instantiate the linker:
    mylinker = linking.Linker(
        method=top_res_method,
        resources_path="/resources/wikidata/",
        linking_resources=dict(),
        base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
        rel_params={
            "base_path": "/resources/rel_db/",
            "wiki_version": "wiki_2019/",
            "training_data": "lwm",  # lwm, aida
            "ranking": link_rank,  # relv, dist, relvdist
            "two_step": two_step,
        },
        overwrite_training=False,
    )

    # --------------------------------------
    # Instantiate the experiment:
    myexperiment = experiment.Experiment(
        dataset=dataset,
        data_path="outputs/data/",
        dataset_df=pd.DataFrame(),
        results_path="outputs/results/",
        myner=myner,
        myranker=myranker,
        mylinker=mylinker,
        overwrite_processing=False,  # If True, do data processing, else load existing processing, if exists.
        processed_data=dict(),  # Dictionary where we'll keep the processed data for the experiments.
        test_split="dev",  # "dev" while experimenting, "test" when running final experiments.
        rel_experiments=False,  # False if we're not interested in running the different experiments with REL, True otherwise.
    )

    # Print experiment information:
    print(myexperiment)
    print(myner)
    print(myranker)
    print(mylinker)

    # Load processed data if existing:
    myexperiment.processed_data = myexperiment.load_data()

    # Perform data postprocessing:
    myexperiment.processed_data = myexperiment.prepare_data()

    # Do the linking experiments:
    myexperiment.linking_experiments()
