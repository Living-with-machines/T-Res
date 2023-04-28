import os
import sys
import pandas as pd
from pathlib import Path
import sqlite3

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import recogniser, ranking, linking
from experiments import experiment

# Choose test scenario:
test_scenario = "test"  # "dev" while experimenting, "test" for the final numbers

# List of experiments:
experiments = [
    ["lwm", "perfectmatch", "mostpopular", "fine", "", ""],
    # ["lwm", "perfectmatch", "bydistance", "fine", "", ""],
    # ["lwm", "deezymatch", "mostpopular", "fine", "", ""],
    # ["lwm", "deezymatch", "bydistance", "fine", "", ""],
    # ["lwm", "deezymatch", "reldisamb", "fine", False, False],
    # ["lwm", "deezymatch", "reldisamb", "fine", True, False],
    # ["lwm", "deezymatch", "reldisamb", "fine", False, True],
    # ["lwm", "deezymatch", "reldisamb", "fine", True, True],
    # ["hipe", "perfectmatch", "mostpopular", "coarse", "", ""],
    # ["hipe", "perfectmatch", "bydistance", "coarse", "", ""],
    # ["hipe", "deezymatch", "mostpopular", "coarse", "", ""],
    # ["hipe", "deezymatch", "bydistance", "coarse", "", ""],
    # ["hipe", "deezymatch", "reldisamb", "coarse", False, False],
    # ["hipe", "deezymatch", "reldisamb", "coarse", True, False],
    # ["hipe", "perfectmatch", "mostpopular", "fine", "", ""],
    # ["hipe", "perfectmatch", "bydistance", "fine", "", ""],
    # ["hipe", "deezymatch", "mostpopular", "fine", "", ""],
    # ["hipe", "deezymatch", "bydistance", "fine", "", ""],
    # ["hipe", "deezymatch", "reldisamb", "fine", True, True],
    # ["hipe", "deezymatch", "reldisamb", "fine", True, False],
    # ["hipe", "deezymatch", "reldisamb", "fine", False, True],
    # ["hipe", "deezymatch", "reldisamb", "fine", False, False],
]

# Mapping experiment parameters:
for exp_param in experiments:
    print("============")
    print(exp_param)
    print("============")
    dataset = exp_param[0]
    cand_select_method = exp_param[1]
    top_res_method = exp_param[2]
    granularity = exp_param[3]
    wpubl = exp_param[4]
    wmtops = exp_param[5]

    # --------------------------------------
    # Instantiate the recogniser:
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-" + granularity,
        train_dataset="../experiments/outputs/data/lwm/ner_"
        + granularity
        + "_train.json",  # Path to the json file containing the training set (see note above).
        test_dataset="../experiments/outputs/data/lwm/ner_"
        + granularity
        + "_dev.json",  # Path to the json file containing the test set (see note above).
        pipe=None,  # We'll store the NER pipeline here, leave this empty.
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune for NER. The value can be: either
        # your local path to a model or the huggingface path.
        # In this case, we use the huggingface path:
        # https://huggingface.co/khosseini/bert_1760_1900). You can
        # chose any other model from the HuggingFace hub, as long as it's
        # trained on the "Fill-Mask" objective (filter by task).
        model_path="../resources/models/",  # Path where the NER model will be stored.
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },  # Training arguments: you can change them.
        overwrite_training=False,  # Set to True if you want to overwrite an existing model with the same name.
        do_test=False,  # Set to True if you want to perform the training on test mode (the string "_test" will be appended to your model name).
        load_from_hub=False,  # Whether the model should be loaded from the HuggingFace hub
    )

    # --------------------------------------
    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method=cand_select_method,
        resources_path="../resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("../resources/models/w2v/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("../resources/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 25,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    # --------------------------------------
    # Instantiate the linker:
    with sqlite3.connect("../resources/rel_db/embedding_database.db") as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method=top_res_method,
            resources_path="../resources/",
            linking_resources=dict(),
            rel_params={
                "model_path": "../resources/models/disambiguation/",
                "data_path": "../experiments/outputs/data/lwm/",
                "training_split": "originalsplit",
                "context_length": 100,
                "db_embeddings": cursor,
                "with_publication": wpubl,
                "without_microtoponyms": wmtops,
                "do_test": False,
                "default_publname": "",
                "default_publwqid": "",
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
        test_split=test_scenario,  # "dev" while experimenting, "test" when running final experiments.
        rel_experiments=False,  # False if we're not interested in running the different experiments with REL, True otherwise.
    )

    # Print experiment information:
    print(myexperiment)
    print(myner)
    print(myranker)
    print(mylinker)

    # -----------------------------------------
    # NER training and creating pipeline:
    # Train the NER models if needed:
    myner.train()
    # Load the NER pipeline:
    myner.pipe = myner.create_pipeline()

    # -----------------------------------------
    # Ranker loading resources and training a model:
    # Load the resources:
    myranker.mentions_to_wikidata = myranker.load_resources()
    # Train a DeezyMatch model if needed:
    myranker.train()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    mylinker.linking_resources = mylinker.load_resources()

    # -----------------------------------------
    # Prepare experiment:
    # Load processed data if existing:
    myexperiment.processed_data = myexperiment.load_data()

    # Perform data postprocessing:
    myexperiment.processed_data = myexperiment.prepare_data()

    # Do the linking experiments:
    myexperiment.linking_experiments()
