import os
import sqlite3
import sys
from argparse import ArgumentParser
from pathlib import Path

import experiment
import pandas as pd

from t_res.geoparser import linking, ranking, recogniser

parser = ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    dest="path",
    help="path to resources directory",
    action="store",
    type=str,
)

args = parser.parse_args()

resources_dir = args.path
current_dir = Path(__file__).parent.resolve()

# Choose test scenario:
# * "dev" while developing and experimenting,
# * "test" for the final numbers,
# * "apply" to train a model using all the data.
test_scenario = "apply"

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
    # ["hipe", "perfectmatch", "mostpopular", "fine", "", ""],
    # ["hipe", "perfectmatch", "bydistance", "fine", "", ""],
    # ["hipe", "deezymatch", "mostpopular", "fine", "", ""],
    # ["hipe", "deezymatch", "bydistance", "fine", "", ""],
    # ["hipe", "deezymatch", "reldisamb", "fine", False, False],
    # ["hipe", "deezymatch", "reldisamb", "fine", True, False],
    # ["hipe", "deezymatch", "reldisamb", "fine", False, True],
    # ["hipe", "deezymatch", "reldisamb", "fine", True, True],
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
    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-" + granularity,
        train_dataset=str(current_dir)
        + "/outputs/data/lwm/ner_"
        + granularity
        + "_train.json",  # Path to the json file containing the training set (see note above).
        test_dataset=str(current_dir)
        + "/outputs/data/lwm/ner_"
        + granularity
        + "_dev.json",  # Path to the json file containing the test set (see note above).
        pipe=None,  # We'll store the NER pipeline here, leave this empty.
        base_model="Livingwithmachines/bert_1760_1900",  # Base model to fine-tune for NER. The value can be: either
        # your local path to a model or the huggingface path.
        # In this case, we use the huggingface path:
        # https://huggingface.co/Livingwithmachines/bert_1760_1900). You can
        # chose any other model from the HuggingFace hub, as long as it's
        # trained on the "Fill-Mask" objective (filter by task).
        model_path=os.path.join(
            resources_dir, "models/"
        ),  # Path where the NER model will be stored.
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },  # Training arguments: you can change them. These are selected based on: https://github.com/dbmdz/clef-hipe/tree/main/experiments/clef-hipe-2022#topres19th
        overwrite_training=False,  # Set to True if you want to overwrite an existing model with the same name.
        do_test=False,  # Set to True if you want to perform the training on test mode (the string "_test" will be appended to your model name).
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.Ranker(
        method=cand_select_method,
        resources_path=resources_dir,
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": os.path.join(resources_dir, "models/w2v/"),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(resources_dir, "deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    # --------------------------------------
    # Instantiate the linker:
    with sqlite3.connect(
        os.path.join(resources_dir, "rel_db/embeddings_database.db")
    ) as conn:
        cursor = conn.cursor()
        linker = linking.Linker(
            method=top_res_method,
            resources_path=resources_dir,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(resources_dir, "models/disambiguation/"),
                "data_path": os.path.join(current_dir, "outputs/data/lwm/"),
                "training_split": "",
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
        data_path=os.path.join(current_dir, "outputs/data/"),
        dataset_df=pd.DataFrame(),
        results_path=os.path.join(current_dir, "outputs/results/"),
        ner=ner,
        ranker=ranker,
        linker=linker,
        overwrite_processing=False,  # If True, do data processing, else load existing processing, if exists.
        processed_data=dict(),  # Dictionary where we'll keep the processed data for the experiments.
        test_split=test_scenario,  # "dev" while experimenting, "test" when running final experiments.
        rel_experiments=False,  # False if we're not interested in running the different experiments with REL, True otherwise.
        end_to_end_eval=False,  # False if we're not evaluating end-to-end for EL, True if we're evaluating "EL-only"
    )

    # Print experiment information:
    print(myexperiment)
    print(ner)
    print(ranker)
    print(linker)

    # -----------------------------------------
    # NER training and creating pipeline:
    # Train the NER models if needed:
    ner.train()
    # Load the NER pipeline:
    ner.pipe = ner.create_pipeline()

    # -----------------------------------------
    # Ranker loading resources and training a model:
    # Load the resources (and train a DeezyMatch model if needed):
    ranker.mentions_to_wikidata = ranker.load_resources()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    linker.load_resources()

    # -----------------------------------------
    # Prepare experiment:
    # Load processed data if existing:
    myexperiment.processed_data = myexperiment.load_data()

    # Perform data postprocessing:
    myexperiment.processed_data = myexperiment.prepare_data()

    # Do the linking experiments:
    myexperiment.linking_experiments()
