import os
import sys
import pandas as pd
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import experiment, recogniser, ranking, linking

# Choose test scenario:
test_scenario = "dev"  # "dev" while experimenting, "test" for the final numbers

# List of experiments:
experiments = [
    ["lwm", "perfectmatch", "mostpopular", "fine", "", ""],
    # ["lwm", "deezymatch", "mostpopular", "fine", "", ""],
    # ["lwm", "perfectmatch", "bydistance", "fine", "", ""],
    # ["lwm", "deezymatch", "bydistance", "fine", "", ""],
    # ["lwm", "relcs", "reldisamb", "fine", "relv", ""],
    # ["lwm", "deezymatch", "reldisamb", "fine", "relv", ""],
    # ["lwm", "deezymatch", "reldisamb", "fine", "relv", "dist"],
    # ["lwm", "deezymatch", "reldisamb", "fine", "relv", "nil"],
    # ["lwm", "deezymatch", "reldisamb", "fine", "publ", ""],
    # ["lwm", "deezymatch", "reldisamb", "fine", "publ", "dist"],
    # ["lwm", "deezymatch", "reldisamb", "fine", "publ", "nil"],
    # ["hipe", "perfectmatch", "mostpopular", "coarse", "", ""],
    # ["hipe", "deezymatch", "mostpopular", "coarse", "", ""],
    # ["hipe", "perfectmatch", "bydistance", "coarse", "", ""],
    # ["hipe", "deezymatch", "bydistance", "coarse", "", ""],
    # ["hipe", "relcs", "reldisamb", "coarse", "relv", ""],
    # ["hipe", "deezymatch", "reldisamb", "coarse", "relv", ""],
    # ["hipe", "deezymatch", "reldisamb", "coarse", "relv", "dist"],
    # ["hipe", "deezymatch", "reldisamb", "coarse", "relv", "nil"],
    # ["hipe", "deezymatch", "reldisamb", "coarse", "publ", ""],
    # ["hipe", "deezymatch", "reldisamb", "coarse", "publ", "dist"],
    # ["hipe", "deezymatch", "reldisamb", "coarse", "publ", "nil"],
    # ["hipe", "perfectmatch", "mostpopular", "fine", "", ""],
    # ["hipe", "deezymatch", "mostpopular", "fine", "", ""],
    # ["hipe", "perfectmatch", "bydistance", "fine", "", ""],
    # ["hipe", "deezymatch", "bydistance", "fine", "", ""],
    # ["hipe", "relcs", "reldisamb", "fine", "relv", ""],
    # ["hipe", "deezymatch", "reldisamb", "fine", "relv", ""],
    # ["hipe", "deezymatch", "reldisamb", "fine", "relv", "dist"],
    # ["hipe", "deezymatch", "reldisamb", "fine", "relv", "nil"],
    # ["hipe", "deezymatch", "reldisamb", "fine", "publ", ""],
    # ["hipe", "deezymatch", "reldisamb", "fine", "publ", "dist"],
    # ["hipe", "deezymatch", "reldisamb", "fine", "publ", "nil"],
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
    micro_locs = exp_param[5]

    # --------------------------------------
    # Instantiate the recogniser:
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-" + training_tagset,
        train_dataset="../experiments/outputs/data/lwm/ner_"
        + training_tagset
        + "_train.json",  # Path to the json file containing the training set (see note above).
        test_dataset="../experiments/outputs/data/lwm/ner_"
        + training_tagset
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
        method=cand_select_method,  # Here we're telling the ranker to use DeezyMatch.
        resources_path="../resources/wikidata/",  # Here, the path to the Wikidata resources.
        mentions_to_wikidata=dict(),  # We'll load the mentions-to-wikidata model here, leave it empty.
        wikidata_to_mentions=dict(),  # We'll load the wikidata-to-mentions model here, leave it empty.
        # Parameters to create the string pair dataset:
        strvar_parameters={
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("../resources/models/w2v/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        # Parameters to train, load and use a DeezyMatch model:
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(
                Path("../resources/deezymatch/").resolve()
            ),  # Path to the DeezyMatch directory where the model is saved.
            "dm_cands": "wkdtalts",  # Name we'll give to the folder that will contain the wikidata candidate vectors.
            "dm_model": "w2v_ocr",  # Name of the DeezyMatch model.
            "dm_output": "deezymatch_on_the_fly",  # Name of the file where the output of DeezyMatch will be stored. Feel free to change that.
            # Ranking measures:
            "ranking_metric": "faiss",  # Metric used by DeezyMatch to rank the candidates.
            "selection_threshold": 25,  # Threshold for that metric.
            "num_candidates": 3,  # Number of name variations for a string (e.g. "London", "Londra", and "Londres" are three different variations in our gazetteer of "Londcn").
            "search_size": 3,  # That should be the same as `num_candidates`.
            "verbose": False,  # Whether to see the DeezyMatch progress or not.
            # DeezyMatch training:
            "overwrite_training": False,  # You can choose to overwrite the model if it exists: in this case we're training a model, regardless of whether it already exists.
            "do_test": False,  # Whether the DeezyMatch model we're loading was a test, or not.
        },
    )

    # --------------------------------------
    # Instantiate the linker:
    mylinker = linking.Linker(
        method=top_res_method,
        resources_path="../resources/",
        linking_resources=dict(),
        rel_params=dict(),
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
        overwrite_processing=True,  # If True, do data processing, else load existing processing, if exists.
        processed_data=dict(),  # Dictionary where we'll keep the processed data for the experiments.
        test_split=test_scenario,  # "dev" while experimenting, "test" when running final experiments.
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
