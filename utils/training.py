import os
import sys
import pandas as pd
from pathlib import Path
from ast import literal_eval

from utils.REL.entity_disambiguation import EntityDisambiguation

# from utils.REL.generate_train_test import GenTrainingTest
# from utils.REL.training_datasets import TrainingEvaluationDatasets
# from utils.REL.wikipedia import Wikipedia

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))


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


def load_training_lwm_data(myexperiment):
    # ------------------------------------------
    # Try to load the LwM dataset, otherwise raise a warning and create
    # an empty dataframe (we're not raising an error because not all
    # linking approaches will need a training set).
    lwm_processed_df = pd.DataFrame()

    # Load training set (add the candidate experiment info to the path):
    cand_approach = myexperiment.myranker.method
    if myexperiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            myexperiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            myexperiment.myranker.deezy_parameters["selection_threshold"]
        )
    processed_file = os.path.join(
        myexperiment.data_path,
        "lwm/"
        + myexperiment.myner.model.replace("coarse", "fine")
        + "_"
        + cand_approach
        + "_mentions.tsv",
    )
    original_file = os.path.join(myexperiment.data_path, "lwm/linking_df_split.tsv")

    if not Path(processed_file).exists():
        sys.exit(
            (
                "* WARNING! The training set has not been generated yet. To do so,\n"
                "please run the same experiment (i.e. same settings) with the LwM\n"
                "dataset. If the linking method you're using is unsupervised, please\n"
                "ignore this warning.\n"
            )
        )

    lwm_processed_df = pd.read_csv(processed_file, sep="\t")
    lwm_processed_df = lwm_processed_df.drop(columns=["Unnamed: 0"])
    lwm_processed_df["candidates"] = lwm_processed_df["candidates"].apply(
        eval_with_exception
    )
    lwm_original_df = pd.read_csv(original_file, sep="\t")

    return lwm_original_df, lwm_processed_df


# ==============================================================
# --------------------------------------------------------------
# Train REL entity disambiguation
def train_rel_ed(
    mylinker,
    train_original,
    train_processed,
    dev_original,
    dev_processed,
    experiment_name,
    cand_selection,
):
    base_path = mylinker.rel_params["base_path"]
    wiki_version = mylinker.rel_params["wiki_version"]
    training_data = mylinker.rel_params["training_data"]

    experiment_path = os.path.join(
        base_path, wiki_version, "generated", experiment_name
    )

    # Check if the model already exists and has to be overwritten:
    if (
        Path(os.path.join(experiment_path, "lr_model.pkl")).exists()
        and mylinker.overwrite_training == False
    ):
        print("The model already exists. The training won't be overwritten.")
        return None

    print(">>> Start the training of the Entity Disambiguation model...")
    Path(os.path.join(base_path, wiki_version, "generated", "test_train_data")).mkdir(
        parents=True, exist_ok=True
    )
    wikipedia = Wikipedia(base_path, wiki_version)
    data_handler = GenTrainingTest(
        base_path, wiki_version, wikipedia, mylinker=mylinker
    )
    for ds in ["train", "dev"]:
        if training_data == "lwm":
            data_handler.process_lwm(
                ds,
                train_original,
                train_processed,
                dev_original,
                dev_processed,
                cand_selection,
            )
        if training_data == "aida":
            data_handler.process_aida(ds)

    datasets = TrainingEvaluationDatasets(base_path, wiki_version).load()

    # Create a folder for the model:
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "train",
        "model_path": os.path.join(experiment_path, "model"),
    }
    model = EntityDisambiguation(base_path, wiki_version, config)

    # Train the model using lwm_train:
    model.train(
        datasets["lwm_train"],
        {k: v for k, v in datasets.items() if k != "lwm_train"},
    )
    # Train and predict using LR (to obtain confidence scores)
    model.train_LR(datasets, experiment_path)

    print(">>> The Entity Disambiguation model is now trained!")
