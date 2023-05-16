import os
import sys
from ast import literal_eval

import pandas as pd
import pytest

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from experiments import experiment
from geoparser import linking, ranking, recogniser


def test_wrong_dataset_path():
    with pytest.raises(SystemExit) as cm:
        experiment.Experiment(
            dataset="lwm",
            data_path="wrong_path/",
            dataset_df=pd.DataFrame(),
            results_path="experiments/outputs/results/",
            myner="test",
            myranker="test",
            mylinker="test",
            test_split="dev",
        )

    assert (
        cm.value.code
        == "\nError: The dataset has not been created, you should first run the prepare_data.py script.\n"
    )


def test_load_data():
    data = pd.read_csv("experiments/outputs/data/lwm/linking_df_split.tsv", sep="\t")
    ids = set()

    for idx, row in data.iterrows():
        article_id = row["article_id"]
        sents = literal_eval(row["sentences"])
        for sent in sents:
            ids.add(str(article_id) + "_" + str(sent["sentence_pos"]))

    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )

    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    # --------------------------------------
    # Instantiate the linker:
    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    myner.train()
    myner.pipe = myner.create_pipeline()

    myranker.mentions_to_wikidata = myranker.load_resources()
    myranker.train()

    mylinker.linking_resources = mylinker.load_resources()

    # --------------------------------------
    # Instantiate the experiment:
    exp = experiment.Experiment(
        dataset="lwm",
        data_path="experiments/outputs/data/",
        dataset_df=pd.DataFrame(),
        results_path="experiments/outputs/results/",
        myner=myner,
        myranker=myranker,
        mylinker=mylinker,
        overwrite_processing=False,  # If True, do data processing, else load existing processing, if exists.
        processed_data=dict(),  # Dictionary where we'll keep the processed data for the experiments.
        test_split="dev",  # "dev" while experimenting, "test" when running final experiments.
        rel_experiments=False,  # False if we're not interested in running the different experiments with REL, True otherwise.
    )

    # Load processed data if existing:
    exp.processed_data = exp.load_data()
    if not exp.processed_data == dict():
        for k, v in exp.processed_data.items():
            assert len(ids) == len(v)

        not_empty_dMentionsPred = [
            v for k, v in exp.processed_data["dMentionsPred"].items() if len(v) > 0
        ]
        not_empty_dCandidates = [
            v for k, v in exp.processed_data["dCandidates"].items() if len(v) > 0
        ]

        assert len(not_empty_dMentionsPred) == len(not_empty_dCandidates)

    else:
        # If the data is not processed, process it, and do the same tests:
        exp.processed_data = exp.prepare_data()
        for k, v in exp.processed_data.items():
            assert len(ids) == len(v)

        not_empty_dMentionsPred = [
            v for k, v in exp.processed_data["dMentionsPred"].items() if len(v) > 0
        ]
        not_empty_dCandidates = [
            v for k, v in exp.processed_data["dCandidates"].items() if len(v) > 0
        ]

        assert len(not_empty_dMentionsPred) == len(not_empty_dCandidates)


def test_wrong_ranker_method():
    ranker = ranking.Ranker(
        # wrong naming: it should be perfectmatch
        method="perfect_match",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    exp = experiment.Experiment(
        dataset="lwm",
        data_path="experiments/outputs/data/",
        dataset_df=pd.DataFrame(),
        results_path="experiments/outputs/results/",
        myner="test",
        myranker=ranker,
        mylinker="test",
    )
    with pytest.raises(SystemExit) as cm:
        exp.prepare_data()
    assert cm.value.code == 0


def test_apply():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )

    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    # --------------------------------------
    # Instantiate the linker:
    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    myner.train()
    myner.pipe = myner.create_pipeline()

    myranker.mentions_to_wikidata = myranker.load_resources()
    myranker.train()

    mylinker.linking_resources = mylinker.load_resources()

    # --------------------------------------
    # Instantiate the experiment:
    exp = experiment.Experiment(
        dataset="lwm",
        data_path="experiments/outputs/data/",
        dataset_df=pd.DataFrame(),
        results_path="experiments/outputs/results/",
        myner=myner,
        myranker=myranker,
        mylinker=mylinker,
        overwrite_processing=False,  # If True, do data processing, else load existing processing, if exists.
        processed_data=dict(),  # Dictionary where we'll keep the processed data for the experiments.
        test_split="apply",  # "dev" while experimenting, "test" when running final experiments.
        rel_experiments=False,  # False if we're not interested in running the different experiments with REL, True otherwise.
    )

    # Load processed data if existing:
    exp.processed_data = exp.load_data()

    # Perform data postprocessing:
    exp.processed_data = exp.prepare_data()

    exp.linking_experiments()

    assert "apply" in exp.dataset_df.columns
