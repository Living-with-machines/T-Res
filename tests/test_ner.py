import os
import shutil
import sys

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import transformers

from geoparser import recogniser
from utils import ner


def test_training():
    """
    Test that running train() generates a model folder
    """

    test_folder_path = "resources/models/blb_lwm-ner-coarse_test.model"

    if os.path.isdir(test_folder_path):
        shutil.rmtree(test_folder_path)

    myner = recogniser.Recogniser(
        model="blb_lwm-ner-coarse",  # NER model name prefix (will have suffixes appended)
        base_model="Livingwithmachines/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_coarse_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_coarse_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=True,  # Set to True if you want to overwrite model if existing
        do_test=True,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )
    assert os.path.isdir(test_folder_path) == False
    myner.train()
    assert os.path.isdir(test_folder_path) == True


def test_create_pipeline():
    """
    Test that create_pipeline returns a model folder path that exists and an Pipeline object
    """
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-coarse",  # NER model name prefix (will have suffixes appended)
        base_model="Livingwithmachines/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=True,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )
    pipe = myner.create_pipeline()
    assert (
        type(pipe)
        == transformers.pipelines.token_classification.TokenClassificationPipeline
    )


def test_ner_predict():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        base_model="Livingwithmachines/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )
    myner.pipe = myner.create_pipeline()

    preds = myner.ner_predict(
        "I grew up in Bologna, a city near Florence, but way more interesting."
    )
    assert type(preds) == list
    assert (type(preds[0])) == dict
    assert len(preds) == 16
    assert preds[4]["entity"] == "B-LOC"
    assert preds[4]["score"] == 0.9994915723800659

    # Test that ner_predict() can handle hyphens
    preds = myner.ner_predict("- I grew up in Plymouth—Kingston.")
    assert preds[0]["word"] == "-"
    assert preds[6]["word"] == ","


def test_ner_load_from_hub():
    myner = recogniser.Recogniser(
        model="Livingwithmachines/toponym-19thC-en",
        load_from_hub=True,
    )
    pipe = myner.create_pipeline()
    assert (
        type(pipe)
        == transformers.pipelines.token_classification.TokenClassificationPipeline
    )


def test_aggregate_mentions():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        base_model="Livingwithmachines/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )
    myner.pipe = myner.create_pipeline()

    sentence = "I grew up in Bologna, a city near Florence, but way more interesting."
    predictions = myner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner.aggregate_mentions(procpreds, "pred")
    assert mentions[0]["mention"] == "Bologna"
    assert mentions[1]["mention"] == "Florence"
    assert mentions[0]["end_char"] - mentions[0]["start_char"] == len(
        mentions[0]["mention"]
    )
    assert mentions[1]["end_char"] - mentions[1]["start_char"] == len(
        mentions[1]["mention"]
    )
    assert mentions[0]["mention"] in sentence
    assert mentions[1]["mention"] in sentence

    sentence = "I grew up in New York City, a city in the United States."
    predictions = myner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner.aggregate_mentions(procpreds, "pred")
    assert mentions[0]["mention"] == "New York City"
    assert mentions[1]["mention"] == "United States"
    assert mentions[0]["end_char"] - mentions[0]["start_char"] == len(
        mentions[0]["mention"]
    )
    assert mentions[1]["end_char"] - mentions[1]["start_char"] == len(
        mentions[1]["mention"]
    )
    assert mentions[0]["mention"] in sentence
    assert mentions[1]["mention"] in sentence

    sentence = "ARMITAGE, DEM’TIST, may be consulted dally, from 9 a.m., till 8 p.m., at his residence, 95, STAMFORP-9TKEET, Ashton-cnder-Ltne."
    predictions = myner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner.aggregate_mentions(procpreds, "pred")
    assert mentions[-1]["mention"] == "Ashton-cnder-Ltne"
    for i in range(len(mentions)):
        assert mentions[i]["end_char"] - mentions[i]["start_char"] == len(
            mentions[i]["mention"]
        )
        assert mentions[i]["mention"] in sentence
