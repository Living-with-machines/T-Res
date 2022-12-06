import os
import shutil
import sys

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import transformers

from geoparser import recogniser


def test_training():
    """
    Test that running train() generates a model folder
    """

    test_folder_path = "experiments/outputs/models/blb_lwm-ner-coarse_test.model"

    if os.path.isdir(test_folder_path):
        shutil.rmtree(test_folder_path)

    myner = recogniser.Recogniser(
        model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
        model=None,  # We'll store the NER model here
        pipe=None,  # We'll store the NER pipeline here
        base_model="resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
        train_dataset="experiments/outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
        output_model_path="experiments/outputs/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=True,  # Set to True if you want to overwrite model if existing
        do_test=True,  # Set to True if you want to train on test mode
        training_tagset="coarse",  # Options are: "coarse" or "fine"
    )
    assert os.path.isdir(test_folder_path) == False
    myner.train()
    assert os.path.isdir(test_folder_path) == True


def test_create_pipeline():
    """
    Test that create_pipeline returns a model folder path that exists and an Pipeline object
    """
    myner = recogniser.Recogniser(
        model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
        model=None,  # We'll store the NER model here
        pipe=None,  # We'll store the NER pipeline here
        base_model="resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
        train_dataset="experiments/outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
        output_model_path="experiments/outputs/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=True,  # Set to True if you want to train on test mode
        training_tagset="coarse",  # Options are: "coarse" or "fine"
    )
    model, pipe = myner.create_pipeline()
    assert os.path.isdir(model) == True
    assert type(pipe) == transformers.pipelines.token_classification.TokenClassificationPipeline


def test_ner_predict():

    myner = recogniser.Recogniser(
        model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
        model=None,  # We'll store the NER model here
        pipe=None,  # We'll store the NER pipeline here
        base_model="resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
        train_dataset="experiments/outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
        output_model_path="experiments/outputs/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        training_tagset="coarse",  # Options are: "coarse" or "fine"
    )
    myner.model, myner.pipe = myner.create_pipeline()

    preds = myner.ner_predict(
        "I grew up in Bologna, a city near Florence, but way more interesting."
    )
    assert type(preds) == list
    assert (type(preds[0])) == dict
    assert len(preds) == 16
    assert preds[4]["entity"] == "B-LOC"
    assert preds[4]["score"] == 0.9897381663322449
