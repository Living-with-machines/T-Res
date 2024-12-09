import os
from pathlib import Path
import pytest

from transformers.pipelines.token_classification import TokenClassificationPipeline

from t_res.geoparser import recogniser
from t_res.utils import ner_utils

current_dir = Path(__file__).parent.resolve()

def test_ner_local_train(tmp_path):
    model_path = os.path.join(tmp_path,"ner_test.model")
    
    ner = recogniser.CustomRecogniser(
        model_name="ner_test",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        base_model="Livingwithmachines/bert_1760_1900", 
        model_path=f"{tmp_path}/",
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,
        do_test=False,
    )
    assert os.path.exists(model_path) is False
    ner.train()
    print(model_path)
    print(os.listdir(tmp_path))
    assert os.path.exists(model_path) is True

@pytest.mark.skip(reason="Needs large model file")
def test_ner_predict():
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        base_model="Livingwithmachines/bert_1760_1900", 
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,
        do_test=False,
    )
    ner.load()
    assert isinstance(ner.pipe, TokenClassificationPipeline)

    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    predictions = ner.ner_predict(sentence)
    assert isinstance(predictions, list)
    assert len(predictions) == 15
    assert predictions[13] == {'entity': 'B-LOC', 'score': pytest.approx(0.9996446371078491, abs=1e-3), 'word': 'Sheffield', 'start': 74, 'end': 83}

    # Test that ner_predict() can handle hyphens
    sentence = "- I grew up in Plymouth—Kingston."
    predictions = ner.ner_predict(sentence)
    assert predictions[0]["word"] == "-"
    assert predictions[6]["word"] == ","

@pytest.mark.skip(reason="Needs large model file")
def test_run():
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        base_model="Livingwithmachines/bert_1760_1900", 
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,
        do_test=False,
    )
    ner.load()
    assert isinstance(ner.pipe, TokenClassificationPipeline)

    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield, but also in Leeds."
    result = ner.run(sentence)

    assert result.sentence.sentence == sentence

    assert result.len() == 2
    assert result.mentions[0].mention == "Sheffield"
    assert result.mentions[0].start_offset == 13
    assert result.mentions[0].end_offset == 13
    assert result.mentions[0].start_char == 74
    assert result.mentions[0].end_char() == 83

    assert result.mentions[1].mention == "Leeds"
    assert result.mentions[1].start_offset == 18
    assert result.mentions[1].end_offset == 18
    assert result.mentions[1].start_char == 97
    assert result.mentions[1].end_char() == 102

    sentence = ', thence to Emery Down,crowing to Minesteed Manor ; he ther tacked back to Notherwood, and from thence back again to the Manor, where, after a brilliant run (Arnie hour and forty-five minutes, Reynold was compelled to succumb to his pursuers. '
    result = ner.run(sentence)

def test_ner_from_hub():
    ner = recogniser.PretrainedRecogniser(
        model_name="Livingwithmachines/toponym-19thC-en",
    )
    ner.load()
    assert isinstance(ner.pipe, TokenClassificationPipeline)
    
    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    predictions = ner.ner_predict(sentence)
    assert isinstance(predictions, list)
    assert len(predictions) == 15
    assert predictions[13] == {'entity': 'B-LOC', 'score': pytest.approx(0.9996446371078491, abs=1e-3), 'word': 'Sheffield', 'start': 74, 'end': 83}

def test_aggregate_mentions():
    ner = recogniser.PretrainedRecogniser(
        model_name="Livingwithmachines/toponym-19thC-en",
    )
    ner.load()
    
    sentence = "I grew up in Bologna, a city near Florence, but way more interesting."
    predictions = ner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner_utils.aggregate_mentions(procpreds, "pred")
    assert len(mentions) == 2
    assert mentions[1]["mention"] == "Florence"
    assert mentions[0] == {'mention': 'Bologna', 'start_offset': 4, 'end_offset': 4, 'start_char': 13, 'end_char': 20, 'ner_score': 20.0, 'ner_label': 'LOC', 'entity_link': 'O'}
    assert mentions[0]["end_char"] - mentions[0]["start_char"] == len(
        mentions[0]["mention"]
    )
    assert mentions[0]["mention"] in sentence

    sentence = "ARMITAGE, DEM’TIST, may be consulted dally, from 9 a.m., till 8 p.m., at his residence, 95, STAMFORP-9TKEET, Ashton-cnder-Ltne."
    predictions = ner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner_utils.aggregate_mentions(procpreds, "pred")
    assert len(mentions) == 2
    assert mentions[1]["mention"] == "Ashton-cnder-Ltne"
    assert mentions[0] == {'mention': 'STAMFORP-9TKEET', 'start_offset': 31, 'end_offset': 33, 'start_char': 92, 'end_char': 107, 'ner_score': 102.667, 'ner_label': 'STREET', 'entity_link': 'O'}
    assert mentions[0]["end_char"] - mentions[0]["start_char"] == len(
            mentions[0]["mention"]
        )
    assert mentions[0]["mention"] in sentence
