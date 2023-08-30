import os
from pathlib import Path
import pytest

from transformers.pipelines.token_classification import TokenClassificationPipeline

from t_res.geoparser import recogniser
from t_res.utils import ner

current_dir = Path(__file__).parent.resolve()

def test_ner_local_train(tmp_path):
    model_path = os.path.join(tmp_path,"ner_test.model")
    
    myner = recogniser.Recogniser(
        model="ner_test",
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
        load_from_hub=False,
    )
    assert os.path.exists(model_path) is False
    myner.train()
    print(model_path)
    print(os.listdir(tmp_path))
    assert os.path.exists(model_path) is True

@pytest.mark.skip(reason="Needs large model file")
def test_ner_predict():
    model_path = os.path.join(current_dir,"sample_files/resources/models/ner_test.model")
    assert os.path.isdir(model_path) is True

    myner = recogniser.Recogniser(
        model="ner_test",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        base_model="Livingwithmachines/bert_1760_1900", 
        model_path=os.path.join(current_dir,"sample_files/resources/models/"),
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,
        do_test=False,
        load_from_hub=False, # Whether the final model should be loaded from the HuggingFace hub"
    )
    myner.pipe = myner.create_pipeline()
    assert isinstance(myner.pipe, TokenClassificationPipeline)

    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    predictions = myner.ner_predict(sentence)
    assert isinstance(predictions, list)
    assert len(predictions) == 15
    assert predictions[13] == {'entity': 'B-LOC', 'score': 0.7941257357597351, 'word': 'Sheffield', 'start': 74, 'end': 83}

    # Test that ner_predict() can handle hyphens
    sentence = "- I grew up in Plymouth—Kingston."
    predictions = myner.ner_predict(sentence)
    assert predictions[0]["word"] == "-"
    assert predictions[6]["word"] == ","


def test_ner_from_hub():
    myner = recogniser.Recogniser(
        model="Livingwithmachines/toponym-19thC-en",
        load_from_hub=True,
    )
    myner.train()
    myner.pipe = myner.create_pipeline()
    assert isinstance(myner.pipe, TokenClassificationPipeline)
    
    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    predictions = myner.ner_predict(sentence)
    assert isinstance(predictions, list)
    assert len(predictions) == 15
    assert predictions[13] == {'entity': 'B-LOC', 'score': 0.9996446371078491, 'word': 'Sheffield', 'start': 74, 'end': 83}


def test_aggregate_mentions():
    myner = recogniser.Recogniser(
        model="Livingwithmachines/toponym-19thC-en",
        load_from_hub=True,
    )
    myner.pipe = myner.create_pipeline()
    
    sentence = "I grew up in Bologna, a city near Florence, but way more interesting."
    predictions = myner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner.aggregate_mentions(procpreds, "pred")
    assert len(mentions) == 2
    assert mentions[1]["mention"] == "Florence"
    assert mentions[0] == {'mention': 'Bologna', 'start_offset': 4, 'end_offset': 4, 'start_char': 13, 'end_char': 20, 'ner_score': 20.0, 'ner_label': 'LOC', 'entity_link': 'O'}
    assert mentions[0]["end_char"] - mentions[0]["start_char"] == len(
        mentions[0]["mention"]
    )
    assert mentions[0]["mention"] in sentence

    sentence = "ARMITAGE, DEM’TIST, may be consulted dally, from 9 a.m., till 8 p.m., at his residence, 95, STAMFORP-9TKEET, Ashton-cnder-Ltne."
    predictions = myner.ner_predict(sentence)
    # Process predictions:
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"]]
        for x in predictions
    ]
    # Aggregate mentions:
    mentions = ner.aggregate_mentions(procpreds, "pred")
    assert len(mentions) == 2
    assert mentions[1]["mention"] == "Ashton-cnder-Ltne"
    assert mentions[0] == {'mention': 'STAMFORP-9TKEET', 'start_offset': 31, 'end_offset': 33, 'start_char': 92, 'end_char': 107, 'ner_score': 102.667, 'ner_label': 'STREET', 'entity_link': 'O'}
    assert mentions[0]["end_char"] - mentions[0]["start_char"] == len(
            mentions[0]["mention"]
        )
    assert mentions[0]["mention"] in sentence
