import os
import sys
from pathlib import Path
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking, pipeline, ranking, recogniser


def test_deezy_mostpopular():

    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("resources/models/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("resources/deezymatch/").resolve()),
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

    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params={},
        overwrite_training=False,
    )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefrield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
    )

    """
    assert resolved[0]["mention"] == "Shefrield"
    assert resolved[0]["candidates"]["Q665346"] == 0.007
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.893
    assert resolved[0]["ner_score"] == 0.994

    resolved = geoparser.run_sentence("")
    assert resolved == []

    resolved = geoparser.run_sentence(" ")
    assert resolved == []

    # asserting behaviour with • character
    resolved = geoparser.run_text(
        " • - ST G pOllO-P• FERRIS - • - , i ",
    )

    assert resolved[0]["candidates"] == {}
    """


def test_deezy_rel_withoutpubl():

    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("resources/models/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("resources/deezymatch/").resolve()),
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

    """

    mylinker = linking.Linker(
        method="reldisamb",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params={
            "base_path": "resources/rel_db/",
            "wiki_version": "wiki_2019/",
            "training_data": "lwm",  # lwm, aida
            "ranking": "relv",  # relv, publ
            "micro_locs": "nil",  # "dist", "nil", ""
        },
        overwrite_training=False,
    )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefrield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
    )

    assert resolved[0]["mention"] == "Shefrield"
    assert (
        resolved[0]["ed_score"] == resolved[0]["candidates"][resolved[0]["prediction"]]
    )
    assert resolved[0]["candidates"]["Q665346"] == 0.916
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.982
    assert resolved[0]["ner_score"] == 0.994

    resolved = geoparser.run_sentence("")
    assert resolved == []

    resolved = geoparser.run_sentence(" ")
    assert resolved == []

    # asserting behaviour with a NIL
    resolved = geoparser.run_text(
        "Chrixtchurch, June 10 Yesterday being the day appointed for the election of taro gentlemen to tepcoeot this borough in the new Parliament.",
    )
    assert resolved[0]["mention"] == "Chrixtchurch"
    assert resolved[0]["ed_score"] == 0.0
    assert resolved[0]["candidates"] == {}
    assert resolved[0]["prediction"] == "NIL"

    # asserting behaviour with • character
    resolved = geoparser.run_text(
        " • - ST G pOllO-P• FERRIS - • - , i ",
    )

    assert resolved[0]["candidates"] == {}

    """


def test_deezy_rel_wpubl_wmtops():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    # --------------------------------------
    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("resources/models/w2v/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("resources/deezymatch/").resolve()),
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

    with sqlite3.connect("resources/rel_db/embedding_database.db") as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path="resources/",
            linking_resources=dict(),
            rel_params={
                "model_path": "resources/models/disambiguation/",
                "data_path": "experiments/outputs/data/lwm/",
                "training_split": "originalsplit",
                "context_length": 100,
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": False,
                "default_publname": "",
                "default_publwqid": "",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefrield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
        place="Sheffield",
        place_wqid="Q42448",
    )

    assert resolved == []
