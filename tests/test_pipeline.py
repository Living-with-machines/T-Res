import os
import sqlite3
import sys
from pathlib import Path

from t_res.geoparser import linking, pipeline, ranking, recogniser


def test_deezy_mostpopular():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/",
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
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
    )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
    )
    assert resolved[0]["mention"] == "Shefiield"
    assert resolved[0]["prior_cand_score"] == dict()
    assert resolved[0]["cross_cand_score"]["Q42448"] == 0.903
    assert resolved[0]["string_match_score"]["Sheffield"][0] == 0.999
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.903
    assert resolved[0]["ner_score"] == 1.0

    resolved = geoparser.run_sentence("")
    assert resolved == []

    resolved = geoparser.run_sentence(" ")
    assert resolved == []

    # asserting behaviour with • character
    resolved = geoparser.run_text(
        " • - ST G pOllO-P• FERRIS - • - , i ",
    )

    assert resolved == []


def test_deezy_rel_wpubl_wmtops():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    # --------------------------------------
    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/",
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
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    with sqlite3.connect("resources/rel_db/embeddings_database.db") as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path="resources/",
            linking_resources=dict(),
            rel_params={
                "model_path": "resources/models/disambiguation/",
                "data_path": "experiments/outputs/data/lwm/",
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
        place="Sheffield",
        place_wqid="Q42448",
    )

    assert resolved[0]["mention"] == "Shefiield"
    assert resolved[0]["prior_cand_score"]["Q42448"] == 0.891
    assert resolved[0]["cross_cand_score"]["Q42448"] == 0.576
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.039
    assert resolved[0]["ner_score"] == 1.0


def test_perfect_rel_wpubl_wmtops():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    # --------------------------------------
    # Instantiate the ranker:
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="resources/",
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
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    with sqlite3.connect("resources/rel_db/embeddings_database.db") as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path="resources/",
            linking_resources=dict(),
            rel_params={
                "model_path": "resources/models/disambiguation/",
                "data_path": "experiments/outputs/data/lwm/",
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": True,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
        place="Sheffield",
        place_wqid="Q42448",
    )

    assert resolved[0]["mention"] == "Shefiield"
    assert resolved[0]["prior_cand_score"] == dict()
    assert resolved[0]["cross_cand_score"] == dict()
    assert resolved[0]["prediction"] == "NIL"
    assert resolved[0]["ed_score"] == 0.0
    assert resolved[0]["ner_score"] == 1.0


def test_modular_deezy_rel():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # We'll store the NER model here
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="./resources/",
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("./resources/models/w2v/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("./resources/deezymatch/").resolve()),
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

    with sqlite3.connect("./resources/rel_db/embeddings_database.db") as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path="./resources/",
            linking_resources=dict(),
            rel_params={
                "model_path": "./resources/models/disambiguation/",
                "data_path": "./experiments/outputs/data/lwm/",
                "training_split": "apply",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    sentence = "STOCKTON AND MIDDLESBROUGH WATER IVARD.  The monthly meeting of the Sr-id:toe and bladtiltwitrough Water Lkerd was held at the Corp.acit:o.i liniklinga, Middlesbrough, on Monday."
    wikidata_id = "Q989418"
    location = "Stockton-on-Tees, Cleveland, England"

    toponyms = geoparser.run_text_recognition(
        sentence,
        place_wqid=wikidata_id,
        place=location,
    )

    cands = geoparser.run_candidate_selection(toponyms)

    disambiguation = geoparser.run_disambiguation(
        toponyms,
        cands,
        place_wqid=wikidata_id,
        place=location,
    )

    assert type(disambiguation) == list
    assert disambiguation[0]["prediction"] == "Q989418"
    assert disambiguation[-1]["prediction"] == "Q171866"
