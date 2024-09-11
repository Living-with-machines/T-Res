import os
import sqlite3
from pathlib import Path

import pytest

from t_res.geoparser import linking, pipeline, ranking, recogniser

current_dir = Path(__file__).parent.resolve()

def test_pipeline_basic():
    geoparser = pipeline.Pipeline(
        resources_path=os.path.join(current_dir,"sample_files/resources")
    )

    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    resolved = geoparser.run_text(sentence)
    assert len(resolved)==1
    assert resolved[0]["mention"]=="Sheffield"
    assert resolved[0]["ner_score"]==1.0
    assert resolved[0]["prediction"]=="Q42448"

def test_pipeline_modular():
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path=os.path.join(current_dir,"sample_files/resources"),
    )
    
    mylinker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )

    geoparser = pipeline.Pipeline(myranker=myranker, mylinker=mylinker)
    
    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    resolved = geoparser.run_text(sentence)
    assert len(resolved)==1
    assert resolved[0]["mention"]=="Sheffield"
    assert resolved[0]["ner_score"]==1.0
    assert resolved[0]["prediction"]=="Q42448"

@pytest.mark.deezy(reason="Needs deezy model")
def test_deezy_mostpopular(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=str(tmp_path),  # Path where the NER model will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": os.path.join(tmp_path,"resources/models/"),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)
    assert len(geoparser.myranker.mentions_to_wikidata.keys())>0

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though.",
    )
    assert len(resolved) == 3
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
        " • - S G pOllO-P• FERRIS - • - , i ",
    )
    assert resolved == []

@pytest.mark.deezy(reason="Needs deezy model")
def test_deezy_rel_wpubl_wmtops(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=str(tmp_path),  # Path where the NER model will be stored
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
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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

    with sqlite3.connect(os.path.join(current_dir,"sample_files/resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path=os.path.join(current_dir,"sample_files/resources/"),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"sample_files/resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
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
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though.",
        place="Sheffield",
        place_wqid="Q42448",
    )

    assert len(resolved) == 3
    assert resolved[0]["mention"] == "Shefiield"
    assert resolved[0]["prior_cand_score"]["Q42448"] == 0.891
    assert resolved[0]["cross_cand_score"]["Q42448"] == 0.576
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.039
    assert resolved[0]["ner_score"] == 1.0

@pytest.mark.deezy(reason="Needs deezy model")
def test_perfect_rel_wpubl_wmtops(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=str(tmp_path),  # Path where the NER model will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
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
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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

    with sqlite3.connect(os.path.join(current_dir,"sample_files/resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path=os.path.join(current_dir,"sample_files/resources/"),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"sample_files/resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
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
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though.",
        place="Sheffield",
        place_wqid="Q42448",
    )

    assert resolved[0]["mention"] == "Shefiield"
    assert resolved[0]["prior_cand_score"] == dict()
    assert resolved[0]["cross_cand_score"] == dict()
    assert resolved[0]["prediction"] == "NIL"
    assert resolved[0]["ed_score"] == 0.0
    assert resolved[0]["ner_score"] == 1.0

@pytest.mark.deezy(reason="Needs deezy model")
def test_modular_deezy_rel(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=str(tmp_path),  # Path where the NER model will be stored
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
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
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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

    with sqlite3.connect(os.path.join(current_dir,"sample_files/resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        mylinker = linking.Linker(
            method="reldisamb",
            resources_path=os.path.join(current_dir,"sample_files/resources/"),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"sample_files/resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
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
