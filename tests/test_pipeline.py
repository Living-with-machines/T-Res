import os
import sqlite3
from pathlib import Path

import pytest

from t_res.geoparser import linking, pipeline, ranking, recogniser
from t_res.geoparser.dataclasses import Candidates

current_dir = Path(__file__).parent.resolve()

def test_pipeline_constructor():
    resources_path=os.path.join(current_dir, "sample_files/resources")
    geoparser = pipeline.Pipeline(resources_path=resources_path)

    # Check default pipeline components.
    assert isinstance(geoparser.ner, recogniser.PretrainedRecogniser)
    assert geoparser.ner.model_name == "Livingwithmachines/toponym-19thC-en"

    assert isinstance(geoparser.ranker, ranking.PerfectMatchRanker)
    assert geoparser.ranker.resources_path == resources_path

    assert isinstance(geoparser.linker, linking.MostPopularLinker)
    assert geoparser.linker.resources_path == resources_path

def test_pipeline_basic():
    geoparser = pipeline.Pipeline(
        resources_path=os.path.join(current_dir, "sample_files/resources")
    )

    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    resolved = geoparser.run_text(sentence)
    assert len(resolved)==1
    assert resolved[0]["mention"]=="Sheffield"
    assert resolved[0]["ner_score"]==1.0
    assert resolved[0]["prediction"]=="Q42448"

def test_pipeline_modular():
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "sample_files/resources"),
    )
    
    linker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir, "sample_files/resources"),
    )

    geoparser = pipeline.Pipeline(ranker=ranker, linker=linker)
    
    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    resolved = geoparser.run_text(sentence)
    assert len(resolved)==1
    assert resolved[0]["mention"]=="Sheffield"
    assert resolved[0]["ner_score"]==1.0
    assert resolved[0]["prediction"]=="Q42448"

@pytest.mark.skip(reason="Needs deezy model")
def test_deezy_mostpopular(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
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
            "dm_path": os.path.join(current_dir,"../resources/deezymatch/"),
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
        already_collected_cands=dict(),
    )

    linker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir, "../resources/"),
    )

    geoparser = pipeline.Pipeline(ner=ner, ranker=ranker, linker=linker)
    assert len(geoparser.ranker.mentions_to_wikidata.keys())>0

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

@pytest.mark.skip(reason="Needs large resources")
def test_deezy_rel_wpubl_wmtops(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
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
            "dm_path": os.path.join(current_dir,"../resources/deezymatch/"),
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
        already_collected_cands=dict(),
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
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

    geoparser = pipeline.Pipeline(ner=ner, ranker=ranker, linker=linker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though.",
        place="Sheffield",
        place_wqid="Q42448",
    )

    assert len(resolved) == 3
    assert resolved[0]["mention"] == "Shefiield"
    assert resolved[0]["prior_cand_score"]["Q42448"] == pytest.approx(0.891, abs=1e-3)
    assert resolved[0]["cross_cand_score"]["Q42448"] == pytest.approx(0.766, abs=1e-3)
    assert resolved[0]["prediction"] == "Q42448"
    # assert resolved[0]["ed_score"] == 0.039 # TODO: reproduce this number.
    assert resolved[0]["ner_score"] == 1.0

@pytest.mark.skip(reason="Needs large resources")
def test_perfect_rel_wpubl_wmtops(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        already_collected_cands=dict(),
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
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

    geoparser = pipeline.Pipeline(ner=ner, ranker=ranker, linker=linker)

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

@pytest.mark.skip(reason="Needs large resources")
def test_modular_deezy_rel(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
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
            "dm_path": os.path.join(current_dir, "../resources/deezymatch/"),
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
        already_collected_cands=dict(),
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir,"../resources/"),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
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

    geoparser = pipeline.Pipeline(ner=ner, ranker=ranker, linker=linker)

    sentence = "STOCKTON AND MIDDLESBROUGH WATER IVARD.  The monthly meeting of the Sr-id:toe and bladtiltwitrough Water Lkerd was held at the Corp.acit:o.i liniklinga, Middlesbrough, on Monday."
    wikidata_id = "Q989418"
    location = "Stockton-on-Tees, Cleveland, England"

    toponyms = geoparser.run_text_recognition(
        sentence,
        place_wqid=wikidata_id,
        place=location,
    )

    assert isinstance(toponyms, list)
    assert len(toponyms) == 4

    cands = geoparser.run_candidate_selection(toponyms)

    assert isinstance(cands, list)
    assert len(cands) == 4
    for c in cands:
        assert isinstance(c, Candidates)

    # Put the candidates in a dictionary for easier access inside run_disambiguation.
    wk_cands = {c.mention : c for c in cands}

    disambiguation = geoparser.run_disambiguation(
        toponyms,
        wk_cands,
        place_wqid=wikidata_id,
        place=location,
    )

    assert isinstance(disambiguation,list)

    assert disambiguation[0]["prediction"] == "Q989418"
    assert disambiguation[-1]["prediction"] == "Q171866"
