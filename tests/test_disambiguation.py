import os
import sqlite3
import sys
from pathlib import Path

import pytest
import pandas as pd
import pytest

from t_res.geoparser import linking, pipeline, ranking, recogniser
from t_res.utils import rel_utils
from t_res.utils.REL import entity_disambiguation
from t_res.geoparser.dataclasses import Predictions

current_dir = Path(__file__).parent.resolve()

@pytest.mark.skip(reason="Needs embeddings database")
def test_embeddings():
    """
    Test embeddings are loaded correctly.
    """
    # Test 1: Check glove embeddings
    mentions = ["in", "apple"]
    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        embs = rel_utils.get_db_emb(cursor, mentions, "snd")
        assert len(mentions) == len(embs)
        assert len(embs[0]) == 300
        mentions = ["cotxe"]
        embs = rel_utils.get_db_emb(cursor, mentions, "snd")
        assert embs == [None]
        # Test 2: Check wiki2vec word embeddings
        mentions = ["in", "apple"]
        embs = rel_utils.get_db_emb(cursor, mentions, "word")
        assert len(mentions) == len(embs)
        assert len(embs[0]) == 300
        mentions = ["cotxe"]
        embs = rel_utils.get_db_emb(cursor, mentions, "word")
        assert embs == [None]
        # Test 2: Check wiki2vec entity embeddings
        mentions = ["Q84", "Q1492"]
        embs = rel_utils.get_db_emb(cursor, mentions, "entity")
        assert len(mentions) == len(embs)
        assert len(embs[0]) == 300
        mentions = ["Q1"]
        embs = rel_utils.get_db_emb(cursor, mentions, "entity")
        assert embs == [None]

@pytest.mark.skip(reason="Needs large resources")
def test_train(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="ner_test",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
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
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir, "../resources/deezymatch"),
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

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir, "../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": False,
                "without_microtoponyms": True,
                "do_test": True,
            },
            overwrite_training=True,
        )

    # -----------------------------------------
    # NER training and creating pipeline:
    ner.load()

    # -----------------------------------------
    # Ranker loading resources and training a model:
    # Load the resources (and train a DeezyMatch model if needed):
    ranker.load()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    linker.load()

    # Train a linking model if needed (it requires ranker to generate potential
    # candidates to the training set):
    linker.train_load_model(ranker)
    assert isinstance(linker.entity_disambiguation_model, entity_disambiguation.EntityDisambiguation)

    # assert expected performance on test set
    assert linker.entity_disambiguation_model.best_performance["f1"] == pytest.approx(0.8571428571428571, abs=1e-6)

@pytest.mark.skip(reason="Needs embeddings database")
def test_load_eval_model(tmp_path):
    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        model_path=str(tmp_path),  # Path where the NER model is or will be stored
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
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir, "../resources/deezymatch"),
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

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "sample_files/resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir, "sample_files/resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir, "sample_files/experiments/outputs/data/lwm"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": False,
                "without_microtoponyms": False,
                "do_test": True,
            },
            overwrite_training=False,
        )

    # -----------------------------------------
    # NER training and creating pipeline:
    ner.load()

    # -----------------------------------------
    # Ranker loading resources and training a model:
    # Load the resources (and train a DeezyMatch model if needed):
    ranker.load()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    linker.load()

    # Train a linking model if needed (it requires ranker to generate potential
    # candidates to the training set):
    linker.train_load_model(ranker)
    assert isinstance(linker.entity_disambiguation_model, entity_disambiguation.EntityDisambiguation)

@pytest.mark.skip(reason="Needs large resources")
def test_predict(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    ner = recogniser.CustomRecogniser(
        model_name="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        model_path=model_path,
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
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
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": False,
            },
            overwrite_training=False,
        )

    mypipe = pipeline.Pipeline(ner=ner, ranker=ranker, linker=linker)

    predictions = mypipe.run(
        "I live on Market-Street in Liverpool. I don't live in Manchester but in Allerton, near Liverpool. There was an adjourned meeting of miners in Ashton-cnder-Lyne.",
        place_of_pub_wqid="Q84",
        place_of_pub="London",
    )

    assert isinstance(predictions, Predictions)
    assert len(predictions.candidates()) == 6

    assert predictions.candidates()[1].best_wqid() in predictions.candidates()[1].best_match().cross_cand_scores().keys()

    highest_cross_cand_score = max(predictions.candidates()[1].best_match().cross_cand_scores().values())
    assert highest_cross_cand_score == 0.857

    best_disambiguation_score = predictions.candidates()[1].best_match().best_disambiguation_score()
    assert round(best_disambiguation_score, 3) == highest_cross_cand_score
