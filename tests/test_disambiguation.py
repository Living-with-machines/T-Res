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

current_dir = Path(__file__).parent.resolve()

@pytest.mark.skip(reason="Needs embeddings db")
def test_embeddings():
    """
    Test embeddings are loaded correctly.
    """
    # Test 1: Check glove embeddings
    mentions = ["in", "apple"]
    with sqlite3.connect(os.path.join(current_dir,"sample_files/resources/rel_db/embeddings_database.db")) as conn:
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

@pytest.mark.deezy(reason="Needs deezy model")
def test_train(tmp_path):
    myner = recogniser.Recogniser(
        model="ner_test",  # NER model name prefix (will have suffixes appended)
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
        load_from_hub=False,
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
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch"),
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
    # Train the NER models if needed:
    myner.train()
    # Load the NER pipeline:
    myner.pipe = myner.create_pipeline()

    # -----------------------------------------
    # Ranker loading resources and training a model:
    # Load the resources:
    myranker.load_resources()
    # Train a DeezyMatch model if needed:
    myranker.train()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    mylinker.load_resources()
    # Train a linking model if needed (it requires myranker to generate potential
    # candidates to the training set):
    mylinker.rel_params["ed_model"] = mylinker.train_load_model(myranker)

    assert (
        type(mylinker.rel_params["ed_model"])
        == entity_disambiguation.EntityDisambiguation
    )

    # assert expected performance on test set
    assert mylinker.rel_params["ed_model"].best_performance["f1"] == 0.6288416075650118

@pytest.mark.deezy(reason="Needs deezy model")
def test_load_eval_model(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
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
        load_from_hub=False,
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
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch"),
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
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm"),
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
    # Train the NER models if needed:
    myner.train()
    # Load the NER pipeline:
    myner.pipe = myner.create_pipeline()

    # -----------------------------------------
    # Ranker loading resources and training a model:
    # Load the resources:
    myranker.load_resources()
    # Train a DeezyMatch model if needed:
    myranker.train()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    mylinker.load_resources()
    # Train a linking model if needed (it requires myranker to generate potential
    # candidates to the training set):
    mylinker.rel_params["ed_model"] = mylinker.train_load_model(myranker)

    assert (
        type(mylinker.rel_params["ed_model"])
        == entity_disambiguation.EntityDisambiguation
    )

@pytest.mark.deezy(reason="Needs deezy model")
def test_predict(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        model_path=str(tmp_path),  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
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
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": False,
            },
            overwrite_training=False,
        )

    mypipe = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    predictions = mypipe.run_text(
        "I live on Market-Street in Liverpool. I don't live in Manchester but in Allerton, near Liverpool. There was an adjourned meeting of miners in Ashton-cnder-Lyne.",
        place="London",
        place_wqid="Q84",
    )
    assert isinstance(predictions,list)

    assert predictions[1]["prediction"] in predictions[1]["cross_cand_score"]

    highest_cross_cand_score = max(
        predictions[1]["cross_cand_score"], key=predictions[1]["cross_cand_score"].get
    )
    assert predictions[1]["prediction"] == highest_cross_cand_score
