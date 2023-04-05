import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from array import array
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import rel_utils
from utils.REL import entity_disambiguation
from geoparser import recogniser, ranking, linking, pipeline


def test_embeddings():
    """
    Test embeddings are loaded correctly.
    """
    # Test 1: Check glove embeddings
    mentions = ["in", "apple"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "snd")
    assert len(mentions) == len(embs)
    assert len(embs[0]) == 300
    mentions = ["cotxe"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "snd")
    assert embs == [None]
    # Test 2: Check wiki2vec word embeddings
    mentions = ["in", "apple"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "word")
    assert len(mentions) == len(embs)
    assert len(embs[0]) == 300
    mentions = ["cotxe"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "word")
    assert embs == [None]
    # Test 2: Check wiki2vec entity embeddings
    mentions = ["Q84", "Q1492"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "entity")
    assert len(mentions) == len(embs)
    assert len(embs[0]) == 300
    mentions = ["Q1"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "entity")
    assert embs == [None]


def test_compare_embeddings():
    """
    Compare our embeddings database with the old database.
    """

    # Test 1: Check old glove embedding is the same as new glove
    # embedding:
    mentions = ["apple"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "snd")
    with sqlite3.connect("resources/rel_db/generic/common_drawl.db") as conn2:
        c2 = conn2.cursor()
        result = c2.execute("SELECT emb FROM embeddings WHERE word=?", ("apple",)).fetchone()
        result = result if result is None else array("f", result[0]).tolist()

    assert embs[0] == result

    # Test 2: Check old glove embedding is different from new wiki2vec
    # word embedding:
    mentions = ["apple"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "snd")
    with sqlite3.connect("resources/rel_db/generic/entity_word_embedding.db") as conn2:
        c2 = conn2.cursor()
        result = c2.execute("SELECT emb FROM embeddings WHERE word=?", ("apple",)).fetchone()
        result = result if result is None else array("f", result[0]).tolist()

    assert embs[0] != result

    # Test 3: Check old wiki2vec word embedding is the same as new wiki2vec
    # word embedding:
    mentions = ["apple"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "word")
    with sqlite3.connect("resources/rel_db/generic/entity_word_embedding.db") as conn2:
        c2 = conn2.cursor()
        result = c2.execute("SELECT emb FROM embeddings WHERE word=?", ("apple",)).fetchone()
        result = result if result is None else array("f", result[0]).tolist()

    assert embs[0] == result

    # Test 4: Check old wiki2vec entity embedding is the same as new wiki2vec
    # entity embedding (old requires wikipedia input, new requires wikidata
    # input):
    mentions = ["Q84"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "entity")
    with sqlite3.connect("resources/rel_db/generic/entity_word_embedding.db") as conn2:
        c2 = conn2.cursor()
        result = c2.execute(
            "SELECT emb FROM embeddings WHERE word=?", ("ENTITY/London",)
        ).fetchone()
        result = result if result is None else array("f", result[0]).tolist()

    assert embs[0] == result

    # Test 5: Check old wiki2vec entity embedding is the same as new wiki2vec
    # entity embedding (old requires wikipedia input, new requires wikidata
    # input):
    mentions = ["Q84"]
    embs = rel_utils.get_db_emb("resources/rel_db/embedding_database.db", mentions, "entity")
    with sqlite3.connect("resources/rel_db/generic/entity_word_embedding.db") as conn2:
        c2 = conn2.cursor()
        result = c2.execute("SELECT emb FROM embeddings WHERE word=?", ("London",)).fetchone()
        result = result if result is None else array("f", result[0]).tolist()

    assert embs[0] != result


def test_prepare_initial_data():
    df = pd.read_csv("experiments/outputs/data/lwm/linking_df_split.tsv", sep="\t").iloc[:1]
    parsed_doc = rel_utils.prepare_initial_data(df, context_len=100)
    assert parsed_doc["4939308"][0]["mention"] == "STALYBRIDGE"
    assert parsed_doc["4939308"][0]["gold"][0] == "Q1398653"
    assert parsed_doc["4939308"][3]["mention"] == "Market-street"
    assert parsed_doc["4939308"][3]["gold"] == "NIL"


def test_train():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )

    myranker = ranking.Ranker(
        method="perfectmatch",
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

    mylinker = linking.Linker(
        method="reldisamb",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params={
            "model_path": "resources/models/disambiguation/",
            "data_path": "experiments/outputs/data/lwm/",
            "training_split": "originalsplit",
            "context_length": 100,
            "topn_candidates": 10,
            "db_embeddings": "resources/rel_db/embedding_database.db",
            "with_publication": False,
            "with_microtoponyms": False,
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
    myranker.mentions_to_wikidata = myranker.load_resources()
    # Train a DeezyMatch model if needed:
    myranker.train()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    mylinker.linking_resources = mylinker.load_resources()
    # Train a linking model if needed (it requires myranker to generate potential
    # candidates to the training set):
    mylinker.rel_params["ed_model"] = mylinker.train_load_model(myranker)

    assert type(mylinker.rel_params["ed_model"]) == entity_disambiguation.EntityDisambiguation

    # assert expected performance on test set
    assert mylinker.rel_params["ed_model"].best_performance["f1"] == 0.6422976501305483


def test_load_eval_model():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
    )

    myranker = ranking.Ranker(
        method="perfectmatch",
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

    mylinker = linking.Linker(
        method="reldisamb",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params={
            "model_path": "resources/models/disambiguation/",
            "data_path": "experiments/outputs/data/lwm/",
            "training_split": "originalsplit",
            "context_length": 100,
            "topn_candidates": 10,
            "db_embeddings": "resources/rel_db/embedding_database.db",
            "with_publication": False,
            "with_microtoponyms": False,
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
    myranker.mentions_to_wikidata = myranker.load_resources()
    # Train a DeezyMatch model if needed:
    myranker.train()

    # -----------------------------------------
    # Linker loading resources:
    # Load linking resources:
    mylinker.linking_resources = mylinker.load_resources()
    # Train a linking model if needed (it requires myranker to generate potential
    # candidates to the training set):
    mylinker.rel_params["ed_model"] = mylinker.train_load_model(myranker)

    assert type(mylinker.rel_params["ed_model"]) == entity_disambiguation.EntityDisambiguation


def test_predict():
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
        pipe=None,  # We'll store the NER pipeline here
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
        model_path="resources/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,
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

    mylinker = linking.Linker(
        method="reldisamb",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params={
            "model_path": "resources/models/disambiguation/",
            "data_path": "experiments/outputs/data/lwm/",
            "training_split": "originalsplit",
            "context_length": 100,
            "topn_candidates": 10,
            "db_embeddings": "resources/rel_db/embedding_database.db",
            "with_publication": False,
            "with_microtoponyms": False,
            "do_test": False,
        },
        overwrite_training=False,
    )

    mypipe = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    predictions = mypipe.run_text(
        "I live in Liverpool and neither in Barbens. I don't live in Manchester but in Allerton. There was an adjourned meeting of miners in Ashton-cnder-Lyne.",
        place="London",
        place_wqid="Q84",
    )
    assert type(predictions) == list
