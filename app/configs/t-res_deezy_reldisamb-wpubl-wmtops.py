import os
import sys
import sqlite3
from pathlib import Path
# sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import pipeline, ranking, linking

# --------------------------------------
# Instantiate the ranker:
myranker = ranking.Ranker(
    method="deezymatch",
    resources_path="../resources/wikidata/",
    mentions_to_wikidata=dict(),
    wikidata_to_mentions=dict(),
    strvar_parameters={
        # Parameters to create the string pair dataset:
        "ocr_threshold": 60,
        "top_threshold": 85,
        "min_len": 5,
        "max_len": 15,
        "w2v_ocr_path": str(Path("../resources/models/w2v/").resolve()),
        "w2v_ocr_model": "w2v_*_news",
        "overwrite_dataset": False,
    },
    deezy_parameters={
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": str(Path("../resources/deezymatch/").resolve()),
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

with sqlite3.connect("../resources/rel_db/embeddings_database.db") as conn:
    cursor = conn.cursor()
    mylinker = linking.Linker(
        method="reldisamb",
        resources_path="../resources/",
        linking_resources=dict(),
        rel_params={
            "model_path": "../resources/models/disambiguation/",
            # "data_path": "../experiments/outputs/data/lwm/",
            "data_path": "outputs/data/lwm/",
            "training_split": "originalsplit",
            "context_length": 100,
            "db_embeddings": cursor,
            "with_publication": True,
            "without_microtoponyms": True,
            "do_test": False,
            "default_publname": "United Kingdom",
            "default_publwqid": "Q145",
        },
        overwrite_training=False,
    )

# geoparser = pipeline.Pipeline(myranker=myranker, mylinker=mylinker)
CONFIG = {"myranker": myranker, "mylinker": mylinker}
