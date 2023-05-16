import os
import sys
from pathlib import Path

from geoparser import linking, ranking

myranker = ranking.Ranker(
    method="deezymatch",
    resources_path="../resources/wikidata/",
    mentions_to_wikidata=dict(),
    wikidata_to_mentions=dict(),
    wiki_filtering={
        "top_mentions": 3,  # Filter mentions to top N mentions
        "minimum_relv": 0.03,  # Filter mentions with more than X relv
    },
    strvar_parameters={
        # Parameters to create the string pair dataset:
        "ocr_threshold": 60,
        "top_threshold": 85,
        "min_len": 5,
        "max_len": 15,
    },
    deezy_parameters={
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": str(Path("outputs/deezymatch/").resolve()),
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
        "w2v_ocr_path": str(Path("outputs/models/").resolve()),
        "w2v_ocr_model": "w2v_*_news",
        "do_test": False,
    },
)

mylinker = linking.Linker(
    method="reldisamb",
    resources_path="../resources/",
    linking_resources=dict(),
    base_model="to-be-removed",  # Base model for vector extraction
    rel_params={
        "base_path": "../resources/rel_db/",
        "wiki_version": "wiki_2019/",
        "training_data": "lwm",  # lwm, aida
        "ranking": "relv",  # relv, publ
        "micro_locs": "nil",  # "dist", "nil", ""
    },
    overwrite_training=False,
)

CONFIG = {"myranker": myranker, "mylinker": mylinker}
