import os
import sys
from pathlib import Path
from geoparser import ranking, linking

myranker = ranking.Ranker(
    method="perfectmatch",
    resources_path="../resources/wikidata/",
    mentions_to_wikidata=dict(),
    wikidata_to_mentions=dict(),
)

mylinker = linking.Linker(
    method="mostpopular",
    resources_path="../resources/wikidata/",
    linking_resources=dict(),
    base_model="to-be-removed",  # Base model for vector extraction
    rel_params={},
    overwrite_training=False,
)

CONFIG = {"myranker": myranker, "mylinker": mylinker}