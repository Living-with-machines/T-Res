import os
import sys

import pandas as pd

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking
from utils import process_data

lwm_original_df = pd.read_csv(
    "../experiments/outputs/data/lwm/linking_df_split.tsv",
    sep="\t",
)

lwm_processed_df = pd.read_csv(
    "../experiments/outputs/data/lwm/blb_lwm-ner-fine_deezymatch+2+10_mentions.tsv", sep="\t"
)
lwm_processed_df = lwm_processed_df.drop(columns=["Unnamed: 0"])
lwm_processed_df["candidates"] = lwm_processed_df["candidates"].apply(
    process_data.eval_with_exception
)

# Instantiate the linker:
mylinker = linking.Linker(
    method="reldisamb:relcs",
    resources_path="/resources/wikidata/",
    linking_resources=dict(),
    base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
    rel_params={"base_path": "/resources/rel_db/", "wiki_version": "wiki_2019/"},
    gnn_params={
        "level": "sentence_id",
        "max_distance": 200,
        "similarity_threshold": 0.7,
        "model_path": "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/gnn_models/",
    },
    overwrite_training=False,
)
# print(mylinker.rel_disambiguation(lwm_processed_df, lwm_original_df))
