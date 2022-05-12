import pandas as pd
import numpy as np
import json
from tqdm import tqdm

"""
This script selects and exports entity embeddings of entities that appear in our gazetteer.
"""

print("\nLoading resources...")
# These embeddings and mapped entities are downloaded from
# https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html
# Downloaded on Apr 11 2022.
embeddings = np.load(
    "/resources/wikidata/wikidata_translation_v1_vectors.npy", mmap_mode="r"
)
with open("/resources/wikidata/wikidata_translation_v1_names.json") as in_json:
    names = json.load(in_json)

# Load Wikidata-based gazetteer:
gazetteer = pd.read_csv("/resources/wikidata/wikidata_gazetteer.csv", low_memory=False)
gazetteer_ids = set(gazetteer.wikidata_id)

print("\nSize of gazetteer:", len(gazetteer_ids))

print("\nStart!")
wiki_entities = []
wiki_embeddings = []
for i in tqdm(range(len(names))):
    current_wid = names[i].split("/")[-1]
    if current_wid.startswith("Q"):
        current_wid = current_wid[:-1]
        current_emb = embeddings[i]
        if current_wid in gazetteer_ids:
            wiki_entities.append(current_wid)
            wiki_embeddings.append(current_emb)

# Store the embeddings:
np.save(
    open("/resources/wikidata/gazetteer_entity_embeddings.npy", "wb"),
    np.array(wiki_embeddings),
)
# Store the mapped Wikidata IDs:
open("/resources/wikidata/gazetteer_entity_ids.txt", "w").write(
    "\n".join(wiki_entities)
)
