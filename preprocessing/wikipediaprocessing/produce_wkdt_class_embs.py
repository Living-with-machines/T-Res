import pandas as pd
from ast import literal_eval
import json
import numpy as np
from tqdm import tqdm

print("\nLoading resources...")
# These embeddings and mapped entities are downloaded from
# https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html
# Downloaded on Apr 11 2022.
embeddings = np.load(
    "/resources/wikidata/wikidata_translation_v1_vectors.npy", mmap_mode="r"
)
with open("/resources/wikidata/wikidata_translation_v1_names.json") as in_json:
    names = json.load(in_json)

gaz = pd.read_csv("/resources/wikidata/wikidata_gazetteer.csv", low_memory=False)
gazetteer_ids = set(gaz.wikidata_id)


def eval_with_exception(string):
    try:
        return literal_eval(string)
    except ValueError:
        return None


# Get all classes in our gazetteer:
gaz["instance_of"] = gaz["instance_of"].apply(eval_with_exception)
instances_all = [i for l in gaz[~gaz.instance_of.isnull()].instance_of for i in l if l]
instances = set(instances_all)

print("\nSize of gazetteer:", len(gazetteer_ids))
print("\nNumber of classes:", len(instances))

print("\nStart!")
class_entities = []
class_embeddings = []
for i in tqdm(range(len(names))):
    current_wid = names[i].split("/")[-1]
    if current_wid.startswith("Q"):
        current_wid = current_wid[:-1]
        current_emb = embeddings[i]
        if current_wid in instances:
            class_entities.append(current_wid)
            class_embeddings.append(current_emb)


# Store the embeddings:
np.save(
    open("/resources/wikidata/gazetteer_wkdtclass_embeddings.npy", "wb"),
    np.array(class_embeddings),
)
# Store the mapped Wikidata IDs:
open("/resources/wikidata/gazetteer_wkdtclass_ids.txt", "w").write(
    "\n".join(class_entities)
)
