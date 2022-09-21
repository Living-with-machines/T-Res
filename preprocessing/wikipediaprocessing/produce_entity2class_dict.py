import pandas as pd
from ast import literal_eval
import json
from collections import Counter

gaz = pd.read_csv("../../resources/wikidata/wikidata_gazetteer.csv", low_memory=False)
gazetteer_ids = set(gaz.wikidata_id)


def eval_with_exception(string):
    try:
        return literal_eval(string)
    except ValueError:
        return []


# Get all classes in our gazetteer:
gaz["instance_of"] = gaz["instance_of"].apply(eval_with_exception)
instances_all = [i for l in gaz[~gaz.instance_of.isnull()].instance_of for i in l if l]
instances = set(instances_all)
instance_counter = Counter(instances_all)

print("\nSize of gazetteer:", len(gazetteer_ids))
print("Number of classes:", len(instances))

print("\nStart!")

dict_id_to_class = dict()
for i, row in gaz.iterrows():
    entity_classes = row["instance_of"]
    # Get most common class:
    keep_most_common_class = ""
    top_class_relv = 0
    for ec in entity_classes:
        current_class_relv = instance_counter.get(ec)
        if current_class_relv > top_class_relv:
            top_class_relv = current_class_relv
            keep_most_common_class = ec
    if keep_most_common_class:
        dict_id_to_class[row["wikidata_id"]] = keep_most_common_class

with open("../../resources/entity2class.txt", "w") as fw:
    fw.write(json.dumps(dict_id_to_class))
