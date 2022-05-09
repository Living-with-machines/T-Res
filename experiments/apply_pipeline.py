import datetime
import itertools
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))

import pandas as pd
from pandarallel import pandarallel
from resolution_pipeline import ELPipeline
from transformers import pipeline

from utils import process_data, ner, ranking, linking


parser = ArgumentParser()
parser.add_argument(
    "-t", "--test", dest="test", help="run in test mode", action="store_true"
)
args = parser.parse_args()


# Named entity recognition approach, options are:
# * rel
# * lwm
ner_model_id = "lwm"

# Candidate selection approach, options are:
# * perfectmatch
# * partialmatch
# * levenshtein
# * deezymatch
cand_select_method = "deezymatch"

# Toponym resolution approach, options are:
# * mostpopular
# * mostpopularnormalised
top_res_method = "mostpopular"

# Entities considered for linking, options are:
# * all: for experiments comparing with other datasets and methods
# * loc: for application to newspapers
accepted_labels_str = "loc"

# Initiate the ranker object:
myranker = ranking.Ranker(
    method=cand_select_method,
    resources_path="/resources/wikidata/",
    mentions_to_wikidata=dict(),
    deezy_parameters={
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/deezymatch/",
        "dm_cands": "wkdtalts",
        "dm_model": "ocr_avgpool",
        "dm_output": "deezymatch_on_the_fly",
        # Ranking measures:
        "ranking_metric": "faiss",
        "selection_threshold": 10,
        "num_candidates": 3,
        "search_size": 3,
        "use_predict": False,
        "verbose": False,
    },
)

# Initiate the linker object:
mylinker = linking.Linker(
    method=top_res_method,
    accepted_labels=accepted_labels_str,
    do_training=False,
    training_csv="/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/data/lwm/linking_df_train.tsv",
    resources_path="/resources/wikidata/",
    linking_resources=dict(),
    myranker=myranker,
)

# END OF USER INPUT
# -----------------------------------------------------

# Load the ranker and linker resources:
print("*** Loading the resources...")
myranker.mentions_to_wikidata = myranker.load_resources()
mylinker.linking_resources = mylinker.load_resources()
print("*** Resources loaded!\n")


# Parallelize if ranking method is one of the following:
if myranker.method in ["partialmatch", "levenshtein"]:
    pandarallel.initialize(nb_workers=10)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


start = datetime.datetime.now()

ner_model_id = "lwm"
ner_model = "outputs/models/" + ner_model_id + "-ner.model"
ner_pipe = pipeline("ner", model=ner_model)

gold_positions = []
dataset = "hmd"

# Print the contents fo the ranker and linker objects:
print(myranker)
print(mylinker)

# Instantiate the entity linking pipeline:
end_to_end = ELPipeline(
    ner_model_id=ner_model_id,
    myranker=myranker,
    mylinker=mylinker,
    ner_pipe=ner_pipe,
    dataset=dataset,
)

print("Start!")

hmd_files = [
    "0002643_plaintext.csv",
]

folder = "../resources/hmd-samples/hmd_data_extension_words/"
Path(folder + "results/").mkdir(parents=True, exist_ok=True)

for dataset_name in hmd_files:

    dataset = pd.read_csv(folder + dataset_name)

    # Add metadata columns: publication_code, year, month, day, and article_path
    dataset[["publication_code", "year", "monthday", "article_path"]] = dataset[
        "article_path"
    ].str.split("/", expand=True)
    dataset["month"] = dataset["monthday"].str[:2]
    dataset["day"] = dataset["monthday"].str[2:]
    dataset = dataset.drop(columns=["Unnamed: 0", "monthday"])

    months = list(dataset.month.unique())
    years = list(dataset.year.unique())

    for month, year in list(itertools.product(months, years)):
        print(dataset_name)

        output_name_toponyms = (
            dataset_name.replace(".csv", "") + "_" + year + month + "_toponyms.json"
        )
        output_name_metadata = (
            dataset_name.replace(".csv", "") + "_" + year + month + "_metadata.json"
        )

        if not Path(folder + "results/" + output_name_toponyms).exists() or args.test:
            print("*", month, year)

            dataset_tmp = dataset.copy()
            dataset_tmp = dataset_tmp[
                (dataset_tmp["month"] == month) & (dataset_tmp["year"] == year)
            ]

            if not dataset_tmp.empty:
                print(dataset_tmp)
                dataset_tmp["toponyms"] = dataset_tmp.apply(
                    lambda row: end_to_end.run(row["target_sentence"])[
                        "predicted_ents"
                    ],
                    axis=1,
                )

                metadata_dict = dataset_tmp[
                    ["article_path", "hits", "publication_code", "year", "month", "day"]
                ].to_dict("index")
                output_dict = dict(zip(dataset_tmp.index, dataset_tmp.toponyms))

                with open(folder + "results/" + output_name_toponyms, "w") as fp:
                    json.dump(output_dict, fp)
                with open(folder + "results/" + output_name_metadata, "w") as fp:
                    json.dump(metadata_dict, fp)

        # If this is a test, break after having parsed one subset of the data:
        if args.test:
            break

end = datetime.datetime.now()
print(end - start)
