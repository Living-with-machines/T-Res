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
from pipeline import ELPipeline
from transformers import pipeline

parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test", help="run in test mode", action="store_true")
args = parser.parse_args()


pandarallel.initialize(progress_bar=True, nb_workers=24)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Ranking parameters for DeezyMatch:
myranker = dict()
myranker["ranking_metric"] = "faiss"
myranker["selection_threshold"] = 20
myranker["num_candidates"] = 1
myranker["search_size"] = 1
# Path to DeezyMatch model and combined candidate vectors:
myranker["dm_path"] = "outputs/deezymatch/"
myranker["dm_cands"] = "wkdtalts"
myranker["dm_model"] = "ocr_avgpool"
myranker["dm_output"] = "deezymatch_on_the_fly"

# Instantiate a dictionary to keep linking parameters:
mylinker = dict()

accepted_labels = ["loc", "b-loc", "i-loc"]

start = datetime.datetime.now()

ner_model_id = "lwm"
ner_model = "outputs/models/" + ner_model_id + "-ner.model"
ner_pipe = pipeline("ner", model=ner_model)

cand_select_method = "deezymatch"
top_res_method = "mostpopular"

gold_positions = []
dataset = "hmd"

end_to_end = ELPipeline(
    ner_model_id=ner_model_id,
    cand_select_method=cand_select_method,
    top_res_method=top_res_method,
    myranker=myranker,
    mylinker=mylinker,
    accepted_labels=accepted_labels,
    ner_pipe=ner_pipe,
)

print("Start!")

hmd_files = [
    "0002642_plaintext.csv",  # London liberal 1846-1868
    "0002645_plaintext.csv",  # London conservative 1853-1858
    "0002085_plaintext.csv",  # Merseyside conservative 1860-1861
    "0002088_plaintext.csv",  # Merseyside conservative 1832-1854
    "0002194_plaintext.csv",  # The Sun
]

folder = "../resources/hmd-samples/hmd_data_extension_words/"
Path(folder + "results/").mkdir(parents=True, exist_ok=True)

for dataset_name in hmd_files:

    dataset = pd.read_csv(folder + dataset_name)

    if args.test:
        dataset = dataset[:10]

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
                dataset_tmp["toponyms"] = dataset_tmp.apply(
                    lambda row: end_to_end.run(row["target_sentence"])["predicted_ents"], axis=1
                )

                metadata_dict = dataset[
                    ["article_path", "hits", "publication_code", "year", "month", "day"]
                ].to_dict("index")
                output_dict = dict(zip(dataset_tmp.index, dataset_tmp.toponyms))

                with open(folder + "results/" + output_name_toponyms, "w") as fp:
                    json.dump(output_dict, fp)
                with open(folder + "results/" + output_name_metadata, "w") as fp:
                    json.dump(metadata_dict, fp)

end = datetime.datetime.now()
print(end - start)
