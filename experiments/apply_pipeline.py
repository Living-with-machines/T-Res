import os
import re
import sys
import glob
import json
import datetime
import pandas as pd
from tqdm import tqdm
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import recogniser, ranking, linking
from utils import ner

# from pandarallel import pandarallel

# pandarallel.initialize(progress_bar=True, nb_workers=24)
# os.environ["TOKENIZERS_PARALLELISM"] = "true"


# --------------------------------------
# Instantiate the recogniser:
myner = recogniser.Recogniser(
    method="lwm",  # NER method
    model_name="blb_lwm-ner",  # NER model name preffix (will have suffixes appended)
    model=None,  # We'll store the NER model here
    pipe=None,  # We'll store the NER pipeline here
    base_model="/resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
    train_dataset="outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
    test_dataset="outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
    output_model_path="outputs/models/",  # Path where the NER model is or will be stored
    training_args={
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
    },
    overwrite_training=False,  # Set to True if you want to overwrite model if existing
    do_test=False,  # Set to True if you want to train on test mode
    training_tagset="fine",  # Options are: "coarse" or "fine"
)


# --------------------------------------
# Instantiate the ranker:
myranker = ranking.Ranker(
    method="deezymatch",
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
        "selection_threshold": 5,
        "num_candidates": 1,
        "search_size": 1,
        "use_predict": False,
        "verbose": False,
    },
)


# --------------------------------------
# Instantiate the linker:
mylinker = linking.Linker(
    method="mostpopular",
    resources_path="/resources/wikidata/",
    linking_resources=dict(),
    base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
    overwrite_training=False,
)

# --------------------------------------
# Load resources:
myner.model, myner.pipe = myner.create_pipeline()
myranker.mentions_to_wikidata = myranker.load_resources()
mylinker.linking_resources = mylinker.load_resources()


def end_to_end(sentence):

    # Perform NER:
    predictions = myner.ner_predict(sentence)
    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
        for x in predictions
    ]
    mentions = ner.aggregate_mentions(procpreds, "pred")

    # Perform candidate ranking:
    wk_cands, myranker.already_collected_cands = myranker.find_candidates(mentions)

    # Perform entity linking:
    linked_mentions = []
    for mention in mentions:
        linked_mention = dict()
        linked_mention["mention"] = mention["mention"]
        linked_mention["ner_label"] = mention["ner_label"]
        # Run entity linking per mention:
        selected_cand = mylinker.run({"candidates": wk_cands[mention["mention"]]})
        linked_mention["wqid"] = selected_cand[0]
        linked_mention["wqid_score"] = selected_cand[1]
        linked_mentions.append(linked_mention)
    return linked_mentions


# -----------------------------------------------
# Case study: query in newspapers.

print("Start!")
start = datetime.datetime.now()

query = "accident"
datasets = ["hmd", "lwm"]

output_path_csv = "../experiments/outputs/newspapers/csvs/"
output_path_resolved = "../experiments/outputs/newspapers/resolved/"
Path(output_path_csv).mkdir(parents=True, exist_ok=True)
Path(output_path_resolved).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# Filter data by query token:
for dataset in datasets:
    dataset_path = "/resources/" + dataset + "-newspapers/" + dataset + "_sentences/"

    for i in tqdm(glob.glob(dataset_path + "*.csv")):
        news_nlp = i.split("/")[-1].split("_")[0]
        path_nlp_query = output_path_csv + query + "_" + news_nlp + ".csv"

        if not Path(path_nlp_query).exists():
            df_query = pd.DataFrame(
                columns=["nlp_id", "article_id", "date", "sentence_id", "sentence"]
            )

            df = pd.read_csv(i, index_col=0, low_memory=False)
            mask = df["sentence"].str.contains(
                r"\b" + query + r"\b", re.IGNORECASE, na=False
            )
            df_query = pd.concat([df_query, df[mask]])

            df_query[["year", "month", "day"]] = df_query["date"].str.split(
                "-", -1, expand=True
            )

            df_query.to_csv(path_nlp_query, index=False)


# ----------------------------------------------------------
# Run toponym resolution:
for i in tqdm(glob.glob(output_path_csv + "*.csv")):

    if not query in i:
        continue

    news_nlp = i.split("/")[-1].split("_")[-1].split(".csv")[0]
    df_query = pd.read_csv(i)

    months = list(df_query.month.unique())
    years = list(df_query.year.unique())

    for year in years:
        for month in months:

            output_name_toponyms = (
                news_nlp + "/" + str(year) + str(month) + "_toponyms.json"
            )
            output_name_metadata = (
                news_nlp + "/" + str(year) + str(month) + "_metadata.json"
            )
            Path(output_path_resolved + news_nlp + "/").mkdir(
                parents=True, exist_ok=True
            )

            print(news_nlp, year, month)

            if not Path(output_path_resolved + output_name_toponyms).exists():
                df_tmp = df_query[
                    (df_query["month"] == month) & (df_query["year"] == year)
                ]

                if not df_tmp.empty:
                    df_tmp["toponyms"] = df_tmp.apply(
                        lambda row: end_to_end(row["sentence"]), axis=1
                    )

                    metadata_dict = df_tmp[
                        [
                            "nlp_id",
                            "article_id",
                            "date",
                            "sentence_id",
                            "year",
                            "month",
                            "day",
                        ]
                    ].to_dict("index")
                    output_dict = dict(zip(df_tmp.index, df_tmp.toponyms))

                    with open(output_path_resolved + output_name_toponyms, "w") as fp:
                        json.dump(output_dict, fp)
                    with open(output_path_resolved + output_name_metadata, "w") as fp:
                        json.dump(metadata_dict, fp)

end = datetime.datetime.now()
print(end - start)
