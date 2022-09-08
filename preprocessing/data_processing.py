import sys, os

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import pandas as pd
from pathlib import Path
from utils import get_data
from utils import preprocess_data
from sklearn.model_selection import train_test_split
import random
import json
import os

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

large_resources = "../resources/"  # path to large resources
small_resources = "../resources/"  # path to small resources
output_path_lwm = "../experiments/outputs/data/lwm/"
output_path_hipe = "../experiments/outputs/data/hipe/"
# Create output folders for processed data if they do not exist:
Path(output_path_lwm).mkdir(parents=True, exist_ok=True)
Path(output_path_hipe).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------
# Gazetteer data
# ------------------------------------------------------

# Load gazetteer (our knowledge base):
gazetteer_ids = set(
    list(
        pd.read_csv(
            os.path.join(large_resources, "wikidata", "wikidata_gazetteer.csv"),
            low_memory=False,
        )["wikidata_id"].unique()
    )
)


# ------------------------------------------------------
# Publication metadata
# ------------------------------------------------------

# Load publication metadata
with open(os.path.join(f"{small_resources}", "publication_metadata.json")) as jsonfile:
    df_metadata = json.load(jsonfile)

dict_titles = {k: df_metadata[k]["publication_title"] for k in df_metadata}
dict_place = {k: df_metadata[k]["publication_place"] for k in df_metadata}
dict_placewqid = {k: df_metadata[k]["wikidata_qid"] for k in df_metadata}


# ------------------------------------------------------
# LWM dataset
# ------------------------------------------------------

# Path of the manually annotated data:
news_path = os.path.join(f"{small_resources}", "news_datasets")

# Download the annotated data from the BL repository:
get_data.download_lwm_data(news_path)

# Training data from the manually annotated data:
topres_path_train = os.path.join(
    f"{small_resources}", "news_datasets", "topRes19th_v2", "train"
)

# Test data from the manually annotated data:
topres_path_test = os.path.join(
    f"{small_resources}", "news_datasets", "topRes19th_v2", "test"
)

# Process data for training a named entity recognition model:
lwm_df = preprocess_data.process_lwm_for_ner(topres_path_train)

# Split NER-formatted training set into train and dev, and store them.
# They will be used by the ner_training.py script:
lwm_train_ner, lwm_dev_ner = train_test_split(
    lwm_df, test_size=0.2, random_state=RANDOM_SEED
)
lwm_train_ner.to_json(
    output_path_lwm + "ner_df_train.json", orient="records", lines=True
)
lwm_dev_ner.to_json(output_path_lwm + "ner_df_dev.json", orient="records", lines=True)

# Process data for the resolution experiments:
lwm_train_df = preprocess_data.process_lwm_for_linking(topres_path_train, gazetteer_ids)
lwm_test_df = preprocess_data.process_lwm_for_linking(topres_path_test, gazetteer_ids)

# Split train set into train and dev set, by article:
lwm_train_df, lwm_dev_df = train_test_split(
    lwm_train_df, test_size=0.33, random_state=RANDOM_SEED
)

# Concatenate original training and test sets:
lwm_all_df = pd.concat([lwm_train_df, lwm_dev_df, lwm_test_df])
lwm_all_df["place_wqid"] = lwm_all_df["publication_code"].map(dict_placewqid)

# Add a column for the ner_split (i.e. the original split)
lwm_all_df["originalsplit"] = lwm_all_df["article_id"].apply(
    lambda x: "test"
    if x in list(lwm_test_df["article_id"].unique())
    else "train"
    if x in list(lwm_train_df["article_id"].unique())
    else "dev"
)

# Split the train set into train and dev for development
# (i.e. when test is not used):
lwm_train_dev_df, lwm_dev_dev_df = train_test_split(
    lwm_train_df, test_size=0.33, random_state=RANDOM_SEED
)
# Add a column for the ner_split (i.e. the original split)
lwm_all_df["withouttest"] = lwm_all_df["article_id"].apply(
    lambda x: "test"
    if x in list(lwm_dev_df["article_id"].unique())
    else "train"
    if x in list(lwm_train_dev_df["article_id"].unique())
    else "dev"
    if x in list(lwm_dev_dev_df["article_id"].unique())
    else "left_out"
)

groups = [i for i, group in lwm_all_df.groupby(["place", "decade"])]
for group in groups:
    # Remove the split that will be the test set from the list of splits:
    remainers = [other_group for other_group in groups if not other_group == group]
    # Randomly pick one split from the remainers to be the dev set:
    dev_group = random.choice(remainers)
    # Name the experiment after the test split:
    group_name = group[0].split("-")[0] + str(group[1])
    # Assign each file to train, dev, or test for each experiment:
    train_test_column = []
    for i, row in lwm_all_df.iterrows():
        if row["place"] == group[0] and row["decade"] == group[1]:
            train_test_column.append("test")
        elif row["place"] == dev_group[0] and row["decade"] == dev_group[1]:
            train_test_column.append("dev")
        else:
            train_test_column.append("train")
    # Store the split in a column named after the experiment (after the test split):
    lwm_all_df[group_name] = train_test_column

# Store dataframe:
lwm_all_df.to_csv(
    os.path.join(f"{output_path_lwm}", "linking_df_split.tsv"),
    sep="\t",
    index=False,
)

print()
print("===================")
print("### LWM experiments")
print("===================\n")
for group in groups:
    group_name = group[0].split("-")[0] + str(group[1])
    print(lwm_all_df[group_name].value_counts())
print(lwm_all_df["originalsplit"].value_counts())
print(lwm_all_df["withouttest"].value_counts())
print()

# ------------------------------------------------------
# CLEF HIPE dataset
# ------------------------------------------------------

# Path to HIPE data:
hipe_path = os.path.join(f"{news_path}", "hipe")

# Download the annotated data from the BL repository:
get_data.download_hipe_data(hipe_path)

hipe_dev_df = preprocess_data.process_hipe_for_linking(
    os.path.join(f"{hipe_path}", "HIPE-2022-v2.1-hipe2020-dev-en.tsv"), gazetteer_ids
)

hipe_test_df = preprocess_data.process_hipe_for_linking(
    os.path.join(f"{hipe_path}", "HIPE-2022-v2.1-hipe2020-test-en.tsv"), gazetteer_ids
)

hipe_all_df = pd.concat([hipe_dev_df, hipe_test_df])
hipe_all_df["place"] = hipe_all_df["publication_code"].map(dict_place)
hipe_all_df["publication_title"] = hipe_all_df["publication_code"].map(dict_titles)
hipe_all_df["place_wqid"] = hipe_all_df["publication_code"].map(dict_placewqid)

# Split dev set into train and dev set, by article:
hipe_train_df, hipe_dev_df = train_test_split(
    hipe_dev_df, test_size=0.5, random_state=RANDOM_SEED
)
train_ids = list(hipe_train_df.article_id.unique())
dev_ids = list(hipe_dev_df.article_id.unique())
test_ids = list(hipe_dev_df.article_id.unique())

dev_test = []  # Original split: into dev and test only
train_dev_test = []  # Following the original split, but dev split into train and dev
for i, row in hipe_all_df.iterrows():
    if row["article_id"] in list(hipe_train_df.article_id.unique()):
        dev_test.append("dev")
        train_dev_test.append("dev")
    elif row["article_id"] in list(hipe_dev_df.article_id.unique()):
        dev_test.append("dev")
        train_dev_test.append("test")
    else:
        dev_test.append("test")
        train_dev_test.append("left_out")

# Store the split in a column named after the experiment:
hipe_all_df["originalsplit"] = dev_test
hipe_all_df["withouttest"] = train_dev_test

# Store dataframe:
hipe_all_df.to_csv(output_path_hipe + "linking_df_split.tsv", sep="\t", index=False)

print("===================")
print("### HIPE experiments")
print("===================\n")
for group_name in ["originalsplit", "withouttest"]:
    print(hipe_all_df[group_name].value_counts())
print()
