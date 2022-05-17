import sys, os

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import pandas as pd
from pathlib import Path
from utils import get_data
from utils import process_data
from sklearn.model_selection import train_test_split


# ------------------------------------------------------
# LWM dataset
# ------------------------------------------------------

# Download the annotated data from the BL repository:
# get_data.download_lwm_data()

# Path for the output dataset dataframes:
output_path_lwm = "outputs/data/lwm/"

# Path of the annotated data:
# !TODO: Change path to that where downloaded data is stored.
topres_path_train = "/resources/newsdataset/fmp_lwm/train/"
topres_path_test = "/resources/newsdataset/fmp_lwm/test/"

# Process data for training a named entity recognition model:
lwm_df = process_data.process_lwm_for_ner(topres_path_train)

# Split NER-formatted training set into train and dev, and store them.
# They will be used by the ner_training.py script:
lwm_train_ner, lwm_dev_ner = train_test_split(lwm_df, test_size=0.2, random_state=42)
lwm_train_ner.to_json(
    output_path_lwm + "ner_df_train.json", orient="records", lines=True
)
lwm_dev_ner.to_json(output_path_lwm + "ner_df_dev.json", orient="records", lines=True)

# Process data for resolution:
lwm_train_df = process_data.process_lwm_for_linking(topres_path_train)
lwm_test_df = process_data.process_lwm_for_linking(topres_path_test)

# Concatenate original training and test sets:
lwm_all_df = pd.concat([lwm_train_df, lwm_test_df])

groups = [i for i, group in lwm_all_df.groupby(["place", "decade"])]
for group in groups:
    group_name = group[0].split("-")[0] + str(group[1])
    train_test_column = []
    for i, row in lwm_all_df.iterrows():
        if row["place"] == group[0] and row["decade"] == group[1]:
            train_test_column.append("test")
        else:
            train_test_column.append("train")

    lwm_all_df[group_name] = train_test_column

# Store dataframe:
lwm_all_df.to_csv(
    output_path_lwm + "linking_df_split.tsv",
    sep="\t",
    index=False,
)

# ------------------------------------------------------
# CLEF HIPE dataset
# ------------------------------------------------------

# Path for the output dataset dataframes:
output_path_hipe = "outputs/data/hipe/"
Path(output_path_hipe).mkdir(parents=True, exist_ok=True)

# Path to folder with HIPE original data (v1.4):
hipe_path = "/resources/newsdataset/clef_hipe/"

hipe_dev_df = process_data.process_hipe_for_linking(
    hipe_path + "HIPE-data-v1.4-dev-en.tsv"
)
hipe_test_df = process_data.process_hipe_for_linking(
    hipe_path + "HIPE-data-v1.4-test-en.tsv"
)

hipe_all_df = pd.concat([hipe_dev_df, hipe_test_df])

all_publications = sorted(list(hipe_all_df.publication_code.unique()))

exp_n = 0
for comb_group in list(process_data.split(all_publications, 3)):
    exp_n += 1
    train_test_column = []
    for i, row in hipe_all_df.iterrows():
        if row["publication_code"] in comb_group:
            train_test_column.append("test")
        else:
            train_test_column.append("train")

    hipe_all_df["Exp" + str(exp_n)] = train_test_column

# Store dataframe:
hipe_all_df.to_csv(
    output_path_hipe + "linking_df_split.tsv",
    sep="\t",
    index=False,
)
