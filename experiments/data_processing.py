import glob
import json
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
output_path = "outputs/data/lwm/"
Path(output_path).mkdir(parents=True, exist_ok=True)

# Path of the annotated data:
# !TODO: Change path to that where downloaded data is stored.
topres_path_train = "/resources/newsdataset/fmp_lwm/train/"
topres_path_test = "/resources/newsdataset/fmp_lwm/test/"

# Process data for training a named entity recognition model:
lwm_df = process_data.process_lwm_for_ner(topres_path_train)

# Split training set into train and set, for development, and store them:
train, test = train_test_split(lwm_df, test_size=0.2, random_state=42)
train.to_json(output_path + 'ner_lwm_df_train.json', orient="records", lines=True)
test.to_json(output_path + 'ner_lwm_df_test.json', orient="records", lines=True)

# Process data for resolution:
train_df = process_data.process_lwm_for_linking(topres_path_train, output_path)
test_df = process_data.process_lwm_for_linking(topres_path_test, output_path)

# Store dataframes:
train_df.to_csv(output_path + "linking_lwm_df_train.tsv", sep="\t", index=False)
test_df.to_csv(output_path + "linking_lwm_df_test.tsv", sep="\t", index=False)

# ------------------------------------------------------
# CLEF HIPE dataset
# ------------------------------------------------------

# Path for the output dataset dataframes:
output_path = "outputs/data/"
Path(output_path).mkdir(parents=True, exist_ok=True)

hipe_path = "/resources/newsdataset/clef_hipe/"

# Process data for training a named entity recognition model:
hipe_df = process_data.process_hipe_for_linking(hipe_path + "HIPE-data-v1.4-dev-en.tsv", output_path, "dev")
hipe_df = process_data.process_hipe_for_linking(hipe_path + "HIPE-data-v1.4-test-en.tsv", output_path, "test")