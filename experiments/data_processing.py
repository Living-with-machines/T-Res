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
Path(output_path_lwm).mkdir(parents=True, exist_ok=True)

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

# Leave out one location just for test and one for dev.
# We'll leave Ashton-under-Lyne and Poole for that, because
# they are our smaller subsets by place, as you can see
# below (number of annotated articles per place):
# * Manchester           216
# * Dorchester           143
# * Poole                 48
# * Ashton-under-Lyne     48

# Split test set into dev and test set, by article:
lwm_dev_df, lwm_test_df = train_test_split(lwm_test_df, test_size=0.5, random_state=42)

# Add the left-out place of publication articles to the corresponding split
# We keep Ashton exclusively for dev:
mask_ashton_train = lwm_train_df["place"] == "Ashton-under-Lyne"
mask_ashton_test = lwm_test_df["place"] == "Ashton-under-Lyne"
lwm_dev_df = pd.concat(
    [lwm_dev_df, lwm_train_df[mask_ashton_train], lwm_test_df[mask_ashton_test]]
)
lwm_train_df = lwm_train_df[~mask_ashton_train]
lwm_test_df = lwm_test_df[~mask_ashton_test]

# Wee keep Poole exclusively for test:
mask_poole_train = lwm_train_df["place"] == "Poole"
mask_poole_dev = lwm_dev_df["place"] == "Poole"
lwm_test_df = pd.concat(
    [lwm_test_df, lwm_train_df[mask_poole_train], lwm_dev_df[mask_poole_dev]]
)
lwm_train_df = lwm_train_df[~mask_poole_train]
lwm_dev_df = lwm_dev_df[~mask_poole_dev]

print("\nLwM dataset")
print("===========")
print("* Size of the toponym resolution train set:", lwm_train_df.shape[0])
print("* Size of the toponym resolution dev set:", lwm_dev_df.shape[0])
print("* Size of the toponym resolution test set:", lwm_test_df.shape[0])
print(
    "\n* Places of publication in the train set: "
    + ", ".join(list(lwm_train_df.place.unique()))
)
print(
    "* Places of publication in the dev set: "
    + ", ".join(list(lwm_dev_df.place.unique()))
)
print(
    "* Places of publication in the test set: "
    + ", ".join(list(lwm_test_df.place.unique()))
)
print()

# Store dataframes:
lwm_train_df.to_csv(output_path_lwm + "linking_df_train.tsv", sep="\t", index=False)
lwm_dev_df.to_csv(output_path_lwm + "linking_df_dev.tsv", sep="\t", index=False)
lwm_test_df.to_csv(output_path_lwm + "linking_df_test.tsv", sep="\t", index=False)


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

# Split dev set into train and dev set, by article:
hipe_train_df, hipe_dev_df = train_test_split(
    hipe_dev_df, test_size=0.5, random_state=42
)

# Store dataframes:
hipe_train_df.to_csv(output_path_hipe + "hipe_train_df.tsv", sep="\t", index=False)
hipe_dev_df.to_csv(output_path_hipe + "linking_df_dev.tsv", sep="\t", index=False)
hipe_test_df.to_csv(output_path_hipe + "linking_df_test.tsv", sep="\t", index=False)

print("\nHIPE dataset")
print("============")
print("* Size of the toponym resolution train set:", hipe_train_df.shape[0])
print("* Size of the toponym resolution dev set:", hipe_dev_df.shape[0])
print("* Size of the toponym resolution test set:", hipe_test_df.shape[0])
