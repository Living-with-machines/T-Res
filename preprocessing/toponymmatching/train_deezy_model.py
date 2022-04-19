from DeezyMatch import train as dm_train
from DeezyMatch import inference as dm_inference
from DeezyMatch import combine_vecs
from pathlib import Path
import pandas as pd
import numpy as np
import urllib
import time
import json
import re


pairs_dataset = "ocr_string_pairs" # String pairs dataset
dm_model = "ocr_avgpool" # Name of the resulting DeezyMatch model
candidates = "wkdtalts" # Name of the resulting Wikidata candidate list

# Path to DeezyMatch outputs:
deezymatch_outputs_path = "../../experiments/outputs/deezymatch/"

# --------------------------------------
# TRAIN THE OCR DEEZYMATCH MODEL
# --------------------------------------

# If model does not exist already, train a new model:
if not Path(deezymatch_outputs_path + "models/" + dm_model + '/' + dm_model + '.model').is_file():
    # train a new model
    dm_train(input_file_path="./input_dfm.yaml",
         dataset_path=deezymatch_outputs_path + "datasets/" + pairs_dataset + ".txt",
         model_name=dm_model)
    

# --------------------------------------
# GENERATE AND COMBINE CANDIDATE VECTORS
# --------------------------------------

# Obtain Wikidata candidates (relies on having run code in preprocessing/wikipediaprocessing/):
wikidata_path = '/resources/wikidata/'
with open(wikidata_path + 'mentions_to_wikidata.json', 'r') as f:
    mentions_to_wikidata_normalized = json.load(f)
unique_placenames_array = list(set(list(mentions_to_wikidata_normalized.keys())))
unique_placenames_array = [" ".join(x.strip().split("\t")) for x in unique_placenames_array if x]
with open(deezymatch_outputs_path + "datasets/" + candidates + ".txt", 'w') as f:
    f.write("\n".join(map(str, unique_placenames_array)))

# Generate vectors for candidates (specified in dataset_path) 
# using a model stored at pretrained_model_path and pretrained_vocab_path 
if not Path(deezymatch_outputs_path + "candidate_vectors/" + candidates + "_" + dm_model + "/embeddings/").is_dir():
    start_time = time.time()
    dm_inference(input_file_path=deezymatch_outputs_path + "models/" + dm_model + "/input_dfm.yaml",
                    dataset_path=deezymatch_outputs_path + "datasets/" + candidates + ".txt", 
                    pretrained_model_path=deezymatch_outputs_path + "models/" + dm_model + "/" + dm_model + ".model", 
                    pretrained_vocab_path=deezymatch_outputs_path + "models/" + dm_model + "/" + dm_model + ".vocab",
                    inference_mode="vect",
                    scenario=deezymatch_outputs_path + "candidate_vectors/" + candidates + "_" + dm_model)
    elapsed = time.time() - start_time
    print("Generate candidate vectors: %s" % elapsed)

# Combine vectors stored in the scenario in candidates/ and save them in combined/
if not Path(deezymatch_outputs_path + "combined/" + candidates + "_" + dm_model).is_dir():
    start_time = time.time()
    combine_vecs(rnn_passes=["fwd", "bwd"], 
                    input_scenario=deezymatch_outputs_path + "candidate_vectors/" + candidates + "_" + dm_model, 
                    output_scenario=deezymatch_outputs_path + "combined/" + candidates + "_" + dm_model, 
                    print_every=1000)
    elapsed = time.time() - start_time
    print("Combine candidate vectors: %s" % elapsed)