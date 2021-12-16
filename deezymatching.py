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

Path('outputs/deezymatch/').mkdir(parents=True, exist_ok=True)

# --------------------------------------
# TRAIN THE OCR DEEZYMATCH MODEL
# --------------------------------------

# If model does not exist already, train a new model:
if not Path('outputs/deezymatch/models/ocr_faiss/ocr_faiss.model').is_file():
    # train a new model
    dm_train(input_file_path="resources/deezymatch/input_dfm.yaml",
         dataset_path="resources/ocr/ocr_string_pairs.txt",
         model_name="ocr_faiss")
    

# --------------------------------------
# GENERATE AND COMBINE CANDIDATE VECTORS
# --------------------------------------

##### UTILS

# Formatting candidates for DeezyMatch innput:
def format_for_candranker(gazname, unique_placenames_array):
    """
    This function returns the unique alternate names in a given gazetteer
    in the format required by DeezyMatch candidate ranker."""
    Path("/".join(gazname.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    with open(gazname + ".txt", "w") as fw:
        for pl in unique_placenames_array:
            pl = urllib.parse.unquote(pl)
            pl = re.sub("\s+", " ", pl)
            if len(pl) <= 50:
                fw.write(pl.strip() + "\t0\tfalse\n")
                

# Generating and combining candidate vectors:
def findcandidates(candidates, dm_model, inputfile):

    # generate vectors for candidates (specified in dataset_path) 
    # using a model stored at pretrained_model_path and pretrained_vocab_path 
    if not Path("outputs/deezymatch/candidate_vectors/" + candidates + "_" + dm_model + "/embeddings/").is_dir():
        start_time = time.time()
        dm_inference(input_file_path="outputs/deezymatch/models/" + dm_model + "/input_dfm.yaml",
                     dataset_path="outputs/deezymatch/candidate_toponyms/" + candidates + ".txt", 
                     pretrained_model_path="outputs/deezymatch/models/" + dm_model + "/" + dm_model + ".model", 
                     pretrained_vocab_path="outputs/deezymatch/models/" + dm_model + "/" + dm_model + ".vocab",
                     inference_mode="vect",
                     scenario="outputs/deezymatch/candidate_vectors/" + candidates + "_" + dm_model)
        elapsed = time.time() - start_time
        print("Generate candidate vectors: %s" % elapsed)

    # combine vectors stored in the scenario in candidates/ and save them in combined/
    if not Path("outputs/deezymatch/combined/" + candidates + "_" + dm_model).is_dir():
        start_time = time.time()
        combine_vecs(rnn_passes=["fwd", "bwd"], 
                     input_scenario="outputs/deezymatch/candidate_vectors/" + candidates + "_" + dm_model, 
                     output_scenario="outputs/deezymatch/combined/" + candidates + "_" + dm_model, 
                     print_every=1000)
        elapsed = time.time() - start_time
        print("Combine candidate vectors: %s" % elapsed)
        

# --------------
# Generate candidate vectors for the Wikidata gazetteer altnames:

candidates = "wkdtalts"
dm_model = "ocr_faiss"
inputfile = "input_dfm"

wikidata_path = '/resources/wikidata/'
with open(wikidata_path + 'mentions_to_wikidata_normalized.json', 'r') as f:
    mentions_to_wikidata_normalized = json.load(f)
unique_placenames_array = list(set(list(mentions_to_wikidata_normalized.keys())))
unique_placenames_array = [" ".join(x.strip().split("\t")) for x in unique_placenames_array]

format_for_candranker("outputs/deezymatch/candidate_toponyms/" + candidates, unique_placenames_array)

findcandidates(candidates, dm_model, inputfile)