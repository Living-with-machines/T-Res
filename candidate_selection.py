import pandas as pd
from pathlib import Path
import time
from DeezyMatch import inference as dm_inference
from DeezyMatch import combine_vecs
from DeezyMatch import candidate_ranker


# ---------------------
# Create file with queries from the LwM dataset
# ---------------------
lwmdf = pd.read_csv("outputs/data/lwm_df.tsv", sep="\t")

dm_queries = "lwm"
queries_path = "outputs/deezymatch/query_toponyms/"
Path(queries_path).mkdir(parents=True, exist_ok=True)

with open(queries_path + dm_queries + ".txt", "w") as fw:
    lqueries = []
    for i, row in lwmdf.iterrows():
        if row["place_class"] == "LOCWiki":
            wqid = row["place_wqid"]
            if not type(wqid) == str:
                wqid = "UNKNOWN"
            lqueries.append(row["mention"] + "\t" + wqid + "\tfalse\n")
    lqueries = list(set(lqueries))
    for q in lqueries:
        fw.write(q)
        

# ---------------------
# Use DeezyMatch for inference.
# ---------------------

# Specify parameters
dm_model = "ocr_faiss"
dm_queries = "lwm"
dm_cands = "wkdtalts"
inputfile = "input_dfm"
candrank_metric = "faiss" # 'faiss', 'cosine', 'conf'
candrank_thr = 50
num_candidates = 20

# generate vectors for queries (specified in dataset_path) 
# using a model stored at pretrained_model_path and pretrained_vocab_path 
start_time = time.time()
dm_inference(input_file_path="outputs/deezymatch/models/" + dm_model + "/" + inputfile + ".yaml",
             dataset_path="outputs/deezymatch/query_toponyms/" + dm_queries + ".txt", 
             pretrained_model_path="outputs/deezymatch/models/" + dm_model + "/" + dm_model + ".model", 
             pretrained_vocab_path="outputs/deezymatch/models/" + dm_model + "/" + dm_model + ".vocab",
             inference_mode="vect",
             scenario="outputs/deezymatch/query_vectors/" + dm_queries + "_" + dm_model)
elapsed = time.time() - start_time
print("Generate query vectors: %s" % elapsed)


# combine vectors stored in the scenario in queries/ and save them in combined/
start_time = time.time()
combine_vecs(rnn_passes=["fwd", "bwd"], 
             input_scenario="outputs/deezymatch/query_vectors/" + dm_queries + "_" + dm_model, 
             output_scenario="outputs/deezymatch/combined/" + dm_queries + "_" + dm_model, 
             print_every=1000)
elapsed = time.time() - start_time
print("Combine query vectors: %s" % elapsed)


# Select candidates based on L2-norm distance (aka faiss distance):
# find candidates from candidate_scenario 
# for queries specified in query_scenario
start_time = time.time()
candidates_pd = \
    candidate_ranker(query_scenario="outputs/deezymatch/combined/" + dm_queries + "_" + dm_model,
                     candidate_scenario="outputs/deezymatch/combined/" + dm_cands + "_" + dm_model, 
                     ranking_metric=candrank_metric, 
                     selection_threshold=candrank_thr, 
                     num_candidates=num_candidates, 
                     search_size=num_candidates, 
                     output_path="outputs/deezymatch/ranker_results/" + dm_queries + "_" + dm_cands + "_" + dm_model + "_" + candrank_metric + str(num_candidates), 
                     pretrained_model_path="outputs/deezymatch/models/" + dm_model + "/" + dm_model + ".model", 
                     pretrained_vocab_path="outputs/deezymatch/models/" + dm_model + "/" + dm_model + ".vocab")
elapsed = time.time() - start_time
print("Rank candidates: %s" % elapsed)


print(pd.read_pickle("outputs/deezymatch/ranker_results/" + dm_queries + "_" + dm_cands + "_" + dm_model + "_" + candrank_metric + str(num_candidates) + ".pkl"))