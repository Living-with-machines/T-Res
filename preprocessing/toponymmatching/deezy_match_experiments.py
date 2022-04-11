import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml
from DeezyMatch import candidate_ranker, combine_vecs
from DeezyMatch import inference as dm_inference
from DeezyMatch import train as dm_train
from tqdm import tqdm

# Prepare output folders:
Path("./experiments/inputs/").mkdir(parents=True, exist_ok=True)
Path("./experiments/resources/").mkdir(parents=True, exist_ok=True)
Path("./experiments/results/").mkdir(parents=True, exist_ok=True)

# Create a good-sized list of candidates from our training set:
train_df = pd.read_csv(
    "../../experiments/outputs/data/lwm/linking_df_train_cands_deezymatch.tsv", sep="\t"
)

# Load Wikidata mentions-to-wqid:
wikidata_path = "/resources/wikidata/"
with open(wikidata_path + "mentions_to_wikidata.json", "r") as f:
    mentions_to_wikidata = json.load(f)

# Reverse dictionary, to have wikidata id as key and altnames as a list as value:
mentions_to_wikidata_keys = dict()
wikidata_to_mentions = dict()
for k in mentions_to_wikidata:
    mentions_to_wikidata_keys[k] = list(mentions_to_wikidata[k].keys())
    for nk in list(mentions_to_wikidata[k].keys()):
        if nk in wikidata_to_mentions:
            wikidata_to_mentions[nk].append(k)
        else:
            wikidata_to_mentions[nk] = [k]

with open("experiments/resources/wikidata_to_mentions.json", "w") as fp:
    json.dump(wikidata_to_mentions, fp)

with open("experiments/resources/mentions_to_wikidata.json", "w") as fp:
    json.dump(mentions_to_wikidata_keys, fp)

with open("experiments/resources/wikidata_to_mentions.json") as json_file:
    wikidata_to_mentions = json.load(json_file)

with open("experiments/resources/mentions_to_wikidata.json") as json_file:
    mentions_to_wikidata = json.load(json_file)

queries = []
candidates = []
for i, row in train_df.iterrows():
    if row["wkdt_qid"] in wikidata_to_mentions:
        queries.append(row["mention"])
        # Get potential altnames for each mention:
        potential_cands = wikidata_to_mentions[row["wkdt_qid"]]
        # Filter altnames to those that are >0.7 sim from query and max length diff of 3 char:
        candidates += [
            x for x in potential_cands if SequenceMatcher(None, row["mention"], x).ratio() > 0.8
        ]
        queries = list(set(queries))
        candidates = list(set(candidates))

Path("experiments/datasets/").mkdir(parents=True, exist_ok=True)

with open("./experiments/datasets/queries.txt", "w") as f:
    f.write("\n".join(map(str, queries)))

with open("./experiments/datasets/candidates.txt", "w") as f:
    f.write("\n".join(map(str, candidates)))

# -----------------------------------------------------
# First batch of experiments:
# -----------------------------------------------------

list_architectures = ["gru", "lstm"]
list_pooling_modes = [
    "hstates_layers_simple",
    "hstates_layers",
    "hstates_l2_distance",
    "average",
    "max",
]
list_dropouts = [0.01, 0.5]
list_learning_rates = [0.01, 0.001]

experiment_id = 0

for architecture in list_architectures:
    for pooling_mode in list_pooling_modes:
        for dropout in list_dropouts:
            for learning_rate in list_learning_rates:

                experiment_id += 1

                with open("input_dfm_exp_char.yaml", "r") as fr:
                    dl_inputs = yaml.load(fr, Loader=yaml.FullLoader)

                # Tokenization:
                dl_inputs["gru_lstm"]["mode"]["tokenize"] = ["char"]
                dl_inputs["gru_lstm"]["mode"]["prefix_suffix"] = ["<", ">"]

                # Dataset size:
                dl_inputs["gru_lstm"]["train_proportion"] = 0.35
                dl_inputs["gru_lstm"]["val_proportion"] = 0.15
                dl_inputs["gru_lstm"]["test_proportion"] = 0.5

                # Architecture:
                dl_inputs["gru_lstm"]["main_archictecture"] = architecture
                dl_inputs["gru_lstm"]["pooling_mode"] = pooling_mode
                dl_inputs["gru_lstm"]["rnn_dropout"] = dropout
                dl_inputs["gru_lstm"]["fc_dropout"] = [dropout, dropout]
                dl_inputs["gru_lstm"]["att_dropout"] = [dropout, dropout]
                dl_inputs["gru_lstm"]["leraning_rate"] = learning_rate

                with open(
                    "./experiments/inputs/input_dfm_char_" + str(experiment_id) + ".yaml", "w"
                ) as outfile:
                    yaml.dump(dl_inputs, outfile, explicit_start=True, indent=2)

                # --------------------------------------
                # Train the DeezyMatch model:

                dm_model = "char_" + str(experiment_id)

                # If model does not exist already, train a new model:
                if not Path(
                    "./experiments/models/" + dm_model + "/" + dm_model + ".model"
                ).is_file():
                    # train a new model
                    dm_train(
                        input_file_path="./experiments/inputs/input_dfm_char_"
                        + str(experiment_id)
                        + ".yaml",
                        dataset_path="./experiments/datasets/ocr_string_pairs.txt",
                        model_name=dm_model,
                    )

                # --------------------------------------
                # Generate and combine vectors for candidates:

                if not Path(
                    "./experiments/candidate_vectors/candidates_" + dm_model + "/embeddings/"
                ).is_dir():
                    start_time = time.time()
                    dm_inference(
                        input_file_path="./experiments/models/"
                        + dm_model
                        + "/input_dfm_char_"
                        + str(experiment_id)
                        + ".yaml",
                        dataset_path="./experiments/datasets/candidates.txt",
                        pretrained_model_path="./experiments/models/"
                        + dm_model
                        + "/"
                        + dm_model
                        + ".model",
                        pretrained_vocab_path="./experiments/models/"
                        + dm_model
                        + "/"
                        + dm_model
                        + ".vocab",
                        inference_mode="vect",
                        scenario="./experiments/candidate_vectors/candidates_" + dm_model,
                    )
                    elapsed = time.time() - start_time
                    print("Generate candidate vectors: %s" % elapsed)

                if not Path("./experiments/combined/candidates_" + dm_model).is_dir():
                    start_time = time.time()
                    combine_vecs(
                        rnn_passes=["fwd", "bwd"],
                        input_scenario="./experiments/candidate_vectors/candidates_" + dm_model,
                        output_scenario="./experiments/combined/candidates_" + dm_model,
                        print_every=100000,
                    )
                    elapsed = time.time() - start_time
                    print("Combine candidate vectors: %s" % elapsed)

                # --------------------------------------
                # Generate and combine vectors for queries:

                if not Path(
                    "./experiments/query_vectors/queries_" + dm_model + "/embeddings/"
                ).is_dir():
                    start_time = time.time()
                    dm_inference(
                        input_file_path="./experiments/models/"
                        + dm_model
                        + "/input_dfm_char_"
                        + str(experiment_id)
                        + ".yaml",
                        dataset_path="./experiments/datasets/queries.txt",
                        pretrained_model_path="./experiments/models/"
                        + dm_model
                        + "/"
                        + dm_model
                        + ".model",
                        pretrained_vocab_path="./experiments/models/"
                        + dm_model
                        + "/"
                        + dm_model
                        + ".vocab",
                        inference_mode="vect",
                        scenario="./experiments/query_vectors/queries_" + dm_model,
                    )
                    elapsed = time.time() - start_time
                    print("Generate query vectors: %s" % elapsed)

                if not Path("./experiments/combined/queries_" + dm_model).is_dir():
                    start_time = time.time()
                    combine_vecs(
                        rnn_passes=["fwd", "bwd"],
                        input_scenario="./experiments/query_vectors/queries_" + dm_model,
                        output_scenario="./experiments/combined/queries_" + dm_model,
                        print_every=100000,
                    )
                    elapsed = time.time() - start_time
                    print("Combine query vectors: %s" % elapsed)

                # --------------------------------------
                # Rank candidates

                candidates_pd = candidate_ranker(
                    query_scenario="./experiments/combined/queries_" + dm_model,
                    candidate_scenario="./experiments/combined/candidates_" + dm_model,
                    ranking_metric="faiss",
                    selection_threshold=20.0,
                    num_candidates=3,
                    search_size=3,
                    output_path="./experiments/ranker_results/queries_candidates_" + dm_model,
                    pretrained_model_path="./experiments/models/"
                    + dm_model
                    + "/"
                    + dm_model
                    + ".model",
                    pretrained_vocab_path="./experiments/models/"
                    + dm_model
                    + "/"
                    + dm_model
                    + ".vocab",
                )

                # --------------------------------------
                # Provide accuracy

                correct = []
                incorrect = []
                correct_distance = 0.0
                incorrect_distance = 0.0
                for i, row in train_df.iterrows():
                    if row["mention"] in list(candidates_pd["query"]):
                        returned_candidates = (
                            candidates_pd[candidates_pd["query"] == row["mention"]]
                            .iloc[0]
                            .faiss_distance
                        )
                        print(row["mention"])
                        print(returned_candidates)
                        for rc in returned_candidates:
                            if rc in wikidata_to_mentions[row["wkdt_qid"]]:
                                correct.append(rc)
                                correct_distance += returned_candidates[rc]
                            else:
                                incorrect.append(rc)
                                incorrect_distance += returned_candidates[rc]

                with open("./experiments/results/" + dm_model + ".txt", "w") as fw:
                    fw.write("Experiment:" + dm_model + "\n")
                    fw.write("------------------------\n")
                    fw.write("Architecture:" + architecture + "\n")
                    fw.write("Pooling mode:" + pooling_mode + "\n")
                    fw.write("Dropout:" + str(dropout) + "\n")
                    fw.write("Learning rate:" + str(learning_rate) + "\n\n")
                    fw.write("Correct:" + str(len(correct)) + "\n")
                    fw.write("Incorrect:" + str(len(incorrect)) + "\n")
                    fw.write(
                        "Accuracy:" + str(len(correct) / (len(correct) + len(incorrect))) + "\n"
                    )
                    fw.write("Correct distance:" + str(correct_distance) + "\n")
                    fw.write("Incorrect distance:" + str(incorrect_distance) + "\n")
                    fw.write(
                        "Distance accuracy:"
                        + str(correct_distance / (correct_distance + incorrect_distance))
                        + "\n"
                    )
