import json
import os
import sys
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))

import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils import candidate_selection, linking, ner, process_data

# Datasets:'
datasets = ["lwm", "hipe"]

# Approach:
ner_model_id = "lwm"  # lwm or rel
cand_select_method = "deezymatch" # either perfectmatch or deezymatch
top_res_method = "mostpopular"

# Ranking parameters for DeezyMatch:
myranker = dict()
if cand_select_method == "deezymatch":
    myranker["ranking_metric"] = "faiss"
    myranker["selection_threshold"] = 5
    myranker["num_candidates"] = 1
    myranker["search_size"] = 1

for dataset in datasets:

    # Path to dev dataframe:
    dev = pd.read_csv(
        "outputs/data/" + dataset + "/linking_df_dev.tsv",
        sep="\t",
    )

    # Path where to store gold tokenization:
    gold_path = "outputs/results/" + dataset + "/lwm_gold_tokenisation.json"

    # Path where to store REL API output:
    rel_end_to_end = "outputs/results/" + dataset + "/rel_end_to_end.json"

    if ner_model_id == "lwm":
        # Path to NER Model:
        ner_model = "outputs/models/" + ner_model_id + "-ner.model"
        ner_pipe = pipeline("ner", model=ner_model)
        gold_tokenisation = {}

    if ner_model_id == "rel":
        if Path(rel_end_to_end).is_file():
            with open(rel_end_to_end) as f:
                rel_preds = json.load(f)
        else:
            rel_preds = {}

        gold_standard = process_data.read_gold_standard(gold_path)
        # currently it's all based on REL
        # but we could for instance use our pipeline and rely on REL only for disambiguation
        cand_select_method = "rel"
        top_res_method = "rel"

    dAnnotated, dSentences = ner.format_for_ner(dev)

    true_mentions_sents = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()

    for sent_id in tqdm.tqdm(dSentences.keys()):

        if ner_model_id == "rel":
            if Path(rel_end_to_end).is_file():
                pred_ents = rel_preds[sent_id]
            else:
                pred_ents = linking.rel_end_to_end(dSentences[sent_id])
                rel_preds[sent_id] = pred_ents

            sentence_preds = []
            prev_ann = ""
            for token in gold_standard[sent_id]:
                start = token["start"]
                end = token["end"]
                word = token["word"]
                n, el, prev_ann = process_data.match_ent(pred_ents, start, end, prev_ann)
                sentence_preds.append([word, n, el])

            dPreds[sent_id] = sentence_preds

        else:
            # Toponym recognition
            gold_standard, predictions = ner.ner_predict(
                dSentences[sent_id], dAnnotated[sent_id], ner_pipe, dataset
            )
            gold_tokenisation[sent_id] = gold_standard
            sentence_preds = [[x["word"], x["entity"], "O"] for x in predictions]
            sentence_trues = [[x["word"], x["entity"], x["link"]] for x in gold_standard]
            sentence_skys = [[x["word"], x["entity"], "O"] for x in gold_standard]

            pred_mentions_sent = ner.aggregate_mentions(sentence_preds)
            true_mentions_sent = ner.aggregate_mentions(sentence_trues)
            # Candidate selection
            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands = candidate_selection.select(mentions, cand_select_method, myranker)

            # # Toponym resolution
            for mention in pred_mentions_sent:
                text_mention = mention["mention"]
                start_offset = mention["start_offset"]
                end_offset = mention["end_offset"]

                # to be extended so that it can include multiple features
                res = linking.select(cands[text_mention], top_res_method)
                if res:
                    link, score, other_cands = res
                    for x in range(start_offset, end_offset + 1):
                        position_ner = sentence_preds[x][1][:2]
                        sentence_preds[x][2] = position_ner + link
                        sentence_preds[x].append(other_cands)
                        true_label = sentence_trues[x][2].split("-")[-1]
                        if true_label in other_cands:
                            sentence_skys[x][2] = sentence_trues[x][2]

            dPreds[sent_id] = sentence_preds
            dTrues[sent_id] = sentence_trues
            dSkys[sent_id] = sentence_skys

    if ner_model_id == "lwm":
        process_data.store_results_hipe(dataset, "true", dTrues)
        process_data.store_results_hipe(
            dataset,
            "skyline:" + ner_model_id + "+" + cand_select_method + "+" + top_res_method,
            dSkys,
        )
        with open(gold_path, "w") as fp:
            json.dump(gold_tokenisation, fp)

    if ner_model_id == "rel":
        if not Path(rel_end_to_end).is_file():
            with open(rel_end_to_end, "w") as fp:
                json.dump(rel_preds, fp)

    process_data.store_results_hipe(
        dataset, ner_model_id + "+" + cand_select_method + "+" + top_res_method, dPreds
    )
