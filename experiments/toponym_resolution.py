import json
import os
import sys
from pathlib import Path

import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils import candidate_selection, eval, linking, ner, process_data

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))

# Dataset:
dataset = "lwm"

# Approach:
ner_model_id = "rel"  # lwm or rel

if ner_model_id == "lwm":
    # Path to NER Model:
    ner_model = (
        "/resources/develop/mcollardanuy/toponym-resolution/outputs/models/"
        + ner_model_id
        + "-ner.model"
    )
    ner_pipe = pipeline("ner", model=ner_model)
    cand_select_method = "perfectmatch"  # either perfectmatch or deezymatch
    top_res_method = "mostpopular"
    gold_tokenisation = {}

if ner_model_id == "rel":
    rel_end_to_end = "outputs/results/" + dataset + "/rel_end_to_end.json"
    if Path(rel_end_to_end).is_file():
        with open(rel_end_to_end) as f:
            rel_preds = json.load(f)
    else:
        rel_preds = {}

    gold_path = "outputs/results/" + dataset + "/lwm_gold_tokenisation.json"
    gold_standard = process_data.read_gold_standard(gold_path)
    # currently it's all based on REL
    # but we could for instance use our pipeline and rely on REL only for disambiguation
    cand_select_method = "rel"
    top_res_method = "rel"

# Path to test dataframe:
df = pd.read_csv(
    "/resources/develop/mcollardanuy/toponym-resolution/outputs/data/linking_lwm_df_test.tsv",
    sep="\t",
)

# Split test set into dev and test set (by article, not sentence):
dev_ids, test_ids = train_test_split(df.article_id.unique(), test_size=0.5, random_state=42)
dev = df[df["article_id"].isin(dev_ids)]
test = df[df["article_id"].isin(test_ids)]

dAnnotated, dSentences = ner.format_for_ner(dev)

true_mentions_sents = dict()
dPreds = dict()
dTrues = dict()

for sent_id in tqdm.tqdm(dSentences.keys()):

    if ner_model_id == "rel":
        if Path(rel_end_to_end).is_file():
            pred_ents = rel_preds[sent_id]
        else:
            pred_ents = linking.rel_end_to_end(dSentences[sent_id])
            rel_preds[sent_id] = pred_ents

        sentence_preds = []
        for token in gold_standard[sent_id]:
            start = token["start"]
            end = token["end"]
            word = token["word"]
            prev_ann = ""
            n, el, prev_ann = process_data.match_ent(pred_ents, start, end, prev_ann)
            sentence_preds.append([word, n, el])

        dPreds[sent_id] = sentence_preds

    else:
        # Toponym recognition
        gold_standard, predictions = ner.ner_predict(
            dSentences[sent_id], dAnnotated[sent_id], ner_pipe
        )
        gold_tokenisation[sent_id] = gold_standard
        sentence_preds = [[x["word"], x["entity"], "O"] for x in predictions]
        sentence_trues = [[x["word"], x["entity"], x["link"]] for x in gold_standard]

        pred_mentions_sent = ner.aggregate_mentions(sentence_preds)
        true_mentions_sent = ner.aggregate_mentions(sentence_trues)
        # Candidate selection
        mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
        cands = candidate_selection.select(mentions, cand_select_method)

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
        dPreds[sent_id] = sentence_preds
        dTrues[sent_id] = sentence_trues
        true_mentions_sents[sent_id] = true_mentions_sent


if ner_model_id == "lwm":
    process_data.store_results_hipe(dataset, "true", dTrues)
    skyline = eval.eval_selection(true_mentions_sents, dTrues, dPreds)
    process_data.store_resolution_skyline(
        dataset, ner_model_id + "+" + cand_select_method + "+" + top_res_method, skyline
    )
    with open("outputs/results/" + dataset + "/lwm_gold_tokenisation.json", "w") as fp:
        json.dump(gold_tokenisation, fp)

if ner_model_id == "rel":
    if not Path(rel_end_to_end).is_file():
        with open(rel_end_to_end, "w") as fp:
            json.dump(rel_preds, fp)

process_data.store_results_hipe(
    dataset, ner_model_id + "+" + cand_select_method + "+" + top_res_method, dPreds
)
