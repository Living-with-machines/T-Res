import datetime
import json
import os
import sys

import pandas as pd

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))

from transformers import pipeline
from utils import candidate_selection, linking, ner, process_data


class ELPipeline:
    def __init__(self, ner_model_id, cand_select_method, top_res_method, myranker, accepted_tags):
        self.ner_model_id = ner_model_id
        self.cand_select_method = cand_select_method
        self.top_res_method = top_res_method
        self.myranker = myranker
        self.accepted_tags = accepted_tags
        self.already_collected_cands = {}

    def run(self, sent):
        if self.ner_model_id == "rel":
            pred_ents = linking.rel_end_to_end(sent)
            pred_ents = [
                {
                    "wikidata_id": process_data.match_wikipedia_to_wikidata(pred[3]),
                    "ner_conf": pred[4],
                    "el_conf": pred[5],
                }
                for pred in pred_ents
            ]
            return pred_ents
        if self.ner_model_id == "lwm":
            ner_model = "outputs/models/" + self.ner_model_id + "-ner.model"
            ner_pipe = pipeline("ner", model=ner_model)
            gold_standard, predictions = ner.ner_predict(sent, [], ner_pipe, "lwm")
            sentence_preds = [[x["word"], x["entity"], "O", x["score"]] for x in predictions]
            pred_mentions_sent = ner.aggregate_mentions(sentence_preds)

            pred_mentions_sent = [
                x for x in pred_mentions_sent if x["ner_label"] in self.accepted_tags
            ]

            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands, self.already_collected_cands = candidate_selection.select(
                mentions, self.cand_select_method, self.myranker, self.already_collected_cands
            )
            pred_ents = []
            for mention in pred_mentions_sent:
                text_mention = mention["mention"]
                res = linking.select(cands[text_mention], self.top_res_method)
                if res:
                    # entity_candidate, candidate_score = cands[text_mention]
                    link, el_score, other_cands = res
                    # mention["entity_candidate"] = entity_candidate
                    # mention["candidate_score"] = candidate_score
                    mention["wikidata_id"] = link
                    mention["el_score"] = el_score
                    pred_ents.append(mention)

            return pred_ents


# Ranking parameters for DeezyMatch:
myranker = dict()
myranker["ranking_metric"] = "faiss"
myranker["selection_threshold"] = 5
myranker["num_candidates"] = 1
myranker["search_size"] = 1
# Path to DeezyMatch model and combined candidate vectors:
myranker["dm_path"] = "outputs/deezymatch/"
myranker["dm_cands"] = "wkdtalts"
myranker["dm_model"] = "ocr_faiss_l2"
myranker["dm_output"] = "deezymatch_on_the_fly"

accepted_tags = {"LOC"}

start = datetime.datetime.now()

end_to_end = ELPipeline(
    ner_model_id="lwm",
    cand_select_method="perfectmatch",
    top_res_method="mostpopular",
    myranker=myranker,
    accepted_tags=accepted_tags,
)


# dSentences = {
#    "1": "Liverpool is a big city up north",
#    "2": "I do not like London in winter",
#    "3": "We live in L%ndon",
# }

folder = "/resources/hmd-samples/hmd_data_extension_words/"
dataset_name = "0002083_plaintext.csv"
output_name = dataset_name.replace(".csv", ".json")

dataset = pd.read_csv(folder + dataset_name)
dataset = dataset[:10]
dataset["toponyms"] = dataset.parallel_apply(
    lambda row: end_to_end.run(row["target_sentence"]), axis=1
)

output_dict = dict(zip(dataset.index, dataset.toponyms))

with open(folder + output_name, "w") as fp:
    json.dump(output_dict, fp)

end = datetime.datetime.now()
print(end - start)
