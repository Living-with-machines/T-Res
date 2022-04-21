import datetime
import itertools
import json
import os
import sys

import pandas as pd
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))

from transformers import pipeline
from utils import candidate_selection, linking, ner, process_data
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=24)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ELPipeline:
    def __init__(
        self,
        ner_model_id,
        cand_select_method,
        top_res_method,
        myranker,
        accepted_tags,
        ner_pipe,
    ):
        self.ner_model_id = ner_model_id
        self.cand_select_method = cand_select_method
        self.top_res_method = top_res_method
        self.myranker = myranker
        self.accepted_tags = accepted_tags
        self.already_collected_cands = {}
        self.ner_pipe = ner_pipe

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
            gold_standard, predictions = ner.ner_predict(sent, [], self.ner_pipe, "lwm")
            sentence_preds = [
                [x["word"], x["entity"], "O", x["score"]] for x in predictions
            ]
            pred_mentions_sent = ner.aggregate_mentions(sentence_preds)

            pred_mentions_sent = [
                x for x in pred_mentions_sent if x["ner_label"] in self.accepted_tags
            ]

            mentions = list(set([mention["mention"] for mention in pred_mentions_sent]))
            cands, self.already_collected_cands = candidate_selection.select(
                mentions,
                self.cand_select_method,
                self.myranker,
                self.already_collected_cands,
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
myranker["selection_threshold"] = 10
myranker["num_candidates"] = 3
myranker["search_size"] = 3
# Path to DeezyMatch model and combined candidate vectors:
myranker["dm_path"] = "outputs/deezymatch/"
myranker["dm_cands"] = "wkdtalts"
myranker["dm_model"] = "ocr_avgpool"
myranker["dm_output"] = "deezymatch_on_the_fly"

accepted_tags = {"LOC"}

start = datetime.datetime.now()

ner_model_id = "lwm"
ner_model = "outputs/models/" + ner_model_id + "-ner.model"
ner_pipe = pipeline("ner", model=ner_model)

end_to_end = ELPipeline(
    ner_model_id=ner_model_id,
    cand_select_method="perfectmatch",
    top_res_method="mostpopular",
    myranker=myranker,
    accepted_tags=accepted_tags,
    ner_pipe=ner_pipe,
)

# dSentences = {
#    "1": "Liverpool is a big city up north",
#    "2": "I do not like London in winter",
#    "3": "We live in L%ndon",
# }

print("Start!")

hmd_files = [
    "0002642_plaintext.csv",  # London liberal 1846-1868
    "0002645_plaintext.csv",  # London conservative 1853-1858
    "0002085_plaintext.csv",  # Merseyside conservative 1860-1861
    "0002088_plaintext.csv",  # Merseyside conservative 1832-1854
]

folder = "/resources/hmd-samples/hmd_data_extension_words/"
Path(folder).mkdir(parents=True, exist_ok=True)

for dataset_name in hmd_files:

    dataset = pd.read_csv(folder + dataset_name)

    # Add metadata columns: publication_code, year, month, day, and article_path
    dataset[["publication_code", "year", "monthday", "article_path"]] = dataset[
        "article_path"
    ].str.split("/", expand=True)
    dataset["month"] = dataset["monthday"].str[:2]
    dataset["day"] = dataset["monthday"].str[2:]
    dataset = dataset.drop(columns=["Unnamed: 0", "monthday"])

    months = list(dataset.month.unique())
    years = list(dataset.year.unique())

    for month, year in list(itertools.product(months, years)):
        print(dataset_name, month, year)

        output_name_toponyms = (
            dataset_name.replace(".csv", "") + "_" + year + month + "_toponyms.json"
        )
        output_name_metadata = (
            dataset_name.replace(".csv", "") + "_" + year + month + "_metadata.json"
        )

        dataset_tmp = dataset.copy()
        dataset_tmp = dataset_tmp[
            (dataset_tmp["month"] == month) & (dataset_tmp["year"] == year)
        ]

        if not dataset_tmp.empty:
            dataset_tmp["toponyms"] = dataset_tmp.apply(
                lambda row: end_to_end.run(row["target_sentence"]), axis=1
            )

            metadata_dict = dataset[
                ["article_path", "hits", "publication_code", "year", "month", "day"]
            ].to_dict("index")
            output_dict = dict(zip(dataset_tmp.index, dataset_tmp.toponyms))

            with open(folder + "results/" + output_name_toponyms, "w") as fp:
                json.dump(output_dict, fp)
            with open(folder + "results/" + output_name_metadata, "w") as fp:
                json.dump(metadata_dict, fp)

end = datetime.datetime.now()
print(end - start)
