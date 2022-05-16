import os
import sys
import json
import operator
from ast import literal_eval
from typing_extensions import Self

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import math

import torch
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear
import torch.nn.functional as F

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import training, tools_perceptron


class Linker:
    def __init__(
        self,
        method,
        myranker,
        do_training,
        training_csv,
        resources_path,
        linking_resources,
        bert_base_model,
        bert_pipeline,
        bert_tokenizer,
        perceptron,
    ):
        self.do_training = do_training
        self.training_csv = training_csv
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
        self.myranker = myranker
        self.bert_base_model = bert_base_model
        self.bert_pipeline = bert_pipeline
        self.bert_tokenizer = bert_tokenizer
        self.perceptron = perceptron

    def __str__(self):
        s = "Entity Linking:\n* Method: {0}\n* Do training: {1}\n* Linking resources: {2}\n".format(
            self.method,
            str(self.do_training),
            ",".join(list(self.linking_resources.keys())),
        )
        return s

    def create_pipeline(self):
        # Load BERT model and tokenizer, and feature-extraction pipeline:
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_base_model)
        self.bert_pipeline = pipeline(
            "feature-extraction",
            model=self.bert_base_model,
            bert_tokenizer=self.bert_tokenizer,
        )
        return self.bert_tokenizer, self.bert_pipeline

    def load_resources(self):
        """
        Load resources required for linking.
        """

        def eval_with_exception(string):
            try:
                return literal_eval(string)
            except ValueError:
                return None

        # No need to load the resources if the method is REL.
        if self.method != "rel":

            # Load Wikidata gazetteer
            gaz = pd.read_csv(
                self.resources_path + "wikidata_gazetteer.csv", low_memory=False
            )
            gaz["instance_of"] = gaz["instance_of"].apply(eval_with_exception)
            dict_wqid_to_lat = dict(zip(gaz.wikidata_id, gaz.latitude))
            dict_wqid_to_lon = dict(zip(gaz.wikidata_id, gaz.longitude))
            dict_wqid_to_class = dict(zip(gaz.wikidata_id, gaz.instance_of))
            self.linking_resources["gazetteer"] = gaz
            self.linking_resources["dict_wqid_to_lat"] = dict_wqid_to_lat
            self.linking_resources["dict_wqid_to_lon"] = dict_wqid_to_lon

            # When a Wikidata entity is assigned to more than one clas, pick
            # the first class created in Wikidata, assuming it will be more
            # generic:
            def assign_class_to_entity(classes):
                if classes:
                    pick_first_class = "Q" + str(
                        sorted([int(x[1:]) for x in classes])[0]
                    )
                    return pick_first_class
                return "Unknown"

            dict_wqid_to_class = {
                x: assign_class_to_entity(dict_wqid_to_class[x])
                for x in dict_wqid_to_class
            }
            self.linking_resources["dict_wqid_to_class"] = dict_wqid_to_class

            # Load Wikidata mentions-to-wikidata (with normalized counts) to QID dictionary
            if self.method == "mostpopularnormalised":
                with open(
                    self.resources_path + "mentions_to_wikidata_normalized.json", "r"
                ) as f:
                    self.linking_resources["mentions_to_wikidata"] = json.load(f)

            # Load Wikidata mentions-to-wikidata (with absolute counts) to QID dictionary
            if self.method == "mostpopular" or self.method == "lwmperceptron":
                with open(self.resources_path + "mentions_to_wikidata.json", "r") as f:
                    self.linking_resources["mentions_to_wikidata"] = json.load(f)

            # Load class and entity Wikidta embeddings:
            path_cls_embs = "/resources/wikidata/gazetteer_wkdtclass_"  # from BigGraph
            # path_cls_embs = "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/embeddings/gazetteer_wkdtclass_"  # from gaz + Kaspar gnn
            path_ent_embs = "/resources/wikidata/gazetteer_entity_"
            class_emb_ids = open(path_cls_embs + "ids.txt", "r").read().split("\n")
            candidate_emb_ids = open(path_ent_embs + "ids.txt", "r").read().split("\n")
            class_embeddings = np.load(path_cls_embs + "embeddings.npy")
            candidate_embeddings = np.load(path_ent_embs + "embeddings.npy")
            dict_wqid_to_clssemb = dict()
            for i in range(len(class_emb_ids)):
                dict_wqid_to_clssemb[class_emb_ids[i]] = class_embeddings[i]
            dict_wqid_to_entemb = dict()
            for i in range(len(candidate_emb_ids)):
                dict_wqid_to_entemb[candidate_emb_ids[i]] = candidate_embeddings[i]
            self.linking_resources["class_embeddings_ids"] = class_emb_ids
            self.linking_resources["candidate_embeddings_ids"] = candidate_emb_ids
            self.linking_resources["class_embeddings"] = class_embeddings
            self.linking_resources["candidate_embeddings"] = candidate_embeddings
            self.linking_resources["dict_wqid_to_clssemb"] = dict_wqid_to_clssemb
            self.linking_resources["dict_wqid_to_entemb"] = dict_wqid_to_entemb

            ## Load place of publication metadata and assign coordinates:
            with open(
                "/resources/develop/mcollardanuy/toponym-resolution/resources/publication_metadata.json"
            ) as json_file:
                publication_metadata = json.load(json_file)
            for publ in publication_metadata:
                publ_wqid = publication_metadata[publ]["wikidata_qid"]
                publication_metadata[publ]["latitude"] = dict_wqid_to_lat[publ_wqid]
                publication_metadata[publ]["longitude"] = dict_wqid_to_lon[publ_wqid]
            self.linking_resources["publication_metadata"] = publication_metadata

        return self.linking_resources

    def get_candidate_wikidata_ids(self, mention):
        """
        Given the output from the candidate ranking module for a given
        mention (e.g. "Liverpool" for mention "LIVERPOOL"), return
        the wikidata IDs.
        """
        wikidata_cands = self.linking_resources["mentions_to_wikidata"].get(
            mention, None
        )
        return wikidata_cands

    # Get BERT vector in context of the mention:
    def get_mention_vector(self, row, agg=np.mean):
        tokenized = self.bert_tokenizer(row["sentence"])
        start_token = tokenized.char_to_token(row["mention_start"])
        end_token = tokenized.char_to_token(row["mention_end"] - 1)

        if not start_token:
            print("\t>>> Watch out, issue with:", row["mention"])
            start_token = 0

        if not end_token:
            print("\t>>> Watch out, issue with:", row["mention"])
            end_token = start_token

        tokens_positions = list(range(start_token, end_token + 1))
        vectors = np.array(self.bert_pipeline(row["sentence"]))
        vector = list(agg(vectors[0, tokens_positions, :], axis=0))
        return vector

    def train(self, training_df):
        """
        Do the training if necessary.
        """
        data_cands_comb = pd.DataFrame()
        # Prepare training set for training:
        training_set_cands_path = (
            "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/data/lwm/all_cands_for_training_"
            + self.myranker.method
            + ".pkl"
        )
        if not Path(training_set_cands_path).exists():
            print("Start preparing the data for the perceptron.")
            data_cands = []
            for j, row in tqdm(training_df.iterrows()):
                data_cands.append(
                    training.create_data_from_row(
                        self, j, row, features=self.perceptron["features"]
                    )
                )
            data_cands_comb = pd.concat(data_cands)
            data_cands_comb.to_pickle(training_set_cands_path)
        else:
            data_cands_comb = pd.read_pickle(training_set_cands_path)

        training.train_perceptron(data_cands_comb)

    # Select disambiguation method
    def run(self, cands, mention_context):
        if cands:
            if "mostpopular" in self.method:
                link, score, other_cands = self.most_popular(cands)
                return (link, score, other_cands)
            if "lwmperceptron" in self.method:
                link, score, other_cands = self.lwm_perceptron(cands, mention_context)
                return (link, score, other_cands)

    # LwM perceptron method:
    def lwm_perceptron(self, cands, mention_context):
        # For each mention, find its mention matches, the corresponding wikidata
        # entities, and the confidence score.

        publ_code = str(mention_context["metadata"]["publication_code"]).zfill(7)
        publ_latitude = self.linking_resources["publication_metadata"][publ_code][
            "latitude"
        ]
        publ_longitude = self.linking_resources["publication_metadata"][publ_code][
            "longitude"
        ]
        publ_coordinates = (publ_latitude, publ_longitude)

        wk_cands = dict()
        all_candidates = []
        for found_mention in cands:
            found_cands = self.get_candidate_wikidata_ids(found_mention)
            if found_cands:
                for cand in found_cands:
                    log_geo_dist = training.find_geo_distance(
                        cand, publ_coordinates, self
                    )
                    all_candidates.append(cand)
                    if not cand in wk_cands:
                        wk_cands[cand] = {
                            "conf_score": cands[found_mention],
                            "wkdt_relv": found_cands[cand],
                            "log_dist": log_geo_dist,
                        }
        # Collect the mention context:
        dict_candidates = {
            "sentence": mention_context["sentence"],
            "mention": mention_context["mention"],
            "mention_start": mention_context["mention_start"],
            "mention_end": mention_context["mention_end"],
            "year": mention_context["metadata"]["year"],
            "place": mention_context["metadata"]["place"],
            "publ_code": mention_context["metadata"]["publication_code"],
            "candidates": wk_cands,
        }

        # Get the mention embedding:
        dict_candidates["mention_emb"] = self.get_mention_vector(
            dict_candidates, agg=np.mean
        )
        best_candidate, score = self.get_perceptron_prediction(dict_candidates)

        return best_candidate, score, all_candidates

    # Most popular candidate:
    def most_popular(self, cands):
        keep_most_popular = ""
        keep_highest_score = 0.0
        total_score = 0.0
        all_candidates = []
        for candidate in cands:
            wikidata_cands = self.linking_resources["mentions_to_wikidata"][candidate]
            if wikidata_cands:
                # most popular wikidata entry (based on number of time mention points to page)
                most_popular_wikidata_cand, score = sorted(
                    wikidata_cands.items(), key=operator.itemgetter(1), reverse=True
                )[0]
                total_score += score
                if score > keep_highest_score:
                    keep_highest_score = score
                    keep_most_popular = most_popular_wikidata_cand
                all_candidates += wikidata_cands
        # we return the predicted, the score (overall the total), and the other candidates
        final_score = keep_highest_score / total_score
        return keep_most_popular, final_score, set(all_candidates)

    def expand_cands_for_inference(
        self,
        example,
        features=["conf_score", "wkdt_relv", "log_dist"],
    ):
        candidates = {}
        for c, vd in example["candidates"].items():
            candidates[c] = [c] + [vd.get(f, 0.0) for f in features]
        candidates = pd.DataFrame.from_dict(
            candidates, orient="index", columns=["wikidata_id"] + features
        )
        if candidates.shape[0]:
            candidates["wkdt_relv"] = (
                candidates["wkdt_relv"] / candidates["wkdt_relv"].max()
            )
            candidates["ext_vector"] = candidates.apply(
                lambda x: list(x[features]), axis=1
            )

        # Add vector representation for class instances
        _embeddings = []
        for i, candidate in candidates.iterrows():
            _embeddings.append(
                tools_perceptron.get_candidate_representation(
                    example,
                    candidate["wikidata_id"],
                    self.linking_resources["dict_wqid_to_entemb"],
                    self.linking_resources["dict_wqid_to_class"],
                    self.linking_resources["dict_wqid_to_clssemb"],
                )
            )

        candidates["x"] = _embeddings

        return candidates

    def get_perceptron_prediction(self, example):
        model = self.perceptron["perceptron_model"]
        m = torch.nn.Softmax(dim=1)
        model.eval()
        cl_df = self.expand_cands_for_inference(example).reset_index()
        x = torch.tensor(np.stack(cl_df.x.values), dtype=torch.float32)
        ext_features = torch.tensor(
            np.stack(cl_df.ext_vector.values), dtype=torch.float32
        )
        out = m(model(x, ext_features))
        loc = int(out[:, 1].argmax())
        return cl_df.iloc[loc].wikidata_id, float(out[loc, 1])


def rel_end_to_end(sent):
    """REL end-to-end using the API."""
    API_URL = "https://rel.cs.ru.nl/api"
    el_result = requests.post(API_URL, json={"text": sent, "spans": []}).json()
    return el_result
