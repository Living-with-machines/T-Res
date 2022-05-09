import os
import sys
import json
import operator
from ast import literal_eval

import numpy as np
import pandas as pd
import requests
from numpy import NaN

import numpy as np
import pandas as pd
import requests
from scipy import spatial
from transformers import AutoTokenizer, pipeline

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import training


class Linker:
    def __init__(
        self,
        method,
        myranker,
        accepted_labels,
        do_training,
        training_csv,
        resources_path,
        linking_resources,
    ):
        self.do_training = do_training
        self.training_csv = training_csv
        self.accepted_labels = accepted_labels
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
        self.myranker = myranker

    def __str__(self):
        s = "Entity Linking:\n* Method: {0}\n* Accepted labels: {1}\n* Do training: {2}\n* Linking resources: {3}\n".format(
            self.method,
            self.accepted_labels,
            str(self.do_training),
            ",".join(list(self.linking_resources.keys())),
        )
        return s

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
            gaz["hcounties"] = gaz["hcounties"].apply(eval_with_exception)
            gaz["countries"] = gaz["countries"].apply(eval_with_exception)
            self.linking_resources["gazetteer"] = gaz

            # Load Wikidata mentions-to-wikidata (with normalized counts) to QID dictionary
            if self.method == "mostpopularnormalised":
                with open(
                    self.resources_path + "mentions_to_wikidata_normalized.json", "r"
                ) as f:
                    self.linking_resources["mentions_to_wikidata"] = json.load(f)

            # Load Wikidata mentions-to-wikidata (with absolute counts) to QID dictionary
            if self.method == "mostpopular":
                with open(self.resources_path + "mentions_to_wikidata.json", "r") as f:
                    self.linking_resources["mentions_to_wikidata"] = json.load(f)

            # Load and map Wikidata class embeddings to their corresponding Wikidata class id:
            dict_class_to_embedding = dict()
            # To do: change the path to the resources one:
            embeddings = np.load(
                "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/embeddings/embeddings.npy"
            )
            with open(
                "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/embeddings/wikidata_ids.txt"
            ) as fr:
                wikidata_ids = fr.readlines()
                wikidata_ids = np.array([x.strip() for x in wikidata_ids])
            for i in range(len(wikidata_ids)):
                dict_class_to_embedding[wikidata_ids[i]] = embeddings[i]
            self.linking_resources["class_embeddings"] = dict_class_to_embedding

            return self.linking_resources

        return self.linking_resources

    def filtering_labels(self):
        """
        Select entities that will be considered for entity linking.
        """
        if self.accepted_labels == "all":
            self.accepted_labels = [
                "loc",
                "b-loc",
                "i-loc",
                "street",
                "b-street",
                "i-street",
                "building",
                "b-building",
                "i-building",
                "other",
                "b-other",
                "i-other",
            ]
        if self.accepted_labels == "loc":
            self.accepted_labels = ["loc", "b-loc", "i-loc"]
        return self.accepted_labels

    def train(self):
        """
        Do the training if necessary.
        """
        if self.do_training == True:
            # Create a dataset for entity linking, with candidates retrieved
            # using the chosen method:
            training_df = training.create_trainset(self.training_csv, self.myranker)
            return training_df

    # Select disambiguation method
    def run(self, cands, mention_context):
        if cands:
            if "mostpopular" in self.method:
                link, score, other_cands = self.most_popular(cands)
                return (link, score, other_cands)
            # elif "featclassifier" in self.method:
            #     link, score, other_cands = feat_classifier(cands, mylinker)
            #     return (link, score, other_cands)

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

    # def run(self):
    #     """
    #     Run toponym resolution.
    #     """

    #     print("*** Loading the resources...")
    #     self.linking_resources = self.load_resources()
    #     print("*** Resources loaded!\n")

    #     # print(self.linking_resources.keys())
    #     self.train()


# # Ranking parameters for DeezyMatch:
# myranker = dict()
# cand_select_method = "perfectmatch"

# end_to_end = Linker(
#     method="featclassifier",
#     cand_select_method=cand_select_method,
#     myranker=myranker,
#     accepted_labels=["loc", "b-loc", "i-loc"],
#     do_training=True,
#     training_csv="/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/data/lwm/linking_df_train.tsv",
#     resources_path="/resources/wikidata/",
#     linking_resources=dict(),
# )

# end_to_end.run()

# import sys

# sys.exit()

"""

# ---------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------

# # Load Wikidata mentions to QID dictionary
# wikidata_path = "/resources/wikidata/"
# with open(wikidata_path + "mentions_to_wikidata.json", "r") as f:
#     mentions_to_wikidata = json.load(f)

# # Load Wikidata normalized mentions to QID dictionary
# with open(wikidata_path + "mentions_to_wikidata_normalized.json", "r") as f:
#     normalised_mentions_to_wikidata = json.load(f)

# Load Wikidata gazetteer:
# gaz = pd.read_csv("/resources/wikidata/wikidata_gazetteer.csv", low_memory=False)


# def eval_with_exception(string):
#     try:
#         return literal_eval(string)
#     except ValueError:
#         return None


# gaz["instance_of"] = gaz["instance_of"].apply(eval_with_exception)
# gaz["hcounties"] = gaz["hcounties"].apply(eval_with_exception)
# gaz["countries"] = gaz["countries"].apply(eval_with_exception)

# Map Wikidata entries to the list of classes they are instances of:
dict_wqid_to_classes = dict(zip(gaz.wikidata_id, gaz.instance_of))
dict_wqid_to_hcounties = dict(zip(gaz.wikidata_id, gaz.hcounties))
dict_wqid_to_countries = dict(zip(gaz.wikidata_id, gaz.countries))

# # Load and map Wikidata class embeddings to their corresponding Wikidata class id:
# dict_class_to_embedding = dict()
# embeddings = np.load(
#     "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/embeddings/embeddings.npy"
# )
# with open(
#     "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/embeddings/wikidata_ids.txt"
# ) as fr:
#     wikidata_ids = fr.readlines()
#     wikidata_ids = np.array([x.strip() for x in wikidata_ids])
# for i in range(len(wikidata_ids)):
#     dict_class_to_embedding[wikidata_ids[i]] = embeddings[i]

# # List of geographical classes we're interested in, mapped to wikidata IDs: (TO DO: turn to file in resources?)
# relevant_classes = {
#     "Q6256": "country",
#     "Q5107": "continent",
#     "Q56061": "administrative territorial entity",
#     "Q532": "village",
#     "Q3957": "town",
#     "Q515": "city",
#     "Q34442": "road",
#     "Q41176": "building",
#     "Q123705": "neighborhood",
#     "Q4022": "river",
#     "Q165": "sea",
#     "Q82794": "geographic region",
#     "Q13418847": "historical event",
#     "Q8502": "mountain",
#     "Q22698": "park",
# }

# # Dictionary that maps relevant wikidata classes to their embedding:
# relevant_classes_embeddings = dict()
# for rc in relevant_classes:
#     relevant_classes_embeddings[rc] = dict_class_to_embedding[rc]

# Load BERT model and tokenizer, and feature-extraction pipeline:
base_model_path = "/resources/models/bert/bert_1760_1900/"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model_rd = pipeline(
    "feature-extraction", model=base_model_path, tokenizer=base_model_path
)

# Place of publication to Wikidata QID
places_of_publication = {
    "Ashton-under-Lyne": "Q659803",
    "Manchester": "Q18125",
    "Dorchester": "Q503331",
    "Poole": "Q203349",
    "New York": "Q60",
}


# ---------------------------------------------------
# PROCESSING DATA
# ---------------------------------------------------

# Function that assigns the average node embedding of the wikidata classes of an entity:
def find_avg_node_embedding(wkdt_loc_qid):
    avg_embedding = None
    wk_classes = dict_wqid_to_classes.get(wkdt_loc_qid, dict_class_to_embedding)
    list_embeddings = []
    if wk_classes:
        for wkclass in wk_classes:
            if wkclass in dict_class_to_embedding:
                list_embeddings.append(dict_class_to_embedding[wkclass])
    if list_embeddings:
        avg_embedding = np.array(list_embeddings).mean(axis=0).tolist()
    return avg_embedding


# Funtion that assigns the closest class to a wikidata entity:
def assign_closest_class(avg_embedding):
    closest_class = None
    if not avg_embedding is None:
        dict_class_similarity = dict()
        for rce in relevant_classes_embeddings:
            dict_class_similarity[rce] = 1 - spatial.distance.cosine(
                avg_embedding, relevant_classes_embeddings[rce]
            )
        closest_class = relevant_classes[
            max(dict_class_similarity, key=dict_class_similarity.get)
        ]
    return closest_class


# Get BERT vector in context of the mention:
def get_mention_vector(row, agg=np.mean):
    tokenized = tokenizer(row["sentence"])
    start_token = tokenized.char_to_token(row["mention_start"])
    end_token = tokenized.char_to_token(row["mention_end"] - 1)
    if not end_token:
        print(row["mention"])  # issue with these names, check later
        end_token = start_token
    # TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
    if not start_token or not end_token:
        return None
    tokens_positions = list(range(start_token, end_token + 1))
    vectors = np.array(model_rd(row["sentence"]))
    vector = list(agg(vectors[0, tokens_positions, :], axis=0))
    return vector


# Get geoscope of the mention:
def get_geoscope_from_publication(place_publ, wkdt_qid):
    publ_qid = places_of_publication[place_publ]
    proximity = ""
    a = dict_wqid_to_hcounties.get(wkdt_qid, [])
    b = dict_wqid_to_hcounties.get(publ_qid, [])
    c = dict_wqid_to_countries.get(wkdt_qid, [])
    d = dict_wqid_to_countries.get(publ_qid, [])
    if any(i in a for i in b):
        proximity = "same_hcounty"
    elif any(i in c for i in d):
        proximity = "national"
    else:
        proximity = "abroad"
    return proximity


# ---------------------------------------------------
# LINKING METHODS
# ---------------------------------------------------

# Select disambiguation method
def select(cands, method, mylinker):
    if cands:
        if "mostpopular" in method:
            link, score, other_cands = most_popular(cands, method)
            return (link, score, other_cands)
        elif "featclassifier" in method:
            link, score, other_cands = feat_classifier(cands, mylinker)
            return (link, score, other_cands)


# Features MLPClassifier method:
def feat_classifier(cands, mylinker):
    model_geoscope = mylinker["model2scope"]
    model_class = mylinker["model2type"]
    mention_context = mylinker["mention_context"]
    mention_emb = get_mention_vector(mention_context)

    keep_best_candidate = ""
    keep_highest_score = 0.0
    all_candidates = []
    for candidate in cands:
        wikidata_cands = mentions_to_wikidata[candidate]
        if wikidata_cands:
            all_candidates += wikidata_cands
            mention_data = dict()
            for i in range(len(model_geoscope.classes_)):
                mention_data[model_geoscope.classes_[i]] = np.array(
                    model_geoscope.predict_proba([mention_emb])
                )[0][i]
            for i in range(len(model_class.classes_)):
                mention_data[model_class.classes_[i]] = np.array(
                    model_class.predict_proba([mention_emb])
                )[0][i]
            mention_data["mention_embedding"] = mention_emb

            cands_df = pd.DataFrame()
            cands_df["wqid_cand"] = [x for x in wikidata_cands]
            cands_df["candidate_relevance"] = [
                wikidata_cands[x] for x in wikidata_cands
            ]

            cands_df["candrank_confidence"] = cands[candidate]
            cands_df.loc[:, "class_emb"] = cands_df.loc[:, :].apply(
                lambda x: find_avg_node_embedding(x["wqid_cand"]), axis=1
            )
            cands_df.loc[:, "candidate_classtype"] = cands_df.loc[:, :].apply(
                lambda x: assign_closest_class(x["class_emb"]), axis=1
            )
            cands_df.loc[:, "candidate_geoscope"] = cands_df.loc[:, :].apply(
                lambda x: get_geoscope_from_publication(
                    mylinker["metadata"]["place"], x["wqid_cand"]
                ),
                axis=1,
            )
            cands_df["candidate_relevance_norm"] = (
                cands_df["candidate_relevance"] - min(cands_df["candidate_relevance"])
            ) / (
                max(cands_df["candidate_relevance"])
                - min(cands_df["candidate_relevance"])
            )
            cands_df["candidate_relevance_norm"] = cands_df[
                "candidate_relevance_norm"
            ].fillna(0.0)
            cands_df["candidate_relevance_norm"] = cands_df[
                "candidate_relevance_norm"
            ].round(3)

            weight_classtype = []
            weight_geoscope = []
            for i, row in cands_df.iterrows():
                wclass = mention_data.get(row["candidate_classtype"], 0.0)
                wgeosc = mention_data.get(row["candidate_geoscope"], 0.0)
                weight_classtype.append(round(wclass, 3))
                weight_geoscope.append(round(wgeosc, 3))

            cands_df["weight_classtype"] = weight_classtype
            cands_df["weight_geoscope"] = weight_geoscope
            cands_df["weight"] = cands_df[
                [
                    "weight_classtype",
                    "weight_geoscope",
                    "candidate_relevance_norm",
                    "candrank_confidence",
                ]
            ].mean(axis=1)

            best_candidate, score = tuple(
                cands_df.sort_values(by=["weight"], ascending=False).iloc[0][
                    ["wqid_cand", "weight"]
                ]
            )

            if score > keep_highest_score:
                keep_highest_score = score
                keep_best_candidate = best_candidate

    return keep_best_candidate, keep_highest_score, set(all_candidates)


# Most popular candidate:
def most_popular(cands, type):
    keep_most_popular = ""
    keep_highest_score = 0.0
    total_score = 0.0
    all_candidates = []
    for candidate in cands:
        if type == "mostpopular":
            wikidata_cands = mentions_to_wikidata[candidate]
        elif type == "mostpopularnormalised":
            wikidata_cands = normalised_mentions_to_wikidata[candidate]
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




"""

# REL end-to-end using the API
API_URL = "https://rel.cs.ru.nl/api"


def rel_end_to_end(sent):
    el_result = requests.post(API_URL, json={"text": sent, "spans": []}).json()
    return el_result
