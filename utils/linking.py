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
        do_training,
        training_csv,
        resources_path,
        linking_resources,
        base_model,
        tokenizer,
        model_rd,
    ):
        self.do_training = do_training
        self.training_csv = training_csv
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
        self.myranker = myranker
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.model_rd = model_rd

    def __str__(self):
        s = "Entity Linking:\n* Method: {0}\n* Do training: {1}\n* Linking resources: {2}\n".format(
            self.method,
            str(self.do_training),
            ",".join(list(self.linking_resources.keys())),
        )
        return s

    def create_pipeline(self):
        # Load BERT model and tokenizer, and feature-extraction pipeline:
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model_rd = pipeline(
            "feature-extraction", model=self.base_model, tokenizer=self.base_model
        )
        return self.tokenizer, self.model_rd

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
        tokenized = self.tokenizer(row["sentence"])
        start_token = tokenized.char_to_token(row["mention_start"])
        end_token = tokenized.char_to_token(row["mention_end"] - 1)
        if not end_token:
            print("\t>>> Watch out, issue with:", row["mention"])  # check later
            end_token = start_token
        # TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
        if not start_token or not end_token:
            return None
        tokens_positions = list(range(start_token, end_token + 1))
        vectors = np.array(self.model_rd(row["sentence"]))
        vector = list(agg(vectors[0, tokens_positions, :], axis=0))
        return vector

    def train(self):
        """
        Do the training if necessary.
        """
        if self.do_training == True:
            # Create a dataset for entity linking, with candidates retrieved
            # using the chosen method:
            training_df = training.create_trainset(
                self.training_csv, self.myranker, self
            )
            print(training_df)
            return training_df

    # Select disambiguation method
    def run(self, cands, mention_context):
        if cands:
            if "mostpopular" in self.method:
                link, score, other_cands = self.most_popular(cands)
                return (link, score, other_cands)

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


def rel_end_to_end(sent):
    """REL end-to-end using the API."""
    API_URL = "https://rel.cs.ru.nl/api"
    el_result = requests.post(API_URL, json={"text": sent, "spans": []}).json()
    return el_result
