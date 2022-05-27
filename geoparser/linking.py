import os
import sys
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, pipeline

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import utils  # training


class Linker:
    def __init__(
        self,
        method,
        resources_path,
        linking_resources,
        base_model,
        overwrite_training,
    ):
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
        self.base_model = base_model
        self.overwrite_training = overwrite_training

    def __str__(self):
        s = "Entity Linking:\n* Method: {0}\n* Overwrite training: {1}\n* Linking resources: {2}\n".format(
            self.method,
            str(self.overwrite_training),
            ",".join(list(self.linking_resources.keys())),
        )
        return s

    def create_pipeline(self):
        # Load BERT model and tokenizer, and feature-extraction pipeline:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model_rd = pipeline(
            "feature-extraction", model=self.base_model, tokenizer=self.base_model
        )
        self.linking_resources["tokenizer"] = tokenizer
        self.linking_resources["model_rd"] = model_rd
        return self.linking_resources

    def load_resources(self):
        """
        Load resources required for linking.
        Note: different methods will require different resources.

        Returns:
            self.linking_resources (dict): a dictionary storing the resources
                that will be needed for a specific linking method.
        """
        # Load Wikidata gazetteer
        gaz = pd.read_csv(
            self.resources_path + "wikidata_gazetteer.csv", low_memory=False
        )
        gaz["instance_of"] = gaz["instance_of"].apply(utils.eval_with_exception)
        self.linking_resources["gazetteer"] = gaz

        # Load Wikidata mentions-to-wikidata (with normalized counts) to QID dictionary
        if self.method == "mostpopularnormalised":
            with open(
                self.resources_path + "mentions_to_wikidata_normalized.json", "r"
            ) as f:
                self.linking_resources["mentions_to_wikidata"] = json.load(f)

        # Load Wikidata mentions-to-wikidata (with absolute counts) to QID dictionary
        if self.method in ["mostpopular", "contextualized"]:
            with open(self.resources_path + "mentions_to_wikidata.json", "r") as f:
                self.linking_resources["mentions_to_wikidata"] = json.load(f)

        # Create contextualized embedding extraction pipeline:
        if self.method == "contextualized":
            self.linking_resources = self.create_pipeline()

        return self.linking_resources

    def perform_training(self, train_df):
        """
        TODO: Here will go the code to perform training, checking
        variable overwrite_training.

        Arguments:
            training_df (pd.DataFrame): a dataframe with a mention per row, for training.

        Returns:
            xxxxxxx
        """
        return None

    # ----------------------------------------------
    def perform_linking(self, test_df):
        """
        Perform the linking.

        Arguments:
            test_df (pd.DataFrame): a dataframe with a mention per row, for testing.

        Returns:
            test_df_results (pd.DataFrame): the same dataframe, with two additional
                columns: "pred_wqid" for the linked wikidata Id and "pred_wqid_score"
                for the linking score.
        """
        test_df_results = test_df.copy()
        test_df_results[["pred_wqid", "pred_wqid_score"]] = test_df_results.apply(
            lambda row: self.run(row.to_dict()), axis=1, result_type="expand"
        )
        return test_df_results

    # ----------------------------------------------
    # Select disambiguation method
    def run(self, dict_mention):
        """
        Given a mention dictionary, return the link and the score according to the
            method specified when initialising the Linker.

        Arguments:
            dict_mention (dict): dictionary with all the relevant information needed
                to disambiguate a certain mention.

        Returns:
            link (str): the Wikidata ID (e.g. "Q84") or "NIL".
            score (float): the confidence of the predicted link.
        """
        if "mostpopular" in self.method:
            link, score = self.most_popular(dict_mention)
            return link, score

    # ----------------------------------------------
    # Most popular candidate:
    def most_popular(self, dict_mention):
        """
        The most popular disambiguation method, which is a painfully strong baseline.
        Given a set of candidates for a given mention, returns as a prediction the
        candidate that is more relevant in terms of inlink structure in Wikipedia.

        Arguments:
            dict_mention (dict): dictionary with all the relevant information needed
                to disambiguate a certain mention.

        Returns:
            keep_most_popular (str): the Wikidata ID (e.g. "Q84") or "NIL".
            final_score (float): the confidence of the predicted link.
        """
        cands = dict_mention["candidates"]
        keep_most_popular = "NIL"
        keep_highest_score = 0.0
        total_score = 0.0
        final_score = 0.0
        if cands:
            for variation in cands:
                for candidate in cands[variation]["Candidates"]:
                    score = cands[variation]["Candidates"][candidate]
                    total_score += score
                    if score > keep_highest_score:
                        keep_highest_score = score
                        keep_most_popular = candidate
            # we return the predicted and the score (overall the total):
            final_score = keep_highest_score / total_score

        return keep_most_popular, final_score
