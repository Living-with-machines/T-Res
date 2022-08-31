import hashlib
import json
import os
import sys
import urllib

import numpy as np
import pandas as pd
from haversine import haversine
from tqdm import tqdm

tqdm.pandas()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# from transformers import AutoTokenizer, pipeline

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data, process_wikipedia, training
from utils.REL.entity_disambiguation import EntityDisambiguation
from utils.REL.mention_detection import MentionDetection


class Linker:
    def __init__(
        self,
        method,
        resources_path,
        linking_resources,
        base_model,
        overwrite_training,
        rel_params,
    ):
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
        self.base_model = base_model
        self.overwrite_training = overwrite_training
        self.rel_params = rel_params

    def __str__(self):
        s = (
            ">>> Entity Linking:\n"
            "    * Method: {0}\n"
            "    * Overwrite training: {1}\n"
        ).format(
            self.method,
            str(self.overwrite_training),
        )
        return s

    def load_resources(self):
        """
        Load resources required for linking.
        Note: different methods will require different resources.

        Returns:
            self.linking_resources (dict): a dictionary storing the resources
                that will be needed for a specific linking method.
        """
        print("*** Load linking resources.")

        # Load Wikidata mentions-to-QID with absolute counts:
        if self.method in ["mostpopular", "reldisamb"]:
            print("  > Loading mentions to wikidata mapping.")
            with open(self.resources_path + "mentions_to_wikidata.json", "r") as f:
                self.linking_resources["mentions_to_wikidata"] = json.load(f)

        if self.method in ["bydistance"]:
            print("  > Loading gazetteer.")
            gaz = pd.read_csv(
                self.resources_path + "wikidata_gazetteer.csv",
                usecols=["wikidata_id", "latitude", "longitude"],
            )
            gaz["latitude"] = gaz["latitude"].astype(float)
            gaz["longitude"] = gaz["longitude"].astype(float)
            gaz["coords"] = gaz[["latitude", "longitude"]].to_numpy().tolist()
            wqid_to_coords = dict(zip(gaz.wikidata_id, gaz.coords))
            self.linking_resources["wqid_to_coords"] = wqid_to_coords
            gaz = ""

        if self.method in ["reldisamb"]:
            # Load gazetteer
            print("  > Loading gazetteer.")
            gaz_ids = set(
                pd.read_csv(
                    self.resources_path + "wikidata_gazetteer.csv",
                    usecols=["wikidata_id"],
                )["wikidata_id"].tolist()
            )

            # Keep only wikipedia entities in the gazetteer:
            self.linking_resources["wikidata_locs"] = gaz_ids

        print("*** Linking resources loaded!\n")
        return self.linking_resources

    # ----------------------------------------------
    def perform_training(
        self,
        train_original,
        train_processed,
        dev_original,
        dev_processed,
        experiment_name,
        cand_selection,
    ):
        """
        TODO: Here will go the code to perform training, checking
        variable overwrite_training.

        Arguments:
            train_original (pd.DataFrame):
            train_processed (pd.DataFrame): a dataframe with a mention per row, for training.
            dev_original (pd.DataFrame):
            dev_processed (pd.DataFrame): a dataframe with a mention per row, for dev.

        Returns:
            xxxxxxx
        """
        if "mostpopular" in self.method:
            return None
        if "bydistance" in self.method:
            return None
        if "reldisamb" in self.method:
            training.train_rel_ed(
                self,
                train_original,
                train_processed,
                dev_original,
                dev_processed,
                experiment_name,
                cand_selection,
            )
            return self.rel_params

    # ----------------------------------------------
    def perform_linking(
        self, test_df, original_df, experiment_name, cand_selection=None
    ):
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

        if "mostpopular" in self.method:
            test_df_results[["pred_wqid", "pred_wqid_score"]] = test_df_results.apply(
                lambda row: self.most_popular(row.to_dict()),
                axis=1,
                result_type="expand",
            )

        if "bydistance" in self.method:
            test_df_results[["pred_wqid", "pred_wqid_score"]] = test_df_results.apply(
                lambda row: self.by_distance(row.to_dict()),
                axis=1,
                result_type="expand",
            )

        if "reldisamb" in self.method:
            dRELresults = self.rel_disambiguation(
                test_df, original_df, experiment_name, cand_selection
            )
            test_df_results[
                ["pred_wqid", "pred_wqid_score"]
            ] = test_df_results.progress_apply(
                lambda row: dRELresults[row["article_id"]][int(row["sentence_pos"])][
                    row["pred_mention"]
                ],
                axis=1,
                result_type="expand",
            )

        return test_df_results

    def rel_disambiguation(self, test_df, original_df, experiment_name, cand_selection):
        # Warning: no model has been trained, the latest one that's been trained will be loaded.

        base_path = self.rel_params["base_path"]
        wiki_version = self.rel_params["wiki_version"]
        # Instantiate REL mention detection:
        self.rel_params["mention_detection"] = MentionDetection(
            base_path, wiki_version, mylinker=self
        )

        # Instantiate REL entity disambiguation:
        experiment_path = os.path.join(
            base_path, wiki_version, "generated", experiment_name
        )
        config = {
            "mode": "eval",
            "model_path": os.path.join(experiment_path, "model"),
        }
        self.rel_params["model"] = EntityDisambiguation(base_path, wiki_version, config)

        dRELresults = dict()
        mentions_dataset = dict()
        # Given our mentions, use REL candidate selection module:
        if "reldisamb" in self.method:
            mentions_dataset, n_mentions = self.rel_params[
                "mention_detection"
            ].format_detected_spans(
                test_df,
                original_df,
                cand_selection,
                mylinker=self,
            )

        # Given the mentions dataset, predict and return linking:
        for mentions_doc in tqdm(mentions_dataset):
            link_predictions, timing = self.rel_params["model"].predict(
                mentions_dataset[mentions_doc]
            )
            for p in link_predictions:
                mentions_sent = p
                for m in link_predictions[p]:
                    returned_mention = m["mention"]
                    returned_prediction = m["prediction"]

                    # REL returns a wikipedia title, provide the wikidata QID:
                    wikipedia_title = process_wikipedia.make_wikilinks_consistent(
                        returned_prediction
                    )
                    processed_wikipedia_title = (
                        process_wikipedia.make_wikipedia2wikidata_consisent(
                            wikipedia_title
                        )
                    )
                    returned_prediction = process_wikipedia.title_to_id(
                        processed_wikipedia_title, lower=True
                    )
                    if not returned_prediction:
                        returned_prediction = "NIL"

                    # Disambiguation confidence:
                    returned_confidence = round(m.get("conf_ed", 0.0), 3)

                    if mentions_doc in dRELresults:
                        if mentions_sent in dRELresults[mentions_doc]:
                            dRELresults[mentions_doc][mentions_sent][
                                returned_mention
                            ] = (
                                returned_prediction,
                                returned_confidence,
                            )
                        else:
                            dRELresults[mentions_doc][mentions_sent] = {
                                returned_mention: (
                                    returned_prediction,
                                    returned_confidence,
                                )
                            }
                    else:
                        dRELresults[mentions_doc] = {
                            mentions_sent: {
                                returned_mention: (
                                    returned_prediction,
                                    returned_confidence,
                                )
                            }
                        }

        return dRELresults

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
                    score = self.linking_resources["mentions_to_wikidata"][variation][
                        candidate
                    ]
                    total_score += score
                    if score > keep_highest_score:
                        keep_highest_score = score
                        keep_most_popular = candidate
            # we return the predicted and the score (overall the total):
            final_score = keep_highest_score / total_score

        return keep_most_popular, final_score

    # ----------------------------------------------
    # Select candidate to place of publication:
    def by_distance(self, dict_mention):
        """
        The by_distance disambiguation method is another baseline, an unsupervised
        disambiguation approach. Given a set of candidates for a given mention and
        the place of publication of the original text, it returns as a prediction the
        location that is the closest to the place of publication.

        Arguments:
            dict_mention (dict): dictionary with all the relevant information needed
                to disambiguate a certain mention.

        Returns:
            keep_most_popular (str): the Wikidata ID (e.g. "Q84") or "NIL".
            final_score (float): the confidence of the predicted link.
        """
        cands = dict_mention["candidates"]
        origin_wqid = dict_mention["place_wqid"]
        origin_coords = self.linking_resources["wqid_to_coords"][origin_wqid]
        keep_closest_cand = "NIL"
        max_on_earth = 20000  # 20000 km, max on Earth
        keep_lowest_distance = max_on_earth  # 20000 km, max on Earth

        if cands:
            l = [list(cands[x]["Candidates"].keys()) for x in cands]
            l = [item for sublist in l for item in sublist]
            l = list(set(l))
            for candidate in l:
                cand_coords = self.linking_resources["wqid_to_coords"][candidate]
                geodist = haversine(origin_coords, cand_coords)
                if geodist < keep_lowest_distance:
                    keep_lowest_distance = geodist
                    keep_closest_cand = candidate

        if keep_lowest_distance == 0.0:
            keep_lowest_distance = 1.0
        else:
            keep_lowest_distance = 1.0 - (keep_lowest_distance / max_on_earth)

        return keep_closest_cand, round(keep_lowest_distance, 3)
