import json
import os
import sys

import numpy as np
import pandas as pd
from haversine import haversine
from tqdm import tqdm

tqdm.pandas()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_wikipedia, training
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

    def run(self, dict_mention):
        if self.method == "mostpopular":
            return self.most_popular(dict_mention)
        if self.method == "bydistance":
            return self.by_distance(dict_mention)

    def disambiguation_setup(self, experiment_name):
        if self.method == "reldisamb":
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
            self.rel_params["model"] = EntityDisambiguation(
                base_path, wiki_version, config
            )

            return self.rel_params["mention_detection"], self.rel_params["model"]

    def format_linking_dataset(self, mentions_dataset):
        formatted_dataset = []
        for m in mentions_dataset:
            formatted_cands = m.copy()
            mention = m["mention"]
            formatted_cands["candidates"] = dict()
            formatted_cands["candidates"][mention] = {"Score": None}
            formatted_cands["candidates"][mention]["Candidates"] = dict()
            candidates = m["candidates"]
            for c in candidates:
                cand_wiki = c[0]
                cand_wiki = process_wikipedia.title_to_id(cand_wiki)
                cand_score = round(c[1], 3)
                formatted_cands["candidates"][mention]["Candidates"][
                    cand_wiki
                ] = cand_score
            if formatted_cands["gold"][0] != "NONE":
                formatted_cands["gold"] = process_wikipedia.title_to_id(
                    formatted_cands["gold"][0]
                )
            formatted_cands["prediction"] = process_wikipedia.title_to_id(
                formatted_cands["prediction"]
            )
            formatted_dataset.append(formatted_cands)
        return formatted_dataset

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

    def perform_linking_rel(self, mentions_dataset, sentence_id, linking_model):
        publication_entry = dict()
        if self.rel_params["ranking"] == "publ":
            # If "publ", add an artificial publication entry:
            publication_entry = self.add_publication_mention(mentions_dataset)
            mentions_dataset.append(publication_entry)
        # Predict mentions in one sentence:
        predictions, timing = linking_model.predict({sentence_id: mentions_dataset})
        if self.rel_params["ranking"] == "publ":
            # ... and if "publ", now remove the artificial publication entry!
            mentions_dataset.remove(publication_entry)
        # Postprocess the predictions:
        for i in range(len(mentions_dataset)):
            mention_dataset = mentions_dataset[i]
            prediction = predictions[sentence_id][i]
            if mention_dataset["mention"] == prediction["mention"]:
                mentions_dataset[i]["prediction"] = prediction["prediction"]
                # If entity is NIL, conf_ed is 0.0:
                mentions_dataset[i]["ed_score"] = round(
                    prediction.get("conf_ed", 0.0), 3
                )
        # Format the predictions to match the output of the other approaches:
        mentions_dataset = self.format_linking_dataset(mentions_dataset)
        mentions_dataset = {sentence_id: mentions_dataset}
        return mentions_dataset

    def perform_linking_mention(self, mention_data):
        # This predicts one mention at a time (does not look at context):
        prediction = self.run(mention_data)
        mention_data["prediction"] = prediction[0]
        mention_data["ed_score"] = prediction[1]
        return mention_data

    def add_publication_mention(self, mention_data):
        # Add artificial publication entity that is already disambiguated,
        # per sentence:
        sentence = mention_data[0]["sentence"]
        sent_idx = mention_data[0]["sent_idx"]
        context = mention_data[0]["context"]
        place_wqid = mention_data[0]["place_wqid"]
        place = mention_data[0]["place"]
        # Add place of publication as a fake entity in each sentence:
        # Wikipedia title of place of publication QID:
        wiki_gold = "NIL"
        gold_ids = process_wikipedia.id_to_title(place_wqid)
        # Get the first of the wikipedia titles returned (they're sorted
        # by their autoincrement id):
        if gold_ids:
            wiki_gold = [gold_ids[0]]
        sent2 = sentence + " Published in " + place
        pos2 = len(sentence) + len(" Published in ")
        end_pos2 = pos2 + len(place)
        dict_publ = {
            "mention": "publication",
            "sent_idx": sent_idx,
            "sentence": sent2,
            "gold": wiki_gold,
            "ngram": "publication",
            "context": context,
            "pos": pos2,
            "end_pos": end_pos2,
            "candidates": [[wiki_gold[0], 1.0]],
        }
        return dict_publ
