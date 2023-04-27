import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from haversine import haversine
from tqdm import tqdm

tqdm.pandas()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import rel_utils
from utils.REL import entity_disambiguation


class Linker:
    def __init__(
        self,
        method,
        resources_path,
        linking_resources,
        overwrite_training,
        rel_params=dict(),
    ):
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
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
        print("  > Loading mentions to wikidata mapping.")
        with open(self.resources_path + "wikidata/mentions_to_wikidata.json", "r") as f:
            self.linking_resources["mentions_to_wikidata"] = json.load(f)

        print("  > Loading gazetteer.")
        gaz = pd.read_csv(
            self.resources_path + "wikidata/wikidata_gazetteer.csv",
            usecols=["wikidata_id", "latitude", "longitude"],
        )
        gaz["latitude"] = gaz["latitude"].astype(float)
        gaz["longitude"] = gaz["longitude"].astype(float)
        gaz["coords"] = gaz[["latitude", "longitude"]].to_numpy().tolist()
        wqid_to_coords = dict(zip(gaz.wikidata_id, gaz.coords))
        self.linking_resources["wqid_to_coords"] = wqid_to_coords
        gaz_ids = set(gaz["wikidata_id"].tolist())
        # Keep only wikipedia entities in the gazetteer:
        self.linking_resources["wikidata_locs"] = gaz_ids
        gaz_ids = ""
        gaz = ""

        # The entity2class.txt file is created as the last step in wikipediaprocessing:
        with open(self.resources_path + "wikidata/entity2class.txt", "r") as fr:
            self.linking_resources["entity2class"] = json.load(fr)

        print("*** Linking resources loaded!\n")
        return self.linking_resources

    def run(self, dict_mention):
        if self.method == "mostpopular":
            return self.most_popular(dict_mention)
        if self.method == "bydistance":
            return self.by_distance(dict_mention)

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
            all_candidates: dictionary containing all candidates and related score
        """
        cands = dict_mention["candidates"]
        keep_most_popular = "NIL"
        keep_highest_score = 0.0
        total_score = 0.0
        final_score = 0.0
        all_candidates = {}
        if cands:
            for variation in cands:
                for candidate in cands[variation]["Candidates"]:
                    score = self.linking_resources["mentions_to_wikidata"][variation][
                        candidate
                    ]
                    total_score += score
                    all_candidates[candidate] = score
                    if score > keep_highest_score:
                        keep_highest_score = score
                        keep_most_popular = candidate
            # we return the predicted and the score (overall the total):
            final_score = keep_highest_score / total_score

            # we compute scores for all candidates
            all_candidates = {
                cand: (score / total_score) for cand, score in all_candidates.items()
            }

        return keep_most_popular, final_score, all_candidates

    # ----------------------------------------------
    # Select candidate to place of publication:
    def by_distance(self, dict_mention, origin_wqid=""):
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
        origin_coords = self.linking_resources["wqid_to_coords"].get(origin_wqid)
        if not origin_coords:
            origin_coords = self.linking_resources["wqid_to_coords"].get(
                dict_mention["place_wqid"]
            )
        keep_closest_cand = "NIL"
        max_on_gb = 1000  # 1000 km, max on GB
        keep_lowest_distance = max_on_gb  # 20000 km, max on Earth
        keep_lowest_relv = 1.0
        resulting_cands = {}

        if cands:
            for x in cands:
                matching_score = cands[x]["Score"]
                for candidate, score in cands[x]["Candidates"].items():
                    cand_coords = self.linking_resources["wqid_to_coords"][candidate]
                    geodist = 20000
                    # if origin_coords and cand_coords:  # If there are coordinates
                    try:
                        geodist = haversine(origin_coords, cand_coords)
                        resulting_cands[candidate] = geodist
                    except ValueError:  # We have one candidate with coordinates in Venus!
                        pass
                    if geodist < keep_lowest_distance:
                        keep_lowest_distance = geodist
                        keep_closest_cand = candidate
                        keep_lowest_relv = (matching_score + score) / 2.0

        if keep_lowest_distance == 0.0:
            keep_lowest_distance = 1.0
        else:
            keep_lowest_distance = (
                max_on_gb if keep_lowest_distance > max_on_gb else keep_lowest_distance
            )
            keep_lowest_distance = 1.0 - (keep_lowest_distance / max_on_gb)

        resulting_score = 0.0
        if not keep_closest_cand == "NIL":
            resulting_score = round((keep_lowest_relv + keep_lowest_distance) / 2, 3)

        return keep_closest_cand, resulting_score, resulting_cands

    def train_load_model(self, myranker, split="originalsplit"):
        """
        Training an entity disambiguation model. The training will be skipped
        if the model already exists and self.overwrite_training it set to False,
        or if the disambiguation method is unsupervised. The training will
        be run on test mode if self.do_test is set to True.

        Returns:
            A trained DeezyMatch model.
            The DeezyMatch candidate vectors.
        """
        if self.method == "reldisamb":

            # Generate ED model name:
            linker_name = myranker.method
            if myranker.method == "deezymatch":
                linker_name += "+" + str(myranker.deezy_parameters["num_candidates"])
                linker_name += "+" + str(
                    myranker.deezy_parameters["selection_threshold"]
                )
            linker_name += "_" + split
            if self.rel_params["with_publication"]:
                linker_name += "+wpubl"
            if self.rel_params["without_microtoponyms"]:
                linker_name += "+wmtops"
            if self.rel_params["do_test"]:
                linker_name += "_test"
            linker_name = os.path.join(self.rel_params["model_path"], linker_name)

            if self.overwrite_training == True or not Path(linker_name).is_dir():
                print(
                    "The entity disambiguation model does not exist or overwrite_training is set to True."
                )

                print("Creating the dataset.")
                # Create the folder where to store the resulting disambiguation models:
                Path(linker_name).mkdir(parents=True, exist_ok=True)
                # Load the linking dataset, separate training and dev:
                linking_df_path = os.path.join(
                    self.rel_params["data_path"], "linking_df_split.tsv"
                )
                linking_df = pd.read_csv(linking_df_path, sep="\t")
                train_df = linking_df[linking_df[split] == "train"]
                dev_df = linking_df[linking_df[split] == "dev"]

                # If this is a test, use only the first 20 rows of the train and dev sets:
                if self.rel_params["do_test"] == True:
                    train_df = train_df.iloc[:20]
                    dev_df = dev_df.iloc[:20]

                # Prepare the dataset into the format required by REL:
                train_json = rel_utils.prepare_rel_trainset(
                    train_df, self, myranker, "train"
                )
                dev_json = rel_utils.prepare_rel_trainset(dev_df, self, myranker, "dev")
                # Set ED configuration to train mode:
                config_rel = {
                    "mode": "train",
                    "model_path": os.path.join(linker_name, "model"),
                }
                # Instantiate the entity disambiguation model:
                model = entity_disambiguation.EntityDisambiguation(
                    self.rel_params["db_embeddings"],
                    config_rel,
                )
                print("Training the model.")
                # Train the model using lwm_train:
                model.train(train_json, dev_json)
                # Train and predict using LR (to obtain confidence scores)
                model.train_LR(train_json, dev_json, linker_name)
                return model
            else:
                # Setting disambiguation model mode to "eval":
                config_rel = {
                    "mode": "eval",
                    "model_path": os.path.join(linker_name, "model"),
                }
                model = entity_disambiguation.EntityDisambiguation(
                    self.rel_params["db_embeddings"],
                    config_rel,
                )
                return model
