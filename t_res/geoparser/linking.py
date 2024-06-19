import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from haversine import haversine
from tqdm import tqdm

tqdm.pandas()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from ..utils import rel_utils
from ..utils.REL import entity_disambiguation
from . import ranking


class Linker:
    """
    The Linker class provides methods for entity linking, which is the task of
    associating mentions in text with their corresponding entities in a
    knowledge base.

    Arguments:
        method (Literal["mostpopular", "reldisamb", "bydistance"]): The
            linking method to use.
        resources_path (str): The path to the linking resources.
        experiments_path (str, optional): The path to the experiments
            directory. Default is "../experiments/".
        linking_resources (dict, optional): Dictionary containing the
            necessary linking resources. Defaults to ``dict()`` (an empty
            dictionary).
        overwrite_training (bool): Flag indicating whether to overwrite the
            training. Defaults to ``False``.
        rel_params (dict, optional): Dictionary containing the parameters
            for performing entity disambiguation using the ``reldisamb``
            approach (adapted from the Radboud Entityt Linker, REL).
            For the default settings, see Notes below.

    Example:

    .. code-block:: python

       linker = Linker(
         method="mostpopular",
         resources_path="/path/to/resources/",
         experiments_path="/path/to/experiments/",
         linking_resources={},
         overwrite_training=True,
         rel_params={"with_publication": True, "do_test": True}
       )

    Note:

        * Note that, in order to instantiate the Linker with the ``reldisamb``
        method, the Linker needs to be wrapped by a context manager in which
        a connection to the entity embeddings database is established and a
        cursor is created:

        .. code-block:: python

           with sqlite3.connect("../resources/rel_db/embeddings_database.db") as conn:
             cursor = conn.cursor()
             mylinker = linking.Linker(
             method="reldisamb",
             resources_path="../resources/",
             experiments_path="../experiments/",
             linking_resources=dict(),
             rel_params={
               "model_path": "../resources/models/disambiguation/",
               "data_path": "../experiments/outputs/data/lwm/",
               "training_split": "",
               "db_embeddings": cursor,
               "with_publication": wpubl,
               "without_microtoponyms": wmtops,
               "do_test": False,
               "default_publname": "",
               "default_publwqid": "",
             },
             overwrite_training=False,
           )

        See below the default settings for ``rel_params``. Note that
        `db_embeddings` defaults to None, but it should be assigned a
        cursor to the entity embeddings database, as described above:

        .. code-block:: python

           rel_params: Optional[dict] = {
             "model_path": "../resources/models/disambiguation/",
             "data_path": "../experiments/outputs/data/lwm/",
             "training_split": "originalsplit",
             "db_embeddings": None,
             "with_publication": True,
             "without_microtoponyms": True,
             "do_test": False,
             "default_publname": "United Kingdom",
             "default_publwqid": "Q145",
           }

    """

    def __init__(
        self,
        method: Literal["mostpopular", "reldisamb", "bydistance"],
        resources_path: str,
        experiments_path: Optional[str] = "../experiments",
        linking_resources: Optional[dict] = dict(),
        overwrite_training: Optional[bool] = False,
        rel_params: Optional[dict] = None,
        rel_device: Optional[str] = None, 
    ):
        """
        Initialises a Linker object.
        """
        self.method = method
        self.resources_path = resources_path
        self.experiments_path = experiments_path
        self.linking_resources = linking_resources
        self.overwrite_training = overwrite_training

        if rel_params is None:
            rel_params = {
                "model_path": os.path.join(resources_path, "models/disambiguation/"),
                "data_path": os.path.join(experiments_path, "outputs/data/lwm/"),
                "training_split": "originalsplit",
                "db_embeddings": None,  # The cursor to the embeddings database.
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            }

        self.rel_params = rel_params
        self.rel_device = rel_device

    def __str__(self) -> str:
        """
        Returns a string representation of the Linker object.

        Returns:
            str: String representation of the Linker object.
        """
        s = ">>> Entity Linking:\n"
        s += f"    * Method: {self.method}\n"
        s += f"    * Overwrite training: {self.overwrite_training}\n"
        return s

    def load_resources(self) -> dict:
        """
        Loads the linking resources.

        Returns:
            dict: Dictionary containing loaded necessary linking resources.

        Note:
            Different methods will require different resources.
        """
        print("*** Load linking resources.")

        # Load Wikidata mentions-to-QID with absolute counts:
        print("  > Loading mentions to wikidata mapping.")
        with open(
            os.path.join(self.resources_path, "wikidata/mentions_to_wikidata.json"), "r"
        ) as f:
            self.linking_resources["mentions_to_wikidata"] = json.load(f)

        print("  > Loading gazetteer.")
        gaz = pd.read_csv(
            os.path.join(self.resources_path, "wikidata/wikidata_gazetteer.csv"),
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

        # The entity2class.txt file is created as the last step in
        # wikipedia processing:
        with open(
            os.path.join(self.resources_path, "wikidata/entity2class.txt"), "r"
        ) as f:
            self.linking_resources["entity2class"] = json.load(f)

        print("*** Linking resources loaded!\n")
        return self.linking_resources

    def run(self, dict_mention: dict) -> Tuple[str, float, dict]:
        """
        Executes the linking process based on the specified unsupervised
        method.

        Arguments:
            dict_mention: Dictionary containing the mention information.

        Returns:
            Tuple[str, float, dict]:
                The result of the linking process. For details, see below:

                - If the ``method`` provided when initialising the
                  :py:meth:`~geoparser.linking.Linker` object was
                  ``"mostpopular"``, see
                  :py:meth:`~geoparser.linking.Linker.most_popular`.
                - If the ``method`` provided when initialising the
                  :py:meth:`~geoparser.linking.Linker` object was
                  ``"bydistance"``, see
                  :py:meth:`~geoparser.linking.Linker.by_distance`.

        """
        if self.method == "mostpopular":
            return self.most_popular(dict_mention)

        if self.method == "bydistance":
            return self.by_distance(dict_mention)

        raise SyntaxError(f"Unknown method provided: {self.method}")

    def most_popular(self, dict_mention: dict) -> Tuple[str, float, dict]:
        """
        Select most popular candidate, given Wikipedia's in-link structure.

        Arguments:
            dict_mention (dict): dictionary with all the relevant information
                needed to disambiguate a certain mention.

        Returns:
            Tuple[str, float, dict]:
                A tuple containing the most popular candidate's Wikidata ID
                (e.g. ``"Q84"``) or ``"NIL"``, the confidence score of the
                predicted link as a float, and a dictionary of all candidates
                and their confidence scores.

        .. note::

            Applying the "most popular" disambiguation method for linking
            entities. Given a set of candidates for a given mention, the
            function returns as a prediction the more relevant Wikidata
            candidate, determined from the in-link structure of Wikipedia.
        """
        cands = dict_mention["candidates"]
        most_popular_candidate_id = "NIL"
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
                        most_popular_candidate_id = candidate

            # Return the predicted and the score (overall the total):
            final_score = keep_highest_score / total_score

            # Compute scores for all candidates
            all_candidates = {
                cand: (score / total_score) for cand, score in all_candidates.items()
            }

        return most_popular_candidate_id, final_score, all_candidates

    def by_distance(
        self, dict_mention: dict, origin_wqid: Optional[str] = ""
    ) -> Tuple[str, float, dict]:
        """
        Select candidate based on distance to the place of publication.

        Arguments:
            dict_mention (dict): dictionary with all the relevant information
                needed to disambiguate a certain mention.
            origin_wqid (str, optional): The origin Wikidata ID for distance
                calculation. Defaults to ``""``.

        Returns:
            Tuple[str, float, dict]:
                A tuple containing the Wikidata ID of the closest candidate
                to the place of publication (e.g. ``"Q84"``) or ``"NIL"``,
                the confidence score of the predicted link as a float (rounded
                to 3 decimals), and a dictionary of all candidates and their
                confidence scores.

        .. note::

            Applying the "by distance" disambiguation method for linking
            entities, based on geographical distance. It undertakes an
            unsupervised disambiguation, which returns a prediction of a
            location closest to the place of publication, for a provided set
            of candidates and the place of publication of the original text.
        """
        cands = dict_mention["candidates"]
        origin_coords = self.linking_resources["wqid_to_coords"].get(origin_wqid)
        if not origin_coords:
            origin_coords = self.linking_resources["wqid_to_coords"].get(
                dict_mention["place_wqid"]
            )
        closest_candidate_id = "NIL"
        max_on_gb = 1000  # 1000 km, max on GB
        keep_lowest_distance = max_on_gb  # 20000 km, max on Earth
        keep_lowest_relv = 1.0
        all_candidates = {}

        if cands:
            for x in cands:
                matching_score = cands[x]["Score"]
                for candidate, score in cands[x]["Candidates"].items():
                    cand_coords = self.linking_resources["wqid_to_coords"][candidate]
                    geodist = 20000
                    # if origin_coords and cand_coords:  # If there are coordinates
                    try:
                        geodist = haversine(origin_coords, cand_coords)
                        all_candidates[candidate] = geodist
                    except ValueError:
                        # We have one candidate with coordinates in Venus!
                        pass
                    if geodist < keep_lowest_distance:
                        keep_lowest_distance = geodist
                        closest_candidate_id = candidate
                        keep_lowest_relv = (matching_score + score) / 2.0

        if keep_lowest_distance == 0.0:
            keep_lowest_distance = 1.0
        else:
            keep_lowest_distance = (
                max_on_gb if keep_lowest_distance > max_on_gb else keep_lowest_distance
            )
            keep_lowest_distance = 1.0 - (keep_lowest_distance / max_on_gb)

        final_score = 0.0
        if not closest_candidate_id == "NIL":
            final_score = round((keep_lowest_relv + keep_lowest_distance) / 2, 3)

        return closest_candidate_id, final_score, all_candidates

    def train_load_model(
        self, myranker: ranking.Ranker, split: Optional[str] = "originalsplit"
    ) -> entity_disambiguation.EntityDisambiguation:
        """
        Trains or loads the entity disambiguation model.

        Arguments:
            myranker (geoparser.ranking.Ranker): The ranker object used for
                training.
            split (str, optional): The split type for training. Defaults to
                ``"originalsplit"``.

        Returns:
            entity_disambiguation.EntityDisambiguation:
                A trained Entity Disambiguation model.

        .. note::

            The training will be skipped if the model already exists and
            ``overwrite_training`` was set to False when initiating the Linker
            object, or if the disambiguation method is unsupervised. The
            training will be run on test mode if ``rel_params`` had a
            ``do_test`` key's value set to True when initiating the Linker
            object.

        .. note::

            **Credit:**

            This method is adapted from the `REL: Radboud Entity Linker
            <https://github.com/informagi/REL/>`_ Github repository:
            Copyright (c) 2020 Johannes Michael van Hulst. See the `permission
            notice <https://github.com/informagi/REL/blob/main/LICENSE>`_.

            ::

                Reference:

                @inproceedings{vanHulst:2020:REL,
                author =    {van Hulst, Johannes M. and Hasibi, Faegheh and Dercksen, Koen and Balog, Krisztian and de Vries, Arjen P.},
                title =     {REL: An Entity Linker Standing on the Shoulders of Giants},
                booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
                series =    {SIGIR '20},
                year =      {2020},
                publisher = {ACM}
                }
        """
        if self.method == "reldisamb":
            # Generate ED model name:
            linker_name = myranker.method
            if myranker.method == "deezymatch":
                linker_name += "+" + str(myranker.deezy_parameters["num_candidates"])
                linker_name += "+" + str(
                    myranker.deezy_parameters["selection_threshold"]
                )
            linker_name += f"_{split}"
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
                # Create the folder where to store the resulting
                # disambiguation models:
                Path(linker_name).mkdir(parents=True, exist_ok=True)

                # Load the linking dataset, separate training and dev:
                linking_df_path = os.path.join(
                    self.rel_params["data_path"], "linking_df_split.tsv"
                )
                linking_df = pd.read_csv(linking_df_path, sep="\t")
                train_df = linking_df[linking_df[split] == "train"]
                dev_df = linking_df[linking_df[split] == "dev"]

                # If this is a test, use only the first 20 rows of the train
                # and dev sets:
                if self.rel_params["do_test"] == True:
                    train_df = train_df.iloc[:20]
                    dev_df = dev_df.iloc[:20]

                # Prepare the dataset into the format required by REL:
                train_json = rel_utils.prepare_rel_trainset(
                    train_df,
                    self.rel_params,
                    self.linking_resources["mentions_to_wikidata"],
                    myranker,
                    "train",
                )
                dev_json = rel_utils.prepare_rel_trainset(
                    dev_df,
                    self.rel_params,
                    self.linking_resources["mentions_to_wikidata"],
                    myranker,
                    "dev",
                )

                # Set ED configuration to train mode:
                config_rel = {
                    "mode": "train",
                    "model_path": os.path.join(linker_name, "model"),
                }
                if self.rel_device is not None:
                    config_rel["device"] = self.rel_device

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
                if self.rel_device is not None:
                    config_rel["device"] = self.rel_device

                model = entity_disambiguation.EntityDisambiguation(
                    self.rel_params["db_embeddings"],
                    config_rel,
                )

                return model
