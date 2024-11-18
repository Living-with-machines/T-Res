import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict

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
from .dataclasses import StringMatchLinks, Candidates, CandidateMatches, CandidateLinks, MostPopularLink, ByDistanceLink, RelDisambLink

class Linker:
    """
    The Linker class provides methods for entity linking, which is the task of
    associating mentions in text with their corresponding entities in a
    knowledge base.

    Arguments:
        resources_path (str): The path to the linking resources.
        experiments_path (str, optional): The path to the experiments
            directory. Default is "../experiments/".
        linking_resources (dict, optional): Dictionary containing the
            necessary linking resources. Defaults to ``dict()`` (an empty
            dictionary).

    This base class should not be instatiated directly. Instead use a subclass
    constructor.
    """

    def __init__(
        self,
        resources_path: str,
        experiments_path: Optional[str] = "../experiments",
        linking_resources: Optional[dict] = dict(),
    ):
        """
        Initialises a Linker object.
        """
        self.resources_path = resources_path
        self.experiments_path = experiments_path
        self.linking_resources = linking_resources

        # TODO:
        # self.cache = dict()

    def __str__(self) -> str:
        """
        Returns a string representation of the Linker object.

        Returns:
            str: String representation of the Linker object.
        """
        s = ">>> Entity Linking:\n"
        s += f"    * Method: {self.method_name()}\n"
        s += f"    * Overwrite training: {self.overwrite_training}\n"
        return s

    def method_name(self) -> str:
        """
        The name of the entity linking method.

        Returns:
            str: The entity linking method name.
        """
        raise NotImplementedError("Subclass implementation required.")

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

        # Load Wikidata mentions-to-QID with normalized counts:
        print("  > Loading mentions to normalized wikidata mapping.")
        with open(
            os.path.join(self.resources_path, "wikidata/mentions_to_wikidata_normalized.json"), "r"
        ) as f:
            self.linking_resources["mentions_to_wikidata_normalized"] = json.load(f)

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

    # TODO: replace dict_mention argument with two args:
    # 1. Mention dataclass instance (Recogniser output)
    # 2. CandidateMatches dataclass instance (Ranker output)
    def run(self, dict_mention: dict) -> Candidates:
        """
        Execute the linking process. Each Linker subclass must implement a 
        linking method by overriding this function.

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
        raise NotImplementedError("Subclass implementation required.")

class MostPopularLinker(Linker):
    """
    An entity linking method that selects the candidate that is most
    popular in the Wikipedia knowledgebase.

    Example:

    .. code-block:: python

       linker = MostPopularLinker(
         resources_path="/path/to/resources/",
         experiments_path="/path/to/experiments/",
         linking_resources={},
       )
    """

    def method_name(self) -> str:
        return "mostpopular"

    # Define a closure for computing the disambiguation scores.
    def disambiguation_scores(wikidata_links: List[MostPopularLink]) -> Dict[str, float]:
        def closure():
            total = sum([m.freq for m in wikidata_links])
            return {link.wqid: link.freq / total for link in wikidata_links}
        return closure

    # TODO: update docstring
    def run(self, dict_mention: dict) -> Candidates:
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
        candidate_matches = dict_mention["candidates"]

        if not isinstance(candidate_matches, CandidateMatches):
            raise ValueError("Expected CandidateMatches instance")

        if candidate_matches.is_empty():
            return Candidates(candidate_matches.mention, candidate_matches.ranking_method, self.method_name(), list())

        wikidata_links = []
        candidate_links = []
        for match in candidate_matches.matches:
            if not isinstance(match, StringMatchLinks):
                raise ValueError("Expected StringMatchLinks instance.")
            
            for wqid in match.wqid_links:
                freq = self.linking_resources["mentions_to_wikidata"][match.variation][wqid]
                wikidata_links.append(MostPopularLink(wqid, freq))

            closure = MostPopularLinker.disambiguation_scores(wikidata_links)
            candidate_links.append(CandidateLinks(match.as_string_match(), wikidata_links, closure))

        # # TODO: create a Linker cache and add the resulting candidates to it.
        return Candidates(candidate_matches.mention, candidate_matches.ranking_method, self.method_name(), candidate_links)

class ByDistanceLinker(Linker):
    """
    An entity linking method that selects the candidate based on its
    proximity to the place of publication.

    Example:

    .. code-block:: python

       linker = ByDistanceLinker(
         resources_path="/path/to/resources/",
         experiments_path="/path/to/experiments/",
         linking_resources={},
       )
    """

    def method_name(self) -> str:
        return "bydistance"

    # Define a closure for computing the disambiguation scores.
    def disambiguation_scores(wikidata_links: List[ByDistanceLink], matching_score: float) -> Dict[str, float]:
        def closure():
            max_on_gb = 1000  # 1000 km, max on GB
            ret = dict()
            for link in wikidata_links:
                
                distance = min(max_on_gb, link.geodist if link.geodist is not None else max_on_gb)
                
                if distance == 0.0:
                    distance_score = 1.0
                else:
                    distance = (max_on_gb if distance > max_on_gb else distance)
                    distance_score = 1.0 - (distance / max_on_gb)

                relv_score = min(1.0, (matching_score + link.normalized_score) / 2.0)

                final_score = 0.0
                if link.geodist is not None:
                    final_score = round((relv_score + distance_score) / 2, 3)
                ret[link.wqid] = final_score
            return ret
        
        return closure
    
    def run(
        self, dict_mention: dict, origin_wqid: Optional[str] = ""
    ) -> Candidates:
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
        candidate_matches = dict_mention["candidates"]

        if not isinstance(candidate_matches, CandidateMatches):
            raise ValueError("Expected CandidateMatches instance")

        if candidate_matches.is_empty():
            return Candidates(candidate_matches.mention, candidate_matches.ranking_method, self.method_name(), list())

        # TODO: fix this duplication in the method args.
        if not(origin_wqid):
            origin_wqid = dict_mention["place_wqid"]

        origin_coords = self.linking_resources["wqid_to_coords"].get(origin_wqid)
        if not origin_coords:
            origin_coords = self.linking_resources["wqid_to_coords"].get(
                dict_mention["place_wqid"]
            )

        wikidata_links = []
        candidate_links = []
        for match in candidate_matches.matches:
            if not isinstance(match, StringMatchLinks):
                raise ValueError("Expected StringMatchLinks instance.")
            # for i, wikidata_link in enumerate(match.wikidata_links):
            for wqid in match.wqid_links:

                candidate_coords = self.linking_resources["wqid_to_coords"][wqid]
                # If coordinates are known for origin and candidate, compute the geodesic distance.
                try:
                    geodist = haversine(origin_coords, candidate_coords)
                except ValueError:
                    # We have one candidate with coordinates in Venus!
                    geodist = None

                normalized_score = self.linking_resources["mentions_to_wikidata_normalized"][match.variation][
                    wqid
                ]
                wikidata_links.append(ByDistanceLink(wqid, origin_wqid, geodist, normalized_score))

            closure = ByDistanceLinker.disambiguation_scores(wikidata_links, match.string_similarity)
            candidate_links.append(CandidateLinks(match.as_string_match(), wikidata_links, closure))

        # # TODO: create a Linker cache and add the resulting candidates to it.
        return Candidates(candidate_matches.mention, candidate_matches.ranking_method, self.method_name(), candidate_links)

class RelDisambLinker(Linker):
    """
    Linker subclass implementing an entity linking method that selects the 
    candidate using the Radboud Entity Linker (REL) model.

    Arguments:
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

    # Override the constructor to include REL model parameters.
    def __init__(
        self,
        resources_path: str,
        experiments_path: Optional[str] = "../experiments",
        linking_resources: Optional[dict] = dict(),
        overwrite_training: Optional[bool] = False,
        rel_params: Optional[dict] = None,
    ):
        
        super().__init__(resources_path, experiments_path, linking_resources)

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


    def method_name(self) -> str:
        return "reldisamb"

    # Define a closure for computing the disambiguation scores.
    def disambiguation_scores(wikidata_links: List[RelDisambLink]) -> Dict[str, float]:
        # TODO (refactor "reldisamb" score computation into this closure.)
        def closure():
            # TODO: these are dummy scores copied from MostPopularLinker (not yet implemented).
            total = sum([m.freq for m in wikidata_links])
            return {link.wqid: link.freq / total for link in wikidata_links}
            raise NotImplementedError("Not yet implemented.")
        return closure

    # TODO: refactor linking logic into this run method.
    def run(self, dict_mention: dict) -> Candidates:
        
        candidate_matches = dict_mention["candidates"]

        if not isinstance(candidate_matches, CandidateMatches):
            raise ValueError("Expected CandidateMatches instance")

        if candidate_matches.is_empty():
            return Candidates(candidate_matches.mention, candidate_matches.ranking_method, self.method_name(), list())

        wikidata_links = []
        candidate_links = []
        for match in candidate_matches.matches:
            if not isinstance(match, StringMatchLinks):
                raise ValueError("Expected StringMatchLinks instance.")
            
            for wqid in match.wqid_links:
                freq = self.linking_resources["mentions_to_wikidata"][match.variation][wqid]
                normalized_score = self.linking_resources["mentions_to_wikidata_normalized"][match.variation][
                    wqid
                ]
                wikidata_links.append(RelDisambLink(wqid, freq, normalized_score))

            closure = RelDisambLinker.disambiguation_scores(wikidata_links)
            candidate_links.append(CandidateLinks(match.as_string_match(), wikidata_links, closure))

        # # TODO: create a Linker cache and add the resulting candidates to it.
        return Candidates(candidate_matches.mention, candidate_matches.ranking_method, self.method_name(), candidate_links)
        
    def train_load_model(
        self, ranker: ranking.Ranker, split: Optional[str] = "originalsplit"
    ) -> entity_disambiguation.EntityDisambiguation:
        """
        Trains or loads the entity disambiguation model.

        Arguments:
            ranker (geoparser.ranking.Ranker): The ranker object used for
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
        # Generate ED model name:
        linker_name = ranker.method_name
        if ranker.method_name == "deezymatch":
            linker_name += "+" + str(ranker.deezy_parameters["num_candidates"])
            linker_name += "+" + str(
                ranker.deezy_parameters["selection_threshold"]
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
                ranker,
                self,
                "train",
            )
            dev_json = rel_utils.prepare_rel_trainset(
                dev_df,
                self.rel_params,
                self.linking_resources["mentions_to_wikidata"],
                ranker,
                self,
                "dev",
            )

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