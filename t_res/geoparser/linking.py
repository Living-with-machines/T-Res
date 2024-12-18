import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

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
from ..utils.dataclasses import *

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
    # Class attribute for the name of the linking method.
    method_name: str = None

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
        self.resources = linking_resources

        # TODO:
        # self.cache = dict()

    def __str__(self) -> str:
        """
        Returns a string representation of the Linker object.

        Returns:
            str: String representation of the Linker object.
        """
        s = ">>> Entity Linking:\n"
        s += f"    * Method: {self.method_name}\n"
        return s

    def new(**kwargs) -> 'Linker':
        """
        Static constructor.

        Args:
            kwargs (dict): A dictionary of keyword arguments matching the
                arguments to a subclass __init__ constructor, plus a 
                `method_name` argument to specify the desired subclass.

        Returns:
            Linker: A Linker subclass instance.

        """
        if not 'method_name' in kwargs.keys():
            raise ValueError("Expected `method_name` keyword argument.")
        method_name = kwargs['method_name']
        del kwargs['method_name']
        if method_name == 'mostpopular':
            return MostPopularLinker(**kwargs)
        if method_name == 'bydistance':
            return ByDistanceLinker(**kwargs)
        if method_name == 'reldisamb':
            return RelDisambLinker(**kwargs)
        raise ValueError("Invalid linking method: {method_name}")

    def wkdt_class(self, wqid: str) -> Optional[str]:
        """Returns the Wikidata class for the given Wikidata entry, if available."""
        return self.resources["entity2class"].get(wqid, None)
    
    def empty_candidates(self, mention: Mention, ranking_method: str, place_of_pub_wqid: str, place_of_pub: str):
        """Returns an empty `Candidates` instance."""
        return MentionCandidates(
            mention,
            ranking_method,
            self.method_name,
            list(),
            place_of_pub_wqid,
            place_of_pub,
            self.with_publication())

    def load(self):
        """
        Loads the linking resources and assigns them to instance variables.
        """
        print("*** Load linking resources.")

        # TODO: make this more consistent with the Ranker (which has a mentions_to_wikidata attribute).

        # Load Wikidata mentions-to-QID with absolute counts:
        print("  > Loading mentions to wikidata mapping.")
        with open(
            os.path.join(self.resources_path, "wikidata/mentions_to_wikidata.json"), "r"
        ) as f:
            self.resources["mentions_to_wikidata"] = json.load(f)

        # Load Wikidata mentions-to-QID with normalized counts:
        print("  > Loading mentions to normalized wikidata mapping.")
        with open(
            os.path.join(self.resources_path, "wikidata/mentions_to_wikidata_normalized.json"), "r"
        ) as f:
            self.resources["mentions_to_wikidata_normalized"] = json.load(f)

        # The entity2class.txt file is created as the last step in
        # wikipedia processing:
        with open(
            os.path.join(self.resources_path, "wikidata/entity2class.txt"), "r"
        ) as f:
            self.resources["entity2class"] = json.load(f)

        print("*** Linking resources loaded!\n")

    # TODO: docstring
    def run(
            self, 
            matches: CandidateMatches, 
            place_of_pub_wqid: Optional[str]=None,
            place_of_pub: Optional[str]=None,
        ) -> MentionCandidates:
        """
        Execute the linking process. Each Linker subclass must implement a 
        linking method by overriding this function.

        Arguments:
            matches: A CandidatesMatches instance.
            origin_wqid (Optional[str]): The Wikidata ID of the place of publication.

        Returns:
            Candidates: The candidates identified by the linking process.
        """
        if matches.is_empty():
            return self.empty_candidates(matches.mention, matches.ranking_method, place_of_pub_wqid, place_of_pub)

        candidate_links = [CandidateLinks(m.as_string_match(), self.wikidata_links(m, place_of_pub_wqid)) 
                           for m in matches.matches]

        # # TODO: create a Linker cache and add the resulting candidates to it.
        return MentionCandidates(
            matches.mention, 
            matches.ranking_method, 
            self.method_name, 
            candidate_links,
            place_of_pub_wqid,
            place_of_pub,
            self.with_publication(),
        )
    
    # TODO: docstring
    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        raise NotImplementedError("Subclass implementation required.")

    def with_publication(self) -> bool:
        return False

    def disambiguate(self, candidates: List[SentenceCandidates]) -> Predictions:
        """
        Perform entity disambiguation given a list of already identified
        toponyms and selected candidates.

        Arguments:
            candidates: A list of SentenceCandidates instances.

        Returns:
            Predictions: A Predictions instance representing the identified and
                linked toponyms.
        """
        if len(candidates) == 0:
            return Predictions(list())
        # Replace each CandidatesLinks instance with a PredictedLinks instance.
        for scs in candidates:
            for cs in scs.candidates:
                for i, links in enumerate(cs.links):
                    scores = self.disambiguation_scores(links.wikidata_links, links.string_match.string_similarity)
                    cs.links[i] = links.attach_scores(scores)
        return Predictions(candidates)

    def disambiguation_scores(self, links: List[WikidataLink], string_similarity: float) -> Dict[str, float]:
        """
        Compute disambiguation scores for a given list Wikidata links.

        Arguments:
            links: A list of WikidataLink instances.
            string_similarity: (Optional) the string similarity score for the candidate match.

        Returns:
            dict: A dictionary containing disambiguation scores, keyed by Wikidata ID.
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
    # Override the method_name class attribute.
    method_name: str = "mostpopular"

    # TODO: docstring
    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        links = [MostPopularLink(
            wqid=wqid,
            wkdt_class=self.wkdt_class(wqid),
            freq=self.resources["mentions_to_wikidata"][match.variation][wqid]) 
            for wqid in match.wqid_links]
        return links

    # Computes disambiguation scores for a collection of potential Wikidata links.
    def disambiguation_scores(self, links: List[MostPopularLink], string_similarity=None) -> Dict[str, float]:
        total = sum([m.freq for m in links])
        return {link.wqid: link.freq / total for link in links}

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
    # Override the method_name class attribute.
    method_name: str = "bydistance"

    def load(self):
        """
        Loads the linking resources and assigns them to instance variables.
        """
        super().load()

        print("  > Loading gazetteer.")
        gaz = pd.read_csv(
            os.path.join(self.resources_path, "wikidata/wikidata_gazetteer.csv"),
            usecols=["wikidata_id", "latitude", "longitude"],
        )
        gaz["latitude"] = gaz["latitude"].astype(float)
        gaz["longitude"] = gaz["longitude"].astype(float)
        gaz["coords"] = gaz[["latitude", "longitude"]].to_numpy().tolist()
        wqid_to_coords = dict(zip(gaz.wikidata_id, gaz.coords))
        self.resources["wqid_to_coords"] = wqid_to_coords
        gaz_ids = set(gaz["wikidata_id"].tolist())
        # Keep only wikipedia entities in the gazetteer:
        self.resources["wikidata_locs"] = gaz_ids
        gaz_ids = ""
        gaz = ""

    def wkdt_coords(self, wqid: str) -> Optional[Tuple[float, float]]:
        """Returns the lat-lon coordinates for the given Wikidata entry, if available."""
        return self.resources["wqid_to_coords"].get(wqid, None)

    # TODO: docstring
    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        
        origin_coords = self.wkdt_coords(place_of_pub_wqid)
        links = [ByDistanceLink(
            wqid=wqid,
            wkdt_class=self.wkdt_class(wqid),
            coords=self.wkdt_coords(wqid),
            place_of_pub_coords=origin_coords,
            geodist=self.haversine(origin_coords, self.wkdt_coords(wqid)),
            normalized_score=self.resources["mentions_to_wikidata_normalized"][match.variation][
                wqid
            ]) for wqid in match.wqid_links]
        return links
    
    def haversine(self, origin_coords: Optional[Tuple[float, float]], coords: Optional[Tuple[float, float]]) -> Optional[float]:
        if not origin_coords:
            print("Missing place of publication coordinates.")
            return None
        try:
            return haversine(origin_coords, coords)
        except ValueError:
            return None

    def disambiguation_scores(self, wikidata_links: List[ByDistanceLink], string_similarity: float) -> Dict[str, float]:
        max_on_gb = 1000  # 1000 km, max on GB
        ret = dict()
        for link in wikidata_links:
            
            distance = min(max_on_gb, link.geodist if link.geodist is not None else max_on_gb)
            
            if distance == 0.0:
                distance_score = 1.0
            else:
                distance = (max_on_gb if distance > max_on_gb else distance)
                distance_score = 1.0 - (distance / max_on_gb)

            relv_score = min(1.0, (string_similarity + link.normalized_score) / 2.0)

            final_score = 0.0
            if link.geodist is not None:
                final_score = round((relv_score + distance_score) / 2, 3)
            ret[link.wqid] = final_score
        return ret
    
# TODO: update docstring.
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
    # Override the method_name class attribute.
    method_name: str = "reldisamb"

    # Override the constructor to include REL model parameters.
    def __init__(
        self,
        resources_path: str,
        ranker: ranking.Ranker,
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
        self.ranker = ranker
        self.entity_disambiguation_model = None

    def __str__(self) -> str:
        """
        Returns a string representation of the Linker object.

        Returns:
            str: String representation of the Linker object.
        """
        s = super().__str__()
        s += f"    * Overwrite training: {self.overwrite_training}\n"
        return s

    # Override the load method to load the entity disambiguation model.
    def load(
        self, split: Optional[str] = "originalsplit"
    ):
        """
        Loads the linking resources and assigns them to instance variables.
        """
        super().load()
        self.train_load_model(split=split)

    # TODO: docstring
    # Override the run method to include handling of REL config parameters.
    def run(
            self, 
            matches: CandidateMatches, 
            place_of_pub_wqid: Optional[str]=None,
            place_of_pub: Optional[str]=None,
        ) -> MentionCandidates:
        """
        Execute the linking process. Each Linker subclass must implement a 
        linking method by overriding this function.

        Arguments:
            matches: A CandidatesMatches instance.
            origin_wqid (Optional[str]): The Wikidata ID of the place of publication.

        Returns:
            Candidates: The candidates identified by the linking process.
        """
        # Skip microtoponyms if configured to do so.
        if self.rel_params["without_microtoponyms"]:
            if matches.mention.is_microtoponym():
                return self.empty_candidates(matches.mention, matches.ranking_method, place_of_pub_wqid, place_of_pub)

        # If configured to link "with publication" (i.e. with an additional sentence
        # containing an artificial mention of the place of publication), use default 
        # values for place_of_pub_wqid and place_of_pub unless they are already populated.
        if self.with_publication():
            if not (place_of_pub_wqid and place_of_pub):
                place_of_pub_wqid = self.rel_params["default_publwqid"]
                place_of_pub = self.rel_params["default_publname"]

        return super().run(matches, place_of_pub_wqid, place_of_pub)

    # TODO: docstring        
    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        links = [RelDisambLink(
            wqid=wqid,
            wkdt_class=self.wkdt_class(wqid),
            freq=self.resources["mentions_to_wikidata"][match.variation][wqid],
            normalized_score=self.resources["mentions_to_wikidata_normalized"][match.variation][
                wqid
            ]) for wqid in match.wqid_links]
        return links

    def with_publication(self) -> bool:
        return self.rel_params["with_publication"]

    # Override the disambiguate method to include REL linking.
    def disambiguate(self, candidates: List[SentenceCandidates], apply_rel: bool=True) -> Predictions:

        # Generate interim predictions as inputs to the REL model.
        predictions = super().disambiguate(candidates)

        if not apply_rel:
            return predictions

        if not self.entity_disambiguation_model:
            ValueError("Entity disambiguation model not yet loaded. Call `load` method.")

        # Apply the REL model to the interim predictions.
        rel_predictions = self.entity_disambiguation_model.predict(
            predictions.as_dict(self.rel_params["with_publication"]))
        # Incorporate the REL model predictions.
        return predictions.apply_rel_disambiguation(rel_predictions, self.rel_params["with_publication"])

    # Computes disambiguation scores for a collection of potential Wikidata links.
    # IMP NOTE: this replaces the rank_candidates function from rel_utils.py:
    def disambiguation_scores(self, links: List[RelDisambLink], string_similarity: float) -> Dict[str, float]:

        ret = dict()
        # copied from rank_candidates (with edits):
        max_cand_freq = max([m.freq for m in links])
        for wikidata_link in links:

            # Mention-to-wikidata absolute relevance:
            qcrlv_score = wikidata_link.freq
            qcm2w_score = wikidata_link.normalized_score
            # Average of CS conf score and mention2wiki norm relv:
            if string_similarity:
                qcm2w_score = (qcm2w_score + string_similarity) / 2
            # tmp_cands.append((wqid, qcrlv_score, qcm2w_score))

            # Normalize absolute mention-to-wikidata relevance by entity:
            qc_score_1 = qcrlv_score / max_cand_freq
            # Candidate selection confidence:
            qc_score_2 = qcm2w_score
            # Averaged relevances and normalize between 0 and 0.9:
            score = ((qc_score_1 + qc_score_2) / 2) * 0.9
            # old: score = round(qc_score, 3)

            ret[wikidata_link.wqid] = score

        # TODO: put the above logic in a function and replace with something like this:
        # return {link.wqid: link.freq / total for link in links}
        return ret

    def train_load_model(self, split: Optional[str] = "originalsplit"):
        """
        Trains or loads the entity disambiguation model and assigns to the
        `entity_disambiguation_model` field.

        Arguments:
            split (str, optional): The split type for training. Defaults to
                ``"originalsplit"``.

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
        linker_name = self.ranker.method_name
        if self.ranker.method_name == "deezymatch":
            linker_name += "+" + str(self.ranker.deezy_parameters["num_candidates"])
            linker_name += "+" + str(
                self.ranker.deezy_parameters["selection_threshold"]
            )
        linker_name += f"_{split}"
        if self.rel_params["with_publication"]:
            linker_name += "+wpubl"
        if self.rel_params["without_microtoponyms"]:
            linker_name += "+wmtops"
        if self.rel_params["do_test"]:
            linker_name += "_test"
        linker_name = os.path.join(self.rel_params["model_path"], linker_name)

        if self.overwrite_training == True or not Path(linker_name).is_dir() or len(os.listdir(linker_name)) == 0:
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
                self.ranker,
                self,
                "train",
            )
            dev_json = rel_utils.prepare_rel_trainset(
                dev_df,
                self.rel_params,
                self.ranker,
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

        self.entity_disambiguation_model = model
