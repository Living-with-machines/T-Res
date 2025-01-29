import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from haversine import haversine
from math import exp
from tqdm import tqdm

tqdm.pandas()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from ..utils import rel_utils
from ..utils.REL import entity_disambiguation
from . import ranking
from ..utils.dataclasses import Mention, MentionCandidates, StringMatchLinks, WikidataLink, MostPopularLink, ByDistanceLink, RelDisambLink, CandidateMatches, CandidateLinks, SentenceCandidates, Predictions, RelPredictions

class Linker:
    """
    The Linker class provides methods for entity linking, which is the task of
    associating mentions in text with their corresponding entities in a
    knowledge base.

    Arguments:
        resources_path (str): The path to the linking resources.
        experiments_path (str, optional): The path to the experiments
            directory.
        linking_resources (dict, optional): Dictionary containing the
            necessary linking resources.

    Note:
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

    def __str__(self) -> str:
        """
        Returns a string representation of the Linker object.

        Returns:
            String representation of the Linker object.
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
            A Linker (subclass) instance.

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
        raise ValueError(f"Invalid linking method: {method_name}")

    def wkdt_class(self, wqid: str) -> Optional[str]:
        """
        Returns the Wikidata class for the given Wikidata entry, if available.
        
        Returns:
            The corresponding Wikidata class if available, otherwise `None`.
        """
        return self.resources["entity2class"].get(wqid, None)
    
    def wkdt_coords(self, wqid: str) -> Optional[Tuple[float, float]]:
        """
        Returns the lat-lon coordinates for the given Wikidata entry, if available.
        
        Returns:
            Latitude and longitude coordinates for the given Wikidata entry, if
                available.
        """
        return self.resources["wqid_to_coords"].get(wqid, None)

    def haversine(self, 
                  origin_coords: Optional[Tuple[float, float]], 
                  coords: Optional[Tuple[float, float]]) -> Optional[float]:
        """
        Calculates the great circle distance between two points on Earth's surface.

        Args:
            origin_coords (Optional[Tuple[float, float]]): coordinates of the origin
            coords (Optional[Tuple[float, float]]): coordinates of the other point

        Returns:
            The great circle distance between the points, or `None` if either pair
                of coordinates is unavailable.
        """
        if not origin_coords:
            print("Missing place of publication coordinates.")
            return None
        try:
            return haversine(origin_coords, coords)
        except ValueError:
            # We have one candidate with coordinates in Venus!
            print(f"Failed to compute haversine distance from {origin_coords} to {coords}")
            return None

    def empty_candidates(self, 
                         mention: Mention, 
                         ranking_method: str, 
                         place_of_pub_wqid: str, 
                         place_of_pub: str) -> MentionCandidates:
        """
        Constructs an empty `MentionCandidates` instance.

        Returns:
            A `MentionCandidates` instance with an empty list of candidate links.
        """
        return MentionCandidates(
            mention,
            ranking_method,
            self.method_name,
            list(),
            place_of_pub_wqid,
            place_of_pub)

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

        print("*** Linking resources loaded!\n")

    def run(
            self, 
            matches: CandidateMatches, 
            place_of_pub_wqid: Optional[str]=None,
            place_of_pub: Optional[str]=None,
        ) -> MentionCandidates:
        """
        Executes the linking process. 
        
        Arguments:
            matches: A `CandidatesMatches` instance containing string matches to be linked.
            place_of_pub_wqid (Optional[str]): The Wikidata ID of the place of publication.
            place_of_pub (Optional[str]): The place of publication.

        Returns:
            The candidates identified by the linking process.
        """
        if matches.is_empty():
            return self.empty_candidates(matches.mention, matches.ranking_method, place_of_pub_wqid, place_of_pub)

        candidate_links = [CandidateLinks(m.as_string_match(), self.wikidata_links(m, place_of_pub_wqid)) 
                           for m in matches.matches]

        return MentionCandidates(
            matches.mention, 
            matches.ranking_method, 
            self.method_name, 
            candidate_links,
            place_of_pub_wqid,
            place_of_pub,
        )
    
    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        """
        Identifies candidate links in the Wikidata knowledgebase.

        Args:
            match (StringMatchLinks): The toponym string match to be linked.
            place_of_pub_wqid (Optional[str], optional): The Wikidata ID of
                the place of publication, if available.

        Raises:
            NotImplementedError: If not implemented in a subclass.

        Returns:
            A list of candidate links in Wikidata.

        Note: 
            Each Linker subclass must implement a linking algorithm by 
                overriding the `wikidata_links` method.
        """
        raise NotImplementedError("Subclass implementation required.")

    def disambiguate(self, candidates: List[SentenceCandidates]) -> Predictions:
        """
        Performs entity disambiguation given a list of already identified
        toponyms and selected candidates.

        Arguments:
            candidates: A list of SentenceCandidates instances.

        Returns:
            A `Predictions` instance representing the identified and
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
        Computes disambiguation scores for a given list of Wikidata links.

        Arguments:
            links: A list of `WikidataLink` instances.
            string_similarity (float): the string similarity score for the candidate match.

        Raises:
            NotImplementedError: If not implemented in a subclass.

        Returns:
            A dictionary containing disambiguation scores, keyed by Wikidata ID.

        Note: 
            Each Linker subclass must implement a linking algorithm by 
                overriding the `disambiguation_scores` method.
        """
        raise NotImplementedError("Subclass implementation required.")
    
class MostPopularLinker(Linker):
    """
    An entity linking method that selects the candidate that is most
    popular in the Wikipedia knowledgebase.

    Example: 
        ```python
        linker = MostPopularLinker(
            resources_path="/path/to/resources/",
            experiments_path="/path/to/experiments/",
            linking_resources={},
        )
        ```

    """
    # Override the method_name class attribute.
    method_name: str = "mostpopular"

    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        """
        Identifies candidate links in the Wikidata knowledgebase.

        Args:
            match (StringMatchLinks): The toponym string match to be linked.
            place_of_pub_wqid (Optional[str], optional): The Wikidata ID of
                the place of publication, if available. **Not used** in this 
                linking method.

        Returns:
            A list of candidate links in Wikidata, each of type 
                [`MostPopularLink`][t_res.utils.dataclasses.MostPopularLink].
        """
        links = [MostPopularLink(
            wqid=wqid,
            wkdt_class=self.wkdt_class(wqid),
            coords=self.wkdt_coords(wqid),
            freq=self.resources["mentions_to_wikidata"][match.variation][wqid]) 
            for wqid in match.wqid_links]
        return links

    def disambiguation_scores(self, links: List[MostPopularLink], string_similarity=None) -> Dict[str, float]:
        """
        Computes disambiguation scores by using the relative mention-to-wikidata 
        link frequencies as a proxy for popularity of the toponym in Wikidata.

        Arguments:
            links: A list of `WikidataLink` instances.
            string_similarity (float): the string similarity score for the candidate match.

        Returns:
            A dictionary containing disambiguation scores, keyed by Wikidata ID.
        """
        total = sum([m.freq for m in links])
        return {link.wqid: link.freq / total for link in links}

class ByDistanceLinker(Linker):
    """
    An entity linking method that selects the candidate based on its
    proximity to the place of publication.

    Example:
        ```python
        linker = ByDistanceLinker(
            resources_path="/path/to/resources/",
            experiments_path="/path/to/experiments/",
            linking_resources={},
        )
        ```
    """
    # Override the method_name class attribute.
    method_name: str = "bydistance"

    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        """
        Identifies candidate links in the Wikidata knowledgebase.

        Args:
            match (StringMatchLinks): The toponym string match to be linked.
            place_of_pub_wqid (Optional[str], optional): The Wikidata ID of
                the place of publication, if available.

        Returns:
            A list of candidate links in Wikidata, each of type 
                [`ByDistanceLink`][t_res.utils.dataclasses.ByDistanceLink].
        """
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
    
    def disambiguation_scores(self, 
                              wikidata_links: List[ByDistanceLink], 
                              string_similarity: float) -> Dict[str, float]:
        """
        Computes disambiguation scores based on the physical proximity of the candidate
        to the place of publication of the source text, also taking into account the 
        string similarity of the match and the relative popularity of the Wikidata entry.

        Arguments:
            links: A list of `WikidataLink` instances.
            string_similarity (float): the string similarity score for the candidate match.

        Returns:
            A dictionary containing disambiguation scores, keyed by Wikidata ID.
        """
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
            final_score = round((relv_score + distance_score) / 2, 3) if link.geodist is not None else 0.0

            ret[link.wqid] = final_score
        return ret
    
class RelDisambLinker(MostPopularLinker):
    """
    An entity linking method that selects the candidate using the [Radboud 
    Entity Linker](https://github.com/informagi/REL/) (REL) model.

    This is a subclass of the MostPopularLinker so that the disambiguation
    score based on Wikidata popularity may be used to compute a combined 
    disambiguation score (if configured to do so).

    Arguments:
        resources_path (str): The path to the linking resources.
        ranker (Ranker): A `Ranker` instance.
        experiments_path (str, optional): The path to the experiments
            directory.
        linking_resources (dict): Dictionary containing the
            necessary linking resources.
        overwrite_training (bool): Flag indicating whether to overwrite the
            training.
        rel_params (dict, optional): Dictionary containing the parameters
            for performing entity disambiguation using the ``reldisamb``
            approach (adapted from the Radboud Entityt Linker, REL).
            For the default settings, see Notes below.

    Example:
        ```python
        linker = Linker(
            resources_path="/path/to/resources/",
            ranker=PerfectMatchRanker(resources_path="/path/to/resources/"),
            experiments_path="/path/to/experiments/",
            linking_resources={},
            overwrite_training=True,
            rel_params={"with_publication": True, "do_test": True}
        )
        ```

    Note:
        Note that, in order to instantiate the Linker with the ``reldisamb``
        method, the Linker needs to be wrapped by a context manager in which
        a connection to the entity embeddings database is established and a
        cursor is created:

        ```python
        with sqlite3.connect("../resources/rel_db/embeddings_database.db") as conn:
            cursor = conn.cursor()
            linker = RelDisambLinker(
                resources_path="../resources/",
                ranker=PerfectMatchRanker(resources_path="../resources/"),
                experiments_path="../experiments/",
                linking_resources=dict(),
                overwrite_training=False,
                rel_params={
                    "model_path": "../resources/models/disambiguation/",
                    "data_path": "../experiments/outputs/data/lwm/",
                    "training_split": "",
                    "db_embeddings": cursor,
                    "with_publication": True,
                    "without_microtoponyms": True,
                    "do_test": False,
                    "default_publname": "",
                    "default_publwqid": "",
                },
            )
        ```

        See below the default settings for ``rel_params``. Note that
        `db_embeddings` defaults to None, but it should be assigned a
        cursor to the entity embeddings database, as described above:

        ```python
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
            "reference_separation": ((49.956739, -8.17751), (60.87, 1.762973))
        }
        ```
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

        # Default linking parameters:
        params = {
            "model_path": os.path.join(resources_path, "models/disambiguation/"),
            "data_path": os.path.join(experiments_path, "outputs/data/lwm/"),
            "training_split": "originalsplit",
            "db_embeddings": None,  # The cursor to the embeddings database.
            "with_publication": True,
            "predict_place_of_publication": True,
            "combined_score": True,
            "without_microtoponyms": True,
            "do_test": False,
            "default_publname": "United Kingdom",
            "default_publwqid": "Q145",
            "reference_separation": ((49.956739, -8.17751), (60.87, 1.762973)),
        }
        if not rel_params is None:
            if not set(rel_params) <= set(params):
                raise ValueError("Invalid REL config parameters.")
            # Update the default parameters with any given parameters.
            params.update(rel_params)

        self.rel_params = params
        self.ranker = ranker
        self.entity_disambiguation_model = None

        reference_separation = self.rel_params['reference_separation']
        self.reference_distance  = self.haversine(reference_separation[0], reference_separation[1])

    def __str__(self) -> str:
        """
        Returns a string representation of the Linker object.

        Returns:
            String representation of the Linker object.
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

    # Override the run method to include handling of REL config parameters.
    def run(
            self, 
            matches: CandidateMatches, 
            place_of_pub_wqid: Optional[str]=None,
            place_of_pub: Optional[str]=None,
        ) -> MentionCandidates:
        """
        Executes the linking process. 
        
        Arguments:
            matches: A `CandidatesMatches` instance containing string matches to be linked.
            place_of_pub_wqid (Optional[str]): The Wikidata ID of the place of publication.
            place_of_pub (Optional[str]): The place of publication.

        Returns:
            The candidates identified by the linking process.
        """
        # If configured to link "with publication" (i.e. with an additional sentence
        # containing an artificial mention of the place of publication), use default 
        # values for place_of_pub_wqid and place_of_pub unless they are already populated.
        if self.rel_params["with_publication"]:
            if not (place_of_pub_wqid and place_of_pub):
                place_of_pub_wqid = self.rel_params["default_publwqid"]
                place_of_pub = self.rel_params["default_publname"]

        # Skip microtoponyms if configured to do so.
        if self.rel_params["without_microtoponyms"]:
            if matches.mention.is_microtoponym():
                return self.empty_candidates(matches.mention, matches.ranking_method, place_of_pub_wqid, place_of_pub)

        return super().run(matches, place_of_pub_wqid, place_of_pub)

    def wikidata_links(
            self, 
            match: StringMatchLinks,
            place_of_pub_wqid: Optional[str]=None,
            ) -> List[WikidataLink]:
        """
        Identifies candidate links in the Wikidata knowledgebase.

        Args:
            match (StringMatchLinks): The toponym string match to be linked.
            place_of_pub_wqid (Optional[str], optional): The Wikidata ID of
                the place of publication, if available.

        Returns:
            A list of candidate links in Wikidata, each of type 
                [`RelDisambLink`][t_res.utils.dataclasses.RelDisambLink].
        """
        links = [RelDisambLink(
            wqid=wqid,
            wkdt_class=self.wkdt_class(wqid),
            coords=self.wkdt_coords(wqid),
            freq=self.resources["mentions_to_wikidata"][match.variation][wqid],
            normalized_score=self.resources["mentions_to_wikidata_normalized"][match.variation][
                wqid
            ]) for wqid in match.wqid_links]
        return links

    # Override the disambiguate method to include REL linking.
    def disambiguate(self, 
                     candidates: List[SentenceCandidates], 
                     apply_rel: bool=True) -> Predictions:
        """
        Performs entity disambiguation given a list of already identified
        toponyms and selected candidates. This method overrides the base
        class implementation to include REL model linking.

        Arguments:
            candidates: A list of SentenceCandidates instances.

        Returns:
            A `Predictions` instance representing the identified and
                linked toponyms.
        """
        # Generate prior predictions as inputs to the REL model.
        predictions = super().disambiguate(candidates)

        # Remove any microtoponyms from the predictions, if configured to do so.
        if self.rel_params["without_microtoponyms"]:
            micro_candidates = [sc for sc in predictions.sentence_candidates for c in sc.candidates if c.mention.is_microtoponym()]
            for sc in micro_candidates:
                sc.remove_microtoponyms()

        if not apply_rel:
            return predictions

        if not self.entity_disambiguation_model:
            ValueError("Entity disambiguation model not yet loaded. Call `load` method.")

        # Apply the REL model to the interim predictions.
        rel_predictions_dict = self.entity_disambiguation_model.predict(
            predictions.as_dict(self.rel_params["with_publication"]))

        # Incorporate the REL model predictions.
        rel_predictions = predictions.apply_rel_disambiguation(rel_predictions_dict, self.rel_params["with_publication"])
    
        # Take into account the `predict_place_of_pub` config parameter.
        if self.rel_params['predict_place_of_publication']:
            self.predict_place_of_publication(rel_predictions)

        # Take into account the `combined_score` config parameter.
        if self.rel_params['combined_score']:
            self.apply_combined_score(rel_predictions)

        return rel_predictions

    # Computes disambiguation scores for a collection of potential Wikidata links.
    # (Note: this replaces the rank_candidates function from rel_utils.py)
    def disambiguation_scores(self, 
                              links: List[RelDisambLink], 
                              string_similarity: float) -> Dict[str, float]:
        """
        Computes *interim* disambiguation scores (i.e. before applying the REL model) 
        by taking into account the string similarity of the match and the relative 
        popularity of the Wikidata entry.

        Arguments:
            links: A list of `WikidataLink` instances.
            string_similarity (float): the string similarity score for the candidate match.

        Returns:
            A dictionary containing disambiguation scores, keyed by Wikidata ID.
        """
        ret = dict()
        if not links:
            return ret
        max_cand_freq = max([m.freq for m in links])
        for wikidata_link in links:

            # Normalize absolute mention-to-Wikidata relevance by entity:
            candidate_score_1 = wikidata_link.freq / max_cand_freq
            # Average of string similarity and mention-to-Wikidata normalized relevance:
            candidate_score_2 = (wikidata_link.normalized_score + string_similarity) / 2

            # Average of two candidate scores, normalized between 0 and 0.9:
            score = ((candidate_score_1 + candidate_score_2) / 2) * 0.9
            ret[wikidata_link.wqid] = score

        return ret

    def predict_place_of_publication(self, rel_predictions: RelPredictions):
        """
        Sets the disambiguation scores for the place of publication to 1.0 inside the given 
        REL predictions, provided the place of publication is known and exists as a candidate link.

        Arguments:
            rel_predictions: An instance of the `RelPredictions` dataclass.
        """
        place_of_pub_wqid = rel_predictions.place_of_pub_wqid()
        if not place_of_pub_wqid:
            return
        for rs in rel_predictions.rel_scores:
            # If the place of publication is not in the list of scored candidates, do nothing.
            if not place_of_pub_wqid in rs.scores.keys():
                return
            rs.scores[place_of_pub_wqid] = 1.0

    def apply_combined_score(self, rel_predictions: RelPredictions):
        """
        Updates all disambiguation scores in the given REL predictions by 
        combining the REL score with place of publication information, if known.

        Arguments:
            rel_predictions: An instance of the `RelPredictions` dataclass.
        """
        place_of_pub_wqid = rel_predictions.place_of_pub_wqid()
        if not place_of_pub_wqid:
            return
        
        def combined_score(rel_score, popularity, proximity):
            if not proximity:
                return rel_score
            return rel_score * max(popularity, proximity)

        # Iterate over the mention candidates and their corresponding REL scores.
        for mc, rs in zip(rel_predictions.candidates(ignore_empty_candidates=False), rel_predictions.rel_scores):
            # Iterate over the predicted Wikidata links.
            for cl in mc.links:
                # Compute popularity and proximity scores for all Wikidata links.
                wqids = [wl.wqid for wl in cl.wikidata_links]
                # Use the MostPopularLinker superclass to compute popularity.
                popularity = super().disambiguation_scores(cl.wikidata_links)
                proximity = {wqid: self.proximity(
                    origin_coords=self.wkdt_coords(place_of_pub_wqid),
                    coords=self.wkdt_coords(wqid)) for wqid in wqids}
                combined = {wqid: combined_score(rs.scores[wqid], popularity[wqid], proximity[wqid]) for wqid in wqids}
                # Update the REL scores.
                rs.scores.update(combined)

    def proximity(self, 
                  origin_coords: Optional[Tuple[float, float]], 
                  coords: Optional[Tuple[float, float]]) -> Optional[float]:
        """Computes the proximity measure between pairs of lat-long coordinates.

        Args:
            origin_coords (Optional[Tuple[float, float]]): _description_
            coords (Optional[Tuple[float, float]]): _description_

        Returns:
            Optional[float]: _description_
        """
        if not coords:
            return None
        distance = self.haversine(origin_coords, coords)
        # Handle caught error in the haversine method.
        if not distance:
            return None
        return exp(-(distance/self.reference_distance)**2)

    def train_load_model(self, split: Optional[str] = "originalsplit"):
        """
        Trains or loads the entity disambiguation model and assigns to the
        `entity_disambiguation_model` field.

        Arguments:
            split (str, optional): The split type for training.

        Note:
            The training will be skipped if the model already exists and
            ``overwrite_training`` was set to False when initiating the Linker
            object, or if the disambiguation method is unsupervised. The
            training will be run on test mode if ``rel_params`` had a
            ``do_test`` key's value set to True when initiating the Linker
            object.

        Note: Credit:
            This class and its methods are adapted from the [REL: Radboud Entity
            Linker](https://github.com/informagi/REL/) Github repository:
            Copyright (c) 2020 Johannes Michael van Hulst. See the [permission
            notice](https://github.com/informagi/REL/blob/main/LICENSE).

            ```
            Reference:

            @inproceedings{vanHulst:2020:REL,
                author =    {van Hulst, Johannes M. and Hasibi, Faegheh and Dercksen, Koen and Balog, Krisztian and de Vries, Arjen P.},
                title =     {REL: An Entity Linker Standing on the Shoulders of Giants},
                booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
                series =    {SIGIR '20},
                year =      {2020},
                publisher = {ACM}
            }
            ```
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
