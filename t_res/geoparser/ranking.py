import json
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
from DeezyMatch import candidate_ranker
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

from ..utils import deezy_processing
from ..utils.dataclasses import StringMatch, StringMatchLinks, CandidateMatches, Mention

class Ranker:
    """
    The Ranker class implements a system for candidate selection through string
    variation ranking. Its subclasses provide methods to select candidates based 
    on different matching approaches, such as perfect match, partial match, 
    Levenshtein distance, and DeezyMatch. The base class handles loading and 
    processing of resources related to candidate selection.

    Arguments:
        resources_path (str): Relative path to the resources directory
            (containing Wikidata resources).
        mentions_to_wikidata (dict, optional): An empty dictionary which
            will store the mapping between mentions and Wikidata IDs,
            which will be loaded through the Ranker's 
            [load method][t_res.geoparser.ranking.Ranker.load].
        wikidata_to_mentions (dict, optional): An empty dictionary which
            will store the mapping between Wikidata IDs and mentions,
            which will be loaded through the Ranker's 
            [load method][t_res.geoparser.ranking.Ranker.load].

    Note:
        This base class should not be instatiated directly. Instead use a subclass
            constructor.

    Example:
        ```python
        # Create a Ranker object:
        ranker = PerfectMatchRanker(resources_path="/path/to/resources/")
        # Load resources
        ranker.load()
        # Perform candidate selection
        queries = ['London', 'Paraguay']
        results = [ranker.matches(query) for query in queries]
        # Print the results
        print("Candidate Selection Results:")
        for matches in results:
            print(matches)
        ```
    """
    # Class attribute for the name of the ranking method.
    method_name: str = None

    # TODO: move wikidata_to_mentions arg to the DeezyMatchRanker __init__ only.
    def __init__(
        self,
        resources_path: str,
        mentions_to_wikidata: Optional[dict] = dict(),
        wikidata_to_mentions: Optional[dict] = dict(),
    ):
        """
        Initialize a Ranker object.
        """
        self.resources_path = resources_path
        self.mentions_to_wikidata = mentions_to_wikidata
        self.wikidata_to_mentions = wikidata_to_mentions
        self.cache = dict()

    def __str__(self) -> str:
        """
        Returns a string representation of the Ranker object, including the method name.
        """
        s = ">>> Candidate selection:\n"
        s += f"    * Method: {self.method_name}\n"
        return s
    
    def new(**kwargs) -> 'Ranker':
        """
        Static constructor.

        Args:
            kwargs (dict): A dictionary of keyword arguments matching the
                arguments to a subclass __init__ constructor, plus a 
                `method_name` argument to specify the desired subclass.

        Returns:
            Ranker: A Ranker subclass instance.

        """
        if not 'method_name' in kwargs.keys():
            raise ValueError("Expected `method_name` keyword argument.")
        method_name = kwargs['method_name']
        del kwargs['method_name']
        if method_name == 'perfectmatch':
            return PerfectMatchRanker(**kwargs)
        if method_name == 'partialmatch':
            return PartialMatchRanker(**kwargs)
        if method_name == 'levenshtein':
            return LevenshteinRanker(**kwargs)
        if method_name == 'deezymatch':
            return DeezyMatchRanker(**kwargs)
        raise ValueError(f"Invalid ranking method: {method_name}")

    def load(self):
        """
        Loads the ranker resources.

        Note:
            This method loads the mentions-to-wikidata and
            wikidata-to-mentions dictionaries from the resources directory,
            specified when initialising the [Ranker][t_res.geoparser.ranking.Ranker]. 
            They are required for performing candidate selection and ranking.

            The loaded mentions-to-wikidata dictionary maps a toponym 
            (e.g. ``"London"``) to the Wikidata entities that are
            referred to by this toponym on Wikipedia (e.g. ``Q84``,
            ``Q2477346``). The data also includes, for each entity, its
            normalized "relevance", i.e. number of in-links across Wikipedia.            

            The loaded dictionaries are filtered to remove noise and the class
            attributes are updated accordingly.
        """
 
        print("*** Loading the ranker resources.")
        # NOTE: these are the *normalized* mentions, *not* relative frequencies.
        files = {
            "mentions_to_wikidata": os.path.join(
                self.resources_path, "wikidata/mentions_to_wikidata_normalized.json"
            ),
            "wikidata_to_mentions": os.path.join(
                self.resources_path, "wikidata/wikidata_to_mentions_normalized.json"
            ),
        }

        with open(files["mentions_to_wikidata"], "r") as f:
            self.mentions_to_wikidata = json.load(f)

        with open(files["wikidata_to_mentions"], "r") as f:
            self.wikidata_to_mentions = json.load(f)

        # Filter mentions to remove noise:
        wikidata_to_mentions_filtered = dict()
        mentions_to_wikidata_filtered = dict()
        for wk in self.wikidata_to_mentions:
            wikipedia_mentions = self.wikidata_to_mentions.get(wk)
            wikipedia_mentions_stripped = dict(
                [
                    (x, wikipedia_mentions[x])
                    for x in wikipedia_mentions
                    if not ", " in x and not " (" in x
                ]
            )

            if wikipedia_mentions_stripped:
                wikipedia_mentions = wikipedia_mentions_stripped

            wikidata_to_mentions_filtered[wk] = dict(
                [(x, wikipedia_mentions[x]) for x in wikipedia_mentions]
            )

            for m in wikidata_to_mentions_filtered[wk]:
                if m in mentions_to_wikidata_filtered:
                    mentions_to_wikidata_filtered[m][
                        wk
                    ] = wikidata_to_mentions_filtered[wk][m]
                else:
                    mentions_to_wikidata_filtered[m] = {
                        wk: wikidata_to_mentions_filtered[wk][m]
                    }

        self.mentions_to_wikidata = mentions_to_wikidata_filtered
        self.wikidata_to_mentions = wikidata_to_mentions_filtered

        del mentions_to_wikidata_filtered
        del wikidata_to_mentions_filtered

    def run(self, mention: Mention) -> CandidateMatches:
        """
        Executes the ranking process for a given toponym mention.

        Arguments:
            mention (Mention): An instance of the Mention dataclass 
                containing a toponym to be matched.

        Returns:
            An instance of the CandidateMatches dataclass 
                containing potential string matches for the given toponym, 
                each with a list of potential Wikidata ID links.

        Note: 
            String matches are added to the cache for efficient retrieval.
        """
        # Use the cache if possible.
        if mention.mention in self.cache:
            return CandidateMatches(mention, self.method_name, self.cache[mention.mention])
        
        # Get the list of candidate string matches for this query.
        string_matches = self.matches(mention.mention)

        # Get the potential Wikidata links for each string match.
        matches = []
        for match in string_matches:
            wqid_links = list(self.mentions_to_wikidata.get(match.variation, dict()).keys())
            matches.append(StringMatchLinks(match.variation, match.string_similarity, wqid_links))

        candidates = CandidateMatches(mention, self.method_name, matches)

        # Update the cache.
        self.cache[mention.mention] = matches
        return candidates

    def matches(self, query: str) -> List[StringMatch]:
        """
        Identifies string matching candidates for the given toponym query.
        
        Each Ranker subclass must implement a ranking method by overriding 
        this function.

        Args:
            query (str): A toponym to be matched.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.

        Returns:
            A list of StringMatch instances containing potential matches for 
                the given toponym.
        """
        raise NotImplementedError("Subclass implementation required.")
    
class PerfectMatchRanker(Ranker):
    """
    A ranking method using perfect string matching.

    Example:
        ```python
        ranker = PerfectMatchRanker(resources_path="/path/to/resources/")
        ranker.load()
        queries = ['London', 'Barcelona', 'Bologna']
        results = [ranker.matches(query) for query in queries]
        # Print the results
        print("Candidate Selection Results:")
        for matches in results:
            print(matches)
        ```
    """
    # Override the method_name class attribute.
    method_name: str = "perfectmatch"

    def matches(self, query: str) -> List[StringMatch]:
        """
        Performs perfect matching between a provided toponym (`query`) and the 
        altnames in the knowledge base.

        Arguments:
            query: A toponym query (string) to be matched.

        Returns:
            A list of StringMatch instances, containing
                potential matches for the given toponym. In the case of 
                perfect string matching, all candidates have string_similarity 
                equal to 1.0.

        Note:
            This method checks if the query has an exact match in the
            mentions_to_wikidata dictionary. If a match is found, it assigns a
            perfect match score of ``1.0`` to the query. Otherwise, an empty
            dictionary is assigned as the list of matches for the query.
        """
        if query in self.mentions_to_wikidata:
            return [StringMatch(query, 1.0)]
        # If no match exists, assign an empty list to matches. 
        return list()
    
class PartialMatchRanker(PerfectMatchRanker):
    """
    A ranking method using partial string matching. 
    
    This class extends PerfectMatchRanker because perfect matches are sought
    before attempting a partial match.

    Example:
        ```python
        # Create a Ranker object:
        ranker = PartialMatchRanker(resources_path="/path/to/resources/")
        # Load resources
        ranker.load()
        # Perform candidate selection
        queries = ['London', 'Paraguay']
        results = [ranker.matches(query) for query in queries]
        # Print the results
        print("Candidate Selection Results:")
        for matches in results:
            print(matches)
        ```
    """
    # Override the method_name class attribute.
    method_name: str = "partialmatch"
    
    # Override the load method to initialise ``pandarellel`` for parallization.
    def load(self):
        super().load()

        pandarallel.initialize(nb_workers=10)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def matches(self, query: str) -> List[StringMatch]:
        """
        Performs partial string matching for a given toponym query.

        Arguments:
            query (str): A toponym to be matched.

        Returns:
            A list of StringMatch instances, containing potential matches for 
                the given toponym.

        Note:
            This method identifies candidates via partial string matching. 
            If a perfect match exists, partial matching is skipped.
        """
        # First attempt a perfect string match.
        candidates = super().matches(query)
        if candidates:
            return candidates
        
        # Seek partial string matches.
        mention_df = pd.DataFrame({"mentions": self.mentions_to_wikidata.keys()})
        mention_df["score"] = mention_df.parallel_apply(
            lambda row: self.matching_score(query, row), axis=1
        )
        mention_df = mention_df.dropna()
        mention_df = mention_df.query('score == score.max()')
        cands_dict = mention_df.set_index("mentions").to_dict()["score"]
        matches = [StringMatch(k, v) for (k, v) in cands_dict.items()]
        return matches

    def matching_score(self, query: str, row: pd.Series) -> float:
        """
        Calculate the partial string matching score as the amount of overlap, 
        if a toponym is contained within a row in the dataset.

        Arguments:
            query (str): A toponym identified in a text.
            row (Series): A pandas Series representing a row in the dataset
                with a "mentions" column, corresponding to a mention in the
                knowledge base.

        Returns:
            The match score indicating the degree of containment,
                ranging from ``0.0`` to ``1.0`` (perfect match).

        Example:
            ```python
            ranker = PartialMatchRanker(...)
            query = 'apple'
            row = pd.Series({'mentions': 'Delicious apple'})
            match_score = ranker.matching_score(query, row)
            print(match_score)
            > 0.3333333333333333
            ```
        """
        # Fix strings
        s1 = query.lower()
        s2 = row["mentions"].lower()

        # E.g. query is 'Dorset' and candidate mention is 'County of Dorset'
        if s1 in s2:
            return len(query) / len(row["mentions"])

        # E.g. query is 'County of Dorset' and candidate mention is 'Dorset'
        if s2 in s1:
            return len(row["mentions"]) / len(query)


class LevenshteinRanker(PartialMatchRanker):
    """
    A ranking method based on partial string matching via the Levenshtein distance.

    This class extends PerfectMatchRanker because perfect matches are sought
    before attempting a partial match.

    Example:
        ```python
        # Create a Ranker object:
        ranker = LevenshteinRanker(resources_path="/path/to/resources/")
        # Load resources
        ranker.load()
        # Perform candidate selection
        queries = ['London', 'Paraguay']
        results = [ranker.matches(query) for query in queries]
        # Print the results
        print("Candidate Selection Results:")
        for matches in results:
            print(matches)
        ```
    """
    # Override the method_name class attribute.
    method_name: str = "levenshtein"

    def matching_score(self, query: str, row: pd.Series) -> float:
        """
        Calculates the partial string matching score as the Damerau-Levenshtein 
        distance between a toponym and a row in the dataset.

        Arguments:
            query (str): A toponym identified in a text.
            row (Series): A pandas Series representing a row in the dataset
                with a "mentions" column, corresponding to an alternate name
                of an entity in the knowledge base.

        Returns:
            The similarity score between the query and the row, ranging
                from ``0.0`` to ``1.0``.

        Note:
            This method computes the Damerau-Levenshtein distance between the
            lowercase versions of a query and the "mentions" column value in
            the given row. The distance is then normalized to a similarity score 
            by subtracting it from ``1.0``.

        Example:
            ```python
            ranker = LevenshteinRanker(...)
            query = 'apple'
            row = pd.Series({'mentions': 'orange'})
            similarity = ranker.matching_score(query, row)
            print(similarity)
            > 0.1666666865348816
            ```
        """
        return 1.0 - normalized_damerau_levenshtein_distance(
            query.lower(), row["mentions"].lower()
        )

class DeezyMatchRanker(PerfectMatchRanker):
    """
    A ranking method using DeezyMatch (a deep neural network approach to 
    fuzzy string matching).

    This class extends PerfectMatchRanker because perfect matches are sought
    before attempting a fuzzy string match.

    Arguments:
        resources_path (str): Relative path to the resources directory
            (containing Wikidata resources).
        mentions_to_wikidata (dict, optional): An empty dictionary which
            will store the mapping between mentions and Wikidata IDs,
            which will be loaded through the Ranker's 
            [load method][t_res.geoparser.ranking.Ranker.load].
        wikidata_to_mentions (dict, optional): An empty dictionary which
            will store the mapping between Wikidata IDs and mentions,
            which will be loaded through Ranker's 
            [load method][t_res.geoparser.ranking.Ranker.load].
        strvar_parameters (dict, optional): Dictionary of string variation
            parameters required to create a DeezyMatch training dataset.
            For the default settings, see Notes below.
        deezy_parameters (dict, optional): Dictionary of DeezyMatch parameters
            for model training. For the default settings, see Notes below.

    Example:
        ```python
        ranker = DeezyMatchRanker(resources_path="/path/to/resources/")
        ranker.load()
        queries = ['London', 'Shefrield']
        results = [ranker.matches(query) for query in queries]
        # Print the results
        print("Candidate Selection Results:")
        for matches in results:
            print(matches)
        ```

    Note:
        - The default settings for ``strvar_parameters``:
            ```python
            strvar_parameters: Optional[dict] = {
                # Parameters to create the string pair dataset:
                "ocr_threshold": 60,
                "top_threshold": 85,
                "min_len": 5,
                "max_len": 15,
                "w2v_ocr_path": str(Path("resources/models/w2v/").resolve()),
                "w2v_ocr_model": "w2v_*_news",
                "overwrite_dataset": False,
            }
            ```

        - The default settings for ``deezy_parameters``:
            ```python
            deezy_parameters: Optional[dict] = {
                "dm_path": str(Path("resources/deezymatch/").resolve()),
                "dm_cands": "wkdtalts",
                "dm_model": "w2v_ocr",
                "dm_output": "deezymatch_on_the_fly",
                "ranking_metric": "faiss",
                "selection_threshold": 50,
                "num_candidates": 1,
                "verbose": False,
                "overwrite_training": False,
                "do_test": False,
            }
            ```
    """
    # Override the method_name class attribute.
    method_name: str = "deezymatch"
    
    # Override the constructor to include DeezyMatch model parameters.
    def __init__(
        self,
        resources_path: str,
        mentions_to_wikidata: Optional[dict] = dict(),
        wikidata_to_mentions: Optional[dict] = dict(),
        strvar_parameters: Optional[dict] = None,
        deezy_parameters: Optional[dict] = None,
    ):
        super().__init__(resources_path, mentions_to_wikidata, wikidata_to_mentions)

        # set paths based on resources path
        if strvar_parameters is None:
            strvar_parameters = {
                # Parameters to create the string pair dataset:
                "ocr_threshold": 60,
                "top_threshold": 85,
                "min_len": 5,
                "max_len": 15,
                "w2v_ocr_path": os.path.join(resources_path, "models/w2v/"),
                "w2v_ocr_model": "w2v_*_news",
                "overwrite_dataset": False,
            }

        if deezy_parameters is None:
            deezy_parameters = {
                # Paths and filenames of DeezyMatch models and data:
                "dm_path": os.path.join(resources_path, "deezymatch/"),
                "dm_cands": "wkdtalts",
                "dm_model": "w2v_ocr",
                "dm_output": "deezymatch_on_the_fly",
                # Ranking measures:
                "ranking_metric": "faiss",
                "selection_threshold": 50,
                "num_candidates": 1,
                "verbose": True,
                # DeezyMatch training:
                "overwrite_training": False,
                "do_test": False,
            }

        self.strvar_parameters = strvar_parameters
        self.deezy_parameters = deezy_parameters

    def __str__(self) -> str:
        """
        Returns a string representation of the Ranker object, including the 
        method name and DeezyMatch training parameters.

        Returns:
            A string representation of the Ranker object.
        """
        s = super().__str__()
        s += "    * DeezyMatch details:\n"
        s += f"      * Model: {self.deezy_parameters['dm_model']}\n"
        s += f"      * Ranking metric: {self.deezy_parameters['ranking_metric']}\n"
        s += f"      * Selection threshold: {self.deezy_parameters['selection_threshold']}\n"
        s += f"      * Num candidates: {self.deezy_parameters['num_candidates']}\n"
        s += f"      * Overwrite training: {self.deezy_parameters['overwrite_training']}\n"
        s += f"      * Overwrite dataset: {self.strvar_parameters['overwrite_dataset']}\n"
        s += f"      * Test mode: {self.deezy_parameters['do_test']}\n"
        return s

    # Override the base class implementation to optionally train the model.
    def load(self, train: bool=True):
        super().load()
        if train or self.deezy_parameters["overwrite_training"]:
            self.train()

    def matches(self, query: str) -> List[StringMatch]:
        """
        Performs DeezyMatch ranking on-the-fly for a given toponym query.

        Arguments:
            query (str): A toponym to be matched.

        Returns:
            A list of StringMatch instances containing potential matches for 
                the given toponym.

        Note:
            This method performs DeezyMatch on-the-fly for the given toponym.
            If a perfect match exists, DeezyMatch matching is skipped.
            Otherwise, it uses the DeezyMatch model to generate candidates and
            ranks them based on the specified ranking metric and selection 
            threshold, provided when initialising the ranker.
        """

        dm_path = self.deezy_parameters["dm_path"]
        dm_cands = self.deezy_parameters["dm_cands"]
        dm_model = self.deezy_parameters["dm_model"]
        dm_output = self.deezy_parameters["dm_output"]

        # First attempt a perfect string match.
        candidates = super().matches(query)
        if candidates:
            return candidates
        
        # Seek fuzzy string matches.
        candidate_scenario = os.path.join(
            dm_path, "combined", dm_cands + "_" + dm_model
        )
        pretrained_model_path = os.path.join(
            f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".model"
        )
        pretrained_vocab_path = os.path.join(
            f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".vocab"
        )

        print(f"candidate_scenario: {candidate_scenario}")
        print(f"pretrained_model_path: {pretrained_model_path}")
        print(f"pretrained_vocab_path: {pretrained_vocab_path}")

        deezy_result = candidate_ranker(
            candidate_scenario=candidate_scenario,
            query=query,
            ranking_metric=self.deezy_parameters["ranking_metric"],
            selection_threshold=self.deezy_parameters["selection_threshold"],
            num_candidates=self.deezy_parameters["num_candidates"],
            search_size=self.deezy_parameters["num_candidates"],
            verbose=self.deezy_parameters["verbose"],
            output_path=os.path.join(dm_path, "ranking", dm_output),
            pretrained_model_path=pretrained_model_path,
            pretrained_vocab_path=pretrained_vocab_path,
        )

        if len(deezy_result.index) != 1:
            raise Exception(f"DeezyMatch result contains {len(deezy_result.index)} rows. Expected 1.")
        row = deezy_result.iloc[0]

        # Reverse cosine distance to cosine similarity:
        returned_cands = dict()
        if self.deezy_parameters["ranking_metric"] == "faiss":
            returned_cands = row["faiss_distance"]
            returned_cands = {
                k: (
                    self.deezy_parameters["selection_threshold"]
                    - returned_cands[k]
                )
                / self.deezy_parameters["selection_threshold"]
                for k in returned_cands
            }
        else:
            returned_cands = row["cosine_dist"]
            returned_cands = {k: 1 - returned_cands[k] for k in returned_cands}

        matches = [StringMatch(k, v) for (k, v) in returned_cands.items()]
        return matches

    def train(self):
        """
        Trains a DeezyMatch model. The training will be skipped if the model
        already exists and the ``overwrite_training`` key in the
        ``deezy_parameters`` passed when initialising the
        [Ranker][t_res.geoparser.ranking.Ranker] object is set to ``False``. The
        training will be run on test mode if the ``do_test`` key in the
        ``deezy_parameters`` passed when initialising the
        [Ranker][t_res.geoparser.ranking.Ranker] object is set to ``True``.
        """

        Path(self.deezy_parameters["dm_path"]).mkdir(parents=True, exist_ok=True)
        if self.deezy_parameters["do_test"] == True:
            if self.deezy_parameters["dm_model"][-5:] != "_test":
                self.deezy_parameters["dm_model"] += "_test"
            if self.deezy_parameters["dm_cands"][-5:] != "_test":
                self.deezy_parameters["dm_cands"] += "_test"
        deezy_processing.train_deezy_model(
            self.deezy_parameters, self.strvar_parameters, self.wikidata_to_mentions
        )
        deezy_processing.generate_candidates(
            self.deezy_parameters, self.mentions_to_wikidata
        )

        # This dictionary is not used anymore:
        self.wikidata_to_mentions = dict()
