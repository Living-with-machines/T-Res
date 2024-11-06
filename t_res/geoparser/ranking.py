import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pandas as pd
from DeezyMatch import candidate_ranker
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

from ..utils import deezy_processing

# TODO: fix docstring.
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
            which will be loaded through the
            :py:meth:`~geoparser.ranking.Ranker.load_resources` method.
        wikidata_to_mentions (dict, optional): An empty dictionary which
            will store the mapping between Wikidata IDs and mentions,
            which will be loaded through the
            :py:meth:`~geoparser.ranking.Ranker.load_resources` method.
        already_collected_cands (dict, optional): Dictionary of already
            collected candidates. Defaults to ``dict()`` (an empty dictionary).

    This base class should not be instatiated directly. Instead use a subclass
    constructor.

    Example:
        >>> # Create a Ranker object:
        >>> ranker = PerfectMatchRanker(
                resources_path="/path/to/resources/",
            )

        >>> # Load resources
        >>> ranker.mentions_to_wikidata = ranker.load_resources()

        >>> # Perform candidate selection
        >>> queries = ['London', 'Paraguay']
        >>> candidates = ranker.run(queries)

        >>> # Print the results
        >>> print("Candidate Selection Results:")
        >>> print(candidates)
        >>> print(ranker.already_collected_cands)

        >>> # Find candidates for mentions
        >>> mentions = [{'mention': 'London'}, {'mention': 'Paraguay'}]
        >>> mention_candidates = ranker.find_candidates(mentions)

        >>> # Print the results
        >>> print("Find Candidates Results:")
        >>> print(mention_candidates)
        >>> print(ranker.already_collected_cands)
    """

    def __init__(
        self,
        resources_path: str,
        mentions_to_wikidata: Optional[dict] = dict(),
        wikidata_to_mentions: Optional[dict] = dict(),
        already_collected_cands: Optional[dict] = dict(),
    ):
        """
        Initialize a Ranker object.
        """
        self.resources_path = resources_path
        self.mentions_to_wikidata = mentions_to_wikidata
        self.wikidata_to_mentions = wikidata_to_mentions
        self.already_collected_cands = already_collected_cands

    def method_name(self) -> str:
        raise NotImplementedError("Subclass implementation required.")

    def __str__(self) -> str:
        """
        Returns a string representation of the Ranker object, including the method name.
        """
        s = ">>> Candidate selection:\n"
        s += f"    * Method: {self.method_name()}\n"
        return s

    def load_resources(self):
        """
        Load the ranker resources.

        Returns:
            dict:
                The loaded mentions-to-wikidata dictionary, which maps a
                mention (e.g. ``"London"``) to the Wikidata entities that are
                referred to by this mention on Wikipedia (e.g. ``Q84``,
                ``Q2477346``). The data also includes, for each entity, their
                normalized "relevance", i.e. number of in-links across Wikipedia.

        Note:
            This method loads the mentions-to-wikidata and
            wikidata-to-mentions dictionaries from the resources directory,
            specified when initialising the
            :py:meth:`~geoparser.ranking.Ranker`. They are required for
            performing candidate selection and ranking.

            It filters the dictionaries to remove noise and updates the class
            attributes accordingly.

            The method also initialises ``pandarallel`` if needed by the
            candidate ranking method (if the ``method`` set in the initialiser
            of the ``Ranker`` was set to "partialmatch" or "levenshtein").
        """
        print("*** Loading the ranker resources.")

        # Load files
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

        # Parallelize if ranking method is one of the following:
        if self.method_name() in ["partialmatch", "levenshtein"]:
            pandarallel.initialize(nb_workers=10)
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def run(self, queries: List[str]) -> dict:
        """
        Execute the ranking process. Each Ranker subclass must implement a 
        ranking method by overriding this function.

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.

        Returns:
            dict: A dictionary whose content will vary depending on the 
                particular ranking method.
        """
        raise NotImplementedError("Subclass implementation required.")

    def find_candidates(self, mentions: List[dict]) -> dict:
        """
        Find candidates for the given mentions using the selected ranking
        method.

        Arguments:
            mentions (list): A list of predicted mentions as dictionaries.

        Returns:
            dict: A dictionary mapping each original mention to a
               sub-dictionary, where the sub-dictionary maps the mention
               variations to a sub-sub-dictionary with two keys: ``"Score"``
               (the string matching similarity score) and ``"Candidates"``
               (a dictionary containing the Wikidata candidates, where the
               key is the Wikidata ID and value is the the relative mention-
               to-wikidata frequency).

               The variation is found by the candidate ranker in the knowledge
               base, and for each variation, the candidate ranking score and
               the candidates from Wikidata are provided. E.g. for mention
               "Guadaloupe" in sentence "sn83030483-1790-03-31-a-i0004_1", the
               candidates will show as follows:

               .. code-block:: json

                  {
                    "Guadaloupe": {
                        "Score": 1.0,
                        "Candidates": {
                            "Q17012": 0.003935458480913026,
                            "Q3153836": 0.07407407407407407
                        }
                    }
                }

        Note:
            This method takes a list of mentions and finds candidates for each
            mention using the selected ranking method. It first extracts the
            queries from the mentions and then calls the appropriate method
            based on the ranking method chosen when initialising the
            :py:meth:`~geoparser.ranking.Ranker` object.

            The method returns a dictionary that maps each original mention to
            a sub-dictionary containing the mention variations as keys and
            their corresponding Wikidata match scores as values.

            Additionally, it updates the already collected candidates
            dictionary (the Ranker object's ``already_collected_cands``
            attribute).
        """
        # Extract the mention
        queries = list(set([mention["mention"] for mention in mentions]))

        # Pass the mentions to :py:meth:`geoparser.ranking.Ranker.run`
        cands = self.run(queries)

        # Get Wikidata candidates
        wk_cands = dict()
        for original_mention in cands:
            wk_cands[original_mention] = dict()
            for variation in cands[original_mention]:
                # If the candidates of the variation of the original mention
                # have already been stored, reuse them:
                stored_value = self.already_collected_cands[original_mention][variation]
                if type(stored_value) == dict:
                    wk_cands[original_mention][variation] = stored_value
                # If the candidates of the variation of the original mention
                # have not yet been found, find them:
                else:
                    match_score = cands[original_mention][variation]
                    # Find Wikidata ID and relv.
                    found_cands = self.mentions_to_wikidata.get(variation, dict())
                    if found_cands and not variation in wk_cands[original_mention]:
                        wk_cands[original_mention][variation] = {
                            "Score": match_score,
                            "Candidates": found_cands,
                        }
                        self.already_collected_cands[original_mention][variation] = {
                            "Score": match_score,
                            "Candidates": found_cands,
                        }

        return wk_cands


# TODO: fix docstring
class PerfectMatchRanker(Ranker):
    """
    A ranking method using perfect string matching.

    Example:
        >>> ranker = PerfectMatchRanker(...)
        >>> ranker.mentions_to_wikidata = ranker.load_resources()
        >>> queries = ['London', 'Barcelona', 'Bologna']
        >>> candidates = ranker.run(queries)
        >>> print(candidates)
        {'London': {'London': 1.0}, 'Barcelona': {'Barcelona': 1.0}, 'Bologna': {'Bologna': 1.0}}
        >>> print(already_collected)
        {'London': {'London': 1.0}, 'Barcelona': {'Barcelona': 1.0}, 'Bologna': {'Bologna': 1.0}}
    """
    def method_name(self) -> str:
        return "perfectmatch"

    def run(self, queries: List[str]) -> dict:
        """
        Perform perfect matching between a provided list of mentions
        (``queries``) and the altnames in the knowledge base.

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.

        Returns:
            dict: A dictionary mapping each mention to its candidate
                list, where the candidate list is a dictionary with the
                mention itself as the key and a perfect match score of
                ``1.0``.

        Note:
            This method checks if each mention has an exact match in the
            mentions_to_wikidata dictionary. If a match is found, it assigns a
            perfect match score of ``1.0`` to the mention. Otherwise, an empty
            dictionary is assigned as the candidate list for the mention.

        Example:
            >>> ranker = PerfectMatchRanker(resources_path="...")
            >>> ranker.mentions_to_wikidata = ranker.load_resources()
            >>> queries = ['London', 'Barcelona', 'Bologna']
            >>> candidates= ranker.run(queries)
            >>> print(candidates)
            {'London': {'London': 1.0}, 'Barcelona': {'Barcelona': 1.0}, 'Bologna': {'Bologna': 1.0}}
            >>> print(already_collected)
            {'London': {'London': 1.0}, 'Barcelona': {'Barcelona': 1.0}, 'Bologna': {'Bologna': 1.0}}
        """
        candidates = {}
        for query in queries:
            if query in self.already_collected_cands:
                candidates[query] = self.already_collected_cands[query]
            else:
                if query in self.mentions_to_wikidata:
                    candidates[query] = {query: 1.0}
                    self.already_collected_cands[query] = {query: 1.0}
                else:
                    candidates[query] = {}
                    self.already_collected_cands[query] = {}

        return candidates


class PartialMatchRanker(PerfectMatchRanker):
    """
    A ranking method using partial string matching. 
    
    This class extends PerfectMatchRanker because perfect matches are sought
    before attempting a partial match.

    Example:

    .. code-block:: python

        ranker = PartialMatchRanker(
            resources_path="/path/to/resources/",
        )
    """

    def method_name(self) -> str:
        return "partialmatch"
    

    def run(self, queries: List[str]) -> dict:
        """
        Perform partial matching for a list of given mentions (``queries``).

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.

        Returns:
            dict: A dictionary mapping each mention to its candidate
                   list, where the candidate list is a dictionary with the
                   mention variations as keys and their match scores as values.

        Note:
            This method performs partial matching for each mention in the given
            list. If a mention has already been matched perfectly, it skips the
            partial matching process for that mention. For the remaining
            mentions, it calculates the match score based on the specified
            partial matching method: Levenshtein distance or containment.

        """
        # First fill in the perfect matches and already collected queries
        candidates = super().run(queries)

        # the rest go through
        remainers = [x for x, y in candidates.items() if len(y) == 0]

        for query in remainers:
            mention_df = pd.DataFrame({"mentions": self.mentions_to_wikidata.keys()})

            mention_df["score"] = mention_df.parallel_apply(
                lambda row: self.matching_score(query, row), axis=1
            )
            mention_df = mention_df.dropna()

            # currently hardcoded cutoff
            top_scores = sorted(
                list(set(list(mention_df["score"].unique()))), reverse=True
            )[:1]
            mention_df = mention_df[mention_df["score"].isin(top_scores)]
            mention_df = mention_df.set_index("mentions").to_dict()["score"]

            candidates[query] = mention_df

            self.already_collected_cands[query] = mention_df

        return candidates
    

    def matching_score(self, query: str, row: pd.Series) -> float:
        """
        Calculate the partial string matching score as the amount of overlap, 
        if a mention is contained within a row in the dataset.

        Arguments:
            query (str): A mention identified in a text.
            row (Series): A pandas Series representing a row in the dataset
                with a "mentions" column, corresponding to a mention in the
                knowledge base.

        Returns:
            float:
                The match score indicating the degree of containment,
                ranging from ``0.0`` to ``1.0`` (perfect match).

        Example:
            >>> ranker = PartialMatchRanker(...)
            >>> query = 'apple'
            >>> row = pd.Series({'mentions': 'Delicious apple'})
            >>> match_score = ranker.matching_score(query, row)
            >>> print(match_score)
            0.3333333333333333
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

    Example:

    .. code-block:: python

        ranker = LevenshteinRanker(
            resources_path="/path/to/resources/",
        )
    """

    def method_name(self) -> str:
        return "levenshtein"

    def matching_score(self, query: str, row: pd.Series) -> float:
        """
        Calculate the partial string matching score as the Damerau-Levenshtein 
        distance between a mention and a row in the dataset.

        Arguments:
            query (str): A mention identified in a text.
            row (Series): A pandas Series representing a row in the dataset
                with a "mentions" column, corresponding to an alternate name
                of an etity in the knowledge base.

        Returns:
            float:
                The similarity score between the query and the row, ranging
                from ``0.0`` to ``1.0``.

        Note:
            This method computes the Damerau-Levenshtein distance between the
            lowercase versions of a mention and the "mentions" column value in
            the given row. The distance is then normalized to a similarity score 
            by subtracting it from ``1.0``. If a mention has already been matched 
            perfectly, it skips the partial matching process for that mention. 

        Example:
            >>> ranker = LevenshteinRanker(...)
            >>> query = 'apple'
            >>> row = pd.Series({'mentions': 'orange'})
            >>> similarity = ranker.matching_score(query, row)
            >>> print(similarity)
            0.1666666865348816
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
            which will be loaded through the
            :py:meth:`~geoparser.ranking.Ranker.load_resources` method.
        wikidata_to_mentions (dict, optional): An empty dictionary which
            will store the mapping between Wikidata IDs and mentions,
            which will be loaded through the
            :py:meth:`~geoparser.ranking.Ranker.load_resources` method.
        strvar_parameters (dict, optional): Dictionary of string variation
            parameters required to create a DeezyMatch training dataset.
            For the default settings, see Notes below.
        deezy_parameters (dict, optional): Dictionary of DeezyMatch parameters
            for model training. For the default settings, see Notes below.
        already_collected_cands (dict, optional): Dictionary of already
            collected candidates. Defaults to ``dict()`` (an empty dictionary).

    Example:

    .. code-block:: python

        ranker = DeezyMatchRanker(
            resources_path="/path/to/resources/",
        )

    Note:
        * The default settings for ``strvar_parameters``:

          .. code-block:: python

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

        * The default settings for ``deezy_parameters``:

          .. code-block:: python

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
    """
    
    # Override the constructor to include DeezyMatch model parameters.
    def __init__(
        self,
        resources_path: str,
        mentions_to_wikidata: Optional[dict] = dict(),
        wikidata_to_mentions: Optional[dict] = dict(),
        strvar_parameters: Optional[dict] = None,
        deezy_parameters: Optional[dict] = None,
        already_collected_cands: Optional[dict] = dict(),
    ):
        super().__init__(resources_path, mentions_to_wikidata, wikidata_to_mentions, already_collected_cands)

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
                "verbose": False,
                # DeezyMatch training:
                "overwrite_training": False,
                "do_test": False,
            }

        self.strvar_parameters = strvar_parameters
        self.deezy_parameters = deezy_parameters

    def method_name(self) -> str:
        return "deezymatch"
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Ranker object, including the 
        method name and DeezyMatch training parameters.
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

    # Override the base class implementation to optionally train the 
    # DeezyMatch model.
    def load_resources(self, train: bool =True) -> dict:
        ret = super().load_resources()
        if train:
            self.train()
        return ret
    
    def run(self, queries: List[str]) -> dict:
        """
        Perform DeezyMatch ranking on-the-fly for a list of given mentions (``queries``).

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.

        Returns:
            dict: A dictionary mapping each mention to its candidate
                list, where the candidate list is a dictionary with the
                mention variations as keys and their match scores as values.

        Example:
            >>> ranker = DeezyMatchRanker(...)
            >>> ranker.load_resources()
            >>> queries = ['London', 'Shefrield']
            >>> candidates = ranker.run(queries)
            >>> print(candidates)
            {'London': {'London': 1.0}, 'Shefrield': {'Sheffield': 0.03382000000000005}}
            >>> print(already_collected)
            {'London': {'London': 1.0}, 'Shefrield': {'Sheffield': 0.03382000000000005}}

        Note:
            This method performs DeezyMatch on-the-fly for each mention in a
            given list of mentions identified in a text. If a query has
            already been matched perfectly, it skips the fuzzy matching
            process for that query. For the remaining queries,
            it uses the DeezyMatch model to generate candidates and ranks them
            based on the specified ranking metric and selection threshold,
            provided when initialising the :py:meth:`~geoparser.ranking.Ranker`
            object.
        """

        dm_path = self.deezy_parameters["dm_path"]
        dm_cands = self.deezy_parameters["dm_cands"]
        dm_model = self.deezy_parameters["dm_model"]
        dm_output = self.deezy_parameters["dm_output"]

        # First fill in the perfect matches and already collected queries
        cands_dict = super().run(queries)

        # the rest go through
        remainers = [x for x, y in cands_dict.items() if len(y) == 0]

        if remainers:
            candidate_scenario = os.path.join(
                dm_path, "combined", dm_cands + "_" + dm_model
            )
            pretrained_model_path = os.path.join(
                f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".model"
            )
            pretrained_vocab_path = os.path.join(
                f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".vocab"
            )

            candidates = candidate_ranker(
                candidate_scenario=candidate_scenario,
                query=remainers,
                ranking_metric=self.deezy_parameters["ranking_metric"],
                selection_threshold=self.deezy_parameters["selection_threshold"],
                num_candidates=self.deezy_parameters["num_candidates"],
                search_size=self.deezy_parameters["num_candidates"],
                verbose=self.deezy_parameters["verbose"],
                output_path=os.path.join(dm_path, "ranking", dm_output),
                pretrained_model_path=pretrained_model_path,
                pretrained_vocab_path=pretrained_vocab_path,
            )

            for _, row in candidates.iterrows():
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

                cands_dict[row["query"]] = returned_cands

                self.already_collected_cands[row["query"]] = returned_cands

        return cands_dict
    
    def train(self) -> None:
        """
        Train a DeezyMatch model. The training will be skipped if the model
        already exists and the ``overwrite_training`` key in the
        ``deezy_parameters`` passed when initialising the
        :py:meth:`~geoparser.ranking.Ranker` object is set to ``False``. The
        training will be run on test mode if the ``do_test`` key in the
        ``deezy_parameters`` passed when initialising the
        :py:meth:`~geoparser.ranking.Ranker` object is set to ``True``.

        Returns:
            None.
        """

        Path(self.deezy_parameters["dm_path"]).mkdir(parents=True, exist_ok=True)
        if self.deezy_parameters["do_test"] == True:
            self.deezy_parameters["dm_model"] += "_test"
            self.deezy_parameters["dm_cands"] += "_test"
        deezy_processing.train_deezy_model(
            self.deezy_parameters, self.strvar_parameters, self.wikidata_to_mentions
        )
        deezy_processing.generate_candidates(
            self.deezy_parameters, self.mentions_to_wikidata
        )

        # This dictionary is not used anymore:
        self.wikidata_to_mentions = dict()
