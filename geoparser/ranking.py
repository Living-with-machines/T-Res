import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pandas as pd
from DeezyMatch import candidate_ranker
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import deezy_processing


class Ranker:
    """
    The Ranker class implements a system for candidate selection through string
    variation ranking. It provides methods to select candidates based on different
    matching approaches, such as perfect match, partial match, Levenshtein distance,
    and DeezyMatch. The class also handles loading and processing of resources
    related to candidate selection.

    Arguments:
        method (str): The candidate selection and ranking method to use.
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
        >>> # Create a Ranker object:
        >>> ranker = Ranker(
                method="perfectmatch",
                resources_path="/path/to/resources/",
            )

        >>> # Load resources
        >>> ranker.load_resources()

        >>> # Train the ranker (if applicable)
        >>> ranker.train()

        >>> # Perform candidate selection
        >>> queries = ['London', 'Paraguay']
        >>> candidates, already_collected = ranker.run(queries)

        >>> # Find candidates for mentions
        >>> mentions = [{'mention': 'London'}, {'mention': 'Paraguay'}]
        >>> mention_candidates, mention_already_collected = ranker.find_candidates(mentions)

        >>> # Print the results
        >>> print("Candidate Selection Results:")
        >>> print(candidates)
        >>> print(already_collected)
        >>> print("Find Candidates Results:")
        >>> print(mention_candidates)
        >>> print(mention_already_collected)

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

    def __init__(
        self,
        method: Literal["perfectmatch", "partialmatch", "levenshtein", "deezymatch"],
        resources_path: str,
        mentions_to_wikidata: Optional[dict] = dict(),
        wikidata_to_mentions: Optional[dict] = dict(),
        strvar_parameters: Optional[dict] = {
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("resources/models/w2v/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "overwrite_dataset": False,
        },
        deezy_parameters: Optional[dict] = {
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("resources/deezymatch/").resolve()),
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
        },
        already_collected_cands: Optional[dict] = dict(),
    ):
        """
        Initialize a Ranker object.
        """
        self.method = method
        self.resources_path = resources_path
        self.mentions_to_wikidata = mentions_to_wikidata
        self.wikidata_to_mentions = wikidata_to_mentions
        self.strvar_parameters = strvar_parameters
        self.deezy_parameters = deezy_parameters
        self.already_collected_cands = already_collected_cands

    def __str__(self) -> str:
        """
        Returns a string representation of the Ranker object.

        Note:
            The string will, at minimum, include the method name, and if the
            ``method`` was set to "deezymatch" in the Ranker initialiser, the
            string will also include the training parameters provided.
        """
        s = ">>> Candidate selection:\n"
        s += f"    * Method: {self.method}\n"

        if self.method == "deezymatch":
            s += "    * DeezyMatch details:\n"
            s += f"      * Model: {self.deezy_parameters['dm_model']}\n"
            s += f"      * Ranking metric: {self.deezy_parameters['ranking_metric']}\n"
            s += f"      * Selection threshold: {self.deezy_parameters['selection_threshold']}\n"
            s += f"      * Num candidates: {self.deezy_parameters['num_candidates']}\n"
            s += f"      * Overwrite training: {self.deezy_parameters['overwrite_training']}\n"
            s += f"      * Overwrite dataset: {self.strvar_parameters['overwrite_dataset']}\n"
            s += f"      * Test mode: {self.deezy_parameters['do_test']}\n"

        return s

    def load_resources(self):
        """
        Load the ranker resources.

        Note:
            This method loads the mentions-to-wikidata and
            wikidata-to-mentions dictionaries from the resources directory,
            specified when initialising the
            :py:meth:`~geoparser.ranking.Ranker`. They are required for
            performing candidate selection and ranking.

            It filters the dictionaries to remove noise and updates the class
            attributes accordingly.

            The loaded mentions-to-wikidata dictionary, which maps a mention 
            (e.g. ``"London"``) to the Wikidata entities that are
            referred to by this mention on Wikipedia (e.g. ``Q84``,
            ``Q2477346``). The data also includes, for each entity, their
            normalized "relevance", i.e. number of in-links across Wikipedia.
            
            The method also initialises ``pandarallel`` if needed by the
            candidate ranking method (if the ``method`` set in the initialiser
            of the ``Ranker`` was set to "partialmatch" or "levenshtein").
        """
        print("*** Loading the ranker resources.")

        # Load files
        files = {
            "mentions_to_wikidata": f"{self.resources_path}mentions_to_wikidata_normalized.json",
            "wikidata_to_mentions": f"{self.resources_path}wikidata_to_mentions_normalized.json",
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
        if self.method in ["partialmatch", "levenshtein"]:
            pandarallel.initialize(nb_workers=10)
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def train(self) -> None:
        """
        Training a DeezyMatch model. The training will be skipped if the model
        already exists and the ``overwrite_training`` key in the
        ``deezy_parameters`` passed when initialising the
        :py:meth:`~geoparser.ranking.Ranker` object is set to ``False``. The
        training will be run on test mode if the ``do_test`` key in the
        ``deezy_parameters`` passed when initialising the
        :py:meth:`~geoparser.ranking.Ranker` object is set to ``True``.

        Returns:
            None.
        """

        if self.method == "deezymatch":
            Path(self.deezy_parameters["dm_path"]).mkdir(parents=True, exist_ok=True)
            if self.deezy_parameters["do_test"] == True:
                self.deezy_parameters["dm_model"] += "_test"
                self.deezy_parameters["dm_cands"] += "_test"
            deezy_processing.train_deezy_model(self.deezy_parameters, self.strvar_parameters, self.wikidata_to_mentions)
            deezy_processing.generate_candidates(
                self.deezy_parameters, self.mentions_to_wikidata
            )

        # This dictionary is not used anymore:
        self.wikidata_to_mentions = dict()

    def perfect_match(self, queries: List[str]) -> Tuple[dict, dict]:
        """
        Perform perfect matching between a provided list of mentions
        (``queries``) and the altnames in the knowledge base.

        Arguments:
            queries (list): A list of mentions (string) identified in a text
                to match.

        Returns:
            Tuple[dict, dict]: A tuple containing two dictionaries:

                #. The first dictionary maps each mention to its candidate
                   list, where the candidate list is a dictionary with the
                   mention itself as the key and a perfect match score of
                   ``1.0``.

                #. The second dictionary stores the already collected
                   candidates for each mention. It is an updated version of the
                   Ranker's ``already_collected_cands`` attribute.

        Note:
            This method checks if each mention has an exact match in the
            mentions_to_wikidata dictionary. If a match is found, it assigns a
            perfect match score of ``1.0`` to the mention. Otherwise, an empty
            dictionary is assigned as the candidate list for the mention.
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

        return candidates, self.already_collected_cands

    def damlev_dist(self, query: str, row: pd.Series) -> float:
        """
        Calculate the Damerau-Levenshtein distance between a mention and a row
        in the dataset.

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
            the given row.

            The distance is then normalized to a similarity score by
            subtracting it from ``1.0``.

        Example:
            >>> ranker = Ranker(...)
            >>> query = 'apple'
            >>> row = pd.Series({'mentions': 'orange'})
            >>> similarity = ranker.damlev_dist(query, row)
            >>> print(similarity)
            0.1666666865348816
        """
        return 1.0 - normalized_damerau_levenshtein_distance(
            query.lower(), row["mentions"].lower()
        )

    def check_if_contained(self, query: str, row: pd.Series) -> float:
        """
        Returns the amount of overlap, if a mention is contained within a row
        in the dataset.

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
            >>> ranker = Ranker(...)
            >>> query = 'apple'
            >>> row = pd.Series({'mentions': 'Delicious apple'})
            >>> match_score = ranker.check_if_contained(query, row)
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

    def partial_match(self, queries: List[str], damlev: bool) -> Tuple[dict, dict]:
        """
        Perform partial matching for a list of given mentions (``queries``).

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.
            damlev (bool): A flag indicating whether to use the
                Damerau-Levenshtein distance for matching (True) or
                containment-based matching (False).

        Returns:
            Tuple[dict, dict]: A tuple containing two dictionaries:

                #. The first dictionary maps each mention to its candidate
                   list, where the candidate list is a dictionary with the
                   mention variations as keys and their match scores as values.

                #. The second dictionary stores the already collected
                   candidates for each mention. It is an updated version of the
                   Ranker's ``already_collected_cands`` attribute.

        Example:
            >>> ranker = Ranker(...)
            >>> queries = ['apple', 'banana', 'orange']
            >>> candidates, already_collected = ranker.partial_match(queries, damlev=False)
            >>> print(candidates)
            {'apple': {'apple': 1.0}, 'banana': {'bananas': 0.5, 'banana split': 0.75}, 'orange': {'orange': 1.0}}
            >>> print(already_collected)
            {'apple': {'apple': 1.0}, 'banana': {'bananas': 0.5, 'banana split': 0.75}, 'orange': {'orange': 1.0}}

        Note:
            This method performs partial matching for each mention in the given
            list. If a mention has already been matched perfectly, it skips the
            partial matching process for that mention. For the remaining
            mentions, it calculates the match score based on the specified
            partial matching method: Levenshtein distance or containment.

        """

        candidates, self.already_collected_cands = self.perfect_match(queries)

        # the rest go through
        remainers = [x for x, y in candidates.items() if len(y) == 0]

        for query in remainers:
            mention_df = pd.DataFrame({"mentions": self.mentions_to_wikidata.keys()})

            if damlev:
                mention_df["score"] = mention_df.parallel_apply(
                    lambda row: self.damlev_dist(query, row), axis=1
                )
            else:
                mention_df["score"] = mention_df.parallel_apply(
                    lambda row: self.check_if_contained(query, row), axis=1
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

        return candidates, self.already_collected_cands

    def deezy_on_the_fly(self, queries: List[str]) -> Tuple[dict, dict]:
        """
        Perform DeezyMatch (a deep neural network approach to fuzzy string
        matching) on-the-fly for a list of given mentions (``queries``).

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.

        Returns:
            Tuple[dict, dict]: A tuple containing two dictionaries:

                #. The first dictionary maps each mention to its candidate
                   list, where the candidate list is a dictionary with the
                   mention variations as keys and their match scores as values.

                #. The second dictionary stores the already collected
                   candidates for each mention. It is an updated version of the
                   Ranker's ``already_collected_cands`` attribute.

        Example:
            >>> ranker = Ranker(...)
            >>> ranker.load_resources()
            >>> queries = ['London', 'Shefrield']
            >>> candidates, already_collected = ranker.deezy_on_the_fly(queries)
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

        # first we fill in the perfect matches and already collected queries
        cands_dict, self.already_collected_cands = self.perfect_match(queries)

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

        return cands_dict, self.already_collected_cands

    def run(self, queries: List[str]) -> Tuple[dict, dict]:
        """
        Run the appropriate ranking method based on the specified method.

        Arguments:
            queries (list): A list of mentions (strings) identified in a text
                to match.

        Returns:
            Tuple[dict, dict]:
                A tuple containing two dictionaries. The resulting dictionaries
                will vary depending on the method set in the Ranker object.
                See Notes below for further information.

        Example:
            >>> myranker = Ranker(method="perfectmatch", ...)
            >>> myranker.load_resources()
            >>> queries = ['London', 'Barcelona', 'Bologna']
            >>> candidates, already_collected = myranker.run(queries)
            >>> print(candidates)
            {'London': {'London': 1.0}, 'Barcelona': {'Barcelona': 1.0}, 'Bologna': {'Bologna': 1.0}}
            >>> print(already_collected)
            {'London': {'London': 1.0}, 'Barcelona': {'Barcelona': 1.0}, 'Bologna': {'Bologna': 1.0}}

        Note:
            This method executes the appropriate ranking method based on the
            ``method`` parameter, selected when initialising the
            :py:meth:`~geoparser.ranking.Ranker` object.

            It delegates the execution to the corresponding method:

            * :py:meth:`~geoparser.ranking.Ranker.perfect_match`
            * :py:meth:`~geoparser.ranking.Ranker.partial_match`
            * :py:meth:`~geoparser.ranking.Ranker.levenshtein`
            * :py:meth:`~geoparser.ranking.Ranker.deezy_on_the_fly`

            See the documentation of those methods for more details about
            their processing if the provided mentions (``queries``).
        """
        if self.method == "perfectmatch":
            return self.perfect_match(queries)
        if self.method == "partialmatch":
            return self.partial_match(queries, damlev=False)
        if self.method == "levenshtein":
            return self.partial_match(queries, damlev=True)
        if self.method == "deezymatch":
            return self.deezy_on_the_fly(queries)
        raise SyntaxError(f"Unknown method: {self.method}")

    def find_candidates(self, mentions: List[dict]) -> Tuple[dict, dict]:
        """
        Find candidates for the given mentions using the selected ranking
        method.

        Arguments:
            mentions (list): A list of predicted mentions as dictionaries.

        Returns:
            Tuple[dict, dict]: A tuple containing two dictionaries:

            #. The first dictionary maps each original mention to a
               sub-dictionary, where the sub-dictionary maps the mention
               variations to a sub-sub-dictionary with two keys: ``"Score"``
               (the string matching similarity score) and ``"Candidates"``
               (a dictionary containing the Wikidata candidates, where the
               key is the Wikidata ID and value is the the relative mention-
               to-wikidata frequency).
            #. The second dictionary stores the already collected candidates
               for each query.

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
        cands, self.already_collected_cands = self.run(queries)

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

        return wk_cands, self.already_collected_cands
