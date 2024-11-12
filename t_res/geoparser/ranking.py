import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from DeezyMatch import candidate_ranker
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from dataclasses import dataclass, field

from ..utils import deezy_processing

@dataclass(order=True, frozen=True)
class StringMatch:
    """Data class representing a potential toponym string match."""
    sort_index: float = field(init=False)
    # The toponym spelling variation.
    variation: str
    # String matching similarly score.
    string_similarity: float

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_similarity)

    def is_empty(self) -> bool:
        return self.variation is None and self.string_similarity is None

    def as_string_match(self):
        return self

# TODO: WikidataMatch (or subclasses) will need to accommodate data relating to 
# all linking methods, with a generic `disambiguation_score` method to get the 
# linking score (previously referred to as `ed_score`).
# And note that the normalized_score is used in rel_utils::rank_candidates method
# when the linking method is `mostpopular`, i.e. when the link freq data must also be
# available. Whereas when the linking method is REL, the freq data is not needed.

# Set frozen=False to allow freq to be set after instantiation.
@dataclass(order=True, frozen=False)
class WikidataMatch:
    """Data class representing a potential toponym match in Wikidata."""
    sort_index: int = field(init=False)
    # The Wikidata ID.
    wqid: str
    # The normalized score.
    normalized_score: float
    # The mention-to-wikidata link frequency.
    freq: int
    # Associated private field to reconcile dataclasses & properties.
    # (See https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python)
    _freq: int = field(init=False, repr=False)

    @property
    def freq(self) -> str:
        return self._freq

    # Custom setter to ensure sort_index is always equal to freq attribute.
    @freq.setter
    def freq(self, freq: int) -> None:
        self._freq = freq
        self.sort_index = freq

    def __post_init__(self):
        # When WikidataMatch instances are sorted in a list, sort by the freq.
        self.sort_index = self.freq

# TODO: rename as Candidate.
@dataclass(order=True, frozen=True)
class CandidateMatch(StringMatch):
    """Data class representing a potential toponym match with Wikidata candidates."""
    sort_index: float = field(init=False)
    # A list of potential matches in Wikidata.
    wikidata_matches: List[WikidataMatch]
    # Associated private field to reconcile dataclasses & properties.
    # (See https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python)
    _wikidata_matches: List[WikidataMatch] = field(init=False, repr=False)

    def __post_init__(self):
        # When CandidateMatch instances are sorted in a list, sort by the string_similarity.
        object.__setattr__(self, 'sort_index', self.string_similarity)

    # Custom getter for the wikidata_matches attribute to ensure correct ordering.
    @property
    def wikidata_matches(self) -> List[WikidataMatch]:
        return sorted(self._wikidata_matches, reverse=True)

    @wikidata_matches.setter
    def wikidata_matches(self, wikidata_matches: List[WikidataMatch]):
        object.__setattr__(self, '_wikidata_matches', wikidata_matches)

    # Returns a StringMatch instance identical to this CandidateMatch
    # instance except with the wikidata_matches attribute removed.
    def as_string_match(self):
        return StringMatch(self.variation, self.string_similarity)
    
    # Returns a list of Wikidata link relative frequencies in descending order.
    # The order is identical to the list in the wikidata_matches attribute.
    def relative_frequencies(self) -> List[float]:
        abs_freqs = [m.freq for m in self._wikidata_matches]
        if any(f is None for f in abs_freqs):
            raise ValueError("Wikidata frequencies not populated.")
        total = sum([m.freq for m in self.wikidata_matches])
        return [m.freq / total for m in self.wikidata_matches]
    
    # Returns the top 7 linking confidence score for each Wikidata candidate
    # as reported as `cross_cand_score` in the T-Res pipeline output.
    # (Currently assumes the `mostpopular` linking method.)
    def cross_cand_score(self, len=7) -> dict:
        tmp = {m.wqid: round(rf, 3) for (m, rf) in zip(self.wikidata_matches, self.relative_frequencies())}
        return dict(sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:len])

    # Returns the WikidataMatch instance with the given Wikidata ID
    # or None if no such match exists.
    def get(self, wqid: str):
        for m in self._wikidata_matches:
            if m.wqid == wqid:
                return m
        return None
    
    # TODO: 
    # (will depend on the linking method, to be recorded in a new field.)
    # Returns a list of disambiguation scores, one for each Wikidata 
    # candidate, in descending order.
    # def disambiguation_scores(self) -> List[float]:
    
@dataclass(order=True, frozen=True)
class Candidates:
    """Data class representing candidate matches for a toponym."""
    # The toponym as mentioned in the text.
    mention: str
    # The string matching method used.
    method: str
    # A dictionary of potential toponym matches, keyed by (each of which may contain a list of Wikidata candidates).
    matches: List[StringMatch]

    def __post_init__(self):
        # Check that the variations are unique in self.matches.
        variations = [match.variation for match in self.matches]
        if len(variations) != len(set(variations)):
            raise ValueError("StringMatch variations must be unique.")
        object.__setattr__(self, 'matches', sorted(self.matches, reverse=True))

    def __str__(self) -> str:
        s = f"Candidates for '{self.mention}':"
        if self.is_empty():
            s += " None"
            return s
        l = max([len(m.variation) for m in self.matches])
        for c in self.matches:
            if c.is_empty():
                continue
            s += f"\n    {c.variation.ljust(l)} [{'{:.3f}'.format(c.string_similarity)}]"
            if isinstance(c, CandidateMatch) and len(c.wikidata_matches) > 0:
                s += ": "
                for w in c.wikidata_matches[:2]:
                    s += f"({w.wqid}, {w.freq}), "
                if len(c.wikidata_matches) > 2:
                    s += "..."
                else:
                    s = s[:-2]
        return s
    
    def is_empty(self) -> bool:
        return len(self.matches) == 0 or self.matches[0].is_empty()
    
    # Strips away information relating to potential Wikidata matches.
    def as_string_matches(self):
        return Candidates(self.mention, 
                          self.method, 
                          [m.as_string_match() for m in self.matches])
    
    # Returns the StringMatch instance with the given spelling variation
    # or None if no such match exists.
    def get(self, variation: str):
        for m in self.matches:
            if m.variation == variation:
                return m
        return None
    
    # Returns the StringMatch with the highest string similarity.
    def best_match(self) -> StringMatch:
        if self.is_empty():
            return None
        return self.matches[0]

    # Returns the Wikidata match with the highest disambiguation score.
    def best_wikidata_match(self) -> WikidataMatch:
        best_match = self.best_match()
        if not best_match or best_match.is_empty():
            return None
        if not isinstance(best_match, CandidateMatch):
            return None
        if len(best_match.wikidata_matches) == 0:
            return None
        return best_match.wikidata_matches[0]

    def best_wqid(self) -> str:
        best_wikidata_match = self.best_wikidata_match()
        if not best_wikidata_match:
            return None
        return best_wikidata_match.wqid

    # TODO:
    # def best_disambiguation_score(self):
    #     best_match = self.best_match()
    #     if not best_match:
    #         return None
    #     return best_match....

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
        already_collected_cands (dict, optional): Dictionary for caching the
            results of queries that have already been executed. Defaults 
            to ``dict()`` (an empty dictionary).

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
        >>> results = [ranker.run(query) for query in queries]
        >>> # Print the results
        >>> print("Candidate Selection Results:")
        >>> for candidates in results:
        >>>     print(candidates)
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
        # TODO: rename as `cache`.
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

        # NOTE: these are the *normalized* mentions, *not* relative frequencies.
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

    # TODO: rename `query` as `mention`.
    def run(self, query: str, attach_wikidata: bool = True) -> Candidates:
        """
        Execute the ranking process.

        Arguments:
            query (str): A toponym to be matched.
            attach_wikidata (bool): If True, potential Wikidata 
                matches are included in the returned candidates.

        Returns:
            Candidates: An instance of the Candidates dataclass, containing
                potential matches for the given toponym.

        Note: if the query result has already been cached with Wikidata
        matches attached, that cached result will be returned (even if
        the `attach_wikidata` flag is set to False). To remove the 
        Wikidata matches, call the `as_string_matches` method on the
        returned Candidates instance.
        """
        if not isinstance(query, str):
            raise ValueError("`query` argument must have type `str`")

        # Use the cache if possible.
        if query in self.already_collected_cands:
            candidates = self.already_collected_cands[query]
        else:
            candidates = self.match_candidates(query)
        if not attach_wikidata:
            return candidates
        return self.attach_wikidata(candidates)

    def match_candidates(self, query: str) -> Candidates:
        """
        Identify matching candidates for the given toponym query.
        
        Each Ranker subclass must implement a ranking method by overriding 
        this function.

        Args:
            query (str): A toponym to be matched.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.

        Returns:
            Candidates: An instance of the Candidates dataclass, containing
                potential matches for the given toponym.
        """
        raise NotImplementedError("Subclass implementation required.")

    def attach_wikidata(self, candidates: Candidates) -> Candidates:
        """
        Replace each `StringMatch` instance in the given candidates by 
        a `CandidateMatch` instance containing potential Wikidata matches.

        Args:
            candidates: An instance of the Candidates dataclass in which
                each match is a StringMatch instance.

        Returns:
            Candidates: An instance of the Candidates dataclass in which
                each match is a CandidateMatch instance.

        The toponym variation is sought in the Wikidata knowledgebase and
        potential matches are attached to the candidates instance.

        Note:
            This method updates the Ranker cache, replacing any existing
            entry for the given toponym.
        """
        # If Wikidata candidates are already attached, there's nothing to do.
        if all([isinstance(m, CandidateMatch) for m in candidates.matches]):
            return candidates

        # Replace each StringMatch instance in the candidates with a CandidateMatch
        # instance (containing potential Wikidata matches with normalized scores).
        for i, match in enumerate(candidates.matches):
            found_cands = self.mentions_to_wikidata.get(match.variation, dict())
            wikidata_matches = [WikidataMatch(wqid=k, 
                                              freq=None, 
                                              normalized_score=v) for (k, v) in found_cands.items()]
            candidates.matches[i] = CandidateMatch(match.variation, match.string_similarity, wikidata_matches)

        # Update the cache.
        self.already_collected_cands[candidates.mention] = candidates
        return candidates

# TODO: fix docstring
class PerfectMatchRanker(Ranker):
    """
    A ranking method using perfect string matching.

    Example:
        >>> ranker = PerfectMatchRanker(...)
        >>> ranker.mentions_to_wikidata = ranker.load_resources()
        >>> queries = ['London', 'Barcelona', 'Bologna']
        >>> results = [ranker.run(query) for query in queries]
        >>> # Print the results
        >>> print("Candidate Selection Results:")
        >>> for candidates in results:
        >>>     print(candidates)
    """
    def method_name(self) -> str:
        return "perfectmatch"

    def match_candidates(self, query: str) -> Candidates:
        """
        Perform perfect matching between a provided list of mentions
        (``queries``) and the altnames in the knowledge base.

        Arguments:
            query: A toponym mention (string) to be matched.

        Returns:
            Candidates: An instance of the Candidates dataclass, containing
                potential matches for the given toponym. In the case of perfect
                matching, all matches have string_similarity equal to 1.0.

        Note:
            This method checks if the query has an exact match in the
            mentions_to_wikidata dictionary. If a match is found, it assigns a
            perfect match score of ``1.0`` to the mention. Otherwise, an empty
            dictionary is assigned as the list of matches for the query.

        Example:
            >>> ranker = PerfectMatchRanker(resources_path="...")
            >>> ranker.mentions_to_wikidata = ranker.load_resources()
            >>> queries = ['London', 'Barcelona', 'Bologna']
            >>> results = [ranker.run(query) for query in queries]
            >>> # Print the results
            >>> print("Candidate Selection Results:")
            >>> for candidates in results:
            >>>     print(candidates)
        """
        if query in self.mentions_to_wikidata:
            matches = [StringMatch(query, 1.0)]
        else:
            # If no match exists, assign an empty list to matches. 
            matches = list()
        candidates = Candidates(query, self.method_name(), matches)
        # Update the cache.
        self.already_collected_cands[query] = candidates
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
    
    def match_candidates(self, query: str) -> Candidates:
        """
        Perform partial string matching for a given toponym query.

        Arguments:
            query (str): A toponym to be matched.

        Returns:
            Candidates: An instance of the Candidates dataclass, containing
                potential matches for the given toponym.

        Note:
            This method performs partial matching for each mention in the given
            list. If a mention has already been matched perfectly, it skips the
            partial matching process for that mention. For the remaining
            mentions, it calculates the match score based on the specified
            partial matching method: Levenshtein distance or containment.
        """
        # First attempt a perfect string match.
        candidates = super().match_candidates(query)
        if not candidates.is_empty():
            return candidates
        
        # Seek partial string matches.
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
        cands_dict = mention_df.set_index("mentions").to_dict()["score"]
        matches = [StringMatch(k, v) for (k, v) in cands_dict.items()]

        # Convert the partial string matches into candidates.
        candidates = Candidates(query, self.method_name(), matches)
        # Update the cache.
        self.already_collected_cands[query] = candidates
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

    # Override the base class implementation to optionally train the model.
    def load_resources(self, train: bool =True) -> dict:
        ret = super().load_resources()
        if train:
            self.train()
        return ret
    
    # TODO: docstring inc. example
    def match_candidates(self, query: str) -> Candidates:
        """
        Perform DeezyMatch ranking on-the-fly for a list of given mentions (``queries``).

        Arguments:
            query (str): A toponym to be matched.

        Returns:
            Candidates: An instance of the Candidates dataclass, containing
                potential matches for the given toponym.

        Example:
            >>> ranker = DeezyMatchRanker(...)
            >>> ranker.load_resources()
            >>> queries = ['London', 'Shefrield']
            >>> results = [ranker.match_candidates(query) for query in queries]
            >>> # Print the results
            >>> print("Candidate Selection Results:")
            >>> for candidates in results:
            >>>     print(candidates)

        Note:
            This method performs DeezyMatch on-the-fly for each mention in a
            given list of mentions identified in a text. If a query has
            already been matched perfectly, it skips the fuzzy matching
            process for that query. For the remaining queries,
            it uses the DeezyMatch model to generate candidates and ranks them
            based on the specified ranking metric and selection threshold,
            provided when initialising the ranker.
        """

        dm_path = self.deezy_parameters["dm_path"]
        dm_cands = self.deezy_parameters["dm_cands"]
        dm_model = self.deezy_parameters["dm_model"]
        dm_output = self.deezy_parameters["dm_output"]

        # First attempt a perfect string match.
        candidates = super().match_candidates(query)
        if not candidates.is_empty():
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
        # Convert the partial string matches into candidates.
        candidates = Candidates(query, self.method_name(), matches)
        # Update the cache.
        self.already_collected_cands[query] = candidates
        return candidates
    
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
