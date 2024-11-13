from typing import List, Dict
from dataclasses import dataclass, field

#################################
# Dataclasses for Ranker output 
#################################

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

# Ranker::run method output type (and Linker::run method input type).
@dataclass(frozen=True)
class CandidateMatches:
    """Data class representing candidate matches for a toponym."""
    # The toponym as mentioned in the text.
    mention: str
    # The string matching method used.
    ranking_method: str
    # A dictionary of potential toponym matches, keyed by (each of which may contain a list of Wikidata candidates).
    matches: List[StringMatch]

#################################
# Dataclasses for Linker output 
#################################

# Base class.
@dataclass(frozen=True)
class WikidataLink:
    """Data class representing a potential toponym link in Wikidata."""
    # The Wikidata ID.
    wqid: str

@dataclass(frozen=True)
class MostPopularLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `mostpopular` linking method."""
    # The mention-to-wikidata link frequency.
    freq: int

@dataclass(frozen=True)
class ByDistanceLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `bydistance` linking method."""
    # TODO: add fields related to wqid_to_coords.
    wqid_to_coords: int

@dataclass(frozen=True)
class RelDisambLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `reldisamb` linking method."""
    # The mention-to-wikidata link frequency.
    freq: int
    # The normalized score from resource `mentions_to_wikidata_normalized.json`.
    normalized_score: float

@dataclass(order=True, frozen=True)
class CandidateLinks:
    """Data class representing a collection of potential links in Wikidata for a given string match."""
    sort_index: float = field(init=False)
    # A StringMatch instance.
    string_match: StringMatch
    # A list of candidate links in Wikidata.
    wikidata_links: List[WikidataLink]
    # Associated private field to reconcile dataclasses & properties.
    # (See https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python)
    _wikidata_links: List[WikidataLink] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_match.string_similarity)

    # Custom getter for the wikidata_matches attribute to ensure correct ordering.
    @property
    def wikidata_links(self) -> List[WikidataLink]:
        return sorted(self._wikidata_matches, reverse=True)

    @wikidata_links.setter
    def wikidata_links(self, wikidata_links: List[WikidataLink]):
        object.__setattr__(self, '_wikidata_links', wikidata_links)

    # Returns a dict of entity disambiguation scores (formerly known as `ed_score`) keyed by wqid.
    def disambiguation_scores() -> Dict[str, float]:
        return "TODO"
    
    def best_disambiguation_score() -> float:
        return "TODO"
    
    def best_wqid() -> float:
        return "TODO"

# Linker::run method output type.
@dataclass(frozen=True)
class CandidatesNew: # TODO: rename as Candidates by retiring the old Candidates dataclass.
    """Data class representing candidate string matches for a toponym, 
    each with candidate Wikidata links."""
    # The toponym as mentioned in the text.
    mention: str
    # The string matching method used.
    ranking_method: str
    # The linking method used.
    linking_method: str
    # A list of CandidateLinks instances.
    matches: List[CandidateLinks]

    def __post_init__(self):
        # Check that the variations are unique in self.matches.
        variations = [m.string_match.variation for m in self.matches]
        if len(variations) != len(set(variations)):
            raise ValueError("StringMatch variations must be unique.")
        object.__setattr__(self, 'matches', sorted(self.matches, reverse=True))

    def __str__(self) -> str:
        s = f"Candidates for '{self.mention}':"
        if self.is_empty():
            s += " None"
            return s
        l = max([len(m.string_match.variation) for m in self.matches])
        for m in self.matches:
            if m.is_empty():
                continue
            s += f"\n    {m.string_match.variation.ljust(l)} [{'{:.3f}'.format(m.string_match.string_similarity)}]"
            if len(m.wikidata_links) > 0:
                s += ": "
                for wqid, score in m.disambiguation_scores().items()[:2]:
                    s += f"({wqid}, {score}), "
                if len(m.wikidata_links) > 2:
                    s += "..."
                else:
                    s = s[:-2]
        return s
    
    def is_empty(self) -> bool:
        return len(self.matches) == 0 or self.matches[0].is_empty()
    
    # Returns the CandidateLinks instance with the given spelling 
    # variation, or None if no such match exists.
    def get(self, variation: str):
        for m in self.matches:
            if m.string_match.variation == variation:
                return m
        return None
    
    # Returns the StringMatch with the highest string similarity.
    def best_match(self) -> StringMatch:
        if self.is_empty():
            return None
        return self.matches[0]

    # Returns the Wikidata match with the highest disambiguation score.
    def best_wikidata_match(self) -> 'WikidataMatch':
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



#####################################################################





# TODO: make this into a base class & rename as WikidataLink
# TODO: set frozen=True after refactoring.
# Set frozen=False to allow freq to be set after instantiation.
@dataclass(order=True, frozen=False)
class WikidataMatch:
    """Data class representing a potential toponym match in Wikidata."""
    # TODO (REMOVE as superseded by `disambiguation_scores` method that will do the sorting): sort_index: int = field(init=False):
    sort_index: int = field(init=False)
    # The Wikidata ID.
    wqid: str
    # # TODO (move to subclasses):
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


# TODO: WikidataMatch (or subclasses) will need to accommodate data relating to 
# all linking methods, with a generic `disambiguation_score` method to get the 
# linking score (previously referred to as `ed_score`).
# And note that the normalized_score is used in rel_utils::rank_candidates method
# when the linking method is `mostpopular`, i.e. when the link freq data must also be
# available. Whereas when the linking method is REL, the freq data is not needed.

# TODO: refactor into CandidateMatches and CandidateLinks (instead of mixed):
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
    def best_wikidata_match(self) -> 'WikidataMatch':
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
