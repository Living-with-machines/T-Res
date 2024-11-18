from typing import List, Dict, Optional
from pydantic.dataclasses import dataclass as pdataclass
from dataclasses import field
from collections.abc import Callable

# TODO: add __str__ methods

################################
# Dataclasses for Ranker
################################

@pdataclass(order=True, frozen=True)
class StringMatch:
    """Data class representing a potential toponym string match."""
    sort_index: float = field(init=False)
    # The toponym spelling variation.
    variation: str
    # String matching similarly score.
    string_similarity: float

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_similarity)

@pdataclass(order=True, frozen=True)
class StringMatchLinks(StringMatch):
    """Data class representing a potential toponym string match 
    with potential Wikidata ID links."""
    # List of potential Wikidata ID links.
    wqid_links: List[str]

    def as_string_match(self) -> StringMatch:
        return StringMatch(self.variation, self.string_similarity)

# Ranker::run method output type.
@pdataclass(frozen=True)
class CandidateMatches:
    """Data class representing candidate matches for a toponym."""
    # The toponym as mentioned in the text.
    mention: str
    # The string matching method used.
    ranking_method: str
    # A dictionary of potential toponym matches, keyed by (each of which may contain a list of Wikidata candidates).
    matches: List[StringMatchLinks]

    def __post_init__(self):
        # Check that the variations are unique in self.matches.
        variations = [match.variation for match in self.matches]
        if len(variations) != len(set(variations)):
            raise ValueError("StringMatch variations must be unique.")
        # Order matches by decreasing string similarity.
        object.__setattr__(self, 'matches', sorted(self.matches, reverse=True))

    def is_empty(self) -> bool:
        return len(self.matches) == 0

    # Returns the StringMatch instance with the given spelling variation
    # or None if no such match exists.
    def get(self, variation: str) -> StringMatchLinks:
        for m in self.matches:
            if m.variation == variation:
                return m
        return None
    

################################
# Dataclasses for Linker
################################

# Base dataclass.
@pdataclass(frozen=True)
class WikidataLink:
    """Data class representing a potential toponym link in Wikidata."""
    # The Wikidata ID.
    wqid: str

@pdataclass(frozen=True)
class MostPopularLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `mostpopular` linking method."""
    # The mention-to-wikidata link frequency.
    freq: int

    def __post_init__(self):
        if not isinstance(self.freq, int):
            raise ValueError("freq must be an integer.")

    # def disambiguation_score(self, total: float) -> float:
    #     return self.freq / total

@pdataclass(frozen=True)
class ByDistanceLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `bydistance` linking method."""
    # The Wikidata ID of the reference point (or "origin"). 
    origin_wqid: str
    # The geodesic distance between the wqid and the origin wqid.
    geodist: Optional[float]
    # The normalized score from resource `mentions_to_wikidata_normalized.json`.
    normalized_score: float

    def __post_init__(self):
        if not isinstance(self.normalized_score, float):
            raise ValueError("normalized_score must be an float.")


@pdataclass(frozen=True)
class RelDisambLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `reldisamb` linking method."""
    # The mention-to-wikidata link frequency.
    freq: int
    # The normalized score from resource `mentions_to_wikidata_normalized.json`.
    normalized_score: float

    def __post_init__(self):
        if not isinstance(self.freq, int):
            raise ValueError("freq must be an integer.")
        if not isinstance(self.normalized_score, float):
            raise ValueError("normalized_score must be an float.")

@pdataclass(order=True, frozen=True)
class CandidateLinks:
    """Data class representing a collection of potential links in Wikidata for a given string match."""
    sort_index: float = field(init=False)
    # A StringMatch instance.
    string_match: StringMatch
    # A list of candidate WikidataLink instances.
    wikidata_links: List[WikidataLink]
    # Closure used to compute disambiguation.
    disambiguation_scores: Callable[..., Dict[str, float]]

    # # Associated private field to reconcile dataclasses & properties.
    # # (See https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python)
    # _wikidata_links: List[WikidataLink] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_match.string_similarity)

    # # Custom getter for the wikidata_matches attribute to ensure correct ordering.
    # @property
    # def wikidata_links(self) -> List[WikidataLink]:
    #     return sorted(self._wikidata_links, reverse=True)

    # @wikidata_links.setter
    # def wikidata_links(self, wikidata_links: List[WikidataLink]):
    #     object.__setattr__(self, '_wikidata_links', wikidata_links)

    def is_empty(self) -> bool:
        return not self.wikidata_links

    def best_disambiguation_score(self) -> float:
        if self.is_empty():
            return None
        return max(self.disambiguation_scores().values())
    
    # TODO: use min(self.wikidata_links, key=lambda link: link....) if poss.
    def best_wikidata_link(self) -> WikidataLink:
        if self.is_empty():
            return None
        for link in self.wikidata_links:
            if link.wqid == self.best_wqid():
                return link
    
    def best_wqid(self) -> float:
        if self.is_empty():
            return None
        scores = self.disambiguation_scores()
        return max(scores, key=lambda key: scores[key])

    # Returns the top 7 Wikidata links in order of their disambiguation score
    # (as reported as `cross_cand_score` in the T-Res pipeline output).
    def cross_cand_score(self, len=7) -> dict:
        scores = {k: round(v, 3) for (k, v) in self.disambiguation_scores().items()}
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:len])

# Linker::run method output type.
@pdataclass(frozen=True)
class Candidates:
    """Data class representing candidate string matches for a toponym, 
    each with candidate Wikidata links."""
    # The toponym as mentioned in the text.
    mention: str
    # The string matching method used.
    ranking_method: str
    # The linking method used.
    linking_method: str
    # A list of CandidateLinks instances.
    links: List[CandidateLinks]

    def __post_init__(self):
        # Check that the variations are unique in self.links.
        variations = [m.string_match.variation for m in self.links]
        if len(variations) != len(set(variations)):
            raise ValueError("StringMatch variations must be unique.")
        object.__setattr__(self, 'links', sorted(self.links, reverse=True))

    def __str__(self) -> str:
        s = f"Candidates for '{self.mention}':"
        if self.is_empty():
            s += " None"
            return s
        l = max([len(m.string_match.variation) for m in self.links])
        for m in self.links:
            if m.is_empty():
                continue
            s += f"\n    {m.string_match.variation.ljust(l)} [{'{:.3f}'.format(m.string_match.string_similarity)}]"
            if len(m.wikidata_links) > 0:
                s += ": "
                # for wqid, score in m.disambiguation_scores().items()[:2]:
                for wqid, score in m.cross_cand_score(len=2).items():
                    s += f"({wqid}, {score}), "
                if len(m.wikidata_links) > 2:
                    s += "..."
                else:
                    s = s[:-2]
        return s
    
    def is_empty(self) -> bool:
        return len(self.links) == 0 or self.links[0].is_empty()
    
    # Returns the CandidateLinks instance with the given spelling 
    # variation, or None if no such match exists.
    def get(self, variation: str):
        for m in self.links:
            if m.string_match.variation == variation:
                return m
        return None
    
    # TODO: rename this as `best_candidate` (and it's understood this means the best 
    # StringMatch with associated candidate WikidataLink instances).
    # Returns the CandidateLinks instance whose StringMatch has the highest string similarity.
    def best_match(self) -> CandidateLinks:
        if self.is_empty():
            return None
        # The list of CandidateLinks instances is ordered by decreasing string similarity.
        return self.links[0]

    # Returns the Wikidata link with the highest disambiguation score.
    def best_wikidata_link(self) -> 'WikidataLink':
        # Get the CandidateLinks instance with highest string similarity.
        best_match = self.best_match()
        if not best_match or best_match.is_empty():
            return None
        if not isinstance(best_match, CandidateLinks):
            raise ValueError("Expected CandidateLinks instance.")
        return best_match.best_wikidata_link()

    def best_wqid(self) -> str:
        best_wikidata_link = self.best_wikidata_link()
        if not best_wikidata_link:
            return None
        return best_wikidata_link.wqid

    # TODO:
    # def best_disambiguation_score(self):
    #     best_match = self.best_match()
    #     if not best_match:
    #         return None
    #     return best_match....

