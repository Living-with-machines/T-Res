"""
The `t_res.utils.dataclasses` module defines all data structures used within the T-Res pipeline, 
implemented as Python dataclasses.
"""

from typing import List, Dict, Tuple, Optional
from pydantic.dataclasses import dataclass as pdataclass
from dataclasses import field

from sentence_splitter import SentenceSplitter

################################
# Dataclasses for Recogniser
################################

@pdataclass(order=True, frozen=True)
class Mention:
    """Dataclass representing a toponym mention in text.
    
    Attributes:
        mention (str): The toponym mention.
        start_offset (int): The token offset inside the text marking the start of the mention.
        end_offset (int): The token offset inside the text marking the end of the mention.
        start_char (int): The character offset inside the text marking the start of the mention.
        ner_score (float): The NER confidence score.
        ner_label (float): The NER label of the mention.
        entity_link (str): The consolidated entity link of the mention ('O' for predicted mentions).
    """
    sort_index: int = field(init=False)
    mention: str
    start_offset: int
    end_offset: int
    start_char: int
    ner_score: float
    ner_label: str
    entity_link: str

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.start_char)

    def __str__(self, pad_mention: int=0, pad_label: int=0):
        s = f"{self.mention.ljust(pad_mention)} {self.ner_label.ljust(pad_label)}"
        s += f" chars: {self.start_char}-{self.end_char()}"
        s += f" confidence: {self.ner_score}"
        return s

    def from_dict(data: dict) -> 'Mention':
        """Constructs a `Mention` instance from a dictionary."""
        if 'sort_index' in data.keys():
            del data['sort_index']
        return Mention(**data)
    
    def end_char(self) -> int:
        """Returns the character offset inside the text marking the end of the mention."""
        return self.start_char + len(self.mention)
    
    def is_microtoponym(self) -> bool:
        """Returns `True` if the `ner_label` is not `LOC`, indicating a microtoponym."""
        return self.ner_label != "LOC"

# Helper class for backwards compatibility with training functions in rel_utils.py
@pdataclass(frozen=True)
class TrainingMention(Mention):
    """Helper class providing backwards compatibility with training functions in `rel_utils.py`.
    
    Attributes:
        gold (str): The Wikidata ID of the known ("gold standard") toponym, or 'NIL' if not known.
    """
    gold: str

    def from_dict(data: dict) -> 'TrainingMention':
        """Constructs a `TrainingMention` instance from a dictionary."""
        if isinstance(data['gold'], list) and len(data['gold']) != 1:
            raise ValueError(f"Multiple gold standard toponymn IDs: {data['gold']}")
        if 'tag' in data.keys() and 'ner_label' not in data.keys():
            data['ner_label'] = data['tag']
        return TrainingMention(
            mention=data['mention'],
            start_offset=-1,
            end_offset=-1,
            start_char=data['pos'],
            ner_score=-1.0,
            ner_label=data['ner_label'],
            entity_link='',
            gold=data['gold'][0] if isinstance(data['gold'], list) else data['gold'],
        )

@pdataclass(frozen=True)
class Sentence:
    """Dataclass representing a sentence.
    
    Attributes:
        sentence (str): The sentence.
    """
    sentence: str

    def __len__(self):
        return len(self.sentence)

@pdataclass(frozen=True)
class SentenceContext(Sentence):
    """Dataclass representing a sentence with (optional) context.
    
    Attributes:
        preceding_sentence (Optional[str]): The preceding sentence (context).
        following_sentence (Optional[str]): The following sentence (context).
        sent_idx (Optional[int]): The sentence index (within a block of text). Defaults to None.
    """
    preceding_sentence: Optional[str]
    following_sentence: Optional[str]
    sent_idx: Optional[int]=None

    def from_text(text: str, language: str="en", non_breaking_prefix_file: str=None) -> List['SentenceContext']:
        """Constructs a list of `SentenceContext` instances from a block of text."""
        splitter = SentenceSplitter(language=language, non_breaking_prefix_file=non_breaking_prefix_file)
        sentences = splitter.split(text)
        return [SentenceContext(s, sentences[i - 1] if i > 0 else None, 
                                sentences[i + 1] if i < len(sentences) - 1 else None) 
                                for i, s in enumerate(sentences)]
    
    def from_sentence(sentence: str) -> 'SentenceContext':
        """Constructs a `SentenceContext` instance from a string."""
        return SentenceContext(sentence, None, None)
    
    # Helper method for the Predictions as_dict method.
    def context_as_list(self) -> List[str]:
        """Converts this instance to a list of strings."""
        preceding = self.preceding_sentence if self.preceding_sentence is not None else ''
        following = self.following_sentence if self.following_sentence is not None else ''
        return [preceding, following]

    # For API deserialisation.
    def from_dict(data: dict) -> Sentence:
        """Constructs a `SentenceContext` instance from a dictionary."""
        ps = data['preceding_sentence'] if 'preceding_sentence' in data.keys() else None
        fs = data['following_sentence'] if 'following_sentence' in data.keys() else None
        sent_idx = data['sent_idx'] if 'sent_idx' in data.keys() else None
        if ps or fs or sent_idx:
            return SentenceContext(data['sentence'], ps, fs, sent_idx)
        return Sentence(data['sentence'])

# Recogniser::run method output type.
@pdataclass(frozen=True)
class SentenceMentions:
    """Dataclass representing toponym mentions in a sentence.
    
    Attributes:
        sentence (Sentence): The sentence.
        mentions (List[Mention]): A list of toponym mentions, ordered by character offset within the sentence.
    """
    sentence: Sentence
    mentions: List[Mention]

    def __post_init__(self):
        if self.is_empty():
            return
        if max([m.end_char() for m in self.mentions]) > len(self.sentence):
            raise ValueError("Max end char exceeds sentence length.")

    def __str__(self):
        s = f"Toponym mentions for sentence: '{self.sentence.sentence}'"
        if self.is_empty():
            s += "\n    None"
            return s
        pad_mention = max([len(m.mention) for m in self.mentions])
        pad_label = max([len(m.ner_label) for m in self.mentions])
        for m in self.mentions:
            s += f"\n    {m.__str__(pad_mention, pad_label)}"
        return s

    def is_empty(self) -> bool:
        """Returns `True` if the list of toponym mentions is empty."""
        return len(self.mentions) == 0

    def len(self) -> int:
        """Returns the length of the list of toponym mentions."""
        return len(self.mentions)
    
    def exclude_microtoponyms(self) -> 'SentenceMentions':
        """Returns this `SentenceMentions` instance omitting any microtoponym mentions."""
        mentions = list(filter(lambda m: not m.is_microtoponym(), self.mentions))
        return SentenceMentions(self.sentence, mentions)
    
    # Helper method for backwards compatibility with training functions in `rel_utils.py`.
    def from_list(data: List[Dict]) -> 'SentenceMentions':
        """Constructs a `SentenceMentions` instance from a list of dictionaries.
        
        Helper method for backwards compatibility with training functions in `rel_utils.py`.
        """
        # The data are assumed to be in the format returned by the 
        # `prepare_initial_data` method in `rel_utils.py`.
        mentions = [TrainingMention.from_dict(d) for d in data]
        # Check that all sentences in the list are identical.
        if {d['sentence'] for d in data} != {data[0]['sentence']}:
            raise ValueError("Inconsistent sentences.")
        d = data[0]
        context = SentenceContext(d['sentence'], d['context'][0], d['context'][1], d['sent_idx'])
        return SentenceMentions(context, mentions)

    # For API deserialisation.
    def from_dict(data: dict) -> 'SentenceMentions':
        """Constructs a `SentenceMentions` instance from a dictionary."""
        return SentenceMentions(
            sentence=SentenceContext.from_dict(data['sentence']),
            mentions=[Mention.from_dict(d) for d in data['mentions']],
            )

    # For API deserialisation.
    def from_json(data: List[Dict]) -> List['SentenceMentions']:
        """Constructs a list of `SentenceMentions` instances from a list of dictionaries."""
        return [SentenceMentions.from_dict(d) for d in data]


################################
# Dataclasses for Ranker
################################

@pdataclass(order=True, frozen=True)
class StringMatch:
    """Dataclass representing a potential toponym string match.
    
    Attributes:
        variation (str): The toponym spelling variation.
        string_similarity (float): String matching similarly score.
    """
    sort_index: float = field(init=False)
    variation: str
    string_similarity: float

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_similarity)

    # For API deserialisation.
    def from_dict(data: dict) -> 'StringMatch':
        """Constructs a `StringMatch` instance from a dictionary."""
        if 'sort_index' in data.keys():
            del data['sort_index']
        if 'wqid_links' in data.keys():
            return StringMatchLinks(**data)
        return StringMatch(**data)

@pdataclass(order=True, frozen=True)
class StringMatchLinks(StringMatch):
    """Dataclass representing a potential toponym string match 
    with potential Wikidata ID links.
    
    Attributes:
        wqid_links (List[str]): List of potential Wikidata ID links.
    """
    wqid_links: List[str]

    def as_string_match(self) -> StringMatch:
        """Converts this `StringMatchLinks` instance into a `StringMatch` instance."""
        return StringMatch(self.variation, self.string_similarity)

# Ranker::run method output type.
@pdataclass(frozen=True)
class CandidateMatches:
    """Dataclass representing candidate matches for a toponym.
    
    Attributes:
        mention (Mention): The toponym mention in the text.
        ranking_method (str): The string matching method used.
        matches (List[StringMatchLinks]): A list of potential toponym matches, each with potential Wikidata links.
    """
    mention: Mention
    ranking_method: str
    matches: List[StringMatchLinks]

    def __post_init__(self):
        # Check that the variations are unique in self.matches.
        variations = [match.variation for match in self.matches]
        if len(variations) != len(set(variations)):
            raise ValueError("StringMatch variations must be unique.")
        # Order matches by decreasing string similarity.
        object.__setattr__(self, 'matches', sorted(self.matches, reverse=True))

    def is_empty(self) -> bool:
        """Returns `True` if the list of toponym matches is empty."""
        return len(self.matches) == 0

    def get(self, variation: str) -> StringMatchLinks:
        """Returns the StringMatch instance with the given spelling variation 
        or None if no such match exists."""
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
    """Dataclass representing a potential toponym link in Wikidata.
    
    Attributes:
        wqid (str): The Wikidata ID.
        wkdt_class (Optional[str]): The Wikidata class of this Wikidata entry (if available).
        coords (Optional[Tuple[float, float]]): The lat-lon coordinates of the link in Wikidata.
    """
    wqid: str
    wkdt_class: Optional[str]
    coords: Optional[Tuple[float, float]]

    # For API deserialisation.
    def from_dict(data: dict) -> 'WikidataLink':
        """Constructs a `WikidataLink` instance from a dictionary."""
        if 'freq' in data.keys():
            if 'normalized_score' in data.keys():
                return RelDisambLink(**data)
            return MostPopularLink(**data)
        return ByDistanceLink(**data)
    
@pdataclass(frozen=True)
class MostPopularLink(WikidataLink):
    """Dataclass representing a string match and potential links in 
    Wikidata under the `mostpopular` linking method.
    
    Attributes:
        freq (int): The mention-to-wikidata link frequency.
    """
    freq: int

    def __post_init__(self):
        if not isinstance(self.freq, int):
            raise ValueError("freq must be an integer.")

@pdataclass(frozen=True)
class ByDistanceLink(WikidataLink):
    """Dataclass representing a string match and potential links in 
    Wikidata under the `bydistance` linking method.
    
    Attributes:
        place_of_pub_coords (Optional[Tuple[float, float]]): The lat-lon coordinates of the place of publication.
        geodist (Optional[float]): The geodesic distance between the wqid and the origin wqid.
        normalized_score (float): The normalized score from resource `mentions_to_wikidata_normalized.json`.
    """
    place_of_pub_coords: Optional[Tuple[float, float]]
    geodist: Optional[float]
    normalized_score: float

    def __post_init__(self):
        if not isinstance(self.normalized_score, float):
            raise ValueError("normalized_score must be an float.")

@pdataclass(frozen=True)
class RelDisambLink(MostPopularLink):
    """Dataclass representing a string match and potential links in 
    Wikidata under the `reldisamb` linking method.
    
    Attributes:
        normalized_score (float): The normalized score from resource `mentions_to_wikidata_normalized.json`.
    """
    normalized_score: float

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.normalized_score, float):
            raise ValueError("normalized_score must be an float.")
        
@pdataclass(order=True, frozen=True)
class CandidateLinks:
    """Dataclass representing a collection of potential links in Wikidata for a given string match.
    
    Attributes:
        string_match (StringMatch): A StringMatch instance.
        wikidata_links (List[WikidataLink]): A list of candidate WikidataLink instances.
    """
    sort_index: float = field(init=False)
    string_match: StringMatch
    wikidata_links: List[WikidataLink]

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_match.string_similarity)

    # One liner.
    def __str__(self, pad_variation: int=0) -> str:
        s = f"{self.string_match.variation.ljust(pad_variation)}"
        s += f" [{'{:.3f}'.format(self.string_match.string_similarity)}]"
        s += f": {self.links_str()}"
        return s
    
    def links_str(self) -> str:
        """Returns a string representation of the list of Wikidata links (for pretty-printing)."""
        if self.is_empty():
            return "None"
        s = ', '.join(link.wqid for link in self.wikidata_links[:3])
        if len(self.wikidata_links) > 3:
            s += ", ..."
        return s

    def is_empty(self) -> bool:
        """Returns `True` if the list of Wikidata links is empty."""
        return len(self.wikidata_links) == 0

    def attach_scores(self, scores: Dict[str, float]) -> 'PredictedLinks':
        """Transforms this CandidateLinks instance into a PredictedLinks instance 
        by attaching disambiguation scores."""
        # Check that there is one score for each link.
        if scores.keys() != {link.wqid for link in self.wikidata_links}:
            raise ValueError("Incompatible disambiguation scores.")
        return PredictedLinks(self.string_match, self.wikidata_links, scores)

    # For API deserialisation.
    def from_dict(data: dict) -> 'CandidateLinks':
        """Constructs a `CandidateLinks` instance from a dictionary."""
        if 'disambiguation_scores' in data.keys():
            return PredictedLinks(
                string_match=StringMatch.from_dict(data['string_match']),
                wikidata_links=[WikidataLink.from_dict(d) for d in data['wikidata_links']],
                disambiguation_scores=data['disambiguation_scores'],
            )
        return CandidateLinks(
            string_match=StringMatch.from_dict(data['string_match']),
            wikidata_links=[WikidataLink.from_dict(d) for d in data['wikidata_links']],
        )

# Extend CandidateLinks to include disambigution scores. Note that we use 
# inheritance, rather than composition, for compatibility with the `links`
# field in the Candidates dataclass.
@pdataclass(order=True, frozen=True)
class PredictedLinks(CandidateLinks):
    """Dataclass representing a collection of potential links in Wikidata with scores for each.
    
    Attributes:
        disambiguation_scores (Dict[str, float]): A disambiguation score for each potential link in Wikidata.
    """
    disambiguation_scores: Dict[str, float]

    def links_str(self) -> str:
        """(Override) Returns a string representation of the list of Wikidata links (for pretty-printing)."""
        if self.is_empty():
            return "None"
        l = [f"{s} ({v})" for s, v in self.cross_cand_scores().items()]
        s = ', '.join(l[:3])
        if len(self.wikidata_links) > 3:
            s += ", ..."
        return s
    
    def best_disambiguation_score(self) -> float:
        """Returns the greatest disambiguation score."""
        if self.is_empty():
            return None
        return max(self.disambiguation_scores.values())
    
    # TODO: use min(self.wikidata_links, key=lambda link: link....) if poss.
    def best_wikidata_link(self) -> WikidataLink:
        """Returns the Wikidata link with the greatest disambiguation score."""
        if self.is_empty():
            return None
        for link in self.wikidata_links:
            if link.wqid == self.best_wqid():
                return link
    
    def best_wqid(self) -> float:
        """Returns the Wikidata ID of the link with the greatest disambiguation score."""
        if self.is_empty():
            return None
        scores = self.disambiguation_scores
        return max(scores, key=lambda key: scores[key])

    def cross_cand_scores(self, len=7) -> dict:
        """Returns the top 7 Wikidata links in order of their disambiguation score 
        (providing backwards compatibility with T-Res pipeline output in previous versions)."""
        scores = {k: round(v, 3) for (k, v) in self.disambiguation_scores.items()}
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:len])
    
    # Helper method for the Predictions as_dict method.
    def scores_as_list(self) -> list:
        """Returns the disambiguation scores as a list.

        Helper method for the Predictions as_dict method."""
        ret = [[k, round(v, 3)] for k, v in self.disambiguation_scores.items()]
        return sorted(ret, key=lambda x: (x[1], x[0]), reverse=True)
    
# Linker::run method output type.
@pdataclass(order=True, frozen=True)
class MentionCandidates:
    """Dataclass representing candidate string matches for a toponym, 
    each with candidate Wikidata links.
    
    Attributes:
        mention (Mention): The toponym mention in the text.
        ranking_method (str): The string matching method used.
        linking_method (str): The linking method used.
        links (List[CandidateLinks]): A list of CandidateLinks instances, ordered by decreasing string similarity.
        place_of_pub_wqid (Optional[str]): Place of publication Wikidata ID.
        place_of_pub (Optional[str]): Place of publication.
    """
    sort_index: float = field(init=False)
    mention: Mention
    ranking_method: str
    linking_method: str
    links: List[CandidateLinks]
    place_of_pub_wqid: Optional[str]
    place_of_pub: Optional[str]

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.mention.start_char)
        # Check that the variations are unique in self.links.
        variations = [m.string_match.variation for m in self.links]
        if len(variations) != len(set(variations)):
            raise ValueError("StringMatch variations must be unique.")
        object.__setattr__(self, 'links', sorted(self.links, reverse=True))
        if self.place_of_pub_wqid:
            if self.place_of_pub_wqid[0] != "Q":
                raise ValueError(f"Invalid Wikidata ID: {self.place_of_pub_wqid}")

    def __str__(self) -> str:
        s = f"Candidates for toponym mention: '{self.mention.mention}':"
        if self.is_empty():
            s += "\n    None"
            return s
        pad_variation = max([len(l.string_match.variation) for l in self.links])
        for link in self.links:
            if link.is_empty():
                continue
            s += f"\n    {link.__str__(pad_variation)}"
        return s
    
    def is_empty(self) -> bool:
        """Returns `True` if the list of `CandidateLinks` is empty *or* the 
        `CandidateLinks` instance with the best string match is empty."""
        return len(self.links) == 0 or self.links[0].is_empty()
    
    def get(self, variation: str) -> Optional[CandidateLinks]:
        """Returns the CandidateLinks instance with the given spelling variation, 
        or None if no such match exists."""
        for m in self.links:
            if m.string_match.variation == variation:
                return m
        return None
    
    def best_match(self) -> Optional[CandidateLinks]:
        """Returns the CandidateLinks instance whose StringMatch has the highest string similarity,
        or None if no such match exists."""
        if self.is_empty():
            return None
        # The list of CandidateLinks instances is ordered by decreasing string similarity.
        return self.links[0]
    
    def best_string_match(self) -> Optional[StringMatch]:
        """Returns the StringMatch instance with the highest string similarity.
        or None if no such match exists."""
        if self.is_empty():
            return None
        return self.best_match().string_match

    def best_wikidata_link(self) -> Optional[WikidataLink]:
        """Returns the Wikidata link with the highest disambiguation score, associated with 
        the best string match candidate, or None if no such match exists."""
        # Get the candidate with highest string similarity.
        best_match = self.best_match()
        if not best_match or best_match.is_empty():
            return None
        if not isinstance(best_match, PredictedLinks):
            raise ValueError(f"Expected PredictedLinks instance. Got {type(best_match)}")
        return best_match.best_wikidata_link()

    def best_wqid(self) -> Optional[str]:
        """Returns the Wikidata ID of the best Wikidata Link, or None if no best link exists."""
        best_wikidata_link = self.best_wikidata_link()
        if not best_wikidata_link:
            return None
        return best_wikidata_link.wqid

    def best_coords(self) -> Optional[Tuple[float, float]]:
        """Returns the lat-long coordinates of the best Wikidata Link, or None if no best link exists."""
        best_wikidata_link = self.best_wikidata_link()
        if not best_wikidata_link:
            return None
        return best_wikidata_link.coords

    def best_disambiguation_score(self) -> Optional[float]:
        """Returns the disambiguation score of the best match, or None if no such match exists."""
        best_match = self.best_match()
        if not best_match or best_match.is_empty():
            return None
        if not isinstance(best_match, PredictedLinks):
            return None
        return best_match.best_disambiguation_score()

    # For API deserialisation.
    def from_dict(data: dict) -> 'MentionCandidates':
        """Constructs a `MentionCandidates` instance from a dictionary."""
        place_of_pub_wqid=data['place_of_pub_wqid'] if 'place_of_pub_wqid' in data.keys() and len(data['place_of_pub_wqid']) > 0 else None
        place_of_pub=data['place_of_pub'] if 'place_of_pub' in data.keys() and len(data['place_of_pub']) > 0 else None
        return MentionCandidates(
            mention=Mention.from_dict(data['mention']),
            ranking_method=data['ranking_method'],
            linking_method=data['linking_method'],
            links=[CandidateLinks.from_dict(d) for d in data['links']],
            place_of_pub_wqid=place_of_pub_wqid,
            place_of_pub=place_of_pub,
        )

################################
# Dataclasses for Pipeline
################################

@pdataclass(frozen=True)
class SentenceCandidates:
    """Dataclass representing candidate matches for all toponym mentions in a sentence.
    
    Attributes:
        sentence (Sentence): The sentence.
        candidates (List[MentionCandidates]): List of candidates for each toponym mention in the sentence.
    """
    sentence: Sentence
    candidates: List[MentionCandidates]

    def __post_init__(self):
        if self.is_empty():
            return
        if max([cs.mention.end_char() for cs in self.candidates]) > len(self.sentence):
            raise ValueError("Inconsistent candidate mentions. Max end char exceeds sentence length.")

    def is_empty(self, ignore_empty_candidates: bool=True) -> bool:
        """Returns `True` if the list of `MentionCandidates` is empty. 
        If `ignore_empty_candidates` is `True`, only non-empty candidates are considered."""
        if ignore_empty_candidates:
            return len(self.candidates) == 0 or all([c.is_empty() for c in self.candidates])
        return len(self.candidates) == 0
    
    def remove_microtoponyms(self):
        """Removes any `MentionCandidates` instances in the `candidates` list that
        refer to a microtoponym mention."""
        indices = [i for i, c in enumerate(self.candidates) if c.mention.is_microtoponym()]
        if not indices:
            return self
        indices.sort(reverse=True)
        for i in indices:
            del self.candidates[i]

    # For API deserialisation.
    def from_dict(data: dict) -> 'SentenceCandidates':
        """Constructs a `SentenceCandidates` instance from a dictionary."""
        return SentenceCandidates(
            sentence=SentenceContext.from_dict(data['sentence']),
            candidates=[MentionCandidates.from_dict(d) for d in data['candidates']]
        )

# Pipeline::run_candidate_selection method output type.
@pdataclass(frozen=True)
class Candidates:
    """Dataclass representing candidate matches for all toponym mentions 
    in a block of text.
    
    Attributes:
        sentence_candidates (List[SentenceCandidates]): List of setence candidates for each sentence in the text.
    """
    sentence_candidates: List[SentenceCandidates]

    def __post_init__(self):
        if self.is_empty():
            return
        # Check that all place of publication data is consistent.
        if {self.place_of_pub_wqid()} != {c.place_of_pub_wqid for c in self.candidates()}:
            raise ValueError("Inconsistent place of publication Wikidata IDs.")
        if {self.place_of_pub()} != {c.place_of_pub for c in self.candidates()}:
            raise ValueError("Inconsistent place of publication data.")

    def __str__(self):
        split = self.text().split(' ')
        s = f"{type(self).__name__} for text: '{' '.join(split[:3])}...{' '.join(split[-3:])}':"
        if self.is_empty():
            s += "\n    None"
            return s
        mention_candidates = self.candidates(ignore_empty_candidates = False)
        def len_variation(c: MentionCandidates) -> int:
            if c.best_match():
                return len(c.best_match().string_match.variation)
            return 0
        pad_mention = max([len(c.mention.mention) for c in mention_candidates])
        pad_variation = max([len_variation(c) for c in mention_candidates])
        for c in mention_candidates:
            s += f"\n    {self.candidates_str(c, pad_mention, pad_variation)}"
        return s
    
    def candidates_str(self, candidates: MentionCandidates, pad_mention: int=0, pad_variation: int=0) -> str:
        """Returns a string representation of a `MentionCandidates` instance (for pretty-printing)."""
        s = f"{candidates.mention.mention.ljust(pad_mention)} => "
        if candidates.best_match():
            s += f"{candidates.best_match().__str__(pad_variation)}"
        else:
            s += f"None"
        return s

    def candidates(self, ignore_empty_candidates: bool=True) -> List[MentionCandidates]:
        """Returns all `MentionCandidates` as a list. If `ignore_empty_candidates` is `True`, 
        only non-empty candidates are considered."""
        if ignore_empty_candidates:
            return [c for sc in self.sentence_candidates for c in sc.candidates if not c.is_empty()]
        return [c for sc in self.sentence_candidates for c in sc.candidates]

    def sentences(self, ignore_empty_candidates: bool=True) -> List[str]:
        """Returns the sentence corresponding to each `MentionCandidates` instance that
        is returned by the `candidates` method."""
        if ignore_empty_candidates:
            return [(sc.sentence.sentence, c)[0] for sc in self.sentence_candidates 
                    for c in sc.candidates if not c.is_empty()]
        return [sc.sentence.sentence for sc in self.sentence_candidates]

    def is_empty(self, ignore_empty_candidates: bool=True) -> bool:
        """Returns `True` if the list of `SentenceCandidates` instances is empty. 
        If `ignore_empty_candidates` is `True`, only non-empty candidates are considered."""
        return len(self.candidates(ignore_empty_candidates)) == 0
    
    def text(self) -> str:
        """Returns the complete text."""
        return " ".join([scs.sentence.sentence for scs in self.sentence_candidates])
    
    # TODO: unit test needed.
    def sentence_contexts(self) -> List[SentenceContext]:
        """Returns a list of `SentenceContext` instances."""
        scs = self.sentence_candidates
        return [SentenceContext(sc.sentence.sentence, 
                         scs[i - 1].sentence.sentence if i > 0 else None, 
                         scs[i + 1].sentence.sentence if i < len(scs) - 1 else None) 
         for i, sc in enumerate(scs)]

    def place_of_pub_wqid(self) -> Optional[str]:
        """Returns the place of publication Wikidata ID, if available."""
        if self.is_empty(ignore_empty_candidates=False):
            return None
        return self.candidates(ignore_empty_candidates=False)[0].place_of_pub_wqid

    def place_of_pub(self) -> Optional[str]:
        """Returns the place of publication, if available."""
        if self.is_empty(ignore_empty_candidates=False):
            return None
        return self.candidates(ignore_empty_candidates=False)[0].place_of_pub

    # For API deserialisation.
    def from_dict(data: dict) -> 'Candidates':
        """Constructs a `Candidates` instance from a dictionary."""
        sentence_candidates = [SentenceCandidates.from_dict(d) for d in data['sentence_candidates']]
        is_predicted_links = [isinstance(links, PredictedLinks) 
                              for scs in sentence_candidates 
                              for mc in scs.candidates 
                              for links in mc.links]
        if any(is_predicted_links):
            return Predictions(sentence_candidates)
        return Candidates(sentence_candidates)

# Pipeline::run_disambiguation method output type.
@pdataclass(frozen=True)
class Predictions(Candidates):
    """Dataclass representing toponym predictions in text."""

    def __post_init__(self):
        super().__post_init__()
        for c in self.candidates():
            if not all([isinstance(links, PredictedLinks) for links in c.links]):
                raise ValueError("Candidate links must be scored.")

    def best_wqids(self) -> List[Optional[str]]:
        """Returns a list of predicted Wikidata IDs (one per toponym mention)."""
        return [c.best_wqid() for c in self.candidates()]

    def best_coords(self) -> List[Optional[Tuple[float, float]]]:
        """Returns a list of predicted lat-long coordinates (one per toponym mention)."""
        return [c.best_coords() for c in self.candidates()]
    
    def best_disambiguation_scores(self) -> List[Optional[float]]:
        """Returns a list of greatest disambiguation scores (one per toponym mention)."""
        return [c.best_disambiguation_score() for c in self.candidates()]

    def apply_rel_disambiguation(
            self, 
            rel_predictions: dict,
            with_publication: bool) -> 'RelPredictions':
        """Incorporates predictions generated by the REL disambiguation method and
        returns an instance of the `RelPredictions` subclass."""

        if not rel_predictions:
            return RelPredictions(self.sentence_candidates, list())

        # If with_publication is True, drop the "artificial" final toponym mention.
        if with_publication and not self.is_empty(ignore_empty_candidates=False):
            del rel_predictions["linking"][-1]

        # Incoroporate the REL model predictions.
        rel_scores = [RelScores(
            mention=d["mention"],
            scores={wqid: score for wqid, score in zip(d["candidates"], d["scores"])},
            confidence=d["conf_ed"]) for d in rel_predictions["linking"]]

        return RelPredictions(self.sentence_candidates, rel_scores)

    def place_of_pub_mention(self) -> dict:
        """Returns a dictionary containing a toponym mention for the place of publication.
        
        Helper method for backward compatibility with training functions in `rel_utils.py`.
        """
        place_of_pub = self.place_of_pub()
        place_of_pub_wqid = self.place_of_pub_wqid()
        if not place_of_pub or not place_of_pub_wqid:
            raise ValueError("Missing place of publication info.")
        prefix = "This article is published in "
        place_of_pub_sentence = f"{prefix}{place_of_pub}."
        # NOTE: this dict is slightly inconsistent versus the mention_dicts 
        # constructed in the as_dict method:
        # - "ner_score" and "conf_md" are missing
        # - "tag" is instead named "ner_label"
        # These inconsistencies are preserved from an earlier version and perhaps
        # should be fixed in future. Note that this format *is* consistent with
        # the keys in the TrainingPredictions.as_list() method.
        return {
            "mention": place_of_pub,
            "sent_idx": 0,
            "sentence": place_of_pub_sentence,
            "gold": [place_of_pub_wqid],
            "ngram": place_of_pub,
            "context": ["", ""],
            "pos": len(prefix),
            "end_pos": len(prefix) + len(place_of_pub),
            "candidates": [[place_of_pub_wqid, 1.0]],
            "place": place_of_pub,
            "place_wqid": place_of_pub_wqid,
            "ner_label": "LOC",
        }

    # Converts to a dictionary for backwards compatibility with entity_disambiguation.py
    # (similar to the deprecated `format_prediction` method in pipeline.py)
    def as_dict(self, with_publication: bool) -> dict:
        """Converts to a dictionary for backwards compatibility with `entity_disambiguation.py`."""
        d = dict()
        d["linking"] = []
        contexts = self.sentence_contexts()
        for i, sc in enumerate(self.sentence_candidates):
            for c in sc.candidates:
                if c.is_empty():
                    candidates = []
                else:
                    candidates = c.best_match().scores_as_list()
                mention_dict = {
                    "mention": c.mention.mention,
                    "context": contexts[i].context_as_list(),
                    "candidates": candidates,
                    "gold": ["NONE"],
                    "ner_score": c.mention.ner_score,
                    "pos": c.mention.start_char,
                    "sent_idx": i,
                    "end_pos": c.mention.end_char(),
                    "ngram": c.mention.mention,
                    "conf_md": c.mention.ner_score,
                    "tag": c.mention.ner_label,
                    "sentence": sc.sentence.sentence,
                    "place": c.place_of_pub,
                    "place_wqid": c.place_of_pub_wqid,
                    # TODO: Do we need to include `string_match_candidates`?  It's not used 
                    # in `entity_disambiguation.py` and entails repetition of the wikidata links:
                    # "string_match_candidates": [link.string_match for link in self.links], 
                }
                d["linking"].append(mention_dict)

        # Append a mention for the place of publication, unless this instance
        # has no sentence candidates, in which case it lacks place of publication info.
        # (NB: Replaces add_publication from rel_utils.py):
        if with_publication and not self.is_empty(ignore_empty_candidates=False):
            d["linking"].append(self.place_of_pub_mention())

        return d
    
    def summary_dict(self) -> List[dict]:
        l = list()
        for c, s in zip(self.candidates(ignore_empty_candidates=True), 
                        self.sentences(ignore_empty_candidates=True)):
            disambiguation_score = c.best_disambiguation_score()
            if disambiguation_score:
                disambiguation_score = round(disambiguation_score, 3)
            d = {
                'mention': c.mention.mention,
                'sentence': s,
                'start_char': c.mention.start_char,
                'end_char': c.mention.end_char(),
                'ner_label': c.mention.ner_label,
                'ner_score': c.mention.ner_score,
                'prediction': c.best_wqid(),
                'toponym_match': c.best_string_match().variation,
                'string_similarity': c.best_string_match().string_similarity,
                'disambiguation_score': disambiguation_score,
            }
            l.append(d)
        return l

@pdataclass(frozen=True)
class TrainingPredictions(Predictions):
    """Dataclass representing toponym predictions for training a REL model."""

    def __post_init__(self):
        super().__post_init__()

    # Similar to the as_dict method in Predictions, but now for backward 
    # compatibility with the `prepare_rel_trainset` function in `rel_utils.py`.
    def as_list(self, with_publication: bool) -> List[dict]:
        """Converts to a list of dictionaries. 
        
        Helper method for backwards compatibility with training functions in `rel_utils.py`."""
        l = list()
        for sc in self.sentence_candidates:
            if not isinstance(sc.sentence, SentenceContext):
                raise ValueError(f"Expected SentenceContext instance. Got: {type(sc)}")
            for c in sc.candidates:
                if not isinstance(c.mention, TrainingMention):
                    raise ValueError(f"Expected TrainingMention instance. Got: {type(c)}")
                if c.is_empty():
                    candidates = []
                else:
                    candidates = c.best_match().scores_as_list()
                mention_dict = {
                    "mention": c.mention.mention,
                    "sent_idx": sc.sentence.sent_idx,
                    "sentence": sc.sentence.sentence,
                    "ngram": c.mention.mention,
                    "context": sc.sentence.context_as_list(),
                    "pos": c.mention.start_char,
                    "end_pos": c.mention.end_char(),
                    "place": c.place_of_pub,
                    "place_wqid": c.place_of_pub_wqid,
                    "candidates": candidates,
                    "ner_label": c.mention.ner_label,
                    "gold": [c.mention.gold] if c.mention.gold != 'NIL' else 'NIL',
                }
                l.append(mention_dict)

        # Append a mention for the place of publication, unless this instance
        # has no sentence candidates, in which case it lacks place of publication info.
        # (NB: Replaces add_publication from rel_utils.py):
        if with_publication and not self.is_empty(ignore_empty_candidates=False):
            l.append(self.place_of_pub_mention())
        return l

@pdataclass(frozen=True)
class RelScores:
    """Dataclass representing scores produced by the REL entity disambiguation model.
    
    Attributes:
        mention (str): The toponym mention.
        scores (Dict[str, float]): REL entity disambiguation scores.
        confidence (float): REL entity disambiguation confidence score.
    """
    mention: str
    scores: Dict[str, float]
    confidence: float

@pdataclass(frozen=True)
class RelPredictions(Predictions):
    """Dataclass representing toponym predictions in text produced by REL entity disambiguation.
    
    Attributes:
        rel_scores (List[RelScores]): A list of REL entity disambiguation scores.
    """
    rel_scores: List[RelScores]

    def __post_init__(self):
        count_candidates = len(super().candidates(ignore_empty_candidates=False))
        if len(self.rel_scores) != count_candidates:
            raise ValueError(f"""Expected one RelScores instance per linked toponym mention.
                             Got {len(self.rel_scores)} instances and {count_candidates} mentions.""")

    # Override the candidates method to return REL linking predictions.
    def candidates(self, ignore_empty_candidates: bool=True) -> List[MentionCandidates]:
        """(Override) Returns all `MentionCandidates` as a list, with REL disambiguation 
        scores determining the predicted Wikidata links. If `ignore_empty_candidates` is `True`, 
        only non-empty candidates are considered."""

        # Construct equivalent Candidate instances but with the REL scores in the PredictedLinks.
        ret = list()
        for c, rs in zip(super().candidates(ignore_empty_candidates=False), self.rel_scores):

            # Check that the mention in the RelScores instance matches that in the candidate.
            if rs.mention != c.mention.mention:
                raise ValueError(f"Inconsistent toponym mentions in RelScores ({rs.mention}) and candidate ({c.mention.mention})")

            if c.is_empty():
                if not ignore_empty_candidates:
                    ret.append(c)
                continue

            predicted_links = c.best_match()
            
            # Get the list of WikidataLink instances for which REL scores are available.
            wikidata_links = [wl for wl in predicted_links.wikidata_links if wl.wqid in rs.scores.keys()]
            links = [PredictedLinks(predicted_links.string_match, wikidata_links, rs.scores)]
            
            ret.append(MentionCandidates(
                c.mention, 
                c.ranking_method, 
                c.linking_method, 
                links, 
                c.place_of_pub_wqid, 
                c.place_of_pub))
        return ret
    
    def interim_candidates(self, ignore_empty_candidates: bool=True) -> List[MentionCandidates]:
        """Returns the list of `MentionCandidates` instances with their interim disambiguation 
        scores, that is, the scores obtained before applying the REL disambiguation method."""
        return super().candidates(ignore_empty_candidates)
