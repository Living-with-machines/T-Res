from typing import List, Dict, Tuple, Optional
from pydantic.dataclasses import dataclass as pdataclass
from dataclasses import field, InitVar

from sentence_splitter import SentenceSplitter

# TODO NEXT: add __str__ methods and check them in the jupyter notebooks.

################################
# Dataclasses for Recogniser
################################

@pdataclass(order=True, frozen=True)
class Mention:
    """Data class representing a toponym mention in text."""
    sort_index: int = field(init=False)
    # The toponym mention.
    mention: str
    # The token offset inside the text marking the start of the mention.
    start_offset: int
    # The token offset inside the text marking the end of the mention.
    end_offset: int
    # The character offset inside the text marking the start of the mention.
    start_char: int
    # The NER confidence score.
    ner_score: float
    # The NER label of the mention.
    ner_label: str
    # The consolidated entity link of the mention ('O' for predicted mentions).
    entity_link: str

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.start_char)

    def from_dict(dict: dict) -> 'Mention':
        return Mention(
            mention=dict['mention'],
            start_offset=dict['start_offset'],
            end_offset=dict['end_offset'],
            start_char=dict['start_char'],
            ner_score=dict['ner_score'],
            ner_label=dict['ner_label'],
            entity_link=dict['entity_link'],
        )
    
    def end_char(self) -> int:
        return self.start_char + len(self.mention)
    
    def is_microtoponym(self) -> bool:
        # A microtoponym is any mention whose `ner_label` is not `LOC`.
        return self.ner_label != "LOC"

# Helper class for backwards compatibility with training functions in rel_utils.py
@pdataclass(frozen=True)
class TrainingMention(Mention):
    # The Wikidata ID of the known ("gold standard") toponym, or 'NIL' if not known.
    gold: str

    def from_dict(dict: dict) -> 'TrainingMention':
        if isinstance(dict['gold'], list) and len(dict['gold']) != 1:
            raise ValueError(f"Multiple gold standard toponymn IDs: {dict['gold']}")
        if 'tag' in dict.keys() and 'ner_label' not in dict.keys():
            dict['ner_label'] = dict['tag']
        return TrainingMention(
            mention=dict['mention'],
            start_offset=-1,
            end_offset=-1,
            start_char=dict['pos'],
            ner_score=-1.0,
            ner_label=dict['ner_label'],
            entity_link='',
            gold=dict['gold'][0] if isinstance(dict['gold'], list) else dict['gold'],
        )

@pdataclass(frozen=True)
class Sentence:
    # The sentence.
    sentence: str

    def __len__(self):
        return len(self.sentence)

@pdataclass(frozen=True)
class SentenceContext(Sentence):
    # The preceding sentence context.
    preceding_sentence: Optional[str]
    # The following sentence context.
    following_sentence: Optional[str]
    # Optional sentence index (within a block of text). Defaults to None.
    sent_idx: Optional[int]=None

    def from_text(text: str, language: str="en", non_breaking_prefix_file: str=None) -> List['SentenceContext']:
        splitter = SentenceSplitter(language=language, non_breaking_prefix_file=non_breaking_prefix_file)
        sentences = splitter.split(text)
        return [SentenceContext(s, sentences[i - 1] if i > 0 else None, 
                                sentences[i + 1] if i < len(sentences) - 1 else None) 
                                for i, s in enumerate(sentences)]
    
    def from_sentence(sentence: str, language: str="en") -> 'SentenceContext':
        return SentenceContext(sentence, None, None)
    
    # Helper method for the Predictions as_dict method.
    def context_as_list(self):
        preceding = self.preceding_sentence if self.preceding_sentence is not None else ''
        following = self.following_sentence if self.following_sentence is not None else ''
        return [preceding, following]

# Recogniser::run method output type.
@pdataclass(frozen=True)
class SentenceMentions:
    # The sentence.
    sentence: Sentence
    # A list of toponym mentions, ordered by character offset within the sentence.
    mentions: List[Mention]

    def __post_init__(self):
        if self.is_empty():
            return
        if max([m.end_char() for m in self.mentions]) > len(self.sentence):
            raise ValueError("Max end char exceeds sentence length.")

    def is_empty(self) -> bool:
        return len(self.mentions) == 0

    def len(self) -> int:
        return len(self.mentions)
    
    def exclude_microtoponyms(self) -> 'SentenceMentions':
        mentions = list(filter(lambda m: not m.is_microtoponym(), self.mentions))
        return SentenceMentions(self.sentence, mentions)
    
    # Helper method for backwards compatibility with training functions in `rel_utils.py`.
    def from_list(data: List[Dict]) -> 'SentenceMentions':
        # The data are assumed to be in the format returned by the 
        # `prepare_initial_data` method in `rel_utils.py`.
        mentions = [TrainingMention.from_dict(d) for d in data]
        # Check that all sentences in the list are identical.
        if {d['sentence'] for d in data} != {data[0]['sentence']}:
            raise ValueError("Inconsistent sentences.")
        d = data[0]
        context = SentenceContext(d['sentence'], d['context'][0], d['context'][1], d['sent_idx'])

        return SentenceMentions(context, mentions)

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
    # The toponym mention in the text.
    mention: Mention
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
    # The Wikidata class of this Wikidata entry (if available).
    wkdt_class: Optional[str]

@pdataclass(frozen=True)
class MostPopularLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `mostpopular` linking method."""
    # The mention-to-wikidata link frequency.
    freq: int

    def __post_init__(self):
        if not isinstance(self.freq, int):
            raise ValueError("freq must be an integer.")

@pdataclass(frozen=True)
class ByDistanceLink(WikidataLink):
    """Data class representing a string match and potential links in 
    Wikidata under the `bydistance` linking method."""
    # The lat-lon coordinates of the link in Wikidata.
    coords: Optional[Tuple[float, float]]
    # The lat-lon coordinates of the place of publication. 
    place_of_pub_coords: Optional[Tuple[float, float]]
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

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.string_match.string_similarity)

    def is_empty(self) -> bool:
        return not self.wikidata_links

    # Transforms this CandidateLinks instance into a PredictedLinks instance
    # by attaching disambiguation scores.
    def attach_scores(self, scores: Dict[str, float]) -> 'PredictedLinks':
        # Check that there is one score for each link.
        if scores.keys() != {link.wqid for link in self.wikidata_links}:
            raise ValueError("Incompatible disambiguation scores.")
        return PredictedLinks(self.string_match, self.wikidata_links, scores)

# Extend CandidateLinks to include disambigution scores. Note that we use 
# inheritance, rather than composition, for compatibility with the `links`
# field in the Candidates dataclass.
@pdataclass(order=True, frozen=True)
class PredictedLinks(CandidateLinks):
    """Data class representing a collection of potential links in Wikidata with scores for each."""
    # A disambiguation score for each potential link in Wikidata.
    disambiguation_scores: Dict[str, float]

    def best_disambiguation_score(self) -> float:
        if self.is_empty():
            return None
        return max(self.disambiguation_scores.values())
    
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
        scores = self.disambiguation_scores
        return max(scores, key=lambda key: scores[key])

    # Returns the top 7 Wikidata links in order of their disambiguation score
    # (as reported as `cross_cand_score` in the T-Res pipeline output).
    def cross_cand_scores(self, len=7) -> dict:
        scores = {k: round(v, 3) for (k, v) in self.disambiguation_scores.items()}
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:len])
    
    # Helper method for the Predictions as_dict method.
    def scores_as_list(self) -> list:
        ret = [[k, round(v, 3)] for k, v in self.disambiguation_scores.items()]
        return sorted(ret, key=lambda x: (x[1], x[0]), reverse=True)

# Linker::run method output type.
@pdataclass(order=True, frozen=True)
class Candidates:
    """Data class representing candidate string matches for a toponym, 
    each with candidate Wikidata links."""
    sort_index: float = field(init=False)
    # The toponym mention in the text.
    mention: Mention
    # The string matching method used.
    ranking_method: str
    # The linking method used.
    linking_method: str
    # A list of CandidateLinks instances.
    links: List[CandidateLinks]
    # Place of publication Wikidata ID.
    place_of_pub_wqid: Optional[str]
    # Place of publication.
    place_of_pub: Optional[str]
    # With publication flag.
    with_publication: bool

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
        s = f"Candidates for '{self.mention.mention}':"
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
                for wqid, score in m.cross_cand_scores(len=2).items():
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
    def get(self, variation: str) -> Optional[CandidateLinks]:
        for m in self.links:
            if m.string_match.variation == variation:
                return m
        return None
    
    # TODO: rename this as `best_candidate` (and it's understood this means the best 
    # StringMatch with associated candidate WikidataLink instances).
    # Returns the CandidateLinks instance whose StringMatch has the highest string similarity.
    def best_match(self) -> Optional[CandidateLinks]:
        if self.is_empty():
            return None
        # The list of CandidateLinks instances is ordered by decreasing string similarity.
        return self.links[0]
    
    def best_string_match(self) -> Optional[StringMatch]:
        if self.is_empty():
            return None
        return self.best_match().string_match

    # Returns the Wikidata link with the highest disambiguation score.
    def best_wikidata_link(self) -> Optional[WikidataLink]:
        # Get the CandidateLinks instance with highest string similarity.
        best_match = self.best_match()
        if not best_match or best_match.is_empty():
            return None
        if not isinstance(best_match, CandidateLinks):
            raise ValueError(f"Expected CandidateLinks instance. Got {type(best_match)}")
        return best_match.best_wikidata_link()

    def best_wqid(self) -> Optional[str]:
        best_wikidata_link = self.best_wikidata_link()
        if not best_wikidata_link:
            return None
        return best_wikidata_link.wqid

    def best_disambiguation_score(self) -> Optional[float]:
        best_match = self.best_match()
        if not best_match or best_match.is_empty():
            return None
        if not isinstance(best_match, PredictedLinks):
            return None
        return best_match.best_disambiguation_score()


################################
# Dataclasses for Pipeline
################################

@pdataclass(frozen=True)
class SentenceCandidates:
    """Data class representing candidate matches for all toponym mentions 
    in a sentence."""
    # The sentence.
    sentence: Sentence
    # List of candidates for each toponym mention in the sentence.
    candidates: List[Candidates]

    def __post_init__(self):
        if self.is_empty():
            return
        if max([cs.mention.end_char() for cs in self.candidates]) > len(self.sentence):
            raise ValueError("Inconsistent candidate mentions. Max end char exceeds sentence length.")

    def is_empty(self, ignore_empty_candidates: bool=True) -> bool:
        if ignore_empty_candidates:
            return len(self.candidates) == 0 or all([c.is_empty() for c in self.candidates])
        return len(self.candidates) == 0

# Pipeline::run_candidate_selection method output type.
@pdataclass(frozen=True)
class TextCandidates:
    """Data class representing candidate matches for all toponym mentions 
    in a block of text."""
    # List of setence candidates for each sentence in the text.
    sentence_candidates: List[SentenceCandidates]

    def __post_init__(self):
        if self.is_empty():
            return
        # Check that all place of publication data is consistent.
        if {self.place_of_pub_wqid()} != {c.place_of_pub_wqid for c in self.candidates()}:
            raise ValueError("Inconsistent place of publication Wikidata IDs.")
        if {self.place_of_pub()} != {c.place_of_pub for c in self.candidates()}:
            raise ValueError("Inconsistent place of publication data.")

    def candidates(self) -> List[Candidates]:
        return [c for sc in self.sentence_candidates for c in sc.candidates]

    def is_empty(self, ignore_empty_candidates: bool=True) -> bool:
        if ignore_empty_candidates:
            return len(self.sentence_candidates) == 0 or all([sc.is_empty() for sc in self.sentence_candidates])
        return len(self.candidates()) == 0
    
    def text(self) -> str:
        return " ".join([scs.sentence.sentence for scs in self.sentence_candidates])
    
    # TODO: unit test needed.
    def sentence_contexts(self) -> List[SentenceContext]:
        scs = self.sentence_candidates
        return [SentenceContext(sc.sentence.sentence, 
                         scs[i - 1].sentence.sentence if i > 0 else None, 
                         scs[i + 1].sentence.sentence if i < len(scs) - 1 else None) 
         for i, sc in enumerate(scs)]

    def place_of_pub_wqid(self) -> Optional[str]:
        if self.is_empty(ignore_empty_candidates=False):
            return None
        return self.candidates()[0].place_of_pub_wqid

    def place_of_pub(self) -> Optional[str]:
        if self.is_empty(ignore_empty_candidates=False):
            return None
        return self.candidates()[0].place_of_pub

# Pipeline::run_disambiguation method output type.
@pdataclass(frozen=True)
class Predictions(TextCandidates):
    """Data class representing toponym predictions in text."""

    def __post_init__(self):
        super().__post_init__()
        for c in self.candidates():
            if not all([isinstance(links, PredictedLinks) for links in c.links]):
                raise ValueError("Candidate links must be scored.")

    def best_wqids(self) -> List[Optional[str]]:
        return [c.best_wqid() for c in self.candidates()]

    def best_disambiguation_scores(self) -> List[Optional[float]]:
        return [c.best_disambiguation_score() for c in self.candidates()]

    def apply_rel_disambiguation(
            self, 
            rel_predictions: dict,
            with_publication: bool) -> 'RelPredictions':
        
        # If with_publication is True, drop the "artificial" final toponym mention.
        if with_publication:
            del rel_predictions["linking"][-1]

        # Incoroporate the REL model predictions.
        rel_scores = [RelScores(
            mention=d["mention"],
            scores={wqid: score for wqid, score in zip(d["candidates"], d["scores"])},
            confidence=d["conf_ed"]) for d in rel_predictions["linking"]]

        return RelPredictions(self.sentence_candidates, rel_scores)

    # Returns a dictionary containing a toponym mention for the place of publication.
    def place_of_pub_mention(self) -> dict:
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

        # Replaces add_publication from rel_utils.py:
        if with_publication:
            d["linking"].append(self.place_of_pub_mention())
        return d

@pdataclass(frozen=True)
class TrainingPredictions(Predictions):
    """Data class representing toponym predictions for training a REL model."""

    def __post_init__(self):
        super().__post_init__()

    # Similar to the as_dict method in Predictions, but now for backward 
    # compatibility with the `prepare_rel_trainset` function in `rel_utils.py`.
    def as_list(self, with_publication: bool) -> dict:
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

        # Replaces add_publication from rel_utils.py:
        if with_publication:
            l.append(self.place_of_pub_mention())
        return l

@pdataclass(frozen=True)
class RelScores:
    """Data class representing scores produced by the REL entity disambiguation model."""
    # The toponym mention.
    mention: str
    # REL entity disambiguation scores.
    scores: Dict[str, float]
    # REL entity disambiguation confidence score.
    confidence: float

@pdataclass(frozen=True)
class RelPredictions(Predictions):
    """Data class representing toponym predictions in text produced by REL entity disambiguation."""
    # A list of REL entity disambiguation scores.
    rel_scores: List[RelScores]

    def __post_init__(self):
        if len(self.rel_scores) != len(super().candidates()):
            raise ValueError(f"""Expected one RelScores instance per linked toponym mention.
                             Got {len(self.rel_scores)} instances and {len(super().candidates())} mentions.""")

    # Override the candidates method to return REL linking predictions.
    def candidates(self) -> List[Candidates]:

        # Construct equivalent Candidate instances but with the REL scores in the PredictedLinks.
        ret = list()
        for c, rs in zip(super().candidates(), self.rel_scores):
            if c.is_empty():
                ret.append(c)
                continue
            predicted_links = c.best_match()
            # Get the list of WikidataLink instances for which REL scores are available.
            wikidata_links = [wl for wl in predicted_links.wikidata_links if wl.wqid in rs.scores.keys()]
            links = [PredictedLinks(predicted_links.string_match, wikidata_links, rs.scores)]
            ret.append(Candidates(
                c.mention, 
                c.ranking_method, 
                c.linking_method, 
                links, 
                c.place_of_pub_wqid, 
                c.place_of_pub, 
                c.with_publication))
        return ret
