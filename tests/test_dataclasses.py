from ast import literal_eval
from t_res.utils.dataclasses import *

def test_sentence_context():

    text = "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though. What is to be done?"
    result = SentenceContext.from_text(text)

    assert len(result) == 3
    assert result[0].sentence == "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds."
    assert result[0].preceding_sentence == None
    assert result[0].following_sentence == "Not in London though."

    assert result[1].sentence == "Not in London though."
    assert result[1].preceding_sentence == "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds."
    assert result[1].following_sentence == "What is to be done?"

    assert result[2].sentence == "What is to be done?"
    assert result[2].preceding_sentence == "Not in London though."
    assert result[2].following_sentence == None

def test_sentence_mentions():

    sentence = ', thence to Emery Down,crowing to Minesteed Manor ; he ther tacked back to Notherwood, and from thence back again to the Manor, where, after a brilliant run (Arnie hour and forty-five minutes, Reynold was compelled to succumb to his pursuers. '
    mentions_str = "[{'mention': 'Emery Down', 'start_offset': 3, 'end_offset': 4, 'start_char': 12, 'ner_score': 0.999, 'ner_label': 'LOC', 'entity_link': 'O'}, {'mention': 'Minesteed Manor', 'start_offset': 8, 'end_offset': 9, 'start_char': 34, 'ner_score': 0.999, 'ner_label': 'BUILDING', 'entity_link': 'O'}, {'mention': 'Notherwood', 'start_offset': 16, 'end_offset': 16, 'start_char': 75, 'ner_score': 0.999, 'ner_label': 'BUILDING', 'entity_link': 'O'}]"
    mentions = [Mention.from_dict(d) for d in literal_eval(mentions_str)]

    sentence_mentions = SentenceMentions(Sentence(sentence), mentions=mentions)
    assert sentence_mentions.len() == 3
    assert sentence_mentions.exclude_microtoponyms().len() == 1
    assert sentence_mentions.exclude_microtoponyms().mentions[0].ner_label == 'LOC'

def test_string_match():

    # Legacy example:
    # {'London': 1.0}
    string_match = StringMatch('London', 1.0)
    assert string_match.variation == 'London'
    assert string_match.string_similarity == 1.0

    # Legacy example:
    # {'Sheftield': {'Shielfield': 0.9387, 'Sheffield': 0.9228, 'Shelfield': 0.8947}}

    # Ranker `string_match` method returns a list[StringMatch].
    matches = [
        StringMatch('Shielfield', 0.9387),
        StringMatch('Sheffield', 0.9228),
        StringMatch('Shelfield', 0.8947),
    ]

    # Inside the Ranker `run` method, these StringMatch instances are 
    # converted into StringMatchLinks instances, by adding to each a
    # list of candidate Wikidata IDs.
    matches = [
        StringMatchLinks('Shielfield', 0.9387, ['Q619055', 'Q5953687']),
        StringMatchLinks('Sheffield', 0.9228, ['Q6707254', 'Q7492778', 'Q1421317']),
        StringMatchLinks('Shelfield', 0.8947, ['Q7493600']),
    ]

def test_candidate_matches():

    matches = [
        StringMatchLinks('Shielfield', 0.9387, ['Q619055', 'Q5953687']),
        StringMatchLinks('Sheffield', 0.9228, ['Q6707254', 'Q7492778', 'Q1421317']),
        StringMatchLinks('Shelfield', 0.8947, ['Q7493600']),
    ]

    # Ranker `run` method returns a CandidateMatches instance.
    mention_str = {'mention': 'Sheftield', 'start_offset': 3, 'end_offset': 4, 'start_char': 12, 'ner_score': 0.699, 'ner_label': 'LOC', 'entity_link': 'O'}
    candidates = CandidateMatches(Mention.from_dict(mention_str), "levenshtein", matches)

    assert candidates.mention.mention == 'Sheftield'
    assert candidates.ranking_method == 'levenshtein'
    assert len(candidates.matches) == 3

    # matches are in order of decreasing string similarity.
    assert candidates.matches[0].variation == 'Shielfield'
    assert candidates.matches[0].string_similarity == 0.9387
    assert len(candidates.matches[0].wqid_links) == 2
    assert candidates.matches[1].variation == 'Sheffield'
    assert candidates.matches[1].string_similarity == 0.9228
    assert len(candidates.matches[1].wqid_links) == 3
    assert candidates.matches[2].variation == 'Shelfield'
    assert candidates.matches[2].string_similarity == 0.8947
    assert len(candidates.matches[2].wqid_links) == 1

def test_wikidata_links():

    wikidata_link = MostPopularLink('Q619055', wkdt_class='Q1076486', freq=22)
    assert wikidata_link.wqid == 'Q619055'
    assert wikidata_link.freq == 22

    wikidata_link = ByDistanceLink(
        'Q619055', 
        wkdt_class='Q1076486',
        coords=(55.76, -2.01583),
        place_of_pub_coords=(51.507222, -0.1275), 
        normalized_score=0.03571428571428571, 
        geodist=1255.45
    )
    assert wikidata_link.wqid == 'Q619055'
    assert wikidata_link.normalized_score == 0.03571428571428571
    assert wikidata_link.geodist == 1255.45

    # Legacy example:
    # {'Q619055': 0.03571428571428571}
    wikidata_link = RelDisambLink(
        'Q619055',
        wkdt_class='Q1076486',
        freq=22,
        normalized_score=0.03571428571428571,
    )
    assert wikidata_link.wqid == 'Q619055'
    assert wikidata_link.normalized_score == 0.03571428571428571
    assert wikidata_link.freq == 22

def test_candidate_links():

    # Legacy example:
    # {'Shielfield': {'Score': 0.9387, 'Candidates': {'Q619055': 0.03571428571428571, 'Q5953687': 0.22857142857142856}}
    wikidata_links = [
        RelDisambLink(
            'Q619055',
            wkdt_class='Q1076486',
            freq=5,
            normalized_score=0.03571428571428571,
        ),
        RelDisambLink(
            'Q5953687',
            wkdt_class='Q23764314',
            freq=33,
            normalized_score=0.22857142857142856,
        ),
    ]
    candidate_links = CandidateLinks(
        StringMatch("Shielfield", 0.9387), 
        wikidata_links, 
    )

    assert candidate_links.string_match.variation == 'Shielfield'
    assert candidate_links.string_match.string_similarity == 0.9387

def test_predicted_links():

    wikidata_links = [
        MostPopularLink(
            'Q619055', 
            wkdt_class='Q1076486',
            freq=5
        ),
        MostPopularLink(
            'Q5953687', 
            wkdt_class='Q23764314',
            freq=33,
        ),
    ]

    candidate_links = CandidateLinks(
        StringMatch("Shielfield", 0.9387), 
        wikidata_links, 
    )

    assert candidate_links.string_match.variation == 'Shielfield'
    assert candidate_links.string_match.string_similarity == 0.9387

    scores = {'Q619055': 5.0 / 38, 'Q5953687': 33.0 / 38}

    # Attach the disambiguation scores to the candidate links 
    # to obtain predicted links.
    predicted_links = candidate_links.attach_scores(scores)

    assert predicted_links.best_wqid() == 'Q5953687'
    assert predicted_links.best_wikidata_link() == MostPopularLink(
        'Q5953687', 
        wkdt_class='Q23764314',
        freq=33
    )

    assert predicted_links.disambiguation_scores == {'Q619055': 5.0 / 38, 'Q5953687': 33.0 / 38}
    assert predicted_links.best_disambiguation_score() == 33.0 / 38

    # Disambiguation scores as a list are ordered from highest to lowest score
    # and rounded to 3 decimal places.
    assert predicted_links.scores_as_list() == [['Q5953687', round(33.0 / 38, 3)], ['Q619055', round(5.0 / 38, 3)]]

