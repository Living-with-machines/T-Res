import os
from pathlib import Path

import numpy as np
import pytest

from t_res.geoparser import linking, ranking
from t_res.geoparser.dataclasses import *

current_dir = Path(__file__).parent.resolve()

def test_linking_data_classes():

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

    # Test disambiguation score methods.
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

    linker = linking.MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
    )

    scores = linker.disambiguation_scores(wikidata_links)
    assert scores == {'Q619055': 5.0 / 38, 'Q5953687': 33.0 / 38}

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

def test_init():

    # Test that parameters passed to the subclass constructor are propagated.
    linker = linking.MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
    )

    assert linker.method_name  == "mostpopular"

    assert linker.resources_path  == "path/to/resources/"
    assert linker.experiments_path  == "path/to/experiments/"
    assert linker.linking_resources['resource'] == 'value'

    linker = linking.MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
    )

    # Test the extra parameters in the RelDisambLinker
    linker = linking.RelDisambLinker(
        resources_path="path/to/resources/",
        ranker=ranking.PerfectMatchRanker("path/to/resources/"),
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
        rel_params={'param': 'value'},
        overwrite_training=True,
    )

    assert linker.method_name  == "reldisamb"

    assert linker.resources_path  == "path/to/resources/"
    assert linker.experiments_path  == "path/to/experiments/"
    assert linker.linking_resources['resource'] == 'value'
    assert linker.rel_params['param'] == 'value'
    assert linker.overwrite_training

    linker = linking.RelDisambLinker(
        resources_path="path/to/resources/",
        ranker=ranking.PerfectMatchRanker("path/to/resources/"),
        experiments_path="path/to/experiments/",
        rel_params={'param': 'value'},
        linking_resources={'resource': 'value'},
    )

    assert not linker.overwrite_training

def test_linking_most_popular():
    linker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
    )
    assert linker.method_name  == "mostpopular"
    linker.load_resources()

    # Construct a CandidateMatches instance (to simulate the output from the Ranker).
    wqid_links = ["Q84", "Q92561"]
    matches = [StringMatchLinks("London", 1.0, wqid_links)]

    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = linker.run(CandidateMatches(mention, "perfectmatch", matches))

    # Check best string match.
    assert candidates.best_match().string_match.variation == "London"
    assert candidates.best_match().string_match.string_similarity == 1.0
    
    # Create a dummy sentence to test the disambiguate method.
    sentence = "A sentence about London."
    sentence_candidates = SentenceCandidates(Sentence(sentence), [candidates])
    predictions = linker.disambiguate([sentence_candidates])

    # Check best Wikidata link.
    candidate = predictions.candidates()[0]
    assert candidate.best_wqid() == "Q84"
    assert isinstance(candidate.best_match(), PredictedLinks)
    assert candidate.best_match().best_disambiguation_score() == pytest.approx(0.9812731647051174, abs=1e-3)
    assert candidate.best_wikidata_link().wkdt_class == "Q515" # London has Wikidata class 'City'

    # Check other Wikidata link.
    assert "Q92561" in candidate.best_match().disambiguation_scores.keys()
    candidate.best_match().disambiguation_scores["Q92561"] == pytest.approx(0.018726835294882633, abs=1e-3)

    candidates = linker.run(CandidateMatches(mention, "perfectmatch", []))
    assert candidates.is_empty()

# This test replaces the legacy unit test named `test_by_distance` and
# reproduces the same results.
def test_disambiguation_scores_by_distance():

    linker = linking.ByDistanceLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
    )
    assert linker.method_name  == "bydistance"

    # Test on London, UK. Wikidata ID "Q84".

    # Note we use normalized_scores taken from a legacy unit test to reproduce 
    # the same values for test assertions. The actual normalized scores in the 
    # Wikidata resources are different.
    wikidata_links = [
        ByDistanceLink(
            "Q84",
            wkdt_class='Q515',
            coords=(51.507222, -0.1275),
            place_of_pub_coords=(51.507222, -0.1275),
            normalized_score=0.9,
            geodist=0.0,
        ),
        ByDistanceLink(
            "Q92561",
            wkdt_class='Q515',
            coords=(42.9837, -81.2497),
            place_of_pub_coords=(51.507222, -0.1275),
            normalized_score=0.1,
            geodist=5876.70916049723,
        )
    ]

    scores = linker.disambiguation_scores(wikidata_links, string_similarity=0.397048)

    assert len(scores) == 2

    # London, UK is the top scoring candidate.
    assert max(scores, key = lambda key: scores[key]) == "Q84"
    assert max(scores.values()) == 0.824

    assert min(scores, key = lambda key: scores[key]) == "Q92561"
    assert min(scores.values()) == 0.124

    # Test on London, Ontario. Wikidata ID "Q92561".
    wikidata_links = [
        ByDistanceLink(
            "Q84", 
            wkdt_class='Q515',
            coords=(51.507222, -0.1275),
            place_of_pub_coords=(42.9837, -81.2497),
            normalized_score=0.9,
            geodist=5876.70916049723, 
        ),
        ByDistanceLink(
            "Q92561",
            wkdt_class='Q515',
            coords=(42.9837, -81.2497),
            place_of_pub_coords=(42.9837, -81.2497),
            normalized_score=0.1,
            geodist=0.0, 
        )
    ]

    scores = linker.disambiguation_scores(wikidata_links, string_similarity=0.397048)

    # London, Ontario is the top scoring candidate.
    assert max(scores, key = lambda key: scores[key]) == "Q92561"
    assert max(scores.values()) == 0.624

    assert min(scores, key = lambda key: scores[key]) == "Q84"
    assert min(scores.values()) == 0.324

    # Test when not all geodesic distances are available.
    wikidata_links = [
        ByDistanceLink(
            "Q84", 
            wkdt_class='Q515',
            coords=(51.507222, -0.1275),
            place_of_pub_coords=(51.507222, -0.1275),
            normalized_score=0.9,
            geodist=0.0, 
        ),
        ByDistanceLink(
            "Q92561", 
            wkdt_class='Q515',
            coords=None,
            place_of_pub_coords=(51.507222, -0.1275),
            normalized_score=0.1, 
            geodist=None, 
        )
    ]
    scores = linker.disambiguation_scores(wikidata_links, string_similarity=0.397048)

    # The score for London UK is unchanged.
    assert max(scores, key = lambda key: scores[key]) == "Q84"
    assert max(scores.values()) == 0.824

    # The score for London, Ontario is now zero (default when no geodist is present).
    assert min(scores, key = lambda key: scores[key]) == "Q92561"
    assert min(scores.values()) == 0.0

    # Implausible case, but worth testing.
    wikidata_links = [
        ByDistanceLink(
            "Q84",
            wkdt_class='Q515',
            coords=None,
            place_of_pub_coords=None,
            normalized_score=0.9,
            geodist=None, 
        ),
        ByDistanceLink(
            "Q92561", 
            wkdt_class='Q515',
            coords=(42.9837, -81.2497),
            place_of_pub_coords=(51.507222, -0.1275),
            normalized_score=0.1,
            geodist=5876.70916049723, 
        )
    ]

    scores = linker.disambiguation_scores(wikidata_links, string_similarity=0.397048)

    # The score for London, Ontario is unchanged.
    assert max(scores, key = lambda key: scores[key]) == "Q92561"
    assert max(scores.values()) == 0.124

    # The score for London UK is now zero (default when no geodist is present).
    assert min(scores, key = lambda key: scores[key]) == "Q84"
    assert min(scores.values()) == 0.0

    # Test when no geodesic distances are available.
    wikidata_links = [
        ByDistanceLink(
            "Q84", 
            wkdt_class='Q515',
            coords=None,
            place_of_pub_coords=None,
            normalized_score=0.9,
            geodist=None, 
        ),
        ByDistanceLink(
            "Q92561", 
            wkdt_class='Q515',
            coords=None,
            place_of_pub_coords=None,
            normalized_score=0.1,
            geodist=None, 
        )
    ]

    scores = linker.disambiguation_scores(wikidata_links, string_similarity=0.397048)

    assert scores == {"Q84": 0.0, "Q92561": 0.0}

    # Test when no candidates are provided.
    wikidata_links = list()
    scores = linker.disambiguation_scores(wikidata_links, string_similarity=0.397048)

    # Expect an empty dictionary.
    assert scores == dict()

def test_linking_by_distance():
    linker = linking.ByDistanceLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
    )
    assert linker.method_name  == "bydistance"
    linker.load_resources()

    # Construct a CandidateMatches instance (to simulate the output from the Ranker).
    wqid_links = ["Q84", "Q92561"]
    matches = [StringMatchLinks("London", 0.397048, wqid_links)]

    origin_wqid = "Q84"
    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = linker.run(CandidateMatches(mention, "perfectmatch", matches), origin_wqid)

    # Check best string match.
    assert candidates.best_match().string_match.variation == "London"
    assert candidates.best_match().string_match.string_similarity == 0.397048
    
    # Create a dummy sentence to test the disambiguate method.
    sentence = "A sentence about London."
    sentence_candidates = SentenceCandidates(Sentence(sentence), [candidates])
    predictions = linker.disambiguate([sentence_candidates])

    # Check that the best Wikidata link is London, UK "Q84".
    candidate = predictions.candidates()[0]
    assert candidate.best_wqid() == "Q84"
    assert candidate.best_match().best_disambiguation_score() == 0.845
    assert candidate.best_match().disambiguation_scores.keys() == {"Q84", "Q92561"}
    assert candidate.best_wikidata_link().coords == (51.507222, -0.1275)
    assert candidate.best_wikidata_link().wkdt_class == "Q515" # London has Wikidata class 'City'

    # Test dependence on the place of publication.
    origin_wqid = "Q92561"
    candidates = linker.run(CandidateMatches(mention, "perfectmatch", matches), origin_wqid)

    sentence_candidates = SentenceCandidates(Sentence(sentence), [candidates])
    predictions = linker.disambiguate([sentence_candidates])

    # Check that the best Wikidata link is now London, Ontario "Q92561".
    candidate = predictions.candidates()[0]
    assert candidate.best_wqid() == "Q92561"
    assert candidate.best_match().best_disambiguation_score() == 0.694
    assert candidate.best_match().disambiguation_scores.keys() == {"Q84", "Q92561"}

    # Test with an empty list of candidates.
    origin_wqid = "Q2365261"
    candidates = linker.run(CandidateMatches(mention, "perfectmatch", []), origin_wqid)

    sentence_candidates = SentenceCandidates(Sentence(sentence), [candidates])
    predictions = linker.disambiguate([sentence_candidates])

    print(predictions)

    assert predictions.is_empty()
