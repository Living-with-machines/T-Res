import os
from pathlib import Path
import sqlite3
from math import exp
import numpy as np
import pytest

from t_res.geoparser import ranking
from t_res.geoparser.linking import *
from t_res.utils.dataclasses import *

current_dir = Path(__file__).parent.resolve()

def test_init():

    # Test that parameters passed to the subclass constructor are propagated.
    linker = MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
    )

    assert linker.method_name  == "mostpopular"

    assert linker.resources_path  == "path/to/resources/"
    assert linker.experiments_path  == "path/to/experiments/"
    assert linker.resources['resource'] == 'value'

    linker = MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
    )

    # Test the extra parameters in the RelDisambLinker
    linker = RelDisambLinker(
        resources_path="path/to/resources/",
        ranker=ranking.PerfectMatchRanker("path/to/resources/"),
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
        rel_params={'with_publication': False},
        overwrite_training=True,
    )

    assert linker.method_name  == "reldisamb"

    assert linker.resources_path  == "path/to/resources/"
    assert linker.experiments_path  == "path/to/experiments/"
    assert linker.resources['resource'] == 'value'
    assert linker.rel_params['with_publication'] == False
    assert linker.overwrite_training

    linker = RelDisambLinker(
        resources_path="path/to/resources/",
        ranker=ranking.PerfectMatchRanker("path/to/resources/"),
        experiments_path="path/to/experiments/",
        rel_params={'with_publication': False},
        linking_resources={'resource': 'value'},
    )

    assert not linker.overwrite_training

    # Test default REL linker parameters

    # Invalid parameter raises ValueError:
    with pytest.raises(ValueError):
        linker = RelDisambLinker(
            resources_path="path/to/resources/",
            ranker=ranking.PerfectMatchRanker("path/to/resources/"),
            experiments_path="path/to/experiments/",
            rel_params={'invalid_param': 'value'},
            linking_resources={'resource': 'value'},
        )

    linker = RelDisambLinker(
        resources_path="path/to/resources/",
        ranker=ranking.PerfectMatchRanker("path/to/resources/"),
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
    )

    # Expect default parameter values:
    assert linker.rel_params['with_publication'] == True
    assert linker.rel_params['do_test'] == False
    assert linker.rel_params["without_microtoponyms"] == True
    

    linker = RelDisambLinker(
        resources_path="path/to/resources/",
        ranker=ranking.PerfectMatchRanker("path/to/resources/"),
        experiments_path="path/to/experiments/",
        rel_params={
            'with_publication': False,
            'do_test': True,
        },
        linking_resources={'resource': 'value'},
    )

    # Default parameter values are overridden:
    assert linker.rel_params['with_publication'] == False
    assert linker.rel_params['do_test'] == True
    # Unspecified parameters have default values:
    assert linker.rel_params["without_microtoponyms"] == True

def test_new():
    # Test Linker construction via string parameters.

    # If a required parameter is omitted, expect a TypeError.
    kwargs = {
        'method_name': 'mostpopular',
        }
    with pytest.raises(TypeError):
        linker = Linker.new(**kwargs)

    kwargs = {
        'method_name': 'mostpopular',
        'resources_path': 'sample_files/resources/',
        'linking_resources': dict(),
        }
    linker = Linker.new(**kwargs)
    assert isinstance(linker, MostPopularLinker)
    assert linker.method_name == 'mostpopular'
    assert linker.resources == dict()

    kwargs = {
        'method_name': 'bydistance',
        'resources_path': 'sample_files/resources/',
        }
    linker = Linker.new(**kwargs)
    assert isinstance(linker, ByDistanceLinker)
    assert linker.method_name == 'bydistance'

    kwargs = {
        'method_name': 'reldisamb',
        'resources_path': 'sample_files/resources/',
        'ranker': ranking.PerfectMatchRanker("sample_files/resources/"),
        'overwrite_training': True,
        }
    linker = Linker.new(**kwargs)
    assert isinstance(linker, RelDisambLinker)
    assert linker.method_name == 'reldisamb'
    assert linker.overwrite_training

    # If the ranking method is invalid, expect a ValueError.
    kwargs = {
        'method_name': 'nosuchlinker',
        }
    with pytest.raises(ValueError):
        linker = Linker.new(**kwargs)

def test_linking_most_popular():
    linker = MostPopularLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
    )
    assert linker.method_name  == "mostpopular"
    linker.load()

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

    linker = ByDistanceLinker(
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
    linker = ByDistanceLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
    )
    assert linker.method_name  == "bydistance"
    linker.load()

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

    # Test with a list of empty candidates.
    origin_wqid = "Q2365261"
    candidates = linker.run(CandidateMatches(mention, "perfectmatch", []), origin_wqid)

    sentence_candidates = SentenceCandidates(Sentence(sentence), [candidates])
    predictions = linker.disambiguate([sentence_candidates])

    # If empty candidates are ignored, the set of predictions is empty:
    assert predictions.is_empty(ignore_empty_candidates=True)
    # If empty candidates are not ignored, the set of predictions is not empty:
    assert not predictions.is_empty(ignore_empty_candidates=False)

@pytest.mark.resources(reason="Needs large resources")
def test_proximity():

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranking.PerfectMatchRanker(os.path.join(current_dir, "../resources/")),
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir, "../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir, "sample_files/experiments/outputs/data/lwm/"),
                "training_split": "apply",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": False,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
                "reference_separation": ((49.956739, -8.17751), (60.87, 1.762973)),
            },
        )
    linker.load()

    place_of_pub_wqid = "Q203349" # Poole, Doset
    wqid = "Q503331" # Dorchester, Dorset

    result = linker.proximity(linker.wkdt_coords(place_of_pub_wqid), linker.wkdt_coords(wqid))

    # Distance from Poole to Dorchester is ~31km
    d = 31.0
    # Reference distance is ~1362km
    reference_d = 1362.0

    assert result == pytest.approx(exp(-(d/reference_d)**2), abs=1e-4)

    # Test with specific coordinates that require normalization.
    origin_coords = [53.067, -2.522]
    coords = [-24.84, 340.47]

    result = linker.proximity(origin_coords, coords)
    assert result == pytest.approx(0, abs=1e-10)
