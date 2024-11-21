import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from t_res.geoparser import linking
from t_res.geoparser.dataclasses import RelDisambLink, StringMatch, StringMatchLinks, MostPopularLink, ByDistanceLink, CandidateMatches, CandidateLinks

current_dir = Path(__file__).parent.resolve()

def test_linking_data_classes():

    wikidata_match = MostPopularLink('Q619055', freq=22, wqid_to_coords=None, entity2class=None)
    assert wikidata_match.wqid == 'Q619055'
    assert wikidata_match.freq == 22

    wikidata_match = ByDistanceLink('Q619055', origin_wqid='Q84', normalized_score=0.03571428571428571, geodist=1255.45, wqid_to_coords=None, entity2class=None)
    assert wikidata_match.wqid == 'Q619055'
    assert wikidata_match.origin_wqid == 'Q84'
    assert wikidata_match.normalized_score == 0.03571428571428571
    assert wikidata_match.geodist == 1255.45

    # Legacy example:
    # {'Q619055': 0.03571428571428571}
    wikidata_match = RelDisambLink('Q619055', normalized_score=0.03571428571428571, freq=22, wqid_to_coords=None, entity2class=None)
    assert wikidata_match.wqid == 'Q619055'
    assert wikidata_match.normalized_score == 0.03571428571428571
    assert wikidata_match.freq == 22

    # old:
    # # Check that the field types prevent accidental mis-ordering of arguments.
    # with pytest.raises(Exception):
    #     wikidata_match = RelDisambLink('Q619055', wqid_to_coords=None, entity2class=None, 0.03571428571428571, 22)

    # Legacy example:
    # {'Shielfield': {'Score': 0.9387, 'Candidates': {'Q619055': 0.03571428571428571, 'Q5953687': 0.22857142857142856}}
    wikidata_links = [
        RelDisambLink('Q619055', freq=5, normalized_score=0.03571428571428571, wqid_to_coords=None, entity2class=None),
        RelDisambLink('Q5953687', freq=33, normalized_score=0.22857142857142856, wqid_to_coords=None, entity2class=None),
    ]
    closure = linking.RelDisambLinker.disambiguation_scores(wikidata_links)
    candidate_links = CandidateLinks(StringMatch("Shielfield", 0.9387), wikidata_links, closure)

    assert candidate_links.string_match.variation == 'Shielfield'
    assert candidate_links.string_match.string_similarity == 0.9387

    # Test disambiguation score methods.
    wikidata_links = [
        MostPopularLink('Q619055', freq=5, wqid_to_coords=None, entity2class=None),
        MostPopularLink('Q5953687', freq=33, wqid_to_coords=None, entity2class=None),
    ]
    closure = linking.MostPopularLinker.disambiguation_scores(wikidata_links)
    candidate_links = CandidateLinks(StringMatch("Shielfield", 0.9387), wikidata_links, closure)

    assert candidate_links.string_match.variation == 'Shielfield'
    assert candidate_links.string_match.string_similarity == 0.9387

    assert candidate_links.best_wqid() == 'Q5953687'
    assert candidate_links.best_wikidata_link() == MostPopularLink('Q5953687', freq=33, wqid_to_coords=None, entity2class=None)

    assert candidate_links.disambiguation_scores() == {'Q619055': 5.0 / 38, 'Q5953687': 33.0 / 38}
    assert candidate_links.best_disambiguation_score() == 33.0 / 38

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

    candidates = linker.run(CandidateMatches("London", "perfectmatch", matches))

    # Check best string match.
    assert candidates.best_match().string_match.variation == "London"
    assert candidates.best_match().string_match.string_similarity == 1.0
    
    # Check best Wikidata link.
    assert candidates.best_wqid() == "Q84"
    assert candidates.best_match().best_disambiguation_score() == pytest.approx(0.9812731647051174, abs=1e-3)

    assert candidates.best_wikidata_link().lat_lon == (51.507222, -0.1275)
    assert candidates.best_wikidata_link().wkdt_class == "Q515" # London has Wikidata class 'City'

    # Check other Wikidata link.
    assert "Q92561" in candidates.best_match().disambiguation_scores().keys()
    candidates.best_match().disambiguation_scores()["Q92561"] == pytest.approx(0.018726835294882633, abs=1e-3)

    candidates = linker.run(CandidateMatches("London", "perfectmatch", []))
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
        ByDistanceLink("Q84", origin_wqid="Q84", geodist=0.0, normalized_score=0.9, wqid_to_coords=None, entity2class=None),
        ByDistanceLink("Q92561", origin_wqid="Q84", geodist=5876.70916049723, normalized_score=0.1, wqid_to_coords=None, entity2class=None)
    ]

    # Get the closure for computing disambiguation scores by distance.
    disambiguation_scores = linking.ByDistanceLinker.disambiguation_scores(wikidata_links, matching_score=0.397048)

    # Compute disambiguation scores.
    result = disambiguation_scores()

    assert len(result) == 2

    # London, UK is the top scoring candidate.
    assert max(result, key = lambda key: result[key]) == "Q84"
    assert max(result.values()) == 0.824

    assert min(result, key = lambda key: result[key]) == "Q92561"
    assert min(result.values()) == 0.124

    # Test on London, Ontario. Wikidata ID "Q92561".
    wikidata_links = [
        ByDistanceLink("Q84", origin_wqid="Q92561", geodist=5876.70916049723, normalized_score=0.9, wqid_to_coords=None, entity2class=None),
        ByDistanceLink("Q92561", origin_wqid="Q92561", geodist=0.0, normalized_score=0.1, wqid_to_coords=None, entity2class=None)
    ]

    # Get the closure for computing disambiguation scores by distance.
    disambiguation_scores = linking.ByDistanceLinker.disambiguation_scores(wikidata_links, matching_score=0.397048)

    # Compute disambiguation scores.
    result = disambiguation_scores()

    # London, Ontario is the top scoring candidate.
    assert max(result, key = lambda key: result[key]) == "Q92561"
    assert max(result.values()) == 0.624

    assert min(result, key = lambda key: result[key]) == "Q84"
    assert min(result.values()) == 0.324

    # Test when not all geodesic distances are available.
    wikidata_links = [
        ByDistanceLink("Q84", origin_wqid="Q84", geodist=0.0, normalized_score=0.9, wqid_to_coords=None, entity2class=None),
        ByDistanceLink("Q92561", origin_wqid="Q84", geodist=None, normalized_score=0.1, wqid_to_coords=None, entity2class=None)
    ]
    disambiguation_scores = linking.ByDistanceLinker.disambiguation_scores(wikidata_links, matching_score=0.397048)
    result = disambiguation_scores()

    # The score for London UK is unchanged.
    assert max(result, key = lambda key: result[key]) == "Q84"
    assert max(result.values()) == 0.824

    # The score for London, Ontario is now zero (default when no geodist is present).
    assert min(result, key = lambda key: result[key]) == "Q92561"
    assert min(result.values()) == 0.0

    # Implausible case, but worth testing.
    wikidata_links = [
        ByDistanceLink("Q84", origin_wqid="Q84", geodist=None, normalized_score=0.9, wqid_to_coords=None, entity2class=None),
        ByDistanceLink("Q92561", origin_wqid="Q84", geodist=5876.70916049723, normalized_score=0.1, wqid_to_coords=None, entity2class=None)
    ]

    disambiguation_scores = linking.ByDistanceLinker.disambiguation_scores(wikidata_links, matching_score=0.397048)
    result = disambiguation_scores()

    # The score for London, Ontario is unchanged.
    assert max(result, key = lambda key: result[key]) == "Q92561"
    assert max(result.values()) == 0.124

    # The score for London UK is now zero (default when no geodist is present).
    assert min(result, key = lambda key: result[key]) == "Q84"
    assert min(result.values()) == 0.0

    # Test when no geodesic distances are available.
    wikidata_links = [
        ByDistanceLink("Q84", origin_wqid="Q84", geodist=None, normalized_score=0.9, wqid_to_coords=None, entity2class=None),
        ByDistanceLink("Q92561", origin_wqid="Q84", geodist=None, normalized_score=0.1, wqid_to_coords=None, entity2class=None)
    ]

    disambiguation_scores = linking.ByDistanceLinker.disambiguation_scores(wikidata_links, matching_score=0.397048)
    result = disambiguation_scores()

    assert result == {"Q84": 0.0, "Q92561": 0.0}

    # Test when no candidates are provided.
    # Get the closure for computing disambiguation scores by distance.
    wikidata_links = list()
    disambiguation_scores = linking.ByDistanceLinker.disambiguation_scores(wikidata_links, matching_score=0.397048)

    # Compute disambiguation scores. Expect an empty result.
    result = disambiguation_scores()
    assert result == dict()

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
    candidates = linker.run(CandidateMatches("London", "perfectmatch", matches), origin_wqid)

    # Check best string match.
    assert candidates.best_match().string_match.variation == "London"
    assert candidates.best_match().string_match.string_similarity == 0.397048
    
    # Check that the best Wikidata link is London, UK "Q84".
    assert candidates.best_wqid() == "Q84"
    assert candidates.best_match().best_disambiguation_score() == 0.845
    assert candidates.best_match().disambiguation_scores().keys() == {"Q84", "Q92561"}

    assert candidates.best_wikidata_link().lat_lon == (51.507222, -0.1275)
    assert candidates.best_wikidata_link().wkdt_class == "Q515" # London has Wikidata class 'City'

    # Test dependence on the place of publication.
    origin_wqid = "Q92561"
    candidates = linker.run(CandidateMatches("London", "perfectmatch", matches), origin_wqid)

    # Check that the best Wikidata link is now London, Ontario "Q92561".
    assert candidates.best_wqid() == "Q92561"
    assert candidates.best_match().best_disambiguation_score() == 0.694
    assert candidates.best_match().disambiguation_scores().keys() == {"Q84", "Q92561"}

    # Test with an empty list of candidates.
    origin_wqid = "Q2365261"
    candidates = linker.run(CandidateMatches("London", "perfectmatch", []), origin_wqid)

    assert candidates.best_wqid() == None
    assert candidates.best_match() == None
