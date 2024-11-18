import os
from pathlib import Path

import pytest

from t_res.geoparser import ranking
from t_res.geoparser.dataclasses import StringMatch, StringMatchLinks, CandidateMatches

current_dir = Path(__file__).parent.resolve()

def test_ranking_data_classes():
    """
    Test the data classes that represent ranking candidates.
    """

    # Legacy example:
    # {'London': 1.0}
    string_match = StringMatch('London', 1.0)
    assert string_match.variation == 'London'
    assert string_match.string_similarity == 1.0

    # Legacy example:
    # {'Sheftield': {'Shielfield': 0.9387, 'Sheffield': 0.9228, 'Shelfield': 0.8947}}

    # Ranker `string_match` method returns a list[StringMatch].
    matches = [
        ranking.StringMatch('Shielfield', 0.9387),
        ranking.StringMatch('Sheffield', 0.9228),
        ranking.StringMatch('Shelfield', 0.8947),
    ]

    # Inside the Ranker `run` method, these StringMatch instances are 
    # converted into StringMatchLinks instances, by adding to each a
    # list of candidate Wikidata IDs.
    matches = [
        ranking.StringMatchLinks('Shielfield', 0.9387, ['Q619055', 'Q5953687']),
        ranking.StringMatchLinks('Sheffield', 0.9228, ['Q6707254', 'Q7492778', 'Q1421317']),
        ranking.StringMatchLinks('Shelfield', 0.8947, ['Q7493600']),
    ]

    # Ranker `run` method returns a CandidateMatches instance.
    candidates = CandidateMatches('Sheftield', "levenshtein", matches)
    assert candidates.mention == 'Sheftield'
    assert len(candidates.matches) == 3

    assert candidates.mention == 'Sheftield'
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

def test_ranking_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    assert ranker.method_name == "perfectmatch"
    
    ranker.load()
    ranker.cache = {}
    
    # Check the cache is empty.
    assert len(ranker.cache) == 0

    candidates = ranker.run("London")

    assert candidates.ranking_method == "perfectmatch"
    assert candidates.mention == "London"
    assert candidates.get("London").variation == "London"
    assert candidates.get("London").string_similarity == 1.0 

    # Check the cache has been updated.
    assert len(ranker.cache) == 1
    assert ranker.cache["London"] == candidates

    # candidates = ranker.run(["Lvndon"])
    # assert candidates["Lvndon"] == {}
    candidates = ranker.run("Lvndon")

    assert candidates.ranking_method == "perfectmatch"
    assert candidates.mention == "Lvndon"
    assert candidates.is_empty()

    # Check the cache has been updated.
    assert len(ranker.cache) == 2
    assert ranker.cache["Lvndon"] == candidates

    candidates = ranker.run("Paperopoli")

    assert candidates.ranking_method == "perfectmatch"
    assert candidates.mention == "Paperopoli"
    assert candidates.is_empty()

    # Check the cache has been updated.
    assert len(ranker.cache) == 3
    assert ranker.cache["Paperopoli"] == candidates

def test_ranking_matching_score():
    """
    Test that matching_score returns score only when there is an overlap
    """

    # Test the overlap matching score.
    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    assert ranker.method_name == "partialmatch"
    
    score_a = ranker.matching_score("New York", {"mentions": "New York City"})
    score_b = ranker.matching_score("New York City", {"mentions": "New York"})
    assert score_a == score_b == 0.6153846153846154

    with pytest.raises(TypeError):
        ranker.matching_score("Lvndon", "London")

    score = ranker.matching_score("London", {"mentions": "New York"})
    assert score is None

    # Test the Levenshtein distance matching score.
    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    score = ranker.matching_score("Lvndon", {"mentions": "London"})
    assert score == 0.8333333283662796

    score = ranker.matching_score("uityity", {"mentions": "asdasd"})
    assert score == 0.0

    with pytest.raises(TypeError):
        ranker.matching_score("Lvndon", "London")

def test_ranking_partial_match():
    """
    Test that partial match either returns results or {}
    """

    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    assert ranker.method_name == "partialmatch"
    
    ranker.load()

    ranker.mentions_to_wikidata = {"London": {"Q84": 0.922}}
    ranker.cache = {}

    candidates = ranker.run("London")

    assert candidates.ranking_method == "partialmatch"
    assert candidates.mention == "London"
    assert candidates.get("London").variation == "London"
    assert candidates.get("London").string_similarity == 1.0 

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": {"Q60": 0.884}}

    ranker.cache = {}
    candidates = ranker.run("New York")

    assert candidates.mention == "New York"
    assert candidates.get("New York City").variation == "New York City"
    assert candidates.get("New York City").string_similarity == pytest.approx(0.615384, abs=10e-6)

    ranker.cache = {}
    candidates = ranker.run("Lvndvn")

    assert candidates.mention == "Lvndvn"
    assert candidates.is_empty()


def test_ranking_levenshtein():
    """
    Test that Levenshtein partial match either returns results or {}
    """

    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    assert ranker.method_name == "levenshtein"
    
    ranker.load()

    ranker.mentions_to_wikidata = {"London": {"Q84": 0.922}}
    ranker.cache = {}

    candidates = ranker.run("London")
    assert candidates.get("London").string_similarity == 1.0

    ranker.cache = {}
    candidates = ranker.run("Lvndvn")
    assert candidates.get("London").string_similarity == pytest.approx(0.66666665, abs=10e-6)

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": {"Q60": 0.884}}

    ranker.cache = {}
    candidates = ranker.run("New York")
    assert candidates.mention == "New York"
    assert candidates.get("New York City").string_similarity == pytest.approx(0.615384615, abs=10e-6)

    ranker.cache = {}
    candidates = ranker.run("Lvndvn")
    assert candidates.mention == "Lvndvn"
    assert candidates.get("New York City").string_similarity == 0.0

    ranker.cache = {}
    candidates = ranker.run("asdasd")
    assert candidates.mention == "asdasd"
    assert candidates.get("New York City").string_similarity == 0.0


@pytest.mark.skip(reason="Needs deezy model")
def test_ranking_deezy_on_the_fly(tmp_path):
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir,"../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir, "../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "cosine",
            "selection_threshold": 0.9,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )
    assert ranker.method_name == "deezymatch"
    
    ranker.load()

    # Test that perfect_match acts before deezy
    candidates = ranker.run("London")
    assert candidates.mention == "London"
    assert candidates.get("London").string_similarity == 1.0

    # Test that deezy works
    ranker.cache = {}
    candidates = ranker.run("Ashton-cnderLyne")
    assert candidates.mention == "Ashton-cnderLyne"
    assert candidates.ranking_method == "deezymatch"

    assert len(candidates.matches) == 3
    assert (0.0 < candidates.get("Ashton under Lyne").string_similarity < 1.0)
    assert (0.0 < candidates.get("Ashton-under-Lyne").string_similarity < 1.0)
    assert (0.0 < candidates.get("Aston-under-Lynne").string_similarity < 1.0)


@pytest.mark.skip(reason="Needs deezy model")
def test_ranking_attach_wikidata(tmp_path):
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir,"../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "cosine",
            "selection_threshold": 0.9,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": True,
        },
    )
    ranker.load(train=False)

    # Check the cache is empty.
    assert len(ranker.cache) == 0

    candidates = ranker.run("London")

    assert candidates.mention == "London"
    assert isinstance(candidates.get("London"), StringMatchLinks)
    assert candidates.get("London").variation == "London"
    # Test that perfect_match acts before deezy
    assert candidates.get("London").string_similarity == 1.0
    assert len(candidates.get("London").wqid_links) == 194
    assert "Q84" in candidates.get("London").wqid_links

    # Check the cache has been updated.
    assert len(ranker.cache) == 1
    assert ranker.cache["London"] == candidates

    # Test that deezy works
    # TODO: add a ranker.clear_cache() method.
    ranker.cache = {}
    candidates = ranker.run("Sheftield")
    assert candidates.mention == "Sheftield"
    assert isinstance(candidates.get("Sheffield"), StringMatchLinks)
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert (0.0 < candidates.get("Sheffield").string_similarity < 1.0)
    assert len(candidates.get("Sheffield").wqid_links) == 50
    assert "Q42448" in candidates.get("Sheffield").wqid_links

    # Test that Perfect Match works
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    candidates = ranker.run("Sheffield")
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert candidates.get("Sheffield").string_similarity == 1.0
    assert "Q42448" in candidates.get("Sheffield").wqid_links

    ranker.cache = {}
    candidates = ranker.run("Sheftield")
    assert candidates.is_empty()

    # Test that check if contained works
    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    # Test that levenshtein works
    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    candidates = ranker.run("Sheffield")
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert candidates.get("Sheffield").string_similarity == 1.0
    assert "Q42448" in candidates.get("Sheffield").wqid_links

    ranker.cache = {}
    candidates = ranker.run("Sheftield")
    assert candidates.mention == "Sheftield"
    assert (0.0 < candidates.get("Sheffield").string_similarity < 1.0)
    assert "Q42448" in candidates.get("Sheffield").wqid_links