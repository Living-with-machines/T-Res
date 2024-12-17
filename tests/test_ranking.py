import os
from pathlib import Path

import pytest

from t_res.geoparser.ranking import *
from t_res.utils.dataclasses import *

current_dir = Path(__file__).parent.resolve()

def test_new():
    # Test Ranker construction via string parameters.

    # If a required parameter is omitted, expect a TypeError.
    kwargs = {
        'method_name': 'perfectmatch',
        }
    with pytest.raises(TypeError):
        ranker = Ranker.new(**kwargs)

    kwargs = {
        'method_name': 'perfectmatch',
        'resources_path': 'sample_files/resources/',
        }
    ranker = Ranker.new(**kwargs)
    assert isinstance(ranker, PerfectMatchRanker)
    assert ranker.method_name == 'perfectmatch'
    assert ranker.mentions_to_wikidata == dict()

    kwargs = {
        'method_name': 'levenshtein',
        'resources_path': 'sample_files/resources/',
        }
    ranker = Ranker.new(**kwargs)
    assert isinstance(ranker, LevenshteinRanker)
    assert ranker.method_name == 'levenshtein'

    kwargs = {
        'method_name': 'deezymatch',
        'resources_path': 'sample_files/resources/',
        }
    ranker = Ranker.new(**kwargs)
    assert isinstance(ranker, DeezyMatchRanker)
    assert ranker.method_name == 'deezymatch'

    # If the ranking method is invalid, expect a ValueError.
    kwargs = {
        'method_name': 'nosuchmatch',
        }
    with pytest.raises(ValueError):
        ranker = Ranker.new(**kwargs)

def test_ranking_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    ranker = PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "sample_files/resources/"),
    )
    assert ranker.method_name == "perfectmatch"
    
    ranker.load()
    ranker.cache = {}
    
    # Check the cache is empty.
    assert len(ranker.cache) == 0

    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.ranking_method == "perfectmatch"
    assert candidates.mention.mention == "London"
    assert candidates.get("London").variation == "London"
    assert candidates.get("London").string_similarity == 1.0 

    # Check the cache has been updated.
    assert len(ranker.cache) == 1
    assert ranker.cache["London"] == candidates.matches

    mention = Mention("Lvndon", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.ranking_method == "perfectmatch"
    assert candidates.mention.mention == "Lvndon"
    assert candidates.is_empty()

    # Check the cache has been updated.
    assert len(ranker.cache) == 2
    assert ranker.cache["Lvndon"] == candidates.matches

    # Construct a dummy mention for the test.
    mention = Mention("Paperopoli", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.ranking_method == "perfectmatch"
    assert candidates.mention.mention == "Paperopoli"
    assert candidates.is_empty()

    # Check the cache has been updated.
    assert len(ranker.cache) == 3
    assert ranker.cache["Paperopoli"] == candidates.matches

def test_ranking_matching_score():
    """
    Test that matching_score returns score only when there is an overlap
    """

    # Test the overlap matching score.
    ranker = PartialMatchRanker(
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
    ranker = LevenshteinRanker(
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

    ranker = PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    assert ranker.method_name == "partialmatch"
    
    ranker.load()

    ranker.mentions_to_wikidata = {"London": {"Q84": 0.922}}
    ranker.cache = {}

    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.ranking_method == "partialmatch"
    assert candidates.mention.mention == "London"
    assert candidates.get("London").variation == "London"
    assert candidates.get("London").string_similarity == 1.0 

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": {"Q60": 0.884}}

    ranker.cache = {}
    # Construct a dummy mention for the test.
    mention = Mention("New York", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.mention.mention == "New York"
    assert candidates.get("New York City").variation == "New York City"
    assert candidates.get("New York City").string_similarity == pytest.approx(0.615384, abs=10e-6)

    ranker.cache = {}
    mention = Mention("Lvndvn", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.mention.mention == "Lvndvn"
    assert candidates.is_empty()


def test_ranking_levenshtein():
    """
    Test that Levenshtein partial match either returns results or {}
    """

    ranker = LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    assert ranker.method_name == "levenshtein"
    
    ranker.load()

    ranker.mentions_to_wikidata = {"London": {"Q84": 0.922}}
    ranker.cache = {}

    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.get("London").string_similarity == 1.0

    ranker.cache = {}
    # Construct a dummy mention for the test.
    mention = Mention("Lvndvn", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.get("London").string_similarity == pytest.approx(0.66666665, abs=10e-6)

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": {"Q60": 0.884}}

    ranker.cache = {}
    # Construct a dummy mention for the test.
    mention = Mention("New York", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.mention.mention == "New York"
    assert candidates.get("New York City").string_similarity == pytest.approx(0.615384615, abs=10e-6)

    ranker.cache = {}
    mention = Mention("Lvndvn", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.mention.mention == "Lvndvn"
    assert candidates.get("New York City").string_similarity == 0.0

    ranker.cache = {}
    mention = Mention("asdasd", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.mention.mention == "asdasd"
    assert candidates.get("New York City").string_similarity == 0.0


@pytest.mark.resources(reason="Needs deezy model")
def test_ranking_deezy_on_the_fly(tmp_path):
    ranker = DeezyMatchRanker(
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

    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    # Test that perfect_match acts before deezy
    assert candidates.mention.mention == "London"
    assert candidates.get("London").string_similarity == 1.0

    # Test that deezy works
    ranker.cache = {}
    # Construct a dummy mention for the test.
    mention = Mention("Ashton-cnderLyne", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.mention.mention == "Ashton-cnderLyne"
    assert candidates.ranking_method == "deezymatch"

    assert len(candidates.matches) == 3
    assert (0.0 < candidates.get("Ashton under Lyne").string_similarity < 1.0)
    assert (0.0 < candidates.get("Ashton-under-Lyne").string_similarity < 1.0)
    assert (0.0 < candidates.get("Aston-under-Lynne").string_similarity < 1.0)


@pytest.mark.resources(reason="Needs deezy model")
def test_ranking_attach_wikidata(tmp_path):
    ranker = DeezyMatchRanker(
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

    # Construct a dummy mention for the test.
    mention = Mention("London", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)

    assert candidates.mention.mention == "London"
    assert isinstance(candidates.get("London"), StringMatchLinks)
    assert candidates.get("London").variation == "London"
    # Test that perfect_match acts before deezy
    assert candidates.get("London").string_similarity == 1.0
    assert len(candidates.get("London").wqid_links) == 194
    assert "Q84" in candidates.get("London").wqid_links

    # Check the cache has been updated.
    assert len(ranker.cache) == 1
    assert ranker.cache["London"] == candidates.matches

    # Test that deezy works
    # TODO: add a ranker.clear_cache() method.
    ranker.cache = {}

    # Construct a dummy mention for the test.
    mention = Mention("Sheftield", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.mention.mention == "Sheftield"
    assert isinstance(candidates.get("Sheffield"), StringMatchLinks)
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert (0.0 < candidates.get("Sheffield").string_similarity < 1.0)
    assert len(candidates.get("Sheffield").wqid_links) == 50
    assert "Q42448" in candidates.get("Sheffield").wqid_links

    # Test that Perfect Match works
    ranker = PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    # Construct a dummy mention for the test.
    mention = Mention("Sheffield", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert candidates.get("Sheffield").string_similarity == 1.0
    assert "Q42448" in candidates.get("Sheffield").wqid_links

    ranker.cache = {}
    mention = Mention("Sheftield", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.is_empty()

    # Test that check if contained works
    ranker = PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    # Test that levenshtein works
    ranker = LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load()

    # Construct a dummy mention for the test.
    mention = Mention("Sheffield", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert candidates.get("Sheffield").string_similarity == 1.0
    assert "Q42448" in candidates.get("Sheffield").wqid_links

    ranker.cache = {}
    mention = Mention("Sheftield", 0, 0, 0, 0.0, 'LOC', 'O')
    candidates = ranker.run(mention)
    assert candidates.mention.mention == "Sheftield"
    assert (0.0 < candidates.get("Sheffield").string_similarity < 1.0)
    assert "Q42448" in candidates.get("Sheffield").wqid_links