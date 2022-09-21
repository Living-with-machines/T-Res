import json
import os
import sys

import pytest

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import ranking


def test_initialise_method():
    """
    Test initialisation works fine
    """
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )
    assert type(myranker.__str__()) == str


def test_load_resources():
    """
    Tests resources are loaded and processed correctly
    """
    with open("/resources/wikidata/mentions_to_wikidata_normalized.json", "r") as f:
        mentions_to_wikidata = json.load(f)

    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
    )
    filtered_mentions = myranker.load_resources()
    assert len(filtered_mentions) < len(mentions_to_wikidata)
    assert len(filtered_mentions["London"]) < len(mentions_to_wikidata["London"])
    assert len(filtered_mentions["London"]) > 0
    assert filtered_mentions["London"]["Q84"] == mentions_to_wikidata["London"]["Q84"]


def test_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
    )
    myranker.load_resources()
    candidates, already_collected_cands = myranker.perfect_match(["London"])
    assert candidates["London"]["London"] == 1.0

    candidates, already_collected_cands = myranker.perfect_match(["Lvndon"])
    assert candidates["Lvndon"] == {}

    candidates, already_collected_cands = myranker.perfect_match(["Paperopoli"])
    assert candidates["Paperopoli"] == {}


def test_damlev():
    """
    Test that damlev returns correctly
    """
    myranker = ranking.Ranker(
        method="partialmatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
    )
    score = myranker.damlev_dist("Lvndon", {"mentions": "London"})
    assert score == 0.8333333283662796

    with pytest.raises(TypeError):
        found = True
        myranker.damlev_dist("Lvndon", "London")
    assert found == True

    assert 0.0 == myranker.damlev_dist("uityity", {"mentions": "asdasd"})


def test_check_if_contained():
    """
    Test that check_if_contained returns score only when there is an overlap
    """

    myranker = ranking.Ranker(
        method="partialmatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
    )
    score_a = myranker.check_if_contained("New York", {"mentions": "New York City"})
    score_b = myranker.check_if_contained("New York City", {"mentions": "New York"})

    assert score_a == score_b == 0.6153846153846154

    with pytest.raises(TypeError):
        found = True
        myranker.check_if_contained("Lvndon", "London")
    assert found == True

    assert None == myranker.check_if_contained("London", {"mentions": "New York"})


def test_partial_match():

    """
    Test that partial match either returns results or {}
    """

    myranker = ranking.Ranker(
        method="partialmatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
    )

    # Test that perfect_match acts before partial match
    myranker.load_resources()
    candidates, already_collected_cands = myranker.partial_match(
        ["London"], damlev=False
    )
    assert candidates["London"]["London"] == 1.0

    # Test that damlev works

    myranker.mentions_to_wikidata = {"London": "Q84"}
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(
        ["Lvndvn"], damlev=True
    )
    assert candidates["Lvndvn"]["London"] == 0.6666666567325592

    # Test that overlap works properly

    myranker.mentions_to_wikidata = {"New York City": "Q60"}
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(
        ["New York"], damlev=False
    )
    assert candidates["New York"]["New York City"] == 0.6153846153846154

    myranker.mentions_to_wikidata = {"New York City": "Q60"}

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(
        ["Lvndvn"], damlev=False
    )
    assert candidates["Lvndvn"] == {}

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(
        ["asdasd"], damlev=True
    )
    assert candidates["asdasd"] == {"New York City": 0.0}


def test_deezy_on_the_fly():
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": "experiments/outputs/deezymatch/",
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
            "w2v_ocr_path": "experiments/outputs/models/",
            "w2v_ocr_model": "w2v_1800s_news",
            "do_test": True,
        },
    )

    # Test that perfect_match acts before deezy
    myranker.load_resources()
    candidates, already_collected_cands = myranker.deezy_on_the_fly(["London"])
    assert candidates["London"]["London"] == 1.0

    # Test that deezy works

    myranker.mentions_to_wikidata = {"London": "Q84"}
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.deezy_on_the_fly(["Lvndon"])
    assert candidates["Lvndon"]["London"] > 0.0 and candidates["Lvndon"]["London"] < 1.0


def test_find_candidates():

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="/resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": "experiments/outputs/deezymatch/",
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
            "w2v_ocr_path": "experiments/outputs/models/",
            "w2v_ocr_model": "w2v_1800s_news",
            "do_test": True,
        },
    )

    # Test that perfect_match acts before deezy
    myranker.load_resources()
    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "London"}]
    )
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    # Test that deezy works
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "Lvndon"}]
    )
    assert (
        candidates["Lvndon"]["London"]["Score"] > 0.0
        and candidates["Lvndon"]["London"]["Score"] < 1.0
    )
    assert "Q84" in candidates["Lvndon"]["London"]["Candidates"]

    # Test that Perfect Match works
    myranker.method = "perfectmatch"

    # Test that perfect_match acts before deezy
    myranker.load_resources()
    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "London"}]
    )
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "Lvndon"}]
    )
    assert candidates["Lvndon"] == {}

    # Test that check if contained works
    myranker.method = "partialmatch"

    # Test that perfect_match acts before partialmatch
    myranker.load_resources()
    myranker.mentions_to_wikidata = {"London": "Q84"}

    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "London"}]
    )
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "Lvndvn"}]
    )
    assert "London" not in candidates["Lvndvn"]

    # Test that levenshtein works
    myranker.method = "levenshtein"

    # Test that perfect_match acts before partialmatch
    myranker.load_resources()
    myranker.mentions_to_wikidata = {"London": "Q84"}

    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "London"}]
    )
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates(
        [{"mention": "Lvndvn"}]
    )
    assert (
        candidates["Lvndvn"]["London"]["Score"] > 0.0
        and candidates["Lvndvn"]["London"]["Score"] < 1.0
    )
    assert "Q84" in candidates["Lvndvn"]["London"]["Candidates"]
