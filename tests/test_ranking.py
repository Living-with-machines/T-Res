import os
from pathlib import Path

import pytest

from t_res.geoparser import ranking

current_dir = Path(__file__).parent.resolve()

def test_ranking_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    myranker.load_resources()

    candidates = myranker.perfect_match(["London"])
    assert candidates["London"] == {'London': 1.0}

    candidates = myranker.perfect_match(["Lvndon"])
    assert candidates["Lvndon"] == {}

    candidates = myranker.perfect_match(["Paperopoli"])
    assert candidates["Paperopoli"] == {}


def test_ranking_damlev():
    """
    Test that damlev returns correctly
    """
    myranker = ranking.Ranker(
        method="partialmatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )

    score = myranker.damlev_dist("Lvndon", {"mentions": "London"})
    assert score == 0.8333333283662796

    score = myranker.damlev_dist("uityity", {"mentions": "asdasd"})
    assert score == 0.0

    with pytest.raises(TypeError):
        myranker.damlev_dist("Lvndon", "London")


def test_ranking_check_if_contained():
    """
    Test that check_if_contained returns score only when there is an overlap
    """

    myranker = ranking.Ranker(
        method="partialmatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    
    score_a = myranker.check_if_contained("New York", {"mentions": "New York City"})
    score_b = myranker.check_if_contained("New York City", {"mentions": "New York"})
    assert score_a == score_b == 0.6153846153846154

    with pytest.raises(TypeError):
        myranker.check_if_contained("Lvndon", "London")

    score = myranker.check_if_contained("London", {"mentions": "New York"})
    assert score is None


def test_ranking_partial_match():
    """
    Test that partial match either returns results or {}
    """

    myranker = ranking.Ranker(
        method="partialmatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    myranker.load_resources()

    # Test that perfect_match acts before partial match
    myranker.mentions_to_wikidata = {"London": "Q84"}
    candidates = myranker.partial_match(["London"], damlev=False)
    assert candidates["London"]["London"] == 1.0

    # Test that damlev works
    myranker.already_collected_cands = {}
    candidates = myranker.partial_match(["Lvndvn"], damlev=True)
    assert candidates["Lvndvn"]["London"] == 0.6666666567325592

    # Test that overlap works properly
    myranker.mentions_to_wikidata = {"New York City": "Q60"}
    myranker.already_collected_cands = {}
    candidates = myranker.partial_match(["New York"], damlev=False)
    assert candidates["New York"]["New York City"] == 0.6153846153846154

    myranker.already_collected_cands = {}
    candidates = myranker.partial_match(["Lvndvn"], damlev=False)
    assert candidates["Lvndvn"] == {}

    myranker.already_collected_cands = {}
    candidates = myranker.partial_match(["asdasd"], damlev=True)
    assert candidates["asdasd"] == {"New York City": 0.0}

@pytest.mark.deezy(reason="Needs deezy model")
def test_ranking_deezy_on_the_fly(tmp_path):
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
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
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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

    # Test that perfect_match acts before deezy
    myranker.load_resources()
    candidates = myranker.deezy_on_the_fly(["London"])
    assert candidates["London"]["London"] == 1.0

    # Test that deezy works
    myranker.already_collected_cands = {}
    candidates = myranker.deezy_on_the_fly(["Ashton-cnderLyne"])
    assert (0.0 < candidates["Ashton-cnderLyne"]["Ashton-under-Lyne"] < 1.0)

@pytest.mark.deezy(reason="Needs deezy model")
def test_ranking_find_candidates(tmp_path):
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
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
            "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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

    # Test that perfect_match acts before deezy
    myranker.load_resources()
    candidates = myranker.find_candidates([{"mention": "London"}])
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    # Test that deezy works
    myranker.already_collected_cands = {}
    candidates = myranker.find_candidates([{"mention": "Sheftield"}])
    assert (0.0 < candidates["Sheftield"]["Sheffield"]["Score"] < 1.0)
    assert "Q42448" in candidates["Sheftield"]["Sheffield"]["Candidates"]

    # Test that Perfect Match works
    myranker.method = "perfectmatch"

    # Test that perfect_match acts before deezy
    myranker.load_resources()
    candidates = myranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    myranker.already_collected_cands = {}
    candidates = myranker.find_candidates([{"mention": "Sheftield"}])
    assert candidates["Sheftield"] == {}

    # Test that check if contained works
    myranker.method = "partialmatch"

    # Test that perfect_match acts before partialmatch
    myranker.load_resources()
    candidates = myranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    myranker.already_collected_cands = {}
    candidates = myranker.find_candidates([{"mention": "Sheftield"}])
    assert "Sheffield" not in candidates["Sheftield"]

    # Test that levenshtein works
    myranker.method = "levenshtein"

    # Test that perfect_match acts before partialmatch
    myranker.load_resources()
    candidates = myranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    myranker.already_collected_cands = {}
    candidates = myranker.find_candidates([{"mention": "Sheftield"}])
    assert (0.0 < candidates["Sheftield"]["Sheffield"]["Score"] < 1.0)
    assert "Q42448" in candidates["Sheftield"]["Sheffield"]["Candidates"]
