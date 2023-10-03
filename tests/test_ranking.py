import json
import os
import sys
from pathlib import Path

import pytest
from DeezyMatch import candidate_ranker

from t_res.geoparser import ranking


def test_initialise_method():
    """
    Test initialisation works fine
    """
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="resources/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )
    assert type(myranker.__str__()) == str


def test_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="resources/",
    )
    myranker.mentions_to_wikidata = myranker.load_resources()
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
        resources_path="resources/",
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
        resources_path="resources/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
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
        resources_path="resources/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    myranker.mentions_to_wikidata = myranker.load_resources()

    # Test that perfect_match acts before partial match
    myranker.mentions_to_wikidata = {"London": "Q84"}
    candidates, already_collected_cands = myranker.partial_match(["London"], damlev=False)
    assert candidates["London"]["London"] == 1.0

    # Test that damlev works
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(["Lvndvn"], damlev=True)
    assert candidates["Lvndvn"]["London"] == 0.6666666567325592

    # Test that overlap works properly
    myranker.mentions_to_wikidata = {"New York City": "Q60"}
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(["New York"], damlev=False)
    assert candidates["New York"]["New York City"] == 0.6153846153846154

    myranker.mentions_to_wikidata = {"New York City": "Q60"}
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(["Lvndvn"], damlev=False)
    assert candidates["Lvndvn"] == {}

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.partial_match(["asdasd"], damlev=True)
    assert candidates["asdasd"] == {"New York City": 0.0}


def test_deezy_on_the_fly():
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": "resources/models/",
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": "resources/deezymatch/",
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
    myranker.mentions_to_wikidata = myranker.load_resources()
    candidates, already_collected_cands = myranker.deezy_on_the_fly(["London"])
    assert candidates["London"]["London"] == 1.0

    # Test that deezy works
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.deezy_on_the_fly(["Ashton-cnderLyne"])
    assert (
        candidates["Ashton-cnderLyne"]["Ashton-under-Lyne"] > 0.0
        and candidates["Ashton-cnderLyne"]["Ashton-under-Lyne"] < 1.0
    )


def test_find_candidates():
    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": "resources/models/",
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": "resources/deezymatch/",
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
    myranker.mentions_to_wikidata = myranker.load_resources()
    candidates, already_collected_cands = myranker.find_candidates([{"mention": "London"}])
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    # Test that deezy works
    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheftield"}])
    assert (
        candidates["Sheftield"]["Sheffield"]["Score"] > 0.0
        and candidates["Sheftield"]["Sheffield"]["Score"] < 1.0
    )
    assert "Q42448" in candidates["Sheftield"]["Sheffield"]["Candidates"]

    # Test that Perfect Match works
    myranker.method = "perfectmatch"

    # Test that perfect_match acts before deezy
    myranker.mentions_to_wikidata = myranker.load_resources()
    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheftield"}])
    assert candidates["Sheftield"] == {}

    # Test that check if contained works
    myranker.method = "partialmatch"

    # Test that perfect_match acts before partialmatch
    myranker.mentions_to_wikidata = myranker.load_resources()

    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheftield"}])
    assert "Sheffield" not in candidates["Sheftield"]

    # Test that levenshtein works
    myranker.method = "levenshtein"

    # Test that perfect_match acts before partialmatch
    myranker.mentions_to_wikidata = myranker.load_resources()

    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    myranker.already_collected_cands = {}

    candidates, already_collected_cands = myranker.find_candidates([{"mention": "Sheftield"}])
    assert (
        candidates["Sheftield"]["Sheffield"]["Score"] > 0.0
        and candidates["Sheftield"]["Sheffield"]["Score"] < 1.0
    )
    assert "Q42448" in candidates["Sheftield"]["Sheffield"]["Candidates"]


def test_deezy_candidate_ranker():
    deezy_parameters = {
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": str(Path("resources/deezymatch/").resolve()),
        "dm_cands": "wkdtalts",
        "dm_model": "w2v_ocr",
        "dm_output": "deezymatch_on_the_fly",
        # Ranking measures:
        "ranking_metric": "faiss",
        "selection_threshold": 50,
        "num_candidates": 1,
        "verbose": False,
        # DeezyMatch training:
        "overwrite_training": False,
        "do_test": False,
    }

    dm_path = deezy_parameters["dm_path"]
    dm_cands = deezy_parameters["dm_cands"]
    dm_model = deezy_parameters["dm_model"]
    dm_output = deezy_parameters["dm_output"]

    query = ["-", "ST G", "• - , i", "- P", "• FERRIS"]

    candidates, already_collected_cands = candidate_ranker(
        candidate_scenario=os.path.join(dm_path, "combined", dm_cands + "_" + dm_model),
        query=query,
        ranking_metric=deezy_parameters["ranking_metric"],
        selection_threshold=deezy_parameters["selection_threshold"],
        num_candidates=deezy_parameters["num_candidates"],
        search_size=deezy_parameters["num_candidates"],
        verbose=deezy_parameters["verbose"],
        output_path=os.path.join(dm_path, "ranking", dm_output),
        pretrained_model_path=os.path.join(
            f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".model"
        ),
        pretrained_vocab_path=os.path.join(
            f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".vocab"
        ),
    )
    assert len(candidates) == len(query)
