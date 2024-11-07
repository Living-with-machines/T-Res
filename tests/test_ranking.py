import os
from pathlib import Path

import pytest

from t_res.geoparser import ranking

current_dir = Path(__file__).parent.resolve()

def test_ranking_data_classes():
    """
    Test the data classes that represent ranking candidates.
    """

    # Legacy example:
    # {'London': 1.0}
    string_match = ranking.StringMatch('London', 1.0)
    assert string_match.variation == 'London'
    assert string_match.string_similarity == 1.0

    # Legacy example:
    # {'Q619055': 0.03571428571428571}
    wikidata_match = ranking.WikidataMatch('Q619055', 0.03571428571428571)
    assert wikidata_match.wqid == 'Q619055'
    assert wikidata_match.freq == 0.03571428571428571

    # Legacy example:
    # {'Shielfield': {'Score': 0.9387, 'Candidates': {'Q619055': 0.03571428571428571, 'Q5953687': 0.22857142857142856}}
    candidate_match = ranking.CandidateMatch('Shielfield', 0.9387, [
        ranking.WikidataMatch('Q619055', 0.03571428571428571),
        ranking.WikidataMatch('Q5953687', 0.22857142857142856),
    ])
    assert candidate_match.variation == 'Shielfield'
    assert candidate_match.string_similarity == 0.9387
    # Wikidata candidates are in order of decreasing freq.
    assert candidate_match.wikidata_candidates[0].wqid == 'Q5953687'
    assert candidate_match.wikidata_candidates[0].freq == 0.22857142857142856
    assert candidate_match.wikidata_candidates[1].wqid == 'Q619055'
    assert candidate_match.wikidata_candidates[1].freq == 0.03571428571428571

    # Legacy example without wikidata:
    # {'Sheftield': {'Shielfield': 0.9387, 'Sheffield': 0.9228, 'Shelfield': 0.8947}}
    candidates = ranking.Candidates('Sheftield', [
        ranking.StringMatch('Shielfield', 0.9387),
        ranking.StringMatch('Sheffield', 0.9228),
        ranking.StringMatch('Shelfield', 0.8947),
    ])
    assert candidates.mention == 'Sheftield'
    assert len(candidates.matches) == 3

    candidates = ranking.Candidates('Sheftield', [
        ranking.CandidateMatch('Sheffield', 0.9228, [
            ranking.WikidataMatch('Q6707254', 0.0410958904109589),
            ranking.WikidataMatch('Q7492778', 0.20202020202020204),
            ranking.WikidataMatch('Q1421317', 0.03875968992248062),
        ]), 
        ranking.CandidateMatch('Shelfield', 0.8947, [
            ranking.WikidataMatch('Q7493600', 1.0),
        ]),
        ranking.CandidateMatch('Shielfield', 0.9387, [
            ranking.WikidataMatch('Q619055', 0.03571428571428571),
            ranking.WikidataMatch('Q5953687', 0.22857142857142856),
        ]),
    ])
    assert candidates.mention == 'Sheftield'
    assert len(candidates.matches) == 3
    # matches are in order of decreasing string similarity.
    assert candidates.matches[0].variation == 'Shielfield'
    assert candidates.matches[0].string_similarity == 0.9387
    assert len(candidates.matches[0].wikidata_candidates) == 2
    assert candidates.matches[1].variation == 'Sheffield'
    assert candidates.matches[1].string_similarity == 0.9228
    assert len(candidates.matches[1].wikidata_candidates) == 3
    assert candidates.matches[2].variation == 'Shelfield'
    assert candidates.matches[2].string_similarity == 0.8947
    assert len(candidates.matches[2].wikidata_candidates) == 1

def test_ranking_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    
    ranker.load_resources()
    ranker.already_collected_cands = {}
    candidates = ranker.run(["London"])
    assert candidates["London"]["London"] == 1.0

    candidates = ranker.run(["Lvndon"])
    assert candidates["Lvndon"] == {}

    candidates = ranker.run(["Paperopoli"])
    assert candidates["Paperopoli"] == {}

def test_ranking_matching_score():
    """
    Test that matching_score returns score only when there is an overlap
    """

    # Test the overlap matching score.
    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    
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
    
    ranker.load_resources()

    score = ranker.matching_score("Lvndon", {"mentions": "London"})
    assert score == 0.8333333283662796

    score = ranker.matching_score("uityity", {"mentions": "asdasd"})
    assert score == 0.0

    with pytest.raises(TypeError):
        ranker.matching_score("Lvndon", "London")

    # ranker.already_collected_cands = {}

    # candidates, already_collected_cands = ranker.run(["asdasd"])
    # assert candidates["asdasd"] == {"New York City": 0.0}


def test_ranking_partial_match():
    """
    Test that partial match either returns results or {}
    """

    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )

    ranker.load_resources()

    # Test that perfect_match acts before partial match
    ranker.mentions_to_wikidata = {"London": "Q84"}
    ranker.already_collected_cands = {}
    candidates = ranker.run(["London"])
    assert candidates["London"]["London"] == 1.0

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": "Q60"}

    ranker.already_collected_cands = {}
    candidates = ranker.run(["New York"])
    assert candidates["New York"]["New York City"] == pytest.approx(0.6153846153846154, abs=10e-6)

    ranker.already_collected_cands = {}
    candidates = ranker.run(["Lvndvn"])
    assert candidates["Lvndvn"] == {}


def test_ranking_levenshtein():
    """
    Test that Levenshtein partial match either returns results or {}
    """

    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )

    ranker.load_resources()

    # Test that perfect_match acts before partial match
    ranker.mentions_to_wikidata = {"London": "Q84"}
    ranker.already_collected_cands = {}
    candidates = ranker.run(["London"])
    assert candidates["London"]["London"] == 1.0

    ranker.already_collected_cands = {}
    candidates = ranker.run(["Lvndvn"])
    assert candidates["Lvndvn"]["London"] == pytest.approx(0.6666666567325592, abs=10e-6)

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": "Q60"}

    ranker.already_collected_cands = {}
    candidates = ranker.run(["New York"])
    assert candidates["New York"]["New York City"] == pytest.approx(0.6153846153846154, abs=10e-6)

    ranker.already_collected_cands = {}
    candidates = ranker.run(["Lvndvn"])
    assert candidates["Lvndvn"] == {"New York City": 0.0}

    ranker.already_collected_cands = {}
    candidates = ranker.run(["asdasd"])
    assert candidates["asdasd"] == {"New York City": 0.0}


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
        already_collected_cands=dict(),
    )

    # Test that perfect_match acts before deezy
    ranker.load_resources()
    candidates = ranker.run(["London"])
    assert candidates["London"]["London"] == 1.0

    # Test that deezy works
    ranker.already_collected_cands = {}
    candidates = ranker.run(["Ashton-cnderLyne"])

    assert (0.0 < candidates["Ashton-cnderLyne"]["Ashton-under-Lyne"] < 1.0)

@pytest.mark.skip(reason="Needs deezy model")
def test_ranking_find_candidates(tmp_path):
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
        already_collected_cands=dict(),
    )

    # Test that perfect_match acts before deezy
    ranker.load_resources(train=False)
    candidates = ranker.find_candidates([{"mention": "London"}])
    assert candidates["London"]["London"]["Score"] == 1.0
    assert "Q84" in candidates["London"]["London"]["Candidates"]

    # Test that deezy works
    ranker.already_collected_cands = {}
    candidates = ranker.find_candidates([{"mention": "Sheftield"}])
    assert (0.0 < candidates["Sheftield"]["Sheffield"]["Score"] < 1.0)
    assert "Q42448" in candidates["Sheftield"]["Sheffield"]["Candidates"]

    # Test that Perfect Match works
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    # Test that perfect_match acts before deezy
    ranker.load_resources()
    candidates = ranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    ranker.already_collected_cands = {}
    candidates = ranker.find_candidates([{"mention": "Sheftield"}])
    assert candidates["Sheftield"] == {}

    # Test that check if contained works
    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    # Test that perfect_match acts before partialmatch
    ranker.load_resources()

    candidates = ranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    ranker.already_collected_cands = {}

    candidates = ranker.find_candidates([{"mention": "Sheftield"}])
    assert "Sheffield" not in candidates["Sheftield"]

    # Test that levenshtein works
    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    # Test that perfect_match acts before partialmatch
    ranker.load_resources()

    candidates = ranker.find_candidates([{"mention": "Sheffield"}])
    assert candidates["Sheffield"]["Sheffield"]["Score"] == 1.0
    assert "Q42448" in candidates["Sheffield"]["Sheffield"]["Candidates"]

    ranker.already_collected_cands = {}
    candidates = ranker.find_candidates([{"mention": "Sheftield"}])
    assert (0.0 < candidates["Sheftield"]["Sheffield"]["Score"] < 1.0)
    assert "Q42448" in candidates["Sheftield"]["Sheffield"]["Candidates"]