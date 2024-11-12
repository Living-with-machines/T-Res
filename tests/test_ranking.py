import os
from pathlib import Path

import pytest

from t_res.geoparser import ranking
from t_res.geoparser.dataclasses import Candidates, StringMatch, WikidataMatch, CandidateMatch

current_dir = Path(__file__).parent.resolve()

def test_ranking_data_classes():
    """
    Test the data classes that represent ranking candidates.
    """

    string_match = StringMatch(None, None)
    assert string_match.is_empty()

    # Legacy example:
    # {'London': 1.0}
    string_match = StringMatch('London', 1.0)
    assert not string_match.is_empty()
    assert string_match.variation == 'London'
    assert string_match.string_similarity == 1.0

    # Legacy example:
    # {'Q619055': 0.03571428571428571}
    wikidata_match = WikidataMatch('Q619055', normalized_score=0.03571428571428571, freq=22)
    assert wikidata_match.wqid == 'Q619055'
    assert wikidata_match.normalized_score == 0.03571428571428571
    assert wikidata_match.freq == 22

    # Legacy example:
    # {'Shielfield': {'Score': 0.9387, 'Candidates': {'Q619055': 0.03571428571428571, 'Q5953687': 0.22857142857142856}}
    wikidata_matches = [
        WikidataMatch('Q619055', 0.03571428571428571, 5),
        WikidataMatch('Q5953687', 0.22857142857142856, 33),
    ]
    candidate_match = CandidateMatch(variation='Shielfield', 
                                             string_similarity=0.9387, 
                                             wikidata_matches=wikidata_matches)
    assert candidate_match.variation == 'Shielfield'
    assert candidate_match.string_similarity == 0.9387
    # Wikidata candidates are in order of decreasing freq.
    assert candidate_match.wikidata_matches[0].wqid == 'Q5953687'
    assert candidate_match.wikidata_matches[0].freq == 33
    assert candidate_match.wikidata_matches[0].normalized_score == 0.22857142857142856
    assert candidate_match.wikidata_matches[1].wqid == 'Q619055'
    assert candidate_match.wikidata_matches[1].normalized_score == 0.03571428571428571
    assert candidate_match.wikidata_matches[1].freq == 5

    # Legacy example without wikidata:
    # {'Sheftield': {'Shielfield': 0.9387, 'Sheffield': 0.9228, 'Shelfield': 0.8947}}
    candidates = Candidates('Sheftield', "levenshtein", [
        ranking.StringMatch('Shielfield', 0.9387),
        ranking.StringMatch('Sheffield', 0.9228),
        ranking.StringMatch('Shelfield', 0.8947),
    ])
    assert candidates.mention == 'Sheftield'
    assert len(candidates.matches) == 3

    candidates = Candidates('Sheftield', "levenshtein", [
        CandidateMatch('Sheffield', 0.9228, [
            WikidataMatch('Q6707254', 0.0410958904109589, 5),
            WikidataMatch('Q7492778', 0.20202020202020204, 33),
            WikidataMatch('Q1421317', 0.03875968992248062, 22),
        ]), 
        CandidateMatch('Shelfield', 0.8947, [
            WikidataMatch('Q7493600', 1.0, 7),
        ]),
        CandidateMatch('Shielfield', 0.9387, [
            WikidataMatch('Q619055', 0.03571428571428571, 2),
            WikidataMatch('Q5953687', 0.22857142857142856, 55),
        ]),
    ])
    assert candidates.mention == 'Sheftield'
    assert len(candidates.matches) == 3
    # matches are in order of decreasing string similarity.
    assert candidates.matches[0].variation == 'Shielfield'
    assert candidates.matches[0].string_similarity == 0.9387
    assert len(candidates.matches[0].wikidata_matches) == 2
    assert candidates.matches[1].variation == 'Sheffield'
    assert candidates.matches[1].string_similarity == 0.9228
    assert len(candidates.matches[1].wikidata_matches) == 3
    assert candidates.matches[2].variation == 'Shelfield'
    assert candidates.matches[2].string_similarity == 0.8947
    assert len(candidates.matches[2].wikidata_matches) == 1

    # Wikidata candidates are in order of decreasing freq.
    assert candidates.matches[0].wikidata_matches[0].wqid == 'Q5953687'
    assert candidates.matches[0].wikidata_matches[0].freq == 55
    assert candidates.matches[0].wikidata_matches[0].normalized_score == 0.22857142857142856
    assert candidates.matches[0].wikidata_matches[1].wqid == 'Q619055'
    assert candidates.matches[0].wikidata_matches[1].freq == 2
    assert candidates.matches[0].wikidata_matches[1].normalized_score == 0.03571428571428571

    assert candidates.matches[1].wikidata_matches[0].wqid == 'Q7492778'
    assert candidates.matches[1].wikidata_matches[0].freq == 33
    assert candidates.matches[1].wikidata_matches[0].normalized_score == 0.20202020202020204
    assert candidates.matches[1].wikidata_matches[1].wqid == 'Q1421317'
    assert candidates.matches[1].wikidata_matches[1].freq == 22
    assert candidates.matches[1].wikidata_matches[1].normalized_score == 0.03875968992248062
    assert candidates.matches[1].wikidata_matches[2].wqid == 'Q6707254'
    assert candidates.matches[1].wikidata_matches[2].freq == 5
    assert candidates.matches[1].wikidata_matches[2].normalized_score == 0.0410958904109589

    # Test sort order of wikidata_matches is correct even when assigned 
    # after instantiation.
    candidates = Candidates('Sheftield', "levenshtein", [
        CandidateMatch('Sheffield', 0.9228, [
            WikidataMatch('Q6707254', freq=None, normalized_score=0.0410958904109589),
            WikidataMatch('Q7492778', freq=None, normalized_score=0.20202020202020204),
            WikidataMatch('Q1421317', freq=None, normalized_score=0.03875968992248062),
        ]), 
        CandidateMatch('Shielfield', 0.9387, [
            WikidataMatch('Q619055', freq=None, normalized_score=0.03571428571428571),
            WikidataMatch('Q5953687', freq=None, normalized_score=0.22857142857142856),
        ]),
    ])

    # Arbitrary order of Wikidata matches when freq is None.
    assert candidates.matches[0].wikidata_matches[0].wqid == 'Q619055'
    assert candidates.matches[0].wikidata_matches[0].freq == None
    assert candidates.matches[0].wikidata_matches[0].normalized_score == 0.03571428571428571
    assert candidates.matches[0].wikidata_matches[1].wqid == 'Q5953687'
    assert candidates.matches[0].wikidata_matches[1].freq == None
    assert candidates.matches[0].wikidata_matches[1].normalized_score == 0.22857142857142856

    # Assign Wikidata link frequencies after instantiation.
    candidates.matches[0].get('Q619055').freq = 2
    candidates.matches[0].get('Q5953687').freq = 55
    
    candidates.matches[1].get('Q6707254').freq = 5
    candidates.matches[1].get('Q7492778').freq = 33
    candidates.matches[1].get('Q1421317').freq = 22

    # Wikidata candidates are in order of decreasing freq 
    # (even when assigned after instantiation).
    assert candidates.matches[0].wikidata_matches[0].wqid == 'Q5953687'
    assert candidates.matches[0].wikidata_matches[0].freq == 55
    assert candidates.matches[0].wikidata_matches[0].normalized_score == 0.22857142857142856
    assert candidates.matches[0].wikidata_matches[1].wqid == 'Q619055'
    assert candidates.matches[0].wikidata_matches[1].freq == 2
    assert candidates.matches[0].wikidata_matches[1].normalized_score == 0.03571428571428571

    assert candidates.matches[1].wikidata_matches[0].wqid == 'Q7492778'
    assert candidates.matches[1].wikidata_matches[0].freq == 33
    assert candidates.matches[1].wikidata_matches[0].normalized_score == 0.20202020202020204
    assert candidates.matches[1].wikidata_matches[1].wqid == 'Q1421317'
    assert candidates.matches[1].wikidata_matches[1].freq == 22
    assert candidates.matches[1].wikidata_matches[1].normalized_score == 0.03875968992248062
    assert candidates.matches[1].wikidata_matches[2].wqid == 'Q6707254'
    assert candidates.matches[1].wikidata_matches[2].freq == 5
    assert candidates.matches[1].wikidata_matches[2].normalized_score == 0.0410958904109589

def test_ranking_perfect_match():
    """
    Test that perfect_match returns only perfect matching cases
    """
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()
    ranker.already_collected_cands = {}
    
    # Check the cache is empty.
    assert len(ranker.already_collected_cands) == 0

    candidates = ranker.run("London", attach_wikidata=False)

    assert candidates.method == "perfectmatch"
    assert candidates.mention == "London"
    assert candidates.get("London").variation == "London"
    assert candidates.get("London").string_similarity == 1.0 

    # Check the cache has been updated.
    assert len(ranker.already_collected_cands) == 1
    assert ranker.already_collected_cands["London"] == candidates

    # candidates = ranker.run(["Lvndon"])
    # assert candidates["Lvndon"] == {}
    candidates = ranker.run("Lvndon", attach_wikidata=False)

    assert candidates.method == "perfectmatch"
    assert candidates.mention == "Lvndon"
    assert candidates.is_empty()

    # Check the cache has been updated.
    assert len(ranker.already_collected_cands) == 2
    assert ranker.already_collected_cands["Lvndon"] == candidates

    candidates = ranker.run("Paperopoli", attach_wikidata=False)

    assert candidates.method == "perfectmatch"
    assert candidates.mention == "Paperopoli"
    assert candidates.is_empty()

    # Check the cache has been updated.
    assert len(ranker.already_collected_cands) == 3
    assert ranker.already_collected_cands["Paperopoli"] == candidates

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

def test_ranking_partial_match():
    """
    Test that partial match either returns results or {}
    """

    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    ranker.mentions_to_wikidata = {"London": "Q84"}
    ranker.already_collected_cands = {}

    candidates = ranker.run("London", attach_wikidata=False)

    assert candidates.method == "partialmatch"
    assert candidates.mention == "London"
    assert candidates.get("London").variation == "London"
    assert candidates.get("London").string_similarity == 1.0 

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": "Q60"}

    ranker.already_collected_cands = {}
    candidates = ranker.run("New York", attach_wikidata=False)

    assert candidates.mention == "New York"
    assert candidates.get("New York City").variation == "New York City"
    assert candidates.get("New York City").string_similarity == pytest.approx(0.615384, abs=10e-6)

    ranker.already_collected_cands = {}
    candidates = ranker.run("Lvndvn", attach_wikidata=False)

    assert candidates.mention == "Lvndvn"
    assert candidates.is_empty()


def test_ranking_levenshtein():
    """
    Test that Levenshtein partial match either returns results or {}
    """

    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    ranker.mentions_to_wikidata = {"London": "Q84"}
    ranker.already_collected_cands = {}

    candidates = ranker.run("London", attach_wikidata=False)
    assert candidates.get("London").string_similarity == 1.0

    ranker.already_collected_cands = {}
    candidates = ranker.run("Lvndvn", attach_wikidata=False)
    assert candidates.get("London").string_similarity == pytest.approx(0.66666665, abs=10e-6)

    # Test that overlap works properly
    ranker.mentions_to_wikidata = {"New York City": "Q60"}

    ranker.already_collected_cands = {}
    candidates = ranker.run("New York", attach_wikidata=False)
    assert candidates.mention == "New York"
    assert candidates.get("New York City").string_similarity == pytest.approx(0.615384615, abs=10e-6)

    ranker.already_collected_cands = {}
    candidates = ranker.run("Lvndvn", attach_wikidata=False)
    assert candidates.mention == "Lvndvn"
    assert candidates.get("New York City").string_similarity == 0.0

    ranker.already_collected_cands = {}
    candidates = ranker.run("asdasd", attach_wikidata=False)
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
        already_collected_cands=dict(),
    )
    ranker.load_resources()

    # Test that perfect_match acts before deezy
    candidates = ranker.run("London", attach_wikidata=False)
    assert candidates.mention == "London"
    assert candidates.get("London").string_similarity == 1.0

    # Test that deezy works
    ranker.already_collected_cands = {}
    candidates = ranker.run("Ashton-cnderLyne", attach_wikidata=False)
    assert candidates.mention == "Ashton-cnderLyne"
    assert candidates.method == "deezymatch"

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
        already_collected_cands=dict(),
    )
    ranker.load_resources(train=False)

    # Check the cache is empty.
    assert len(ranker.already_collected_cands) == 0

    candidates = ranker.run("London")

    assert candidates.mention == "London"
    assert isinstance(candidates.get("London"), CandidateMatch)
    assert candidates.get("London").variation == "London"
    # Test that perfect_match acts before deezy
    assert candidates.get("London").string_similarity == 1.0
    assert len(candidates.get("London").wikidata_matches) == 194
    assert "Q84" in [m.wqid for m in candidates.get("London").wikidata_matches]
    assert candidates.get("London").get("Q84").freq == None
    assert candidates.get("London").get("Q84").normalized_score == pytest.approx(0.9761847, abs=10e-6)

    # Check the cache has been updated.
    assert len(ranker.already_collected_cands) == 1
    assert ranker.already_collected_cands["London"] == candidates

    # Test that deezy works
    # TODO: add a ranker.clear_cache() method.
    ranker.already_collected_cands = {}
    candidates = ranker.run("Sheftield")
    assert candidates.mention == "Sheftield"
    assert isinstance(candidates.get("Sheffield"), CandidateMatch)
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert (0.0 < candidates.get("Sheffield").string_similarity < 1.0)
    assert len(candidates.get("Sheffield").wikidata_matches) == 50
    assert "Q42448" in [m.wqid for m in candidates.get("Sheffield").wikidata_matches]
    assert candidates.get("Sheffield").get("Q42448").freq == None
    assert candidates.get("Sheffield").get("Q42448").normalized_score == pytest.approx(0.96211867, abs=10e-6)

    # Test that Perfect Match works
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    candidates = ranker.run("Sheffield")
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert candidates.get("Sheffield").string_similarity == 1.0
    assert "Q42448" in [m.wqid for m in candidates.get("Sheffield").wikidata_matches]

    ranker.already_collected_cands = {}
    candidates = ranker.run("Sheftield")
    assert candidates.is_empty()

    # Test that check if contained works
    ranker = ranking.PartialMatchRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    # Test that levenshtein works
    ranker = ranking.LevenshteinRanker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
    )
    ranker.load_resources()

    candidates = ranker.run("Sheffield")
    assert candidates.get("Sheffield").variation == "Sheffield"
    assert candidates.get("Sheffield").string_similarity == 1.0
    assert "Q42448" in [m.wqid for m in candidates.get("Sheffield").wikidata_matches]

    ranker.already_collected_cands = {}
    candidates = ranker.run("Sheftield")
    assert candidates.mention == "Sheftield"
    assert (0.0 < candidates.get("Sheffield").string_similarity < 1.0)
    assert "Q42448" in [m.wqid for m in candidates.get("Sheffield").wikidata_matches]