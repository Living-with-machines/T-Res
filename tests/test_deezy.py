import os
from pathlib import Path

import pytest
from DeezyMatch import candidate_ranker

current_dir = Path(__file__).parent.resolve()

@pytest.mark.skip(reason="Needs deezy model")
def test_deezy_match_deezy_candidate_ranker(tmp_path):
    deezy_parameters = {
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": os.path.join(current_dir,"sample_files/resources/deezymatch/"),
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

    candidates = candidate_ranker(
        candidate_scenario=os.path.join(dm_path, "combined", dm_cands + "_" + dm_model),
        query=query,
        ranking_metric=deezy_parameters["ranking_metric"],
        selection_threshold=deezy_parameters["selection_threshold"],
        num_candidates=deezy_parameters["num_candidates"],
        search_size=deezy_parameters["num_candidates"],
        verbose=deezy_parameters["verbose"],
        output_path=os.path.join(tmp_path,dm_output),
    )
    assert len(candidates) == len(query)
