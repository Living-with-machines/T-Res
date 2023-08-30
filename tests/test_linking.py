import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from t_res.geoparser import linking

current_dir = Path(__file__).parent.resolve()

def test_linking_most_popular():
    mylinker = linking.Linker(
        method="mostpopular",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    mylinker.load_resources()
    dict_mention = {
        "candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}}}
    }
    keep_most_popular, final_score, candidates = mylinker.most_popular(dict_mention)
    assert keep_most_popular == "Q84"
    assert final_score == 0.9812731647051174
    assert candidates == {"Q84": 0.9812731647051174, "Q92561": 0.018726835294882633}

    dict_mention = {"candidates": {}}
    keep_most_popular, final_score, candidates = mylinker.most_popular(dict_mention)
    assert keep_most_popular == "NIL"
    assert final_score == 0.0
    assert candidates == {}


def test_by_distance():
    mylinker = linking.Linker(
        method="bydistance",
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    mylinker.load_resources()

    #test it finds London, UK
    dict_mention = {
        "candidates": {
            "London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}, "Score": 0.397048}
        },
        "place_wqid": "Q84",
    }
    pred, final_score, resulting_cands = mylinker.by_distance(dict_mention)
    assert pred == "Q84"
    assert final_score == 0.824
    assert "Q84" in resulting_cands

    #test it finds London, CA
    dict_mention = {
        "candidates": {
            "London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}, "Score": 0.397048}
        },
        "place_wqid": "Q92561",
    }
    pred, final_score, resulting_cands = mylinker.by_distance(dict_mention)
    assert pred == "Q92561"
    assert final_score == 0.624
    assert "Q84" in resulting_cands

    #check it finds none
    dict_mention = {
        "candidates": {"London": {"Candidates": {}, "Score": 0.397048}},
        "place_wqid": "Q2365261",
    }
    pred, final_score, resulting_cands = mylinker.by_distance(dict_mention)
    assert pred == "NIL"
    assert final_score == 0.0
    assert "Q84" not in resulting_cands
