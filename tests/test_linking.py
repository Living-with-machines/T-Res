import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from t_res.geoparser import linking

current_dir = Path(__file__).parent.resolve()

def test_init():

    # Test that parameters passed to the subclass constructor are propagated.
    mylinker = linking.MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
        rel_params={'param': 'value'},
        overwrite_training=True,
    )

    assert mylinker.method_name()  == "mostpopular"

    assert mylinker.resources_path  == "path/to/resources/"
    assert mylinker.experiments_path  == "path/to/experiments/"
    assert mylinker.linking_resources['resource'] == 'value'
    assert mylinker.rel_params['param'] == 'value'
    assert mylinker.overwrite_training

    mylinker = linking.MostPopularLinker(
        resources_path="path/to/resources/",
        experiments_path="path/to/experiments/",
        linking_resources={'resource': 'value'},
        rel_params={'param': 'value'},
    )

    assert not mylinker.overwrite_training

def test_linking_most_popular():
    mylinker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    assert mylinker.method_name()  == "mostpopular"

    mylinker.load_resources()
    dict_mention = {
        "candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}}}
    }
    keep_most_popular, final_score, candidates = mylinker.run(dict_mention)
    assert keep_most_popular == "Q84"
    assert final_score == 0.9812731647051174
    assert candidates == {"Q84": 0.9812731647051174, "Q92561": 0.018726835294882633}

    dict_mention = {"candidates": {}}
    keep_most_popular, final_score, candidates = mylinker.run(dict_mention)
    assert keep_most_popular == "NIL"
    assert final_score == 0.0
    assert candidates == {}


def test_linking_by_distance():
    mylinker = linking.ByDistanceLinker(
        resources_path=os.path.join(current_dir,"sample_files/resources/"),
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    assert mylinker.method_name()  == "bydistance"

    mylinker.load_resources()

    #test it finds London, UK
    dict_mention = {
        "candidates": {
            "London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}, "Score": 0.397048}
        },
        "place_wqid": "Q84",
    }
    pred, final_score, resulting_cands = mylinker.run(dict_mention)
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
    pred, final_score, resulting_cands = mylinker.run(dict_mention)
    assert pred == "Q92561"
    assert final_score == 0.624
    assert "Q84" in resulting_cands

    #check it finds none
    dict_mention = {
        "candidates": {"London": {"Candidates": {}, "Score": 0.397048}},
        "place_wqid": "Q2365261",
    }
    pred, final_score, resulting_cands = mylinker.run(dict_mention)
    assert pred == "NIL"
    assert final_score == 0.0
    assert "Q84" not in resulting_cands
