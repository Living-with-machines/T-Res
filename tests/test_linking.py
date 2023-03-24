import os
import sys
import sqlite3
import numpy as np

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking


def test_initialise_method():
    """
    Test initialisation works fine
    """
    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    assert type(mylinker.__str__()) == str


def test_load_resources():
    """
    Test initialisation works fine
    """
    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )
    mylinker.load_resources()

    conn = sqlite3.connect(mylinker.resources_path + "/rel_db/generic/common_drawl.db")

    c = conn.cursor()

    word = "apple"
    c.execute("SELECT emb FROM embeddings WHERE word=?", (word,))
    result = c.fetchone()

    embedding = np.fromstring(result[0], dtype="float32")

    assert type(embedding)==np.ndarray
    


def test_most_popular():
    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    mylinker.load_resources()
    dict_mention = {"candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}}}}
    keep_most_popular, final_score, candidates = mylinker.most_popular(dict_mention)
    assert keep_most_popular == "Q84"
    assert final_score == 0.9895689976719958
    assert candidates == {"Q84": 0.9895689976719958, "Q92561": 0.01043100232800422}

    dict_mention = {"candidates": {}}
    keep_most_popular, final_score, candidates = mylinker.most_popular(dict_mention)
    assert keep_most_popular == "NIL"
    assert final_score == 0.0
    assert candidates == {}


def test_by_distance():
    mylinker = linking.Linker(
        method="bydistance",
        resources_path="resources/",
        linking_resources=dict(),
        rel_params=dict(),
        overwrite_training=False,
    )

    mylinker.load_resources()

    dict_mention = {
        "candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}, "Score": 0.397048}},
        "place_wqid": "Q84",
    }
    pred, final_score, resulting_cands = mylinker.by_distance(dict_mention)
    assert pred == "Q84"
    assert final_score == 0.824
    assert "Q84" in resulting_cands

    dict_mention = {
        "candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}, "Score": 0.397048}},
        "place_wqid": "Q172",
    }
    pred, final_score, resulting_cands = mylinker.by_distance(dict_mention)
    assert pred == "Q92561"
    assert final_score == 0.54
    assert "Q84" in resulting_cands

    dict_mention = {
        "candidates": {"London": {"Candidates": {}, "Score": 0.397048}},
        "place_wqid": "Q172",
    }
    pred, final_score, resulting_cands = mylinker.by_distance(dict_mention)
    assert pred == "NIL"
    assert final_score == 0.0
    assert "Q84" not in resulting_cands
