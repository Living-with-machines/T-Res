import os
import sys

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking


def test_initialise_method():
    """
    Test initialisation works fine
    """
    myranker = linking.Linker(
        method="mostpopular",
        resources_path="/resources/wikidata/",
        linking_resources=dict(),
        base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
        rel_params={
            "base_path": "/resources/rel_db/",
            "wiki_version": "wiki_2019/",
        },
        overwrite_training=False,
    )

    assert type(myranker.__str__()) == str


def test_most_popular():
    myranker = linking.Linker(
        method="mostpopular",
        resources_path="/resources/wikidata/",
        linking_resources=dict(),
        base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
        rel_params={
            "base_path": "/resources/rel_db/",
            "wiki_version": "wiki_2019/",
        },
        overwrite_training=False,
    )

    myranker.load_resources()
    dict_mention = {"candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}}}}
    keep_most_popular, final_score = myranker.most_popular(dict_mention)
    assert keep_most_popular == "Q84"
    assert final_score == 0.9895689976719958

    dict_mention = {"candidates": {}}
    keep_most_popular, final_score = myranker.most_popular(dict_mention)
    assert keep_most_popular == "NIL"
    assert final_score == 0.0


def test_by_distance():
    myranker = linking.Linker(
        method="bydistance",
        resources_path="/resources/wikidata/",
        linking_resources=dict(),
        base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
        rel_params={
            "base_path": "/resources/rel_db/",
            "wiki_version": "wiki_2019/",
        },
        overwrite_training=False,
    )

    myranker.load_resources()

    dict_mention = {
        "candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}}},
        "place_wqid": "Q84",
    }
    pred, final_score = myranker.by_distance(dict_mention)
    assert pred == "Q84"
    assert final_score == 1.0

    myranker.load_resources()

    dict_mention = {
        "candidates": {"London": {"Candidates": {"Q84": 0.9, "Q92561": 0.1}}},
        "place_wqid": "Q172",
    }
    pred, final_score = myranker.by_distance(dict_mention)
    assert pred == "Q92561"
    assert final_score == 0.992
