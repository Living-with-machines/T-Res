import urllib

from utils import process_wikipedia


def test_make_links_consistent():
    string_a = "Havannah%2C_Cheshire"
    string_b = "Havannah%2C%20Cheshire"

    string_c = "New_York"

    assert process_wikipedia.make_wikilinks_consistent(
        string_a
    ) == process_wikipedia.make_wikilinks_consistent(string_b)
    assert (process_wikipedia.make_wikilinks_consistent(string_a) == string_a) is False
    assert process_wikipedia.make_wikilinks_consistent(string_c) == "new%20york"


def test_wikidata2wikipedia():
    db = "/resources/wikipedia/wikidata2wikipedia/index_enwiki-latest.db"
    assert process_wikipedia.title_to_id("BOLOGNA", lower=True, path_to_db=db) == None
    assert process_wikipedia.title_to_id("Bologna", lower=True, path_to_db=db) == None
    assert (
        process_wikipedia.title_to_id("bologna", lower=True, path_to_db=db) == "Q1891"
    )
    assert (
        process_wikipedia.title_to_id("new_york_city", lower=True, path_to_db=db)
        == "Q60"
    )
    assert (
        process_wikipedia.title_to_id("new%20york%20city", lower=True, path_to_db=db)
        == None
    )
    prepare_url = process_wikipedia.make_wikipedia2wikidata_consisent("New York City")
    assert (
        process_wikipedia.title_to_id(prepare_url, lower=True, path_to_db=db) == "Q60"
    )
