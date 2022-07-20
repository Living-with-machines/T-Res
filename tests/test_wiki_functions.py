from utils import process_wikipedia


def test_make_links_consistent():
    string_a = "Havannah%2C_Cheshire"
    string_b = "Havannah%2C%20Cheshire"

    assert process_wikipedia.make_links_consistent(
        string_a
    ) == process_wikipedia.make_links_consistent(string_b)
    assert (process_wikipedia.make_links_consistent(string_a) == string_a) is False
