import os
import pytest
import requests

from t_res.utils.dataclasses import SentenceMentions, Candidates

API_URL = "http://127.0.0.1:8123"
# API_URL = f"http://{os.getenv('HOST_URL')}:8000/v2/t-res_deezy_reldisamb-wpubl-wmtops"

@pytest.mark.skip(reason="integration test")
def test_root():
    response = requests.get(f'{API_URL}/')
    assert response.status_code == 200
    assert 'Title' in response.json().keys()

@pytest.mark.skip(reason="integration test")
def test_health():
    response = requests.get(f'{API_URL}/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

@pytest.mark.skip(reason="integration test")
def test_run_ner():
    test_body = {"text": "Harvey, from London;Thomas and Elizabeth, Barnett."}
    expected_response = [{'sentence': {'sentence': 'Harvey, from London;Thomas and Elizabeth, Barnett.'}, 'mentions': [{'sort_index': 13, 'mention': 'London', 'start_offset': 3, 'end_offset': 3, 'start_char': 13, 'ner_score': 0.997, 'ner_label': 'LOC', 'entity_link': 'O'}]}]

    response = requests.get(f'{API_URL}/run_ner', json=test_body)

    assert response.status_code == 200
    assert response.json() == expected_response
    
    # Test deserialisation:
    result = SentenceMentions.from_json(response.json())
    assert len(result) == 1
    assert result[0].sentence.sentence == test_body['text']
    assert len(result[0].mentions) == 1
    assert result[0].mentions[0].mention == "London"

@pytest.mark.skip(reason="integration test")
def test_run_candidate_selection():
    test_body = {"sentence_mentions": [{'sentence': {'sentence': 'Harvey, from London;Thomas and Elizabeth, Barnett.'}, 'mentions': [{'sort_index': 13, 'mention': 'London', 'start_offset': 3, 'end_offset': 3, 'start_char': 13, 'ner_score': 0.997, 'ner_label': 'LOC', 'entity_link': 'O'}]}]}

    response = requests.get(f'{API_URL}/run_candidate_selection', json=test_body)

    assert response.status_code == 200

    # Test deserialisation:
    result = Candidates.from_dict(response.json())
    assert result.text() == 'Harvey, from London;Thomas and Elizabeth, Barnett.'
    assert result.place_of_pub_wqid() is None
    assert result.place_of_pub() is None
    assert len(result.candidates()) == 1
    assert result.candidates()[0].mention.mention == "London"
    assert result.candidates()[0].best_string_match().variation == "London"
    assert result.candidates()[0].best_string_match().string_similarity == 1.0

    # Test with place of publication info.
    test_body['place_of_pub'] = 'Poole, Dorset'
    test_body['place_of_pub_wqid'] = 'Q203349'

    response = requests.get(f'{API_URL}/run_candidate_selection', json=test_body)

    assert response.status_code == 200

    # Test deserialisation:
    result = Candidates.from_dict(response.json())
    assert result.text() == 'Harvey, from London;Thomas and Elizabeth, Barnett.'
    assert result.place_of_pub_wqid() == 'Q203349'
    assert result.place_of_pub() == 'Poole, Dorset'

### OLD:

@pytest.mark.skip(reason="integration test")
def test_t_res():
    
    test_body = {
        "sentence": "A remarkable case of rattening has just occurred in the building trade at Newtown.",
        "place": "Powys",
        "place_wqid": "Q156150"
        }
    expected_response = [{'mention': 'Newtown', 'ner_score': 0.996, 'pos': 74, 'sent_idx': 0, 'end_pos': 81, 'tag': 'LOC', 'sentence': 'A remarkable case of rattening has just occurred in the building trade at Newtown.', 'prediction': 'Q669171', 'ed_score': 0.034, 'cross_cand_score': {'Q669171': 0.41, 'Q1851145': 0.298, 'Q5355774': 0.143, 'Q738356': 0.107, 'Q15262210': 0.024, 'Q7020654': 0.018, 'Q18748305': 0.0}, 'prior_cand_score': {'Q1851145': 0.86, 'Q669171': 0.734, 'Q5355774': 0.537, 'Q738356': 0.516, 'Q15262210': 0.485, 'Q7020654': 0.483, 'Q18748305': 0.476}, 'latlon': [52.5132, -3.3141], 'wkdt_class': 'Q3957'}]
    
    response = requests.get(f'{API_URL}/toponym_resolution', json=test_body)
    assert response.status_code == 200
    assert response.json() == expected_response


@pytest.mark.skip(reason="integration test")
def test_ner():

    test_body = {"sentence": "Harvey, from London;Thomas and Elizabeth, Barnett."}
    expected_response = [{"entity":"B-LOC","score":0.990628182888031,"word":"London","start":13,"end":19}]
    response = requests.get(f'{API_URL}/ner', json=test_body)

    assert response.status_code == 200
    assert response.json() == expected_response

# 