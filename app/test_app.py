import os
import requests


# API_URL = "http://127.0.0.1:8123"
API_URL = f"http://{os.getenv('HOST_URL')}:8000/v2/t-res_deezy_reldisamb-wpubl-wmtops"


def test_health():
    response = requests.get(f'{API_URL}/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


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

