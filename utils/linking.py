import json
import requests
import operator
from numpy import NaN

wikidata_path = '/resources/wikidata/'
with open(wikidata_path + 'mentions_to_wikidata.json', 'r') as f:
    mentions_to_wikidata = json.load(f)

with open(wikidata_path + 'mentions_to_wikidata_normalized.json', 'r') as f:
    normalised_mentions_to_wikidata = json.load(f)

def select(cands,method):
    if cands:
        if "mostpopular" in method:
            link, score, other_cands = most_popular(cands,type=method)
            return (link,score,other_cands)


def most_popular(cands,type):
    # most popular candidate
    keep_most_popular = tuple()
    keep_highest_score = 0.0
    all_candidates = []
    for candidate in cands:
        if type == 'mostpopular':
            wikidata_cands = mentions_to_wikidata[candidate]
        elif type == 'mostpopularnormalised':
            wikidata_cands = normalised_mentions_to_wikidata[candidate]
        if wikidata_cands:
            # most popular wikidata entry (based on number of time mention points to page)
            most_popular_wikidata_cand, score = sorted(wikidata_cands.items(), key=operator.itemgetter(1),reverse=True)[0]
            if score > keep_highest_score:
                keep_highest_score = score
                keep_most_popular = most_popular_wikidata_cand
            all_candidates += wikidata_cands
    # we return the predicted, the score, and the other candidates
    return keep_most_popular, keep_highest_score, set(all_candidates)

API_URL = "https://rel.cs.ru.nl/api"
def rel_end_to_end(sent):
    
    el_result = requests.post(API_URL, json={
        "text": sent,
        "spans": []
    }).json()
    return el_result