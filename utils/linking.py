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
    most_popular_cand = sorted(cands.items(), key=operator.itemgetter(1),reverse=True)[0][0]
    if type == 'mostpopular':
        wikidata_cands = mentions_to_wikidata[most_popular_cand]
    elif type == 'mostpopularnormalised':
        wikidata_cands = normalised_mentions_to_wikidata[most_popular_cand]
    if wikidata_cands:
        # most popular wikidata entry (based on number of time mention points to page)
        most_popular_wikidata_cand, score = sorted(wikidata_cands.items(), key=operator.itemgetter(1),reverse=True)[0]
        # we return the predicted, the score, and the other candidates
        return most_popular_wikidata_cand, score,wikidata_cands

API_URL = "https://rel.cs.ru.nl/api"
def rel_end_to_end(sent):
    
    el_result = requests.post(API_URL, json={
        "text": sent,
        "spans": []
    }).json()
    return el_result