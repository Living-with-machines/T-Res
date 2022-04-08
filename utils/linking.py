import json
import requests
import operator
import pandas as pd
import numpy as np
from numpy import NaN
from ast import literal_eval

# Load Wikidata mentions to QID dictionary
wikidata_path = '/resources/wikidata/'
with open(wikidata_path + 'mentions_to_wikidata.json', 'r') as f:
    mentions_to_wikidata = json.load(f)

# Load Wikidata normalized mentions to QID dictionary
with open(wikidata_path + 'mentions_to_wikidata_normalized.json', 'r') as f:
    normalised_mentions_to_wikidata = json.load(f)


# -----------------------------------------
# LINKING METHODS

# Select disambiguation method
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


def training_class_disambiguation(df):
    
    # Create dataframe by mention:
    rows = []
    for i, row in df.iterrows():
        article_id = row["article_id"]
        sentences = literal_eval(row["sentences"])
        annotations = literal_eval(row["annotations"])
        for s in sentences:
            for a in annotations:
                if s["sentence_pos"] == a["sent_pos"]:
                    rows.append((article_id, s["sentence_pos"], s["sentence_text"], a["mention_pos"], a["mention"], a["wkdt_qid"], a["mention_start"], a["mention_end"]))

    training_df = pd.DataFrame(columns=["article_id", "sentence_pos", "sentence", "mention_pos", "mention", "wkdt_qid", "mention_start", "mention_end"], data=rows)
    # Drop rows with unlinked entities (DO WE WANT THIS?):
    training_df = training_df[~training_df["wkdt_qid"].isnull()]

    return training_df


def get_candidate_wikidata_ids(cands):
    dCands = dict()
    for k in cands:
        if k in mentions_to_wikidata:
            dCands[k] = list(mentions_to_wikidata[k].keys())

    return dCands