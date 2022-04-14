import json
import os
from collections import OrderedDict

import pandas as pd
from DeezyMatch import candidate_ranker
from numpy import NaN
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

pandarallel.initialize()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load Wikidata mentions-to-wqid:
wikidata_path = "/resources/wikidata/"
with open(wikidata_path + "mentions_to_wikidata.json", "r") as f:
    mentions_to_wikidata = json.load(f)

# Get candidates QIDs from strings:
def get_candidate_wikidata_ids(cands):
    dCands = dict()
    for k in cands:
        if k in mentions_to_wikidata:
            dCands[k] = list(mentions_to_wikidata[k].keys())

    return dCands


### Overall select function


def select(queries, approach, myranker, already_collected_cands):

    if approach == "perfectmatch":
        return perfect_match(queries, already_collected_cands)

    if approach == "partialmatch":
        return partial_match(queries, already_collected_cands, damlev=False)

    if approach == "levenshtein":
        return partial_match(queries, already_collected_cands, damlev=True)

    if approach == "deezymatch":

        return deezy_on_the_fly(queries, myranker, already_collected_cands)


#### PerfectMatch ####


def perfect_match(queries, already_collected_cands):
    candidates = {}
    for query in queries:
        if query in already_collected_cands:
            candidates[query] = already_collected_cands[query]
        else:
            if query in mentions_to_wikidata:
                candidates[query] = {query: 1.0}
                already_collected_cands[query] = {query: 1.0}
            else:
                candidates[query] = {}
                already_collected_cands[query] = {}
    return candidates, already_collected_cands


#### PartialMatch ####


def partial_match(queries: list, already_collected_cands: dict, damlev: bool) -> tuple[dict, dict]:

    """
    Given a list of queries return a dict of partial matches for each of them
    and enrich another overall dictionary of candidatees

    Args:
        queries (list): list of mentions identified in a given sentence
        already_collected_cands (dict): dictionary of already processed mentions
        damlev (bool): either damerau_levenshtein or simple overlap

    Returns:
        tuple: two dictionaries: a (partial) match between queries->mentions
                                an enriched version of already_collected_cands
    """

    candidates, already_collected_cands = perfect_match(queries, already_collected_cands)

    # the rest go through
    remainers = [x for x, y in candidates.items() if len(y) == 0]

    for query in remainers:
        mention_df = pd.DataFrame({"mentions": mentions_to_wikidata.keys()})
        if damlev:
            mention_df["score"] = mention_df.parallel_apply(
                lambda row: damlev_dist(query, row), axis=1
            )
        else:
            mention_df["score"] = mention_df.parallel_apply(
                lambda row: check_if_contained(query, row), axis=1
            )
        mention_df = mention_df.dropna()
        # currently hardcoded cutoff
        top_scores = sorted(list(set(list(mention_df["score"].unique()))), reverse=True)[:1]
        mention_df = mention_df[mention_df["score"].isin(top_scores)]
        mention_df = mention_df.set_index("mentions").to_dict()["score"]
        candidates[query] = mention_df
        already_collected_cands[query] = mention_df
    return candidates, already_collected_cands


def damlev_dist(query: str, row: pd.Series) -> float:
    """
    Compute damerau levenshtein distance between query and Series

    Args:
        query (str): the mention identified in text
        row (Series): the row corresponding to a mention in the KB

    Returns:
        float: the similarity score, between 1.0 and 0.0
    """
    return 1.0 - normalized_damerau_levenshtein_distance(query.lower(), row["mentions"].lower())


def check_if_contained(query: str, row: pd.Series) -> float:
    """
    Takes a query and a Series and return the amount of overlap, if any

    Args:
        query (str): the mention identified in text
        row (Series): the row corresponding to a mention in the KB

    Returns:
        float: the size of overlap between query and mention, max 1.0 (perfect match)
    """
    s1 = query.lower()
    s2 = row["mentions"].lower()
    # E.g. query is 'Dorset' and candidate mention is 'County of Dorset'
    if s1 in s2:
        return len(query) / len(row["mentions"])
    # E.g. query is 'County of Dorset' and candidate mention is 'Dorset'
    if s2 in s1:
        return len(row["mentions"]) / len(query)

#### DeezyMatch ####


def deezy_on_the_fly(queries, myranker, already_collected_cands):

    dm_path = myranker["dm_path"]
    dm_cands = myranker["dm_cands"]
    dm_model = myranker["dm_model"]
    dm_output = myranker["dm_output"]

    # first we fill in the perfect matches and already collected queries
    cands_dict, already_collected_cands = perfect_match(queries, already_collected_cands)

    # the rest go through
    remainers = [x for x, y in cands_dict.items() if len(y) == 0]
    if remainers:
        try:
            candidates = candidate_ranker(
                candidate_scenario=dm_path + "combined/" + dm_cands + "_" + dm_model,
                query=remainers,
                ranking_metric=myranker["ranking_metric"],
                selection_threshold=myranker["selection_threshold"],
                num_candidates=myranker["num_candidates"],
                search_size=myranker["search_size"],
                output_path=dm_path + "ranking/" + dm_output,
                pretrained_model_path=dm_path + "models/" + dm_model + "/" + dm_model + ".model",
                pretrained_vocab_path=dm_path + "models/" + dm_model + "/" + dm_model + ".vocab",
            )

            for idx, row in candidates.iterrows():
                cands_dict[row["query"]] = dict(row["cosine_dist"])
                already_collected_cands[row["query"]] = dict(row["cosine_dist"])
        except TypeError:
            pass

    return cands_dict, already_collected_cands
