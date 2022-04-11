import json
from DeezyMatch import candidate_ranker
from collections import OrderedDict


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

def select(queries, approach, myranker):

    if approach == "perfectmatch":
        return perfect_match(queries)

    if approach == "deezymatch":

        return deezy_on_the_fly(queries, myranker)


#### PerfectMatch ####

def perfect_match(queries):
    candidates = {}
    for query in queries:
        if query in mentions_to_wikidata:
            candidates[query] = {query:1.0}
        else:
            candidates[query] = {}
    return candidates


#### DeezyMatch ####

def deezy_on_the_fly(queries, myranker):

    dm_path = myranker["dm_path"]
    dm_cands = myranker["dm_cands"]
    dm_model = myranker["dm_model"]
    dm_output = myranker["dm_output"]

    # first we fill in the perfect matches
    cands_dict = perfect_match(queries)

    # the rest go through deezymatch
    remainers = [x for x,y in cands_dict.items() if len(y)==0]
    if remainers:
        try:
            candidates = candidate_ranker(candidate_scenario=dm_path + "combined/" + dm_cands + "_" + dm_model,
                                        query=remainers,
                                        ranking_metric=myranker["ranking_metric"], 
                                        selection_threshold=myranker["selection_threshold"], 
                                        num_candidates=myranker["num_candidates"],
                                        search_size=myranker["search_size"],
                                        output_path=dm_path + "ranking/" + dm_output, 
                                        pretrained_model_path=dm_path + "models/" + dm_model + "/" + dm_model + ".model", 
                                        pretrained_vocab_path=dm_path + "models/" + dm_model + "/" + dm_model + ".vocab")

            for idx,row in candidates.iterrows():
                cands_dict[row['query']] = dict(row['cosine_dist'])
        except TypeError:
            pass

    return cands_dict