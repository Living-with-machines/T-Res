import json
from DeezyMatch import candidate_ranker
from collections import OrderedDict

### Overall select function

wikidata_path = '/resources/wikidata/'
with open(wikidata_path + 'mentions_to_wikidata.json', 'r') as f:
    mentions_to_wikidata = json.load(f)

# Path to DeezyMatch model and combined candidate vectors:
dm_path = "/resources/develop/mcollardanuy/toponym-resolution/outputs/deezymatch/"
dm_cands = "wkdtalts"
dm_model = "ocr_faiss_cur085_l2"
dm_output = "deezymatch_on_the_fly"

def select(queries,approach):

    if approach == 'perfectmatch':
        return perfect_match(queries)

    if approach == 'deezymatch':
        return deezy_on_the_fly(queries, dm_cands, dm_model,
                                                dm_output, dm_path, thr=10, cands=10,
                                                cdiff=2)

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

def filter_by_length(q, lfd, cdiff):
    # Filter deezymatch results by those with similar length to query (i.e. two chars difference)
    return OrderedDict([(x, lfd[x]) for x in lfd if abs(len(q) - len(x)) <= cdiff])

def deezy_on_the_fly(queries, dm_cands, dm_model, dm_output, dm_path, thr, cands, cdiff):

    # first we fill in the perfect matches
    cands_dict = perfect_match(queries)

    # the rest go through deezymatch
    remainers = [x for x,y in cands_dict.items() if len(y)==0]

    candidates = candidate_ranker(candidate_scenario=dm_path + "combined/" + dm_cands + "_" + dm_model,
                         query=remainers,
                         ranking_metric="faiss", 
                         selection_threshold=thr, 
                         num_candidates=cands, 
                         search_size=cands, 
                         output_path= './outputs/deezymatch/' + dm_output, 
                         pretrained_model_path=dm_path + "models/" + dm_model + "/" + dm_model + ".model", 
                         pretrained_vocab_path=dm_path + "models/" + dm_model + "/" + dm_model + ".vocab")

    for idx,row in candidates.iterrows():
        cands_dict[row['query']] = row['pred_score']

    return cands_dict

