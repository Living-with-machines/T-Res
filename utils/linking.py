from DeezyMatch import candidate_ranker
from collections import OrderedDict
from haversine import haversine
import pandas as pd


def filter_by_length(q, lfd, cdiff):
    # Filter deezymatch results by those with similar length to query (i.e. two chars difference)
    return OrderedDict([(x, lfd[x]) for x in lfd if abs(len(q) - len(x)) <= cdiff])


def deezy_on_the_fly(found_entities, dm_cands, dm_model, dm_output, dm_path, thr, cands, cdiff):
    query_list=[x["toponym"] for x in found_entities if x["place_class"] == "LOC"]
    dEntities = dict()
    if query_list:
        # Ranking on-the-fly
        # find candidates from candidate_scenario 
        # for queries specified by the `query` argument
        candidates_pd = candidate_ranker(candidate_scenario=dm_path + "combined/" + dm_cands + "_" + dm_model,
                         query=query_list,
                         ranking_metric="faiss", 
                         selection_threshold=thr, 
                         num_candidates=cands, 
                         search_size=cands, 
                         output_path=dm_path + dm_output, 
                         pretrained_model_path=dm_path + "models/" + dm_model + "/" + dm_model + ".model", 
                         pretrained_vocab_path=dm_path + "models/" + dm_model + "/" + dm_model + ".vocab")
        df = pd.read_pickle(dm_path + dm_output + ".pkl")
        for i, row in df.iterrows():
            # Filter deezymatch results by those with similar length to query:
            dEntities[row["query"]] = filter_by_length(row["query"], row["faiss_distance"], cdiff)
    return dEntities


def resolve_baseline1(candidate_mentions, mentions_to_wikidata_normalized, overall_entity_freq_wikidata, gazdf, place_of_publication, max_relv, max_dist, dmthr, max_mentions):
        
    dEntities = dict()
    
    # Latitude and longitude of the place of publication given as a Wikidata ID:
    publ_latitude = gazdf[gazdf["wikidata_id"] == place_of_publication].iloc[0]["latitude"]
    publ_longitude = gazdf[gazdf["wikidata_id"] == place_of_publication].iloc[0]["longitude"]
    
    for m in candidate_mentions:
        
        best_score = 0.0
        best_match = ""

        tmkeys = [] # List of wikidata IDs that are candidates given a query
        entity_name_fit = dict() # Dictionary where for each Wikidata entry we keep
                                 # the probability of it being referred to by this mention.
        entity_deezy_conf = dict() # Dictionary where for each Wikidata entry we keep
                                   # the DeezyMatch confidence score.
        top_candidate_mentions = list(candidate_mentions[m])[:max_mentions] # Max number of mentions to keep.
        
        for tm in top_candidate_mentions:
            for tmk in mentions_to_wikidata_normalized[tm]:
                tmkeys.append(tmk)
                entity_name_fit[tmk] = mentions_to_wikidata_normalized[tm][tmk]
                entity_deezy_conf[tmk] = candidate_mentions[m][tm]
        
        # Temporary gazetteer with only candidate entries: 
        tmpdf = gazdf[gazdf["wikidata_id"].isin(tmkeys)]
        
        # We provide a score to each candidate, based on features:
        for i, row in tmpdf.iterrows():
            # Feature 1: Probability of Wikidata entry being referred with this name:
            feature1 = entity_name_fit[row["wikidata_id"]]
            # Feature 2: Normalized relevance of Wikidata entry:
            relv = overall_entity_freq_wikidata[row["wikidata_id"]]
            relv = max_relv if relv > max_relv else relv
            feature2 = relv/max_relv
            # Feature 3: Normalized distance from Wikidata entry to place of publication:
            havdist = haversine((publ_latitude, publ_longitude), (row["latitude"], row["longitude"]))
            havdist = max_dist if havdist > max_dist else havdist
            feature3 = 1 - havdist/max_dist
            # Feature 4: Normalized DeezyMatch confidence score:
            feature4 = 1 - (entity_deezy_conf[row["wikidata_id"]]/dmthr)
            
            features = [feature1, feature2, feature3, feature4]
            score = sum(features)/len(features)
            if score > best_score:
                best_score = score
                best_match = (row["wikidata_id"], row["english_label"], row["latitude"], row["longitude"], best_score)
        
        dEntities[m] = best_match
    
    return dEntities