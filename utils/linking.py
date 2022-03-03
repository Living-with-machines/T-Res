import json
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



# from haversine import haversine

# def resolve_baseline1(candidate_mentions, mentions_to_wikidata_normalized, overall_entity_freq_wikidata, gazdf, place_of_publication, max_relv, max_dist, dmthr, max_mentions):
        
#     dEntities = dict()
    
#     # Latitude and longitude of the place of publication given as a Wikidata ID:
#     publ_latitude = gazdf[gazdf["wikidata_id"] == place_of_publication].iloc[0]["latitude"]
#     publ_longitude = gazdf[gazdf["wikidata_id"] == place_of_publication].iloc[0]["longitude"]
    
#     for m in candidate_mentions:
        
#         best_score = 0.0
#         best_match = ""

#         tmkeys = [] # List of wikidata IDs that are candidates given a query
#         entity_name_fit = dict() # Dictionary where for each Wikidata entry we keep
#                                  # the probability of it being referred to by this mention.
#         entity_deezy_conf = dict() # Dictionary where for each Wikidata entry we keep
#                                    # the DeezyMatch confidence score.
#         top_candidate_mentions = list(candidate_mentions[m])[:max_mentions] # Max number of mentions to keep.
        
#         for tm in top_candidate_mentions:
#             for tmk in mentions_to_wikidata_normalized[tm]:
#                 tmkeys.append(tmk)
#                 entity_name_fit[tmk] = mentions_to_wikidata_normalized[tm][tmk]
#                 entity_deezy_conf[tmk] = candidate_mentions[m][tm]
        
#         # Temporary gazetteer with only candidate entries: 
#         tmpdf = gazdf[gazdf["wikidata_id"].isin(tmkeys)]
        
#         # We provide a score to each candidate, based on features:
#         for i, row in tmpdf.iterrows():
#             # Feature 1: Probability of Wikidata entry being referred with this name:
#             feature1 = entity_name_fit[row["wikidata_id"]]
#             # Feature 2: Normalized relevance of Wikidata entry:
#             relv = overall_entity_freq_wikidata[row["wikidata_id"]]
#             relv = max_relv if relv > max_relv else relv
#             feature2 = relv/max_relv
#             # Feature 3: Normalized distance from Wikidata entry to place of publication:
#             havdist = haversine((publ_latitude, publ_longitude), (row["latitude"], row["longitude"]))
#             havdist = max_dist if havdist > max_dist else havdist
#             feature3 = 1 - havdist/max_dist
#             # Feature 4: Normalized DeezyMatch confidence score:
#             feature4 = 1 - (entity_deezy_conf[row["wikidata_id"]]/dmthr)
            
#             features = [feature1, feature2, feature3, feature4]
#             score = sum(features)/len(features)
#             if score > best_score:
#                 best_score = score
#                 best_match = (row["wikidata_id"], row["english_label"], row["latitude"], row["longitude"], best_score)
        
#         dEntities[m] = best_match
    
#     return dEntities