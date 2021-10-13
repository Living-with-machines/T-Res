import time,os,json
from collections import Counter
import pickle

def fill_dicts(res):    
    global mentions_freq, entity_freq, mention_overall_dict

    url,box_mentions, box_entities, mentions_dict = res[0],res[1],res[2],res[3]
    mentions_freq+= box_mentions
    entity_freq+= box_entities
    
    for k,v in box_entities.items():
        if k in entity_overall_dict:
            entity_overall_dict[k].append(url)
        else:
            entity_overall_dict[k] = [url]
    
    for k,v in mentions_dict.items():
        if k in mention_overall_dict:
            mention_overall_dict[k]+=v
        else:
            mention_overall_dict[k] = Counter()
            mention_overall_dict[k]+= v


overall_mentions_freq = Counter()
overall_entity_freq = Counter()

mention_overall_dict = {}
entity_overall_dict = {}


with open("../resources/wikipedia/extractedResources/overall_mentions_freq.pickle", "wb") as fp:
    pickle.dump(overall_mentions_freq, fp)

with open("../resources/wikipedia/extractedResources/overall_entity_freq.pickle", "wb") as fp:
    pickle.dump(overall_entity_freq, fp)
    
json_folder = "../resources/wikipedia/extractedResources/Store-Counts/"

start_time = time.time()
previous = start_time
jsons = [x for x in os.listdir(json_folder) if ".json" in x]

for filename in jsons:
    
    mentions_freq = Counter()
    entity_freq = Counter()
    
    with open(json_folder+filename) as f:
        data = json.load(f)
        
        for res in data:
            fill_dicts(res)
         
    with open("../resources/wikipedia/extractedResources/overall_mentions_freq.pickle", "rb") as f:
        overall_mentions_freq = pickle.load(f)
        
    overall_mentions_freq += mentions_freq
    
    with open("../resources/wikipedia/extractedResources/overall_mentions_freq.pickle", "wb") as fp:
        pickle.dump(overall_mentions_freq, fp)
        
        
    with open("../resources/wikipedia/extractedResources/overall_entity_freq.pickle", "rb") as f:
        overall_entity_freq = pickle.load(f)
        
    overall_entity_freq += entity_freq
    
    with open("../resources/wikipedia/extractedResources/overall_entity_freq.pickle", "wb") as fp:
        pickle.dump(overall_entity_freq, fp)

        
    with open("../resources/wikipedia/extractedResources/mention_overall_dict.pickle", "wb") as fp:
        pickle.dump(mention_overall_dict, fp)
    
    with open("../resources/wikipedia/extractedResources/entity_overall_dict.pickle", "wb") as fp:
        pickle.dump(entity_overall_dict, fp)
    
    print ("done:", filename)
    
    elapsed_time = time.time() - start_time
    diff = time.time() - previous
    since_beginning = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    last_step = time.strftime("%H:%M:%S", time.gmtime(diff))
    
    print("Since beginning: %s, Last step: %s" % (since_beginning,last_step))
    previous = time.time()
    
print ("all done.")


