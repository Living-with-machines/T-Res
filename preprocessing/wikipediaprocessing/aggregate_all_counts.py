import pickle
import time,os,json,pathlib
from tqdm import tqdm
from utils import process_wikipedia
from collections import Counter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test",
                    help="run in test mode",action='store_true')

args = parser.parse_args()

if args.test:
    path = '/resources/wikipedia/test-extractedResources/'
    if pathlib.Path(path).is_dir() == False:
        print ("Error! To run in test mode, you need to have extracted entity and mention counts in "+path)
        exit()
else:
    path = '/resources/wikipedia/extractedResources/'

# start by defining the objects you'll be filling with counts
overall_mentions_freq = Counter()
overall_entity_freq = Counter()
mention_overall_dict = {}
entity_overall_dict = {}

json_folder = path+'Store-Counts/'
jsons = [filename for filename in os.listdir(json_folder) if '.json' in filename]

start_time = time.time()
previous = start_time

for i in tqdm(range(len(jsons))):
    filename = jsons[i]
    
    # we setup specific counters for this folder (just because it's faster instead of updating directly the large ones)
    mentions_freq = Counter()
    entity_freq = Counter()
    
    with open(json_folder+filename) as f:
        entity_counts = json.load(f)
        
        # we update the dictionaries and the local counters
        for entity_count in entity_counts:
            mentions_freq, entity_freq, mention_overall_dict,entity_overall_dict= process_wikipedia.fill_dicts(entity_count,mentions_freq, entity_freq, mention_overall_dict,entity_overall_dict)
    
    # we then update the overall counts
    overall_mentions_freq += mentions_freq
    overall_entity_freq += entity_freq
    
    elapsed_time = time.time() - start_time
    diff = time.time() - previous
    since_beginning = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    last_step = time.strftime("%H:%M:%S", time.gmtime(diff))
    
    #print('Since beginning: %s, Last step: %s' % (since_beginning,last_step))
    previous = time.time()

with open(path+'overall_mentions_freq.pickle', 'wb') as fp:
    pickle.dump(overall_mentions_freq, fp)
                
with open(path+'overall_entity_freq.pickle', 'wb') as fp:
    pickle.dump(overall_entity_freq, fp)

# here we simply save the dictionaries as well
with open(path+'mention_overall_dict.pickle', 'wb') as fp:
    pickle.dump(mention_overall_dict, fp)

with open(path+'entity_overall_dict.pickle', 'wb') as fp:
    pickle.dump(entity_overall_dict, fp)

print ('all done.')


