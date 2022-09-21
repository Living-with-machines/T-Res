import json
import os
import pathlib
import sys
import time
from argparse import ArgumentParser
from collections import Counter

from tqdm import tqdm

# Add "../.." to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.pardir, os.path.pardir)))

from utils import process_wikipedia

parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test", help="run in test mode", action="store_true")

args = parser.parse_args()

if args.test:
    path = "/resources/wikipedia/test-extractedResources/"
else:
    path = "/resources/wikipedia/extractedResources/"

if pathlib.Path(path).is_dir() == False:
    print("Error! You need to have extracted entity and mention counts in " + path)
    exit()

# start by defining the objects you'll be filling with counts

# overall_mentions_freq and overall_entity_freq are simply mentions and entity counters.
# For each entity as key (say the entity "London"), we have as value how many times this entity appears mentioned in Wikipedia.
# The same for each mention, say "NYC", we have as value how many times this appears as a mention (i.e., the text of a hyperlink) of an entity.

overall_mentions_freq = Counter()
overall_entity_freq = Counter()

# mention_overall_dict maps a mention to a Counter object containing all entities associated with it, for instance for the mention "London":
# 'London': 76511, 'London%2C%20Ontario': 790, 'London%2C%20England': 350, 'London%20GAA': 321, 'City%20of%20London': 144, etc
mention_overall_dict = {}

# in case of entity_inlink_dict, the dictionary maps an entity, say "London", with a Counter of all other entities which link to it
entity_inlink_dict = {}
# in case of entity_outlink_dict, the dictionary maps an entity, say "London", with a Counter of all other entities which are linked from its page
entity_outlink_dict = {}

json_folder = path + "Store-Counts/"
jsons = [filename for filename in os.listdir(json_folder) if ".json" in filename]

start_time = time.time()
previous = start_time

for i in tqdm(range(len(jsons))):
    filename = jsons[i]

    # we setup specific counters for this folder (just because it's faster instead of updating directly the large ones)
    mentions_freq = Counter()
    entity_freq = Counter()

    with open(json_folder + filename) as f:
        entity_counts = json.load(f)

        # we update the dictionaries and the local counters
        for entity_count in entity_counts:
            (
                mentions_freq,
                entity_freq,
                mention_overall_dict,
                entity_inlink_dict,
                entity_outlink_dict,
            ) = process_wikipedia.fill_dicts(
                entity_count,
                mentions_freq,
                entity_freq,
                mention_overall_dict,
                entity_inlink_dict,
                entity_outlink_dict,
            )
    # we then update the overall counts
    overall_mentions_freq += mentions_freq
    overall_entity_freq += entity_freq

    elapsed_time = time.time() - start_time
    diff = time.time() - previous
    since_beginning = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    last_step = time.strftime("%H:%M:%S", time.gmtime(diff))

    # print('Since beginning: %s, Last step: %s' % (since_beginning,last_step))
    previous = time.time()


with open(path + "overall_mentions_freq.json", "w") as fp:
    json.dump(overall_mentions_freq, fp)

with open(path + "overall_entity_freq.json", "w") as fp:
    json.dump(overall_entity_freq, fp)

# we just convert the dictionaries of counts to a Counter
mention_overall_dict = {x: Counter(y) for x, y in mention_overall_dict.items()}

with open(path + "mention_overall_dict.json", "w") as fp:
    json.dump(mention_overall_dict, fp)

# we just convert the dictionaries of counts to a Counter
entity_inlink_dict = {x: Counter(y) for x, y in entity_inlink_dict.items()}

with open(path + "entity_inlink_dict.json", "w") as fp:
    json.dump(entity_inlink_dict, fp)

with open(path + "entity_outlink_dict.json", "w") as fp:
    json.dump(entity_outlink_dict, fp)

# finally, we create a dictionary, mapping each entity to a Counter of the frequency of associated mentions
entities_overall_dict = {x: dict() for x in overall_entity_freq.keys()}

for mention, entities in tqdm(mention_overall_dict.items()):
    for entity, freq in entities.items():
        entities_overall_dict[entity][mention] = freq

entities_overall_dict = {x: Counter(y) for x, y in entities_overall_dict.items()}

with open(path + "entities_overall_dict.json", "w") as fp:
    json.dump(entities_overall_dict, fp)

print("all done.")
