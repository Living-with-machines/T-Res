import pathlib
from tqdm import tqdm
import json
from wikimapper import WikiMapper
from argparse import ArgumentParser

# as usual, we can run it in test mode if the extracted folders are available
parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test",
                    help="run in test mode",action='store_true')

args = parser.parse_args()

if args.test:
    path = '../../resources/wikipedia/test-extractedResources/'
else:
    path = '/resources/wikipedia/extractedResources/'

if pathlib.Path(path).is_dir() == False:
    print ("Error! TYou need to have extracted entity and mention counts in "+path)
    exit()


# we take the list of available entities
with open(path+'overall_entity_freq.json', 'r') as f:
    overall_entity_freq = json.load(f)

# and a mapper of wikipedia / wikidata
mapper = WikiMapper('/resources/wikipedia/wikidata2wikipedia/index_enwiki-latest.db')

wikidata2wikipedia = {}

wikipedia2wikidata = {}

for entity in tqdm(overall_entity_freq.keys()):
    # we also take the entity frequency in wikipedia in case of multiple wikipedia pages for a single wikidata id
    freq = overall_entity_freq[entity]
    wikidata_id = mapper.title_to_id(entity.replace("%20","_"))

    # for each wikidata id we save a list of related wikipedia pages, with their frequency (they are probably multiple redicteds of the same entry)
    if wikidata_id:
        if wikidata_id not in wikidata2wikipedia:
            wikidata2wikipedia[wikidata_id] = [{"title":entity,"freq":freq}]
        else:
            wikidata2wikipedia[wikidata_id].append({"title":entity,"freq":freq})

        # and here we map the opposite, each wikipedia page to a single wikidata id
        wikipedia2wikidata[entity] = wikidata_id

with open(path+'wikidata2wikipedia.json', 'w') as fp:
    json.dump(wikidata2wikipedia, fp)

with open(path+'wikipedia2wikidata.json', 'w') as fp:
    json.dump(wikipedia2wikidata, fp)

