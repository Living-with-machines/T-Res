import wptools
import pathlib
from tqdm import tqdm
import os, pickle, json
from wikimapper import WikiMapper

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

with open(path+'overall_entity_freq.pickle', 'rb') as f:
    overall_entity_freq = pickle.load(f)

mapper = WikiMapper('/resources/wikipedia/wikidata2wikipedia/index_enwiki-latest.db')

wikidata2wikipedia = {}

wikipedia2wikidata = {}

for entity in tqdm(overall_entity_freq.keys()):
    freq = overall_entity_freq[entity]
    wikidata_id = mapper.title_to_id(entity.replace("%20","_"))

    # just to be sure we also check whether the id is missing in the DB but exists through the API
    if wikidata_id is None:
        try:
            wikidata_id = wptools.page(entity,silent=True,verbose=False).get_wikidata().data["wikibase"]
        except LookupError:
            wikidata_id = None
    if wikidata_id:
        if wikidata_id not in wikidata2wikipedia:
            wikidata2wikipedia[wikidata_id] = [{"title":entity,"freq":freq}]
        else:
            wikidata2wikipedia[wikidata_id].append({"title":entity,"freq":freq})

        wikipedia2wikidata[entity] = wikidata_id

with open(path+'wikidata2wikipedia.json', 'w') as fp:
    json.dump(wikidata2wikipedia, fp)

with open(path+'wikipedia2wikidata.json', 'w') as fp:
    json.dump(wikipedia2wikidata, fp)

