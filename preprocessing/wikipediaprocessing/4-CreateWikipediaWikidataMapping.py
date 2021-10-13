import wptools
from tqdm import tqdm
import os, pickle, json
from wikimapper import WikiMapper


with open("/resources/wikipedia/extractedResources/overall_entity_freq.pickle", "rb") as f:
    overall_entity_freq = pickle.load(f)

mapper = WikiMapper("/resources/wikidata2wikipedia/index_enwiki-20190420.db")

wikidata2wikipedia = {}

wikipedia2wikidata = {}

folder = os.listdir("/resources/wikipedia/extractedResources/Pages/")

for i in tqdm(range(len(folder))):
    page = folder[i]
    title = page.replace(".json","")
    freq = overall_entity_freq[title]
    wikidata_id = mapper.title_to_id(title.replace(" ","_"))
    if wikidata_id is None:
        try:
            wikidata_id = wptools.page(title,silent=True,verbose=False).get_wikidata().data["wikibase"]
        except LookupError:
            wikidata_id = None
    if wikidata_id:
        if wikidata_id not in wikidata2wikipedia:
            wikidata2wikipedia[wikidata_id] = [{"title":title,"freq":freq}]
        else:
            wikidata2wikipedia[wikidata_id].append({"title":title,"freq":freq})

        wikipedia2wikidata[page] = wikidata_id

with open('/resources/wikipedia/extractedResources/wikidata2wikipedia.json', 'w') as fp:
    json.dump(wikidata2wikipedia, fp)

with open('/resources/wikipedia/extractedResources/wikipedia2wikidata.json', 'w') as fp:
    json.dump(wikipedia2wikidata, fp)

