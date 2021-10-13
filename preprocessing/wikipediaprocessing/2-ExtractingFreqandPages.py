import hashlib
import os, json,pathlib
from bs4 import BeautifulSoup
from collections import Counter
import multiprocessing as mp

def clean_page(page):
        
    entities = [x for x in page.findAll("a") if x.has_attr("href")]
         
    box_mentions = Counter([x.text for x in entities])
    box_entities = Counter([x["href"] for x in entities])
        
    mentions_dict = {x:[] for x in box_mentions}
    for e in entities:
        mentions_dict[e.text].append(e["href"])
    
    mentions_dict = {x:Counter(y) for x,y in mentions_dict.items()} 
    
    return [box_mentions,box_entities,mentions_dict]

def get_sections (page):
    page = page.text.strip().split("\n")
    sections = {"Main":{"order":1,"content":[]}}
    dict_key = "Main"
    ct = 1
    for line in page:
        if not "Section::::" in line:
            sections[dict_key]["content"].append(line)
        else:
            ct+=1
            dict_key = line.replace("Section::::","")[:-1]
            sections[dict_key] = {"order":ct,"content":[]}
            
    sections = {x:y for x,y in sections.items() if len(y["content"])>0}
    return sections

def process_doc(doc):
    content = open(processed_docs+folder+"/"+doc,).read()
    content = BeautifulSoup(content,"html.parser").findAll("doc")
    pages = []
    for page in content:
        title = page["title"]
        sections = {"title":title,"sections": get_sections(page)}
        r = [title]+ clean_page(page) + [sections]
        pages.append([r])
    return pages

processed_docs = "/resources/wikipedia/test-processedWiki/"

pathlib.Path('/resources/wikipedia/extractedResources/Pages/').mkdir(parents=True, exist_ok=True)
pathlib.Path('/resources/wikipedia/extractedResources/Store-Counts/').mkdir(parents=True, exist_ok=True)

# the number of cpu
N= mp.cpu_count()-2

if __name__ == '__main__':
    
    step = 1

    for folder in os.listdir(processed_docs):
        with mp.Pool(processes = N) as p:
            res = p.map(process_doc, os.listdir(processed_docs+folder))

        res = [y for x in res for y in x]
        
        # separating frequency counts from aspects
        freq_res = [x[0][:-1] for x in res]
        sections = [x[0][-1] for x in res]
        
        # saving sections independently
        for sect in sections:
            hashed_title = hashlib.sha224(sect["title"].encode('utf-8')).hexdigest()
#                wikidata_id = sect["wikidata_id"]
#                if wikidata_id is not None:
            s = sect["sections"]
            if hashed_title+".json" not in os.listdir('/resources/wikipedia/extractedResources/Pages/'):
                with open('/resources/wikipedia/extractedResources/Pages/'+hashed_title+".json", 'w') as fp:
                    json.dump(s, fp)
            else:
                print ("issue with hash!",sect["title"])
        # storing counts, still divided in folders       
        with open('/resources/wikipedia/extractedResources/Store-Counts/'+str(step)+".json", 'w') as fp:
            json.dump(freq_res, fp)
        
        print("Done %s folders over %s" % (step, len(os.listdir(processed_docs))))
        step+=1