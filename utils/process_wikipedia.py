import urllib
from bs4 import BeautifulSoup
from collections import Counter

### Processing pages ####

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

# we open each file and extract the title, the entity frequencies and the sections
def process_doc(filename):
    content = open(filename).read()
    content = BeautifulSoup(content,"html.parser").findAll("doc")
    pages = []
    for page in content:
        title = urllib.parse.quote(page["title"])
        sections = {"title":title,"sections": get_sections(page)}
        r = [title]+ clean_page(page) + [sections]
        pages.append([r])
    return pages

### Aggregating statistics pages ####

def fill_dicts(res,mentions_freq, entity_freq, mention_overall_dict,entity_inlink_dict,entity_outlink_dict):    

    title,box_mentions, box_entities, mentions_dict = res[0],res[1],res[2],res[3]
    # to make it percent encoded as the other references to the same entity
    mentions_freq+= box_mentions
    entity_freq+= box_entities

    entity_outlink_dict[title] = box_entities
    
    for k,v in box_entities.items():
        if k in entity_inlink_dict:
            entity_inlink_dict[k].append(title)
        else:
            entity_inlink_dict[k] = [title]
    
    for k,v in mentions_dict.items():
        if k in mention_overall_dict:
            mention_overall_dict[k]+=v
        else:
            mention_overall_dict[k] = Counter()
            mention_overall_dict[k]+= v
    return mentions_freq, entity_freq, mention_overall_dict,entity_inlink_dict,entity_outlink_dict