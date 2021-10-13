import os,pickle, pathlib
import multiprocessing as mp
from bs4 import BeautifulSoup

def process_doc(doc):
    content = open(processed_docs+folder+"/"+doc).read()
    soup = BeautifulSoup(content,features="html.parser").findAll("doc")
    pages = []
    for page in soup:
        mentions = find_mentions(page)
        pages.append(mentions)
    return pages

def find_mentions(page):
    mentions = set([link.text for link in page.findAll("a") if link.has_attr("href")])
    return mentions


processed_docs = "/resources/wikipedia/test-processedWiki/"

N= mp.cpu_count()-2

all_mentions = set()

if __name__ == '__main__':

    for folder in os.listdir(processed_docs):
        if folder != ".gitkeep":
            with mp.Pool(processes = N) as p:
                groups = p.map(process_doc, os.listdir(processed_docs+folder))
        
            all_mentions.update([mention for group in groups for page in group for mention in page])


out = '/resources/wikipedia/extractedResources/'

pathlib.Path(out).mkdir(parents=True, exist_ok=True)

with open(out+'all_mentions.pickle', "wb") as fp:
    pickle.dump(all_mentions, fp)


