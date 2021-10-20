import hashlib
from tqdm import tqdm
import os, json,pathlib
from utils import process_wikipedia
import multiprocessing as mp
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-t", "--test", dest="test",
                    help="run in test mode",action='store_true')

args = parser.parse_args()

# we setup these folders for acquiring the output of this script
if args.test:
    path = '/resources/wikipedia/test-extractedResources/'
    processed_docs = '/resources/wikipedia/test-processedWiki/'
else:
    path = '/resources/wikipedia/extractedResources/'
    processed_docs = '/resources/wikipedia/processedWiki/'

pathlib.Path(path+'Pages/').mkdir(parents=True, exist_ok=True)
pathlib.Path(path+'Store-Counts/').mkdir(parents=True, exist_ok=True)

# this is where we have stored the output of the WikiExtractor

# the number of cpus
N= mp.cpu_count()

# this is just to doublecheck in case we have duplicatec hashed files (it shouldn't happen)
out = open(path+'hashed_duplicates.csv','w')

if __name__ == '__main__':
    
    # a Wikipedia dump is divided in subparts, which the WikiExtractor maps to different folders
    folders = list(os.listdir(processed_docs))
    for i in tqdm(range(len(folders))):
        folder = folders[i]
        # we set up multiple processes to go through all files in each folder
        with mp.Pool(processes = N) as p:
            paths = [processed_docs+folder+"/"+filename for filename in os.listdir(processed_docs+folder)]
            pages = p.map(process_wikipedia.process_doc, paths)

        pages = [page for group in pages for page in group]
        
        # separating frequency counts from aspects
        freq_counts = [x[0][:-1] for x in pages]
        pages_with_sections = [x[0][-1] for x in pages]
        
        # saving pages with sections
        for page_with_sect in pages_with_sections:
            # hashing the title and checking if too long or containing /
            if "/" in page_with_sect["title"] or len(page_with_sect["title"])>200:
                pagetitle = hashlib.sha224(page_with_sect["title"].encode('utf-8')).hexdigest()
                # just checking if there are multiple titles with the same hash (it should not happen)
                if pagetitle+".json" in set(os.listdir(path+'Pages/')):
                    out.write(page_with_sect["title"]+","+pagetitle+"\n")
                    continue
            else:
                pagetitle = page_with_sect["title"]

            sections = page_with_sect["sections"]

            # saving the page with sections
            with open(path+'Pages/'+pagetitle+".json", 'w') as fp:
                json.dump(sections, fp)
    
        # storing counts, still divided in folders       
        with open(path+'Store-Counts/'+str(i)+".json", 'w') as fp:
            json.dump(freq_counts, fp)
        
out.close()