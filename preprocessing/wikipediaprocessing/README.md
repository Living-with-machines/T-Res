# Overview

These scripts process a Wikipedia dump, extracting and structuring:
- pages
- mention/entity statistics 
- in- /out-link information 
- and provides an alignment between Wikipedia pages and Wikidata IDs.


## 1. Pre-process a Wikipedia Dump

First, download a Wikipedia dump from [here](https://dumps.wikimedia.org/enwiki/) (we used `enwiki-20211001`). Then process it with the [WikiExtractor](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) and store it into the `../resources/wikipedia/processedWiki/` folder. As in the current version of the WikiExtractor the possibility of keeping sections is not offered anymore, we have used [this version](https://github.com/attardi/wikiextractor/tree/e4abb4cbd019b0257824ee47c23dd163919b731b) of the code from March 2020. To obtain it:

```
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
git checkout e4abb4cbd019b0257824ee47c23dd163919b731b 
```

Then you can run the tool with the following command:

```
python WikiExtractor.py -l -s -o ../resources/wikipedia/processedWiki/ [here put the path to the Wikipedia Dump .xml.bz2]
```

Note that the flag -s will keep the sections and the flag -l the links.

### Run the code in test mode

If you want to run the scripts in `test` mode you should download first only one batch of .xml.bz2 articles (instead of a full dump). In our setting we used `enwiki-20211001-pages-articles-multistream27.xml-p68475910p68864378.bz2` (which is around 200MB instead of over 18GB). You should process it the same way we describe above for the full dump and store it in `../resources/wikipedia/test-processedWiki/`.

## 2. Extract entities/mentions frequency counts and pages

Having the Wiki dump processed by the WikiExtractor in the `../resources/wikipedia/processedWiki/` folder, the first step is to extract entity and mention statistics (e.g., how many times the mention `London` is pointing to Wikipedia page of the capital of the UK and how many times to `London,_Ontario`). Statistics are still divided in the n-folders consituting the output of the WikiExtractor and will be saved in the `../resources/wikipedia/extractedResources/Store-Counts/` folder as json files. The script will also store a .json file for each entity, with all its aspects (i.e., sections, see [here](https://madoc.bib.uni-mannheim.de/49596/1/EAL.pdf) to know more about Entity-Aspect Linking) in the `Pages/` folder. You should run the script as:

```
python extract_freq_and_pages.py
```
Note that if you have set up the `test` mode, you can run the script using the flag `-t`, which will consider only a sub-part of the corpus.

Next, you can aggregate all entity and mention counts in single `.json` file and save them in the `extractedResources/` folder by running:
```
python aggregate_all_counts.py
```
As above, you can run the script in test mode using the flag `-t`, if you have set it up.

## 3. Map Wikipedia and Wikidata

Finally, to align the Wikipedia pages extracted from the dump to Wikidata you can use:
```
python map_wikidata_wikipedia.py
```
This script relies on the use of the WikiMapper and in particular to the availability of a specific Wikipedia/Wikidata index, which we have created following [these instructions](https://github.com/jcklie/wikimapper#create-your-own-index), using a SQL dump from October 2021. It will produce two json files, mapping wikidata ids to wikipedia pages and viceversa. As above, you can run it in `test` mode as well, if you have set this up.

## Final outputs

These scripts will produce the following outputs (note that entities are percent encoded across all files):

In the 
- A `Pages/` folder, containing a `.json` file for each page available in the input Wikipedia dump. Note that due to the presence of specific characters of to the length of some pages titles, some titles have been hashed.
- `hashed_duplicates.csv`: just to check in case there are issues with duplicate hashed filenames. This file should remain empty.  
- A `Store-Counts/` folder, containing partial counts as `.json` files.
- `entities_overall_dict.json`: this is a dictionary which maps each entity to a `Counter` object of all possible mentions  
- `mention_overall_dict.json`: this is a dictionary which maps each mention to a `Counter` object of all possible associated entities.
- `overall_entity_freq.json`: this is a dictionary which simply maps an entity to its overall frequency in the Wikipedia corpus.
- `overall_mentions_freq.json`: this is a dictionary which simply maps a mention to its overall frequency in the Wikipedia corpus.
- `entity_inlink_dict.json`: this dictionary gives you a list of pages linking to each Wikipedia page.
- `entity_outlink_dict.json`: this dictionary gives you a list of pages linked from each Wikipedia page.
- `wikipedia2wikidata.json`: a dictionary mapping Wikipedia pages to Wikidata ids.
- `wikidata2wikipedia.json`: a dictionary mapping Wikidata ids to a list of Wikipedia pages with associated frequency.
