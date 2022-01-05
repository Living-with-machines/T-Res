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

## 2. Extract entities/mentions frequency counts and pages

Having the Wiki dump processed by the WikiExtractor in the `../resources/wikipedia/processedWiki/` folder, the first step is to extract entity and mention statistics (e.g., how many times the mention `London` is pointing to Wikipedia page of the capital of the UK and how many times to `London,_Ontario`). Statistics are still divided in the n-folders consituting the output of the WikiExtractor and will be saved in the `../resources/wikipedia/extractedResources/Store-Counts/` folder as json files. The script will also store a .json file for each entity, with all its aspects (i.e., sections, see [here](https://madoc.bib.uni-mannheim.de/49596/1/EAL.pdf) to know more about Entity-Aspect Linking) in the `Pages/` folder. You should run the script as:

```
python extract_freq_and_pages.py
```
Note that you can run the script in `test` mode using the flag `-t`, which will consider only a sub-part of the corpus.

Next, you can aggregate all entity and mention counts in single `.json` file and save them in the `extractedResources/` folder by running:
```
python aggregate_all_counts.py
```
As above, you can run the script in test mode using the flag `-t`.

## 3. Map Wikipedia and Wikidata

To align the Wikipedia pages extracted from the dump to Wikidata you can use:
```
python map_wikidata_wikipedia.py
```
This script relies on the use of the WikiMapper and in particular to the availability of a specific Wikipedia/Wikidata index, which we have created following [these instructions](https://github.com/jcklie/wikimapper#create-your-own-index), using a SQL dump from October 2021. It will produce two json files, mapping wikidata ids to wikipedia pages and viceversa. As above, you can run it in `test` mode as well, using the flag `-t`.

## 4. Create a Wikidata-based gazetteer

Finally, to extract locations from Wikidata (and their relevant properties) if they have a corresponding page on Wikipedia, you can use:
```
python entity_extraction.py -t ['True'|'False']
```

This script is partially based on [this code](https://akbaritabar.netlify.app/how_to_use_a_wikidata_dump).

The script assumes that you have already downloaded a full Wikidata dump (`latest-all.json.bz2`) from [here](https://dumps.wikimedia.org/wikidatawiki/entities/). We assume the downloaded `bz2` file is stored in `../resources/wikidata/`.

By default, the script runs on test mode. You can change this behaviour by setting `-t` to `'False'`. Beware that this step will take about 2 full days.

The output is in the form of `.csv` files that will be created in `../resources/wikidata/extracted/`, each containing 5,000 rows corresponding to geographical entities extracted from Wikidata (if they have a corresponding Wikipedia page) with the following fields (corresponding to wikidata properties, e.g. `P7959` for [historical county](https://www.wikidata.org/wiki/Property:P7959); a description of each can be found as comments in the [code](https://github.com/Living-with-machines/toponym-resolution/blob/main/preprocessing/wikipediaprocessing/wikidata_extraction.py#L91-L393)):

```
'wikidata_id', 'english_label', 'instance_of', 'description_set', 'alias_dict', 'nativelabel', 'population_dict', 'area', 'hcounties', 'date_opening', 'date_closing': date_closing, 'inception_date', 'dissolved_date', 'follows', 'replaces', 'adm_regions', 'countries', 'continents', 'capital_of', 'borders', 'near_water', 'latitude', 'longitude', 'wikititle', 'geonamesIDs', 'connectswith', 'street_address', 'street_located', 'postal_code'
```

## Final outputs

These scripts will produce the following outputs (note that entities are percent encoded across all files):

- In `/resources/wikipedia/extractedResources/`:
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
- In `/resources/wikidata/extracted/`:
  - A list of `.csv` files, each containing 5,000 rows corresponding to geographical entities extracted from Wikidata.
