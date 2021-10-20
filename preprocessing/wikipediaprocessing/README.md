The scripts here process a Wikipedia dump, extracting and structuring pages, mention/entity statistics and in- /out-link information.


### 1. Pre-process a Wikipedia Dump

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

### 2. Extract entity, mention and ngram frequencies + sections

Having the Wiki dump processed by the WikiExtractor in the "processedWiki/" folder, the first step is to extract entity and mention statistics (e.g., how many times the mention "London" is pointing to "London" the capital of the UK and how many times to "London,_Ontario"). Statistics are still divided in the n-folders consituting the output of the WikiExtractor and will be saved in the `../resources/wikipedia/extractedResources/Store-Counts/` folder as json files. The script will also store a .json file for each entity, with all its aspects (i.e., sections, see [here](https://madoc.bib.uni-mannheim.de/49596/1/EAL.pdf) to know more about Entity-Aspect Linking):

```
extract_freq_and_pages.py
```

Next, you can aggregate all entity and mention counts in single .pickle file and save them in the `extractedResources/` folder by running:
```
aggregate_all_counts.py
```
### 3. Extract entity, mention and ngram frequencies + sections

