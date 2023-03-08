# Reproducing the Experiments

Follow these steps to reproduce the experiments in our paper.

### 1. Obtain the external resources [DRAFT]

You will need the following resources, which are created using the code in the wiki2gaz repository ([TODO: add link]) or can be downloaded from [TODO: add link]:
```
../resources/wikidata/mentions_to_wikidata.json
../resources/wikidata/wikidata_gazetteer.csv
../resources/wikidata/mentions_to_wikidata_normalized.json
../resources/wikidata/wikidata_to_mentions_normalized.json
../resources/wikipedia/wikidata2wikipedia/index_enwiki-latest.db
```

### 2. Preparing the data

To create the datasets that we use in the experiments presented in the paper, run the following command:
```bash
python prepare_data.py
```
This script takes care of downloading the LwM and HIPE datasets and format them as needed in the experiments.

### 3. Running the experiments