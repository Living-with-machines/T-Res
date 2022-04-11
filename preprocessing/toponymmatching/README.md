# Create a dataset of OCR aligned pairs

## Obtaining the aligned dataframe

These are the steps to reproduce the creation of a dataset of aligned OCR token pairs. See [this repository](https://github.com/Living-with-machines/lwm_ARTIDIGH_2020_OCR_impact_downstream_NLP_tasks) for more information.

Run the following notebooks:
* [Create the `trove` dataframe](https://github.com/Living-with-machines/lwm_ARTIDIGH_2020_OCR_impact_downstream_NLP_tasks/blob/master/create_trove_dataframe.ipynb), which covers the steps required to download Overproof data and process this data into a Pandas Dataframe. It will take some time to run the first time. The notebook will output the Pandas Dataframe as a pickle file.
* [Align `trove` tokens](https://github.com/Living-with-machines/lwm_ARTIDIGH_2020_OCR_impact_downstream_NLP_tasks/blob/master/aligning_trove.ipynb), which performs alignment between the two versions of the text. Further explanation of the approach taken is outlined in the notebook. Store the output dataframe `trove_subsample_aligned.pkl` in `../../resources/ocr/`.

## Creating a string matching dataset

Run `create_ocr_pairs.py` to create a string matching dataset, that will be stored as `../../experiments/outputs/deezymatch/ocr_string_pairs.txt`.

## Training a DeezyMatch model and generating candidate vectors

The `train_deezy_model.py` script does the following:
1. Trains a DeezyMatch model. It requires the following:
    * An input file, provided in this folder.
    * A character vocabulary file, provided in this folder.
    * The string pairs dataset generated in the previous step.
2. Generates candidate vectors from Wikidata. It requires the following:
    * The `mentions_to_wikidata.json` file that is generated when processing wikipedia and wikidata (in particular, generated in `../wikipediaprocessing/create_wk_gazetteer.py`), which contains all mentions in our resource, matched to all possible corresponding wikidata QIDs.
    * The DeezyMatch model just trained and additional files created in the previous step.

The outputs of this step are stored in different subfolders of the `../../experiments/outputs/deezymatch`.