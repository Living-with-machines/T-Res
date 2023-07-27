Reproducing the Experiments: ``experiments`` module
===================================================

Follow these steps to reproduce the experiments in our paper.

1. Obtain the external resources [DRAFT]
----------------------------------------

You will need the following resources, which are created using the code in the [wiki2gaz](https://github.com/Living-with-machines/wiki2gaz) or can be downloaded from [TODO: add link]:

..

    ../resources/wikidata/wikidata_gazetteer.csv
    ../resources/wikidata/entity2class.txt
    ../resources/wikidata/mentions_to_wikidata.json
    ../resources/wikidata/mentions_to_wikidata_normalized.json
    ../resources/wikidata/wikidata_to_mentions_normalized.json
    ../resources/wikipedia/wikidata2wikipedia/index_enwiki-latest.db

You will also need the [word2vec embeddings](TODO: add link) trained from 19th Century data. These embeddings have been created by Nilo Pedrazzini. For more information, check https://github.com/Living-with-machines/DiachronicEmb-BigHistData.

2. Preparing the data
-------------------------

To create the datasets that we use in the experiments presented in the paper, run the following command:

.. code-block:: bash

    $ python prepare_data.py

This script takes care of downloading the LwM and HIPE datasets and format them as needed in the experiments.

3. Running the experiments
--------------------------

To run the experiments, run the following script:

.. code-block:: bash

    $ python toponym_resolution.py

This script does runs for all different scenarios reported in the experiments in the paper.

4. Evaluate
-----------

To evaluate the different approaches and obtain a table with results such as the one provided in the paper, go to the `../evaluation/` directory. There, you should clone the [HIPE scorer](https://github.com/hipe-eval/HIPE-scorer). We are using the code version at commit 50dff4e, and have added the line `return eval_stats` at the end of the `get_results()` function. From `../evaluation/`, run the following script to obtain the results in latex format:

.. code-block:: bash

    $ python display_results.py

.. toctree::
   :maxdepth: 2
   :caption: Table of contents:

   experiment
   prepare_data
   toponym_resolution