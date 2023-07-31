<div style="text-align: center">
<h1>T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers</h1>
</div>

## Overview

T-Res is an end-to-end pipeline for toponym detection, linking, and resolution on digitised historical newspapers. Given an input text, T-Res identifies the places that are mentioned in it, links them to their corresponding Wikidata IDs, and provides their geographic coordinates. T-Res has been developed to assist researchers explore large collections of digitised historical newspapers, and has been designed to tackle common problems often found when dealing with this type of data.

The pipeline has three main components:

* **The Recogniser** performs named entity recognition.
* **The Ranker** performs candidate selection and ranking.
* **The Linker** performs entity linking and resolution.

The three components are used in combination in the **Pipeline** class.

We also provide the code to deploy T-Res as an API, and show how to use it. Each of these elements are described in the documentation.

The repository contains the code for the experiments described in our paper.

## Documentation

The T-Res documentation can be found at https://living-with-machines.github.io/T-Res/.

## Resources and directory structure

T-Res relies on several resources in the following directory structure:

```
T-Res/
├── app/
├── evaluation/
├── examples/
├── experiments/
│   └── outputs/
│       └── data/
│           └── lwm/
│               ├── linking_df_split.tsv [*?]
│               ├── ner_fine_dev.json [*+?]
│               └── ner_fine_train.json [*+?]
├── geoparser/
├── resources/
│   ├── deezymatch/
│   │   └── data/
│   │       └── w2v_ocr_pairs.txt [*+?]
│   ├── models/
│   ├── news_datasets/
│   ├── rel_db/
│   │   └── embeddings_database.db [*+?]
│   └── wikidata/
│       ├── entity2class.txt [*]
│       ├── mentions_to_wikidata_normalized.json [*]
│       ├── mentions_to_wikidata.json [*]
│       ├── wikidta_gazetteer.csv [*]
│       └── wikidata_to_mentions_normalized.json [*]
├── tests/
└── utils/
```

These resources are described in detail in the documentation. A question mark (`?`) is used to indicate resources which are only required for some approaches (for example, the `rel_db/embeddings_database.db` file is only required by the REL-based disambiguation approaches). Note that an asterisk (`*`) next to the resource means that the path can be changed when instantiating the T-Res objects, and a plus sign (`+`) if the name of the file can be changed in the instantiation.

By default, T-Res expects to be run from the `experiments/` folder, or a directory in the same level (for example, the `examples/` folder).

## Example

This is an example on how to use the default T-Res pipeline:

```python
from geoparser import pipeline

geoparser = pipeline.Pipeline()

output = geoparser.run_text("She was on a visit at Chippenham.")
```

This returns:

```python
[{'mention': 'Chippenham',
  'ner_score': 1.0,
  'pos': 22,
  'sent_idx': 0,
  'end_pos': 32,
  'tag': 'LOC',
  'sentence': 'She was on a visit at Chippenham.',
  'prediction': 'Q775299',
  'ed_score': 0.651,
  'string_match_score': {'Chippenham': (1.0,
    ['Q775299',
     'Q3138621',
     'Q2178517',
     'Q1913461',
     'Q7592323',
     'Q5101644',
     'Q67149348'])},
  'prior_cand_score': {},
  'cross_cand_score': {'Q775299': 0.651,
   'Q3138621': 0.274,
   'Q2178517': 0.035,
   'Q1913461': 0.033,
   'Q5101644': 0.003,
   'Q7592323': 0.002,
   'Q67149348': 0.002},
  'latlon': [51.4585, -2.1158],
  'wkdt_class': 'Q3957'}]
```

Note that T-Res allows the user to use their own knowledge base, and to choose among different approaches for performing each of the steps in the pipeline. Please refer to the documentation to learn how.

## Acknowledgements

This work was supported by Living with Machines (AHRC grant AH/S01179X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1).

Living with Machines, funded by the UK Research and Innovation (UKRI) Strategic Priority Fund, is a multidisciplinary collaboration delivered by the Arts and Humanities Research Council (AHRC), with The Alan Turing Institute, the British Library and Cambridge, King's College London, East Anglia, Exeter, and Queen Mary University of London.

## Credits

This work has been inspired by many previous projects, but particularly the [Radboud Entity Linker (REL)](https://github.com/informagi/REL).

We adapt some code from:
* Huggingface tutorials: [Apache License 2.0](https://github.com/huggingface/notebooks/blob/main/LICENSE)
* DeezyMatch tutorials: [MIT License](https://github.com/Living-with-machines/DeezyMatch/blob/master/LICENSE)
* Radboud Entity Linker: [MIT License](https://github.com/informagi/REL/blob/main/LICENSE)
* Wikimapper: [Apache License 2.0](https://github.com/jcklie/wikimapper/blob/master/LICENSE)

Classes, methods and functions that have been taken or adapted from above are credited in the docstrings.

In our experiments, we have used resources built from Wikidata and Wikipedia for linking. In order to assess T-Res performance, we have used the [topRes19th](https://doi.org/10.23636/r7d4-kw08) and the [HIPE-2020](https://impresso.github.io/CLEF-HIPE-2020/datasets.html) datasets, and the [HIPE-scorer](https://github.com/hipe-eval/HIPE-scorer/blob/master/LICENSE) for evaluation.

## Cite

**[TODO]**
