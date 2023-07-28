<div style="text-align: center">
<h1>T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers</h1>
</div>

## Overview

T-Res is an end-to-end pipeline for toponym resolution for digitised historical newspapers. Given an input text, T-Res identifies the places that are mentioned in it, links them to their corresponding Wikidata IDs, and provides their geographic coordinates. T-Res has been designed to tackle common problems of working with digitised historical newspapers.

The pipeline has three main components:

* **The Recogniser** performs named entity recognition.
* **The Ranker** performs candidate selection and ranking.
* **The Linker** performs entity linking and resolution.

The three components are used in combination in the **Pipeline** class.

We also provide the code to deploy T-Res as an API, and show how to use it. Each of these elements are described in this documentation.

## Directory structure

```
toponym-resolution/
   ├── app/
   ├── docs/
   ├── evaluation/
   ├── examples/
   ├── experiments/
   │   └── outputs/
   ├── geoparser/
   ├── resources/
   │   ├── deezymatch/
   │   ├── models/
   │   ├── news_datasets/
   │   ├── wikidata/
   │   └── wikipedia/
   ├── tests/
   └── utils/
```

## Documentation

The T-Res documentation can be found at **[TODO]**.

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

Finally, this work has used the [HIPE-scorer](https://github.com/hipe-eval/HIPE-scorer/blob/master/LICENSE) for assessing the performance of T-Res.

## Cite

**[TODO]**
