# T-Res: A Toponym Resolution Pipeline for Digitised Historical Newspapers

<div align="center">
    <figure>
    <img src="https://user-images.githubusercontent.com/8415204/234827786-ff796b7d-5773-427c-98d5-006a108506a8.png"
        alt="A cartoon of a funny T-Rex reading a map with a lense"
        width="70%">
    </figure>
</div>


## Table of contents

* [Overview](#overview)
* [Directory structure](#directory-structure)
* [Resources](#resources)
* [Installation](#installation)

## Overview

T-Res is an end-to-end pipeline for toponym resolution for digitised historical newspapers. Given an input text (a sentence or a text), T-Res identifies the places that are mentioned in it, links them to their corresponding Wikidata IDs, and provides their geographic coordinates. T-Res has been designed to tackle common problems of working with digitised historical newspapers.

The pipeline has three main components:
* The Recogniser: performs named entity recognition.
* The Ranker: performs candidate selection and ranking.
* The Linker: performs entity linking and resolution.

The three components are used in combination in the Pipeline class.

We also provide the code to deploy T-Res as an API, and show how to use it. We will describe each of these elements below.

## Directory structure

```
toponym-resolution/
   ├── app/
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

### The Recogniser

### The Ranker

### The Linker

### The Pipeline

### The T-Res API




## Resources

### Generic resources required by T-Res

[TODO]

### Toponym recognition using BLERT

[TODO]

### Candidate ranking using DeezyMatch

We provide different options for using a DeezyMatch model for OCR:
1. Training a DeezyMatch model from scratch, including generating a string pairs dataset.
2. Training a DeezyMatch model, given an existing pairs dataset.
3. Using an existing DeezyMatch model.

If you're working on 18th-20th century digitised texts in English, you can use option 2. Otherwise, we recommend you generate your own pairs dataset (from your own word2vec embeddings).

#### **Option 1:** Training a DeezyMatch model from scratch, including generating a pairs dataset

To do so, the `resources/` folder should (at least) contain the following files, in the following locations:
```
toponym-resolution/
   ├── ...
   ├── resources/
   │   ├── deezymatch/
   │   │   ├── data/
   │   │   └── inputs/
   │   │       ├── characters_v001.vocab
   │   │       └── input_dfm.yaml
   │   ├── models/
   │   │   └── w2v
   │   │       ├── w2v_[XXX]_news
   │   │       │   ├── w2v.model
   │   │       │   ├── w2v.model.syn1neg.npy
   │   │       │   └── w2v.model.wv.vectors.npy
   │   │       └── ...
   │   ├── news_datasets/
   │   ├── wikidata/
   │   │   └── mentions_to_wikidata.json
   │   └── wikipedia/
   └── ...
```
Note that the `resources/models/w2v/` folder may contain several word2vec models: each should be named `w2v_[XXXX]_news`, where `[XXXX]` can any identifier. If the goal is to learn models that work well with OCR errors, these word2vec embeddings should ideally be trained from noisy OCR data (digitised historical newspapers, for example).

See this with an example in [this notebook](https://github.com/Living-with-machines/toponym-resolution/blob/release/examples/examples/train_use_deezy_model_1.ipynb).

[TODO: add link to the w2v models on Zenodo]

#### **Option 2:** Training a DeezyMatch model, given an existing pairs dataset

To do so, the `resources/` folder should (at least) contain the following files, in the following locations:
```
toponym-resolution/
   ├── ...
   ├── resources/
   │   ├── deezymatch/
   │   │   ├── data/
   │   │       └── w2v_ocr_pairs.txt
   │   │   └── inputs/
   │   │       ├── characters_v001.vocab
   │   │       └── input_dfm.yaml
   │   ├── models/
   │   ├── news_datasets/
   │   ├── wikidata/
   │   │   └── mentions_to_wikidata.json
   │   └── wikipedia/
   └── ...
```

See this with an example in [this notebook](https://github.com/Living-with-machines/toponym-resolution/blob/release/examples/examples/train_use_deezy_model_2.ipynb).

[TODO: add link to the w2v_ocr_pairs.txt or upload to github]

#### **Option 3:** Using an existing DeezyMatch model

To do so, the `resources/` folder should **(at least)** contain the following folders, in the following locations:
```
toponym-resolution/
   ├── ...
   ├── resources/
   │   ├── deezymatch/
   │   │   └── models/
   │   │       └── w2v_ocr/
   │   │           ├── input_dfm.yaml
   │   │           ├── w2v_ocr.model
   │   │           ├── w2v_ocr.model_state_dict
   │   │           └── w2v_ocr.vocab
   │   ├── models/
   │   ├── news_datasets/
   │   ├── wikidata/
   │   │   └── mentions_to_wikidata.json
   │   └── wikipedia/
   └── ...
```

See this with an example in [this notebook](https://github.com/Living-with-machines/toponym-resolution/blob/release/examples/examples/train_use_deezy_model_3.ipynb).

[TODO: add link to the DeezyMatch model on Zenodo]

### Toponym linking using REL

[TODO]

## Installation

If you want to work directly on the code base, we suggest to install T-Res following these instructions (which have been tested Linux (ubuntu 20.04)). 

### First, update the system

First, you need to make sure the system is up to date and all essential libraries are installed.

```
sudo apt update
sudo apt install     build-essential     curl     libbz2-dev     libffi-dev     liblzma-dev     libncursesw5-dev     libreadline-dev     libsqlite3-dev     libssl-dev     libxml2-dev     libxmlsec1-dev     llvm     make     tk-dev     wget     xz-utils     zlib1g-dev
```

### Install pyenv

Then you need to install pyenv, which we use to manage virtual environments:

```
curl https://pyenv.run | bash
```
And also to make sure paths are properly exported:

```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init --path)"\nfi' >> ~/.bashrc
```
Then you can restart your bash session, to make sure all changes are updated:

```
source ~/.bashrc
```
And then you run the following commands to update `pyenv` and create the needed environemnt.

```
pyenv update

pyenv install 3.9.7
pyenv global 3.9.7
```

### Install poetry

To manage dipendencies across libraries, we use Poetry. To install it, do the following:

```
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH=$PATH:$HOME/.poetry/bin' >> ~/.bashrc
```

### Project Installation

You can now clone the repo and `cd` into it:

```
git clone https://github.com/Living-with-machines/toponym-resolution.git
cd toponym-resolution
```

Explicitly tell poetry to use the python version defined above:

```
poetry env use python
```

Install all dependencies using `poetry`:

```
poetry install
```

Create a kernel:
```
poetry run ipython kernel install --user --name=<KERNEL_NAME>
```

### How to use poetry

To activate the environment:

```
poetry shell
```

Now you can run a script as usual, for instance :

```
python experiments/toponym_resolution.py
```

To add a package:

```
poetry add [package name]
```

To run the Python tests:

```
poetry run pytest
```

If you want to use Jupyter notebook, run it as usual, and then select the created kernel in "Kernel" > "Change kernel".

```
jupyter notebook
```
