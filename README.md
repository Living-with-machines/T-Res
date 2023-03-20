# Toponym resolution for digitised historical newspapers

A toponym resolution pipeline for digitised historical newspapers.

## Table of contents

* [Overview](#overview)
* [Directory structure](#directory-structure)
* [Resources](#resources)
* [Installation](#installation)

## Overview

T-Res is an end-to-end pipeline for toponym resolution in digitised historical newspapers. Given an input text (a sentence or a text), it identifies the places that are mentioned in it, links them to their corresponding Wikidata IDs, and provides their geographic coordinates. T-Res has been designed to tackle common problems of working with digitised historical newspapers.

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

The setup relies on the integration of `pyenv` and `poetry`. The following are the commands you should run if you are setting this up on Ubuntu (to know more, see [these guidelines](https://www.adaltas.com/en/2021/06/09/pyrepo-project-initialization/)). To install them on a different OS, you can follow [these guidelines](https://github.com/pyenv/pyenv#installation) for `pyenv` (but remember to first of all install the prerequisites listed in the link!) and then [these guidelines](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions) for `poetry`.

If you haven't already `pyenv` and `poetry` installed, first you need to ensure the following packages are installed:

```
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
```

Then you can install `pyenv` with the `pyenv-installer`:

```
curl https://pyenv.run | bash
```
Then to properly configure pyenv for use on the system, you need:

```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init --path)"\nfi' >> ~/.bashrc
```

Restart the terminal:
```
source ~/.bashrc
```

Check that `pyenv` is correctly installed by typing:
```
pyenv
```

Install Python 3.9.7 and set it as the global Python version:

```
pyenv install 3.9.7
pyenv global 3.9.7
```

Now you can install `poetry` the following way:

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

To configure poetry for use in the system:
```
echo 'export PATH=$PATH:$HOME/.poetry/bin' >> ~/.bashrc
```

Restart the terminal:
```
source ~/.bashrc
```

And check whether `poetry` works:
```
poetry
```

## Project Installation

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

You then need to add a couple of NLTK resources, needed by DeezyMatch:

```
poetry run python -m nltk.downloader brown words
```

To use Jupyter notebooks, you will need to add the Jupyter package:
```
poetry add -D jupyter
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

Now you can run a script as usual:

```
python processing.py
```

Add a package:

```
poetry add [package name]
```

Run the Python tests:

```
poetry run pytest
```

If you want to use Jupyter notebook, run it as usual, and then select the created kernel in "Kernel" > "Change kernel".

```
jupyter notebook
```
