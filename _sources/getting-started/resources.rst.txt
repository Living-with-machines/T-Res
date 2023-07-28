.. _top-resources:

=================================
Resources and directory structure
=================================

T-Res requires several resources to work. Some resources can be downloaded
and loaded directly from the web. Others will need to be generated, following
the instructions provided in this section.

Toponym recognition and disambiguation training data
----------------------------------------------------

We provide the dataset we used to train T-Res for the tasks of toponym recognition
(i.e. a named entity recognition task) and toponym disambiguation (i.e. an entity
linking task focused on geographical entities). The dataset is based on the
`TopRes19th dataset <https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.56>`_.

.. note::

    You can download the data (in the format required by T-Res) from the `British
    Library research repository <https://bl.iro.bl.uk/concern/datasets/ef537c70-87cb-495a-86c8-edffefa6bdc6>`_.

By default, T-Res assumes the files are stored in the following location:

::

    T-Res/
    └── experiments/
        └── outputs/
            └── data/
                └── lwm/
                    ├── ner_fine_dev.json
                    ├── ner_fine_test.json
                    └── linking_df_split.tsv

Continue reading the sections below to learn more about the datasets, and for a
description of the format expected by T-Res.

1. Toponym recognition dataset
##############################

T-Res allows directly loading a pre-trained BERT-based NER model, either locally
or from the HuggingFace models hub. If this is your option, you can skip this
section. Otherwise, if you want to train your own NER model using either our
dataset or a different dataset, you should continue reading.

T-Res requires that the data for training a NER model is provided as two json files
(one for training, one for testing) in the JSON Lines format, where each line
corresponds to a sentence. Each sentence is a dictionary with three key-value
pairs: ``id`` (an identifier of the sentence, a string), ``tokens`` (the list of
tokens into which the sentence has been split), and ``ner_tags`` (the list of
annotations per token, in the BIO format). The length of ``tokens`` and ``ner_tags``
is therefore always the same. See below an example of three lines from one of
the JSON files, corresponding to three annotated sentences:

.. code-block:: json

    {"id":"3896239_29","ner_tags":["O","B-STREET","I-STREET","O","O","O","B-BUILDING","I-BUILDING","O","O","O","O","O","O","O","O","O","O"],"tokens":[",","Old","Millgate",",","to","the","Collegiate","Church",",","where","they","arrived","a","little","after","ten","oclock","."]}
    {"id":"8262498_11","ner_tags":["O","O","O","O","O","O","O","O","O","O","O","B-LOC","O","B-LOC","O","O","O","O","O","O"],"tokens":["On","the","'","JSth","November","the","ship","Santo","Christo",",","from","Monteveido","to","Cadiz",",","with","hides","and","copper","."]}
    {"id":"10715509_7","ner_tags":["O","O","O","B-LOC","O","O","O","O","O","O","O","O","O","O","O","O"],"tokens":["A","COACH","to","SOUTHAMPTON",",","every","morning","at","a","quarter","before","6",",","Sundays","excepted","."]}

Note that the list of NER labels will be automatically detected from the training
data.

2. Toponym disambiguation dataset
#################################

Train and test data examples are required for training a new entity
disambiguation (ED) model. They should be provided in a single tsv file, named
``linking_df_split.tsv``, one document per row, with the following required
columns:

* ``article_id``: article identifier, which consists of the number in the
  document file in the original dataset (for example, the ``article_id`` of
  ``1218_Poole1860.tsv`` is ``1218``).
* ``sentences``: list of dictionaries, each dictionary corresponding to a
  sentence in the article, with two fields: ``sentence_pos`` (the position
  of the sentence in the article) and ``sentence_text`` (the text of the
  sentence). For example:

  .. code-block:: json

    [
        {
            "sentence_pos": 1,
            "sentence_text": "DUKINFIELD.  "
        },
        {
            "sentence_pos": 2,
            "sentence_text": "Knutsford Sessions."
        },
        {
            "sentence_pos": 3,
            "sentence_text": "—The servant girl, Eliza Ann Byrom, who stole a quantity of clothes from the house where she lodged, in Dukiafield, was sentenced to two months’ imprisonment. "
        }
    ]

* ``annotations``: list of dictionaries containing the annotated place names.
  Each dictionary corresponds to a named entity mentioned in the text, with the
  following fields (at least): ``mention_pos`` (order of the mention in the article),
  ``mention`` (the actual mention), ``entity_type`` (the type of named entity),
  ``wkdt_qid`` (the Wikidata ID of the resolved entity), ``mention_start``
  (the character start position of the mention in the sentence), ``mention_end``
  (the character end position of the mention in the sentence), ``sent_pos``
  (the sentence index in which the mention is found).

  For example:

  .. code-block:: json

    [
        {
            "mention_pos": 0,
            "mention": "DUKINFIELD",
            "entity_type": "LOC",
            "wkdt_qid": "Q1976179",
            "mention_start": 0,
            "mention_end": 10,
            "sent_pos": 1
        },
        {
            "mention_pos": 1,
            "mention": "Knutsford",
            "entity_type": "LOC",
            "wkdt_qid": "Q1470791",
            "mention_start": 0,
            "mention_end": 9,
            "sent_pos": 2
        },
        {
            "mention_pos": 2,
            "mention": "Dukiafield",
            "entity_type": "LOC",
            "wkdt_qid": "Q1976179",
            "mention_start": 104,
            "mention_end": 114,
            "sent_pos": 3
        }
    ]

* ``place``: A string containing the place of publication of the newspaper to
  which the article belongs. For example, "Manchester" or "Ashton-under-Lyne".

* ``place_wqid``: A string with the Wikidata ID of the place of publication.
  For example, if ``place`` is London UK, then ``place_wqid`` should be ``Q84``.

Finally, the TSV contains a set of columns which can be used to indicate how
to split the dataset into training (``train``), development (``dev``), testing
(``test``), or documents to leave out (``left_out``). The Linker requires that
the user specifies which column should be used for training the ED model.
The code assumes the following columns:

* ``originalsplit``: The articles maintain the ``test`` set of the original
  dataset. Train is split into ``train`` (0.66) and ``dev`` (0.33).

* ``apply``: The articles are divided into ``train`` and ``dev``, with no articles
  left for testing. This split can be used to train the final entity disambiguation
  model, after the experiments.

* ``withouttest``: This split can be used for development. The articles in the
  test set of the original dataset are left out. The training set is split into
  ``train``, ``dev`` and ``test``.

`back to top <#top-resources>`_

Wikipedia- and Wikidata-based resources
---------------------------------------

T-Res requires a series of Wikipedia- and Wikidata-based resources:

* ``mentions_to_wikidata.json``
* ``mentions_to_wikidata_normalized.json``
* ``wikidata_to_mentions_normalized.json``
* ``wikidata_gazetteer.csv``
* ``entity2class.txt``

.. note::

    These files can be generated using the
    `wiki2gaz <https://github.com/Living-with-machines/wiki2gaz>`_ GitHub
    repository (**[coming soon]**). For more information on how they are built,
    refer to the ``wiki2gaz`` documentation.

T-Res assumes these files in the following default location:

::

    T-Res/
    └── resources/
        └── wikidata/
            ├── entity2class.txt
            ├── mentions_to_wikidata_normalized.json
            ├── mentions_to_wikidata.json
            ├── wikidata_gazetteer.csv
            └── wikidata_to_mentions_normalized.json

The sections below describe the contents of the files, as well as their
format, in case you prefer to provide your own resources (which should be
in the same format).

``mentions_to_wikidata.json``
#############################

A JSON file consisting of a python dictionary in which the key is a mention
of a place in Wikipedia (by means of an anchor text) and the value is an inner
dictionary, where the inner keys are the QIDs of all Wikidata entities that
can be referred to by the mention in question, and the inner values are the
absolute counts (i.e. the number of times such mention is used in Wikipedia
to refer to this particular entity).

You can load the dictionary, and access it, as follows:

::

    >>> import json
    >>> with open('mentions_to_wikidata.json', 'r') as f:
    ...     mentions_to_wikidata = json.load(f)
    ...
    >>> mentions_to_wikidata["Wiltshire"]


In the example, the value assigned to the key "Wiltshire" is:

.. code-block:: json

    {
        "Q23183": 4457,
        "Q55448990": 5,
        "Q8023421": 1
    }

In the example, we see that the mention "Wiltshire" is assigned a mapping
between key ``Q23183`` and value 4457. This means that, on Wikipedia,
"Wiltshire" appears 4457 times to refer to entity `Q23183
<https://www.wikidata.org/wiki/Q23183>`_ (through the mapping between
Wikidata entity ``Q23183`` and its `corresponding Wikipedia page
<https://en.wikipedia.org/wiki/Wiltshire>`_).

``mentions_to_wikidata_normalized.json``
########################################

A JSON file containing the normalised version of the ``mentions_to_wikidata.json``
dictionary. For example, the value of the mention "Wiltshire" is now:

.. code-block:: json

    {
        "Q23183": 0.9767696690773614,
        "Q55448990": 1.0,
        "Q8023421": 0.03125
    }

Note that these scores do not add up to one, as they are normalised by entity,
not by mention. They are a measure of how likely an entity is to be referred to
by a mention. In the example, we see that entity ``Q55448990`` is always referred
to as ``Wiltshire``.

``wikidata_to_mentions_normalized.json``
########################################

A JSON file consisting of a python dictionary in which the key is a Wikidata QID
and the value is an inner dictionary, in which the inner keys are the mentions
used in Wikipedia to refer to such Wikidata entity, and the values are their
relative frequencies.

You can load the dictionary, and access it, as follows:

::

    >>> import json
    >>> with open('wikidata_to_mentions_normalized.json', 'r') as f:
    ...     wikidata_to_mentions_normalized = json.load(f)
    ...
    >>> wikidata_to_mentions_normalized["Q23183"]

In this example, the value of entity `Q23183 <https://www.wikidata.org/wiki/Q23183>`_ is:

.. code-block:: json

    {
        "Wiltshire, England": 0.005478851632697786,
        "Wilton": 0.00021915406530791147,
        "Wiltshire": 0.9767696690773614,
        "College": 0.00021915406530791147,
        "Wiltshire Council": 0.0015340784571553803,
        "West Wiltshire": 0.00021915406530791147,
        "North Wiltshire": 0.00021915406530791147,
        "Wilts": 0.0015340784571553803,
        "County of Wilts": 0.0026298487836949377,
        "County of Wiltshire": 0.010081087004163929,
        "Wilts.": 0.00021915406530791147,
        "Wiltshire county": 0.00021915406530791147,
        "Wiltshire, United Kingdom": 0.00021915406530791147,
        "Wiltshire plains": 0.00021915406530791147,
        "Wiltshire England": 0.00021915406530791147
    }

In this example, we can see that entity ``Q23183`` is referred to as "Wiltshire,
England" in Wikipedia 0.5% of the times and as "Wiltshire" 97.7% of the times.
These values add up to one.

``wikidata_gazetteer.csv``
##########################

A csv file consisting of (at least) the following four columns:

* a Wikidata ID (QID) of a location,
* its English label,
* its latitude, and
* its longitude.

You can load the csv, and show the first five rows, as follows:

::

    >>> import pandas as pd
    >>> df = pd.read_csv("wikidata_gazetteer.csv")
    >>> df. head()
      wikidata_id                     english_label  latitude  longitude
    0    Q5059107                        Centennial  40.01140  -87.24330
    1    Q5059144                Centennial Grounds  39.99270  -75.19380
    2    Q5059153            Centennial High School  40.06170  -83.05780
    3    Q5059162            Centennial High School  38.30440 -104.63800
    4    Q5059178  Centennial Memorial Samsung Hall  37.58949  127.03434

Each row corresponds to a Wikidata geographic entity (i.e. a Wikidata entity
with coordinates).

``entity2class.txt``
####################

A python dictionary in which each entity in Wikidata is mapped to its most
common Wikidata class.

You can load the dictionary, and access it, as follows:

::

    >>> with open('entity2class.txt', 'r') as f:
    ...     entity2class = json.load(f)
    ...
    >>> entity2class["Q23183"]
    'Q180673'
    >>> entity2class["Q84"]
    'Q515'

For example, Wiltshire (`Q23183 <https://www.wikidata.org/wiki/Q23183>`_) is
mapped to `Q180673 <https://www.wikidata.org/wiki/Q180673>`_, i.e. "cerimonial
county  of England", whereas London (`Q84 <https://www.wikidata.org/wiki/Q84>`_)
is mapped to `Q515 <https://www.wikidata.org/wiki/Q515>`_, i.e. "city".

`back to top <#top-resources>`_

Entity and word embeddings
--------------------------

In order to perform toponym linking and resolution using the REL-based approaches,
T-Res requires a database of word2vec and wiki2vec embeddings. Note that you will
not need this if you use the ``mostpopular`` disambiguation approach.

By default, T-Res expects a database file called ``embeddings_database.db`` with,
at least, one table (``entity_embeddings``) with at least the following columns:

* ``word``: Either a lower-cased token (i.e. a word on Wikipedia) or a Wikidata QID
  preceded by ``ENTITY/``. The database should also contain the following two wildcard
  tokens: ``#ENTITY/UNK#`` and ``#WORD/UNK#``.
* ``emb``: The corresponding word or entity embedding.

Generate the embeddings database
################################

In our experiments, we derived the embeddings database from REL's shared resources.

.. note::

    We are working towards improving this step in the pipeline. Meanwhile, to generate
    the ``embeddings_database.db``, please follow these steps:

    #. Make sure you have ``wikidata_gazetteer.csv`` in ``./resources/wikidata/`` (see
    `above <#wikipedia-and-wikidata-based-resources>`_).
    #. Generate a Wikipedia-to-Wikidata index, following `this instructions
    <https://github.com/jcklie/wikimapper#create-your-own-index>`_, save it as: ``./resources/wikipedia/index_enwiki-latest.db``.
    #. Run `this script <https://github.com/Living-with-machines/wiki2gaz/blob/main/download_and_merge_embeddings_databases.py>`_
    to create the embeddings database.

You can load the file, and access a token embedding, as follows:

::

    >>> import array
    >>> from array import array
    >>> with sqlite3.connect("embeddings_database.db") as conn:
    ...     cursor = conn.cursor()
    ...     result = cursor.execute("SELECT emb FROM entity_embeddings WHERE word='lerwick'").fetchone()
    ...     result = result if result is None else array("f", result[0]).tolist()
    ...
    >>> result
    [-0.3257000148296356, -0.00989999994635582, -0.13420000672340393, ...]

You can load the file, and access an entity embedding, as follows:

::

    >>> import array
    >>> from array import array
    >>> with sqlite3.connect("embeddings_database.db") as conn:
    ...     cursor = conn.cursor()
    ...     result = cursor.execute("SELECT emb FROM entity_embeddings WHERE word='ENTITY/Q84'").fetchone()
    ...     result = result if result is None else array("f", result[0]).tolist()
    ...
    >>> result
    [-0.014700000174343586, 0.007899999618530273, -0.1808999925851822, ...]

T-Res expects the ``embeddings_database.db`` file to be stored as follows:

::

    T-Res/
    └── resources/
        └── rel_db/
            └── embeddings_database.db

`back to top <#top-resources>`_

DeezyMatch training set
---------------------------------------

In order to train a DeezyMatch model, a training set consisting of positive and
negative string pairs is required. We provide a dataset of positive and negative
OCR variations, which can be used to train a DeezyMatch model, which can then be
used to perform fuzzy string matching to find candidates for entity linking.

.. note::

    The DeezyMatch training set can be downloaded from the `British Library research
    repository <https://bl.iro.bl.uk/concern/datasets/12208b77-74d6-44b5-88f9-df04db881d63>`_.

T-Res assumes by default the DeezyMatch training set to be named ``w2v_ocr_pairs.txt``
and to be in the following location:

::

    T-Res/
    └── resources/
        └── deezymatch/
            └── data/
                └── w2v_ocr_pairs.txt

Optionally, T-Res also provides the option to generate a DeezyMatch training set
from word2vec embeddings trained on digitised texts. Continue reading the sections
below for more information about both types of resources.

1. DeezyMatch training set
##########################

T-Res can directly load the string pairs dataset required to train a new DeezyMatch
model. By default, the code assumes the dataset to be called ``w2v_ocr_pairs.txt``.
The dataset consists of three columns: ``word1``, ``word2``, and a boolean describing
whether ``word2`` is an OCR variation of ``word1``. For example:

  .. code-block::

    could   might   FALSE
    could   wished  FALSE
    could   hardly  FALSE
    could   didnot  FALSE
    could   never   FALSE
    could   reusing FALSE
    could   could   TRUE
    could   coeld   TRUE
    could   could   TRUE
    could   conld   TRUE
    could   could   TRUE
    could   couid   TRUE

This dataset has been automatically generated from word2vec embeddings trained on
digitised historical news texts (i.e. with OCR noise), and has been expanded with
toponym alternate names extracted from Wikipedia.

The dataset we provide consists of 1,085,514 string pairs.

2. Word2Vec embeddings trained on noisy data
############################################

The 19thC word2vec embeddings **are not needed** if you already have the DeezyMatch
training set ``w2v_ocr_pairs.txt`` (described in the `section above
<#deezymatch-training-set>`_).

To create a new DeezyMatch training set using T-Res, you need to provide Word2Vec
models that have been trained on digitised historical news texts. In our experiments,
we used the embeddings trained on a 4.2-billion-word corpus of 19th-century British
newspapers using Word2Vec (you can download them from `Zenodo
<https://doi.org/10.5281/zenodo.7887305>`_), but you can also do this with your
own word2vec embeddings. The embeddings are divided into periods of ten years each.
By default, T-Res assumes that the word2vec models are stored in
``./resources/models/w2v/``, in directories named ``w2v_xxxxs_news/``, where
``xxxx`` corresponds to the decade (e.g. 1800 or 1810) of the models.

See the expected directory structure below:

::

    T-Res/
    └── resources/
        └── models/
            └── w2v/
                ├── w2v_1800_news/
                │     ├── w2v.model
                │     ├── w2v.model.syn1neg.npy
                │     └── w2v.model.wv.vectors.npy
                ├── w2v_1810_news/
                │     ├── w2v.model
                │     ├── w2v.model.syn1neg.npy
                │     └── w2v.model.wv.vectors.npy
                └── .../

Summary of resources and directory structure
--------------------------------------------

In the code and our tutorials, we assume the following directory structure
for the mentioned resources that are required in order to run the pipeline.

::

    T-Res/
    ├── app/
    ├── evaluation/
    ├── examples/
    ├── experiments/
    │   └── outputs/
    │       └── data/
    │           └── lwm/
    │               ├── linking_df_split.tsv [*]
    │               ├── ner_fine_dev.json [*+]
    │               └── ner_fine_train.json [*+]
    ├── geoparser/
    ├── resources/
    │   ├── deezymatch/
    │   │   └── data/
    │   │       └── w2v_ocr_pairs.txt
    │   ├── models/
    │   ├── news_datasets/
    │   ├── rel_db/
    │   │   └── embeddings_database.db [*+]
    │   └── wikidata/
    │       ├── entity2class.txt [*]
    │       ├── mentions_to_wikidata_normalized.json [*]
    │       ├── mentions_to_wikidata.json [*]
    │       ├── wikidta_gazetteer.csv [*]
    │       └── wikidata_to_mentions_normalized.json [*]
    ├── tests/
    └── utils/

Note that an asterisk (``*``) next to the resource means that the path can
be changed when instantiating the T-Res objects, and a plus sign (``+``) if
the name of the file can be changed in the instantiation.

`back to top <#top-resources>`_
