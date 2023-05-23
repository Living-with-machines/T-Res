.. _top:

=================
The complete tour
=================

The T-Res has three main classes: the Recogniser class (which performs named
entity recognition---NER), the Ranker class (which performs candidate
selection and ranking for the named entities identified by the Recogniser),
and the Linker class (which selectes the most likely candidate from those
provided by the Ranker). An additional class, the Pipeline, wraps these three
components into one, therefore making end-to-end T-Res easier to use.

The Recogniser
--------------

The Recogniser allows (1) loading an existing model (either directly
downloading a model from the HuggingFace hub or loading a locally stored NER
model) and (2) training a new model and loading it if it is already trained.

The following notebooks show examples using the Recogniser:

* ``./examples/load_use_ner_model.ipynb``
* ``./examples/train_use_ner_model.ipynb``

1. Instantiate the Recogniser
#############################

To load an already trained model (both from HuggingFace or a local model), you
can just instantiate the recogniser as follows:

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="path-to-model",
        load_from_hub=True,
    )

For example, to load the `dslim/bert-base-NER <https://huggingface.co/dslim/bert-base-NER>`_
NER model from the HuggingFace hub:

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="dslim/bert-base-NER",
        load_from_hub=True,
    )

To load a NER model that is stored locally (for example, let's suppose we have
a NER model in this relative location
``../resources/models/blb_lwm-ner-fine``), you can also load it in the same
way (notice that ``load_from_hub`` should still be ``True``, probably a better
name would be ``load_from_path``):

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="resources/models/blb_lwm-ner-fine",
        load_from_hub=True,
    )

Alternatively, you can use the Recogniser to train a new model (and load it,
once it's trained). To instantiate the Recogniser for training a new model and
loading it once it's trained, you can do it as in the example (see the
description of each parameter below):

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",
        pipe=None,
        base_model="khosseini/bert_1760_1900",
        model_path="resources/models/",
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,
        do_test=False,
        load_from_hub=False,
    )

Description of the arguments:

* ``load_from_hub`` set to ``False`` indicates we're not using an off-the-shelf
  model. It will prepare the Recogniser to train a new model, unless the model
  already exists or if ``overwrite_training`` is set to ``True``. If
  ``overwrite_training`` is set to ``False`` and ``load_from_hub`` is set to
  ``False``, the Recogniser will be prepared to first try to load the model
  and—if it does not exist—will train it. If ``overwrite_training`` is set to
  ``True`` and ``load_from_hub`` is set to ``False``, the Recogniser will be
  ready to directly try to train a model.
* ``base_model`` is the path to the model that will be used as base to
  train our NER model. This can be the path to a HuggingFace model (we are
  using `khosseini/bert_1760_1900 <https://huggingface.co/khosseini/bert_1760_1900>`_,
  a BERT model trained on 19th Century texts) or the path to a model stored
  locally.
* ``train_dataset`` and ``test_dataset`` contain the path to the train
  and test data sets necessary for training the NER model. The paths point to a
  json file (one for training, one for testing), in which each line is a
  dictionary corresponding to a sentence. Each sentence-dictionary has three
  key-value pairs: ``id`` is an ID of the sentence (a string), ``tokens`` is
  the list of tokens into which the sentence has been split, and ``ner_tags``
  is the list of annotations per token (in BIO format). The length of
  ``tokens`` and ``ner_tags`` should always be the same. This is an example of
  two lines from either the training or test json files:
  
  .. code-block:: json

    {
        "id":"3896239_29",
        "ner_tags": [
            "O",
            "B-STREET",
            "I-STREET",
            "O",
            "O",
            "O",
            "B-BUILDING",
            "I-BUILDING",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
        ],
        "tokens": [
            ",",
            "Old",
            "Millgate",
            ",",
            "to",
            "the",
            "Collegiate",
            "Church",
            ",",
            "where",
            "they",
            "arrived",
            "a",
            "little",
            "after",
            "ten",
            "oclock",
            "."
        ]
    }

    {
        "id":"8262498_11",
        "ner_tags": [
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-LOC",
            "O",
            "B-LOC",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
        ],
        "tokens": [
            "On",
            "the",
            "\u2018",
            "JSth",
            "November",
            "the",
            "ship",
            "Santo",
            "Christo",
            ",",
            "from",
            "Monteveido",
            "to",
            "Cadiz",
            ",",
            "with",
            "hides",
            "and",
            "copper",
            "."
        ]
    }

* ``model_path`` is the path where the Recogniser should store the model,
  and ``model`` is the name of the model. The ``pipe`` argument can be
  left empty: that's where we will store the NER pipeline, once the model is
  trained and loaded.
* The training arguments can be modified in ``training_args``: you can
  change the learning rate, batch size, number of training epochs, and weight
  decay.
* Finally, ``do_test`` allows you to train a mock model and then load it
  (the suffix `_test` will be added to the model name). As mentioned above,
  ``overwrite_training`` forces retraining a model, even if a model with
  the same name and characteristics already exists.

This instantiation prepares a new model
(``resources/models/blb_lwm-ner-fine.model``) to be trained, unless the model
already exists (``overwrite_training`` is ``False``), in which case it will
just load it.

1. Train the NER model
######################

After having instantiated the Recogniser, to train the model, run:

.. code-block:: python

    myner.train()

Note that if ``load_to_hub`` is set to ``True`` or the model already exists
(and ``overwrite_training`` is set to ``False``), the training will be skipped,
even if you call the ``train()`` method.

3. Create a NER pipeline
########################

In order to create a NER pipeline, run:

.. code-block:: python

    myner.pipe = myner.create_pipeline()

This loads the NER model into a
`Transformers pipeline <https://huggingface.co/docs/transformers/main_classes/pipelines>`_,
to use it for inference.

1. Use the NER pipeline
#######################

In order to run the NER pipeline on a sentence, use the ``ner_predict()``
method of the Recogniser as follows:

.. code-block:: python

    sentence = "I ought to be at Dewsbury Moor."
    predictions = myner.ner_predict(sentence)
    print(predictions)

This returns all words in the sentence, with their detected entity type,
confidence score, and start and end characters in the sentence, as follows:

.. code-block:: json

    [
        {
            "entity": "O",
            "score": 0.9997773766517639,
            "word": "I",
            "start": 0,
            "end": 1
        },
        {
            "entity": "O", "score": 0.9997766613960266,
            "word": "ought",
            "start": 2,
            "end": 7
        },
        {
            "entity": "O",
            "score": 0.9997838139533997,
            "word": "to",
            "start": 8,
            "end": 10
        },
        {
            "entity": "O",
            "score": 0.9997853636741638,
            "word": "be",
            "start": 11,
            "end": 13
        },
        {
            "entity": "O",
            "score": 0.9997740387916565,
            "word": "at",
            "start": 14,
            "end": 16
        },
        {
            "entity": "B-LOC",
            "score": 0.9603037536144257,
            "word": "Dewsbury",
            "start": 17,
            "end": 25
        },
        {
            "entity": "I-LOC",
            "score": 0.9753544330596924,
            "word": "Moor",
            "start": 26,
            "end": 30
        },
        {
            "entity": "O",
            "score": 0.9997835755348206,
            "word": ".",
            "start": 30,
            "end": 3
        1}
    ]


To return the named entities in a user-friendlier format, run:

.. code-block:: python

    from utils import ner

    # Process predictions:
    processed_predictions = [
        [
            x["word"], x["entity"], "O", x["start"], x["end"], x["score"]
        ]
        for x in predictions
    ]

    # Aggretate mentions:
    mentions = ner.aggregate_mentions(processed_predictions, "pred")

This returns only the named entities, aggregating multiple tokens together:

.. code-block:: json

    [
        {
            "mention": "Dewsbury Moor",
            "start_offset": 5,
            "end_offset": 6,
            "start_char": 17,
            "end_char": 30,
            "ner_score": 0.968,
            "ner_label": "LOC",
            "entity_link": "O"
        }
    ]

`back to top <#top>`_

The Ranker
----------

The Ranker takes the named entities detected by the Recogniser as input. Given
a knowledge base, it ranks the entities according to their string similarity to
the named entity, and selects a subset of candidates that will be passed on to
the next component, the Linker, to do the disambiguation and select the most
likely entity.

In order to use the Ranker and the Linker, we need a knowledge base, a
gazetteer. T-Res uses a gazetteer which combines data from Wikipedia and
Wikidata. The steps to create it are described in the
`wiki2gaz <https://github.com/Living-with-machines/wiki2gaz>`_ GitHub
repository.

The following files are needed to run the Ranker:

* ``wikidata_to_mentions_normalized.json``: dictionary of Wikidata entities
  (by their QID) mapped to the mentions used in Wikipedia to refer to them
  (obtained through Wikipedia anchor texts) and the normalised score. For
  example, the value of entity `Q23183 <https://www.wikidata.org/wiki/Q23183>`_
  is the following:

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

* ``mentions_to_wikidata_normalized.json``: the reverse dictionary to the one
* above, it maps a mention to all the Wikidata entities that are referred to
* by this mention in Wikipedia. For example, the value of `"Wiltshire"` is:

  .. code-block:: json
  
      {
          "Q23183": 0.9767696690773614,
          "Q55448990": 1.0,
          "Q8023421": 0.03125
      }

  These scores don't add up to one, as they are normalised per entity,
  therefore indicating how often an entity is referred to by this mention. For
  example, ``Q55448990`` is always referred to as ``Wiltshire``.

We provide four different strategies for selecting candidates:

* ``perfectmatch`` retrieves candidates from the knowledge base if one of
  their alternate names is identical to the detected named entity. For example,
  given the mention "Wiltshire", the following Wikidata entities will be
  retrieved: `Q23183 <https://www.wikidata.org/wiki/Q23183>`_,
  `Q55448990 <https://www.wikidata.org/wiki/Q55448990>`_, and
  `Q8023421 <https://www.wikidata.org/wiki/Q8023421>`_, because all these
  entities are referred to as "Wiltshire" in Wikipedia anchor texts.
* ``partialmatch`` retrieves candidates from the knowledge base if there is
  a (partial) match between the query and the candidate names, based on string
  overlap. Therefore, the mention "Ashton-under" returns candidates for
  "Ashton-under-Lyne".
* ``levenshtein`` retrieves candidates from the knowledge base if there is
  a fuzzy match between the query and the candidate names, based on levenshtein
  distance. Therefore, if the mention "Wiltshrre" would still return the
  candidates for "Wiltshire". This method is often quite accurate when it comes
  to OCR variations, but it is very slow.
* ``deezymatch`` retrieves candidates from the knowledge base if there is a
  fuzzy match between the query and the candidate names, based on
  `DeezyMatch <https://github.com/Living-with-machines/DeezyMatch>`_ embeddings.
  Significantly more complex than the other methods to set up from scratch, but
  the fastest approach.

1. Instantiate the Ranker
#########################

To use the Ranker for exact matching (``perfectmatch``) or fuzzy string
matching based either on overlap or Levenshtein distance (``partialmatch`` and
``levenshtein`` respectively), instantiate it as follows, changing the
``method`` argument accordingly:

.. code-block:: python

    from geoparser import ranking

    myranker = ranking.Ranker(
        method="perfectmatch", # or "partialmatch" or "levenshtein"
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

Note that ``resources_path`` should contain the path to the directory
where the resources are stored, namely ``wikidata_to_mentions_normalized.json``
and ``mentions_to_wikidata.json``. The ``mentions_to_wikidata`` and
``wikidata_to_mentions`` dictionaries should be left empty, as they will be
populated when the Ranker loads the resources.

DeezyMatch instantiation is trickier, as it requires training a model that,
ideally, should capture the types of string variations that can be found in
your data (such as OCR errrors). Using the Ranker, you can:

#. Train a DeezyMatch model from scratch, including generating a string pairs
   dataset.
#. Train a DeezyMatch model, given an existing string pairs dataset.
#. Use an existing DeezyMatch model.

See below each of them in detail.

1. Use an existing DeezyMatch model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use an existing DeezyMatch model, you wil need to have the following
``resources`` file structure (where ``wkdtalts`` is the name given to the set
of all Wikidata alternate names and ``w2v_ocr`` is the name given to the
DeezyMatch model).

::

    toponym-resolution/
    ├── ...
    ├── resources/
    │   ├── deezymatch/
    │   │   ├── combined/
    │   │   │   └── wkdtalts_w2v_ocr/
    │   │   │       ├── bwd.pt
    │   │   │       ├── bwd_id.pt
    │   │   │       ├── bwd_items.npy
    │   │   │       ├── fwd.pt
    │   │   │       ├── fwd_id.pt
    │   │   │       ├── fwd_items.npy
    │   │   │       └── input_dfm.yaml
    │   │   └── models/
    │   │       └── w2v_ocr/
    │   │           ├── input_dfm.yaml
    │   │           ├── w2v_ocr.model
    │   │           ├── w2v_ocr.model_state_dict
    │   │           └── w2v_ocr.vocab
    │   ├── models/
    │   ├── news_datasets/
    │   ├── wikidata/
    │   │   ├── mentions_to_wikidata.json
    │   │   └── wikidata_to_mentions.json
    │   └── wikipedia/
    └── ...

The Ranker can then be instantiated as follows:

.. code-block:: python

    from pathlib import Path
    from geoparser import ranking

    myranker = ranking.Ranker(
        # Generic Ranker parameters:
        method="deezymatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        # Parameters to create the string pair dataset:
        strvar_parameters={
            "overwrite_dataset": False,
        },
        # Parameters to train, load and use a DeezyMatch model:
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("resources/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 25,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": True,
            "do_test": True,
        },
    )

Description of the arguments (to learn more, refer to the
`DeezyMatch readme <https://github.com/Living-with-machines/DeezyMatch/blob/master/README.md>`_:

* ``strvar_parameters`` contains the parameters needed to generate the
  DeezyMatch training set. In this scenario, the DeezyMatch model is already
  trained and there is therefore no need to generate the training set.
* ``deezy_parameters`` contains the set of parameters to train or load a
  DeezyMatch model:

  * ``dm_path``: The path to the folder where the DeezyMatch model and
    data will be stored.
  * ``dm_cands``: The name given to the set of alternate names from which
    DeezyMatch will try to find a match for a given mention.
  * ``dm_model``: Name of the DeezyMatch model to train or load.
  * ``ranking_metric`` Metric used to TODO

You can download these resources from:

* ``resources/deezymatch/combined/wkdtalts_w2v_ocr/``: **[TODO]**
* ``resources/deezymatch/models/w2v_ocr/``: **[TODO]**
* ``wikidata/mentions_to_wikidata.json``: **[TODO]**
* ``wikidata/wikidata_to_mentions.json``: **[TODO]**

1. Train a DeezyMatch model from scratch, including generating a string pairs dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO

2. Train a DeezyMatch model, given an existing string pairs dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

.. code-block:: python

    myranker = ranking.Ranker(
        method="perfectmatch",
        resources_path="../resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        # Parameters to create the string pair dataset:
        strvar_parameters={
            "overwrite_dataset": False,
        },
        deezy_parameters={
            "dm_path": str(Path("../resources/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 25,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

2. Load the resources
#####################

The following line loads the resources (i.e. the ``mentions-to-wikidata`` and
``wikidata_to_mentions`` dictionaries) required to perform candidate selection
and ranking, regardless of the Ranker method.

.. code-block:: python

    myranker.mentions_to_wikidata = myranker.load_resources()

3. Train a DeezyMatch model
###########################

The following line will train a DeezyMatch model, given the arguments specified
when instantiating the Ranker.

.. code-block:: python

    myranker.train()

Note that if the model already exists and overwrite_training is set to
``False``, the training will be skipped, even if you call the train() method.
The training will also be skipped if the Ranker is not instantiated for
DeezyMatch.

4. Retrieve candidates for a given mention
##########################################

.. code-block:: python

    toponym = "Manchefter"
    print(myranker.find_candidates([{"mention": toponym}])[0][toponym])

`back to top <#top>`_

The Linker
----------

TODO

`back to top <#top>`_

The Pipeline
------------

TODO

`back to top <#top>`_
