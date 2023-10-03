.. _top-tour:

=================
The complete tour
=================

The T-Res has three main classes: the **Recogniser** class (which performs
toponym recognition, which is a named entity recognition task), the **Ranker**
class (which performs candidate selection and ranking for the named entities
identified by the Recogniser), and the **Linker** class (which selects the
most likely candidate from those provided by the Ranker).

An additional class, the **Pipeline**, wraps these three components into one,
therefore making it easier for the user to perform end-to-end entity linking.

In the following sections, we provide a complete tour: including an in-depth
description of each of the four classes. We recommend that you start with the
Pipeline, which wraps the three other classes, and refer to the description of
each of the other classes to learn more about them. We also recommend that
you first try to run T-Res using the default pipeline, and then change it
accordingly to your needs.

.. warning::

    Note that, before being able to run the pipeline, you will need to make sure
    you have all the required resources. Refer to the ":doc:`resources`" page
    in the documentation.

The Pipeline
------------

The Pipeline wraps the Recogniser, the Ranker and the Linker into one object,
to make it easier to use T-Res for end-to-end entity linking.

1. Instantiate the Pipeline
###########################

By default, the Pipeline instantiates:

* a Recogniser (from a HuggingFace model),
* a Ranker (using the `perfectmatch` approach), and
* a Linker (using the `mostpopular` approach).

To instantiate the default T-Res pipeline, do:

.. code-block:: python

    from geoparser import pipeline

    geoparser = pipeline.Pipeline()

You can also instantiate a pipeline using a customised Recogniser, Ranker and
Linker. To see the different options, refer to the sections on instantiating
each of them: :ref:`Recogniser <The Recogniser>`, :ref:`Ranker <The Ranker>`
and :ref:`Linker <The Linker>`.

In order to instantiate a pipeline using a customised Recogniser, Ranker and
Linker, just instantiate them beforehand, and then pass them as arguments to
the Pipeline, as follows:

.. code-block:: python

    from geoparser import pipeline, recogniser, ranking, linking

    myner = recogniser.Recogniser(...)
    myranker = ranking.Ranker(...)
    mylinker = linking.Linker(...)

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

.. warning::

    Note that the default Pipeline expects to be run from the ``experiments/``
    or the ``examples`` folder (or any other folder in the same level). The
    Pipeline will look for the resources at ``../resources/``. Make sure all
    the required resources are in the right locations.

.. note::

    If a model needs to be trained, the Pipeline itself will take care of it.
    Therefore, you should expect that the first time the Pipeline is used (or
    if you change certain input parameters) T-Res will take long to be ready
    to be used for prediction, as it will train the models if the approaches
    require so.

2. Use the Pipeline
###################

Once instantiated (and once all the models have been trained or loaded, if needed),
the Pipeline can be used to perform end-to-end toponym recognition and linking
(given an input text) or to perform each of the three steps individually: (1)
toponym recognition given an input text, (2) candidate selection given a toponym
or list of toponyms, and (3) toponym disambiguation given the output from the
first two steps.

End-to-end pipeline
^^^^^^^^^^^^^^^^^^^

The Pipeline can be used to perform end-to-end toponym recognition and linking
given an input text, using the ``run_sentence()`` method (which applies the
T-Res pipeline to the input text) or the ``run_text()`` method (which takes
care of splitting a text into sentences, before running ``run_sentence()``
on each sentence).

See this with examples:

.. code-block:: python

    output = geoparser.run_text("Inspector Liddle said: I am an inspector of police, living in the city of Durham.")

.. code-block:: python

    output = geoparser.run_sentence("Inspector Liddle said: I am an inspector of police, living in the city of Durham.")

In both cases, the following parameters are optional **[TODO: link to docstrings]**:

* ``place``: The place of publication associated with the text document as a
  human-legible string (e.g. ``"London"``). This defaults to ``""``.
* ``place_wqid``: The Wikidata ID of the place of publication provided in
  ``place`` (e.g. ``"Q84"``). This defaults to ``""``.

For example:

.. code-block:: python

    output = geoparser.run_text("Inspector Liddle said: I am an inspector of police, living in the city of Durham.",
        place="Alston, Cumbria, England",
        place_wqid="Q2560190"
        )

The output of this example is the following:

.. code-block:: json

    [{"mention": "Durham",
      "ner_score": 0.999,
      "pos": 74,
      "sent_idx": 0,
      "end_pos": 80,
      "tag": "LOC",
      "sentence": "Inspector Liddle said: I am an inspector of police, living in the city of Durham.",
      "prediction": "Q179815",
      "ed_score": 0.039,
      "cross_cand_score": {
        "Q179815": 0.396,
        "Q23082": 0.327,
        "Q49229": 0.141,
        "Q5316459": 0.049,
        "Q458393": 0.045,
        "Q17003433": 0.042,
        "Q1075483": 0.0
      },
      "string_match_score": {"Durham": [1.0, ["Q1137286", "Q5316477", "Q752266", "..."]]},
      "prior_cand_score": {
        "Q179815": 0.881,
        "Q49229": 0.522,
        "Q5316459": 0.457,
        "Q17003433": 0.455,
        "Q23082": 0.313,
        "Q458393": 0.295,
        "Q1075483": 0.293
      },
      "latlon": [54.783333, -1.566667],
      "wkdt_class": "Q515"}]

Step-by-step pipeline
^^^^^^^^^^^^^^^^^^^^^

See how to perform toponym recognition with the Pipeline, with an example:

.. code-block:: python

    output = geoparser.run_text_recognition(
        "Inspector Liddle said: I am an inspector of police, living in the city of Durham.",
        place="Alston, Cumbria, England",
        place_wqid="Q2560190"
    )

This is the output for this example:

.. code-block:: json

    [{"mention": "Durham",
      "context": ["", ""],
      "candidates": [],
      "gold": ["NONE"],
      "ner_score": 0.999,
      "pos": 74,
      "sent_idx": 0,
      "end_pos": 80,
      "ngram": "Durham",
      "conf_md": 0.999,
      "tag": "LOC",
      "sentence": "Inspector Liddle said: I am an inspector of police, living in the city of Durham.",
      "place": "Alston, Cumbria, England",
      "place_wqid": "Q2560190"
      }]

See how to perform candidate selection given the output from the previous
step, with an example:

.. code-block:: python

    ner_output = [
        {
            'mention': 'Durham',
            'context': ['', ''],
            'candidates': [],
            'gold': ['NONE'],
            'ner_score': 0.999,
            'pos': 74,
            'sent_idx': 0,
            'end_pos': 80,
            'ngram': 'Durham',
            'conf_md': 0.999,
            'tag': 'LOC',
            'sentence': 'Inspector Liddle said: I am an inspector of police, living in the city of Durham.',
            'place': 'Alston, Cumbria, England',
            'place_wqid': 'Q2560190'
        }
    ]

    cands = geoparser.run_candidate_selection(ner_output)

This is the output for this example:

.. code-block:: json

    {"Durham":
        {"Durham":
            {
              "Score": 1.0,
              "Candidates":
                {
                    "Q1137286": 0.022222222222222223,
                    "Q5316477": 0.3157894736842105,
                    "Q752266": 0.013513513513513514,
                    "Q23082": 0.06484443152079093,
                }
            }
        }
    }

Finally, see how to perform toponym disambiguation given the output from
the two previous steps, with an example:

.. code-block:: python

    ner_output = [
        {
            'mention': 'Durham',
            'context': ['', ''],
            'candidates': [],
            'gold': ['NONE'],
            'ner_score': 0.999,
            'pos': 74,
            'sent_idx': 0,
            'end_pos': 80,
            'ngram': 'Durham',
            'conf_md': 0.999,
            'tag': 'LOC',
            'sentence': 'Inspector Liddle said: I am an inspector of police, living in the city of Durham.',
            'place': 'Alston, Cumbria, England',
            'place_wqid': 'Q2560190'
        }
    ]

    cands = {'Durham': {'Durham': {'Score': 1.0,
                                   'Candidates': {
                                      'Q1137286': 0.022222222222222223,
                                      'Q5316477': 0.3157894736842105,
                                      'Q752266': 0.013513513513513514,
                                      'Q23082': 0.06484443152079093}}}}

    disamb_output = geoparser.run_disambiguation(ner_output, cands)

This will return the exact same output as running the pipeline end-to-end.

Description of the output
^^^^^^^^^^^^^^^^^^^^^^^^^

The output of running the pipeline (both using the end-to-end method or
in a step-wise manner, regardless of the methods used for each of the
three components), will have the following format:

.. code-block:: json

    [{"mention": "Durham",
      "ner_score": 0.999,
      "pos": 74,
      "sent_idx": 0,
      "end_pos": 80,
      "tag": "LOC",
      "sentence": "Inspector Liddle said: I am an inspector of police, living in the city of Durham.",
      "prediction": "Q179815",
      "ed_score": 0.039,
      "cross_cand_score": {
        "Q179815": 0.396,
        "Q23082": 0.327,
        "Q49229": 0.141,
        "Q5316459": 0.049,
        "Q458393": 0.045,
        "Q17003433": 0.042,
        "Q1075483": 0.0
      },
      "string_match_score": {"Durham": [1.0, ["Q1137286", "Q5316477", "Q752266", "..."]]},
      "prior_cand_score": {
        "Q179815": 0.881,
        "Q49229": 0.522,
        "Q5316459": 0.457,
        "Q17003433": 0.455,
        "Q23082": 0.313,
        "Q458393": 0.295,
        "Q1075483": 0.293
      },
      "latlon": [54.783333, -1.566667],
      "wkdt_class": "Q515"}]

Description of the fields:

* ``mention``: The mention text.
* ``ner_score``: The NER confidence score of the mention.
* ``pos``: The starting position of the mention in the sentence.
* ``sent_idx``: The index of the sentence.
* ``end_pos``: The ending position of the mention in the sentence.
* ``tag``: The NER label of the mention.
* ``sentence``: The input sentence.
* ``prediction``: The predicted entity linking result (a Wikidata QID or NIL).
* ``ed_score``: The entity disambiguation score.
* ``string_match_score``: A dictionary of candidate entities and their string
  matching confidence scores.
* ``prior_cand_score``: A dictionary of candidate entities and their prior
  confidence scores.
* ``cross_cand_score``: A dictionary of candidate entities and their
  cross-candidate confidence scores.
* ``latlon``: The latitude and longitude coordinates of the predicted entity.
* ``wkdt_class``: The Wikidata class of the predicted entity.

Pipeline recommendations
^^^^^^^^^^^^^^^^^^^^^^^^

* To get started with T-Res, we recommend to start using the default pipeline,
  as its significantly less complex than the better performing approaches.
* The default pipeline may not be a bad option if you are planning to perform
  toponym recognition on modern global clean data. However, take into account
  that it uses context-agnostic approaches, which often perform quantitavively
  quite well just because of the higher probability of the most common sense
  to appear in texts.
* Running T-Res with DeezyMatch for candidate selection and ``reldisamb`` for
  entity disambiguation takes considerably longer than using the default
  pipeline. If you want to run T-Res on a few sentences, you can use the
  end-to-end ``run_text()`` or ``run_sentence()`` methods. If, however, you
  have a large number of texts on which to run T-Res, then we recommend that
  you use the step-wise approach. If done efficiently, this can save a lot
  of time. Using this approach, you should:

  #. Perform toponym recognition on all the texts,
  #. Obtain the set of all unique toponyms identified in the full dataset,
     and perform candidate selection on the unique set of toponyms,
  #. Perform toponym disambiguation on a per-text basis, passing as argument
     the dictionary of candidates returned in the previous step.

  See an example, assuming the dataset is in a ``CSV`` format, with one text
  per row:

  .. code-block:: python

    # Load the data:
    df = pd.read_pickle("1880-1900-LwM-HMD-subsample.csv")
    location = "London"
    wikidata_id = "Q84"

    # Instantiate the recogniser, ranker and linker:
    myner = recogniser.Recogniser(...)
    myranker = ranking.Ranker(...)
    mylinker = linking.Linker(...)

    # Instantiate the pipeline:
    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    # Find mentions for each text in the dataframe:
    nlp_df["identified_toponyms"] = nlp_df.progress_apply(
        lambda x: geoparser.run_text_recognition(
            x["text"],
            place_wqid=wikidata_id,
            place=location,
        ),
        axis=1,
    )

    # Obtain the set of unique mentions in the whole dataset and find their candidates:
    all_toponyms = [item for l in nlp_df["identified_toponyms"] for item in l]
    all_cands = geoparser.run_candidate_selection(all_toponyms)

    # Disambiguate the mentions for each text in the dataframe, taking as an input the
    # recognised mentions and the mention-to-candidate dictionaries:
    nlp_df["identified_toponyms"] = nlp_df.progress_apply(
        lambda x: geoparser.run_disambiguation(
            x["identified_toponyms"],
            all_cands,
            place_wqid=wikidata_id,
            place=location,
        ),
        axis=1,
    )

`back to top <#top-tour>`_

.. _The Recogniser:

The Recogniser
--------------

The Recogniser performs toponym recognition (i.e. geographic named entity
recognition), using HuggingFace's ``transformers`` library. Users can either:

#. Load an existing model (either directly downloading a model from the
   HuggingFace hub or loading a locally stored NER model), or
#. Fine-tune a new model on top of a base model and loading it, or directly
   load it if it is already pre-trained.

The following notebooks provide examples of both training or loading a
NER model using the Recogniser, and using it for detecting entities:

::

    ./examples/train_use_ner_model.ipynb
    ./examples/load_use_ner_model.ipynb

1. Instantiate the Recogniser
#############################

To load an already trained model (both from HuggingFace or a locally stored
pre-trained model), you can just instantiate the recogniser as follows:

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="path-to-model",
        load_from_hub=True,
    )

For example, in order to load the `Livingwithmachines/toponym-19thC-en
<https://huggingface.co/Livingwithmachines/toponym-19thC-en>`_ NER model
from the HuggingFace hub, initialise the Recogniser as follows:

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="Livingwithmachines/toponym-19thC-en",
        load_from_hub=True,
    )

You can also load a model that is stored locally in the same way. For example,
let's suppose the user has a NER model stored in the relative location
``../resources/models/blb_lwm-ner-fine``. The user could load it as follows
(notice that ``load_from_hub`` should still be True, a better name for this
would probably be ``load_from_path``):

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="resources/models/blb_lwm-ner-fine",
        load_from_hub=True,
    )

Alternatively, you can use the Recogniser to train a new model (and load it,
once it's trained). The model will be trained using HuggingFace's
``transformers`` library. To instantiate the Recogniser for training a new
model and loading it once it's trained, you can do it as in the example
(see the description of each parameter below):

.. code-block:: python

    import recogniser

    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset="experiments/outputs/data/lwm/ner_fine_train.json",
        test_dataset="experiments/outputs/data/lwm/ner_fine_dev.json",
        base_model="Livingwithmachines/bert_1760_1900",
        model_path="resources/models/",
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,
        do_test=False,
        load_from_hub=False,
    )

Description of the parameters:

* ``load_from_hub``: it indicates whether to load a pre-trained NER model. If it is
  set to ``False``, the Recogniser will be prepared to train a new model, unless
  the model already exists.
* ``overwrite_training``: it indicates whether a model should be re-trained, even if
  there already is a model with the same name in the pre-specified output folder.
  If ``load_from_hub`` is set to ``False`` and ``overwrite_training`` is also set
  to ``False``, then the Recogniser will be prepared to first try to load the model
  and---if it does not exist---to train it. If ``overwrite_training`` is set to
  ``True``, it will prepare the Recogniser to train a model, even if a model with
  the same name already exists.
* ``base_model``: the path to the model that will be used as base to train our NER
  model. This can be the path to a HuggingFace model (for example, we are using
  `Livingwithmachines/bert_1760_1900 <https://huggingface.co/Livingwithmachines/bert_1760_1900>`_,
  a BERT model trained on nineteenth-century texts) or the path to a pre-trained
  model from a local folder.
* ``train_dataset`` and ``test_dataset``: the path to the train and test data sets
  necessary for training the NER model. You can find more information about the
  format of this data in the ":doc:`resources`" page in the documentation.
* ``model_path``: the path folder where the Recogniser will store the model (and
  try to load it from).
* ``model``: the name of the NER model.
* ``training_args``: the training arguments: the user can change the learning rate,
  batch size, number of training epochs, and weight decay.
* ``do_test``: it allows the user to train a mock model and then load it (note that
  the suffix ``_test`` will be added to the model name).

2. Train the NER model
######################

Once the Recogniser has been initialised, you can train the model by running:

.. code-block:: python

    myner.train()

Note that if ``load_to_hub`` is set to ``True`` or the model already exists
(and ``overwrite_training`` is set to ``False``), the training will be skipped,
even if you call the ``train()`` method.

.. note::

    Note that this step is already taken care of if you use the T-Res ``Pipeline``.

`back to top <#top-tour>`_

.. _The Ranker:

The Ranker
----------

The Ranker takes the named entities detected by the Recogniser as input.
Given a knowledge base, it ranks the entities names according to their string
similarity to the target named entity, and selects a subset of candidates that
will be passed on to the next component, the Linker, to do the disambiguation
and select the most likely entity.

In order to use the Ranker and the Linker, we need a knowledge base, a gazetteer.
T-Res uses a gazetteer which combines data from Wikipedia and Wikidata. See how
to obtain the Wikidata-based resources in the ":doc:`resources`" page in the
documentation.

T-Res provides four different strategies for selecting candidates:

* ``perfectmatch`` retrieves candidates from the knowledge base if one of their
  alternate names is identical to the detected named entity. For example, given
  the mention "Wiltshire", the following Wikidata entities will be retrieved:
  `Q23183 <https://www.wikidata.org/wiki/Q23183>`_,
  `Q55448990 <https://www.wikidata.org/wiki/Q55448990>`_, and
  `Q8023421 <https://www.wikidata.org/wiki/Q8023421>`_, because all these
  entities are referred to as "Wiltshire" in Wikipedia anchor texts.
* ``partialmatch`` retrieves candidates from the knowledge base if there is a
  (partial) match between the query and the candidate names, based on string
  overlap. Therefore, the mention "Ashton-under" returns candidates for
  "Ashton-under-Lyne".
* ``levenshtein`` retrieves candidates from the knowledge base if there is a
  fuzzy match between the query and the candidate names, based on levenshtein
  distance. Therefore, mention "Wiltshrre" would still return the candidates
  for "Wiltshire". This method is often quite accurate when it comes to OCR
  variations, but it is very slow.
* ``deezymatch`` retrieves candidates from the knowledge base if there is a
  fuzzy match between the query and the candidate names, based on similarity
  between `DeezyMatch <https://github.com/Living-with-machines/DeezyMatch>`_
  embeddings. It is significantly more complex than the other methods to set
  up from scratch, and you will need to train a DeezyMatch model (which takes
  about two hours), but once it is set up, it is the fastest approach (except
  for ``perfectmatch``).

1. Instantiate the Ranker
#########################

1.1. Perfectmatch, partialmatch, and levenshtein
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the Ranker for exact matching (``perfectmatch``) or fuzzy string
matching based either on overlap or Levenshtein distance (``partialmatch``
and ``levenshtein`` respectively), instantiate it as follows, changing the
``method`` argument accordingly:

.. code-block:: python

    from geoparser import ranking

    myranker = ranking.Ranker(
        method="perfectmatch", # or "partialmatch" or "levenshtein"
        resources_path="resources/wikidata/",
    )

Note that ``resources_path`` should contain the path to the directory
where the Wikidata- and Wikipedia-based resources are stored, as described
in the ":doc:`resources`" page in the documentation.

1.2. DeezyMatch
^^^^^^^^^^^^^^^

DeezyMatch instantiation is trickier, as it requires training a model that,
ideally, should capture the types of string variations that can be found in
your data (such as OCR errrors). Using the Ranker, you can:

* **Option 1:** Train a DeezyMatch model from scratch, including generating
  a string pairs dataset.
* **Option 2:** Train a DeezyMatch model, given an existing string pairs dataset.

Once a DeezyMatch has been trained, you can load it and use it. The following
notebooks provide examples of each case:

::

    ./examples/train_use_deezy_model_1.ipynb # Option 1
    ./examples/train_use_deezy_model_2.ipynb # Option 2
    ./examples/train_use_deezy_model_3.ipynb # Load an existing DeezyMatch model.

See below each option in detail.

Option 1. Train a DeezyMatch model from scratch, given an existing string pairs dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To train a DeezyMatch model from scratch, using an existing string pairs dataset,
you will need to have the following `resources` file structure (as described in
the ":doc:`resources`" page in the documentation):

::

    T-RES/
    ├── ...
    ├── resources/
    │   ├── deezymatch/
    │   │   ├── data/
    │   │   │   └── w2v_ocr_pairs.txt
    │   │   └── inputs/
    │   │       ├── characters_v001.vocab
    │   │       └── input_dfm.yaml
    │   ├── models/
    │   ├── news_datasets/
    │   ├── wikidata/
    │   │   ├── mentions_to_wikidata_normalized.json
    │   │   └── wikidata_to_mentions_normalized.json
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
        # Parameters to create the string pair dataset:
        strvar_parameters=dict(),
        # Parameters to train, load and use a DeezyMatch model:
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("resources/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

Description of the parameters (to learn more, refer to the `DeezyMatch readme
<https://github.com/Living-with-machines/DeezyMatch/blob/master/README.md#candidate-ranking>`_):

* ``strvar_parameters`` contains the parameters needed to generate the
  DeezyMatch training set. It can be left empty, since the training set
  already exists.
* ``deezy_parameters``: contains the set of parameters to train or load a
  DeezyMatch model:

  * ``dm_path``: The path to the folder where the DeezyMatch model and data will
    be stored.
  * ``dm_cands``: The name given to the set of alternate names from which DeezyMatch
    will try to find a match for a given mention.
  * ``dm_model``: Name of the DeezyMatch model to train (or load if the
    model already exists).
  * ``dm_output``: Name of the DeezyMatch output file (not really needed).
  * ``ranking_metric``: DeezyMatch parameter: the metric used to rank the string
    variations based on their vectors.
  * ``selection_threshold``: DeezyMatch parameter: selection threshold based on
    the ranking metric.
  * ``num_candidates``: DeezyMatch parameter: maximum number of string variations
    that will be retrieved.
  * ``verbose``: DeezyMatch parameter: verbose output or not.
  * ``overwrite_training``: Whether to overwrite the training of a DeezyMatch
    model provided it already exists.
  * ``do_test``: Whether to train a model in test mode.

Option 2. Train a DeezyMatch model from scratch, including generating a string pairs dataset
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To train a DeezyMatch model from scratch, including generating a string pairs
dataset, you will need to have the following ``resources`` file structure (as
described in the ":doc:`resources`" page in the documentation):

::

    T-RES/
    ├── ...
    ├── resources/
    │   ├── deezymatch/
    │   ├── models/
    │   │   └── w2v/
    │   │       ├── w2v_1800s_news
    │   │       │   ├── w2v.model
    │   │       │   ├── w2v.model.syn1neg.npy
    │   │       │   └── w2v.model.wv.vectors.npy
    │   │       ├── ...
    │   │       └── w2v_1860s_news
    │   │           ├── w2v.model
    │   │           ├── w2v.model.syn1neg.npy
    │   │           └── w2v.model.wv.vectors.npy
    │   ├── news_datasets/
    │   ├── wikidata/
    │   │   ├── mentions_to_wikidata_normalized.json
    │   │   └── wikidata_to_mentions_normalized.json
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
        # Parameters to create the string pair dataset:
        strvar_parameters={
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(Path("../resources/models/w2v/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
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
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

Description of the parameters (to learn more, refer to the `DeezyMatch readme
<https://github.com/Living-with-machines/DeezyMatch/blob/master/README.md#candidate-ranking>`_):

* ``strvar_parameters`` contains the parameters needed to generate the
  DeezyMatch training set:

  * ``ocr_threshold``: Maximum `FuzzyWuzzy <https://pypi.org/project/fuzzywuzzy/>`_
    ratio for two strings to be considered negative variations of each other.
  * ``top_threshold``: Minimum `FuzzyWuzzy <https://pypi.org/project/fuzzywuzzy/>`_
    ratio for two strings to be considered positive variations of each other.
  * ``min_len``: Minimum length for a word to be included in the dataset.
  * ``max_len``: Maximum length for a word to be included in the dataset.
  * ``w2v_ocr_path``: The path to the word2vec embeddings folders.
  * ``w2v_ocr_model``: The folder name of the word2vec embeddings (it can be a
    regular expression).
  * ``overwrite_dataset``: Whether to overwrite the dataset if it already exists.

* ``deezy_parameters``: contains the set of parameters to train or load a
  DeezyMatch model:

  * ``dm_path``: The path to the folder where the DeezyMatch model and data will
    be stored.
  * ``dm_cands``: The name given to the set of alternate names from which DeezyMatch
    will try to find a match for a given mention.
  * ``dm_model``: Name of the DeezyMatch model to train or load.
  * ``dm_output``: Name of the DeezyMatch output file (not really needed).
  * ``ranking_metric``: DeezyMatch parameter: the metric used to rank the string
    variations based on their vectors.
  * ``selection_threshold``: DeezyMatch parameter: selection threshold based on
    the ranking metric.
  * ``num_candidates``: DeezyMatch parameter: maximum number of string variations
    that will be retrieved.
  * ``verbose``: DeezyMatch parameter: verbose output or not.
  * ``overwrite_training``: Whether to overwrite the training of a DeezyMatch
    model provided it already exists.
  * ``do_test``: Whether to train a model in test mode.

2. Load the resources
#####################

The following line of code loads the resources (i.e. the
``mentions-to-wikidata_normalized.json`` and
``wikidata_to_mentions_normalized.json`` files into dictionaries). They are
required in order to perform candidate selection and ranking, regardless of
the Ranker method.

.. code-block:: python

    myranker.load_resources()

.. note::

    Note that this step is already taken care of if you use the ``Pipeline``.

1. Train a DeezyMatch model
###########################

The following line will train a DeezyMatch model, given the arguments specified
when instantiating the Ranker.

.. code-block:: python

    myranker.train()

Note that if the model already exists and ``overwrite_training`` is set to
``False``, the training will be skipped, even if you call the ``train()``
method. The training will also be skipped if the Ranker is instantiated for
a different method than DeezyMatch.

The resulting model will be stored in the specified path. In this case, the
resulting DeezyMatch model that the Ranker will use is called ``w2v_ocr``:

::

    T-RES/
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
    │   │   ├── mentions_to_wikidata_normalized.json
    │   │   └── wikidata_to_mentions_normalized.json
    │   └── wikipedia/
    └── ...

.. note::

    Note that this step is already taken care of if you use the ``Pipeline``.

4. Retrieve candidates for a given mention
##########################################

In order to use the Ranker to retrieve candidates for a given mention, follow
the example. The ``find_candidates`` Ranker method requires that the input is
a list of dictionaries, where the key is always ``"mention"`` and the value
is the toponym in question.

.. code-block:: python

    toponym = "Manchefter"
    print(myranker.find_candidates([{"mention": toponym}])[0][toponym])

`back to top <#top-tour>`_

.. _The Linker:

The Linker
----------

The Linker takes as input the set of candidates selected by the Ranker and
disambiguates them, selecting the best matching entity depending on the
approach selected for disambiguation.

We provide two different strategies for disambiguation:

* ``mostpopular``: Unsupervised method, which, given a set of candidates
  for a given mention, returns as a prediction the candidate that is most
  popular in terms of inlink structure in Wikipedia.
* ``reldisamb``: Given a set of candidates, this approach uses the
  `REL re-implementation <https://github.com/informagi/REL/>`_ of the
  `ment-norm algorithm <https://github.com/lephong/mulrel-nel>`_ proposed
  by Le and Titov (2018) and partially based on Ganea and Hofmann (2017),
  and adapts it. To know more:

  ::

      Van Hulst, Johannes M., Faegheh Hasibi, Koen Dercksen, Krisztian Balog, and
      Arjen P. de Vries. "Rel: An entity linker standing on the shoulders of giants."
      In Proceedings of the 43rd International ACM SIGIR Conference on Research and
      Development in Information Retrieval, pp. 2197-2200. 2020.

      Le, Phong, and Ivan Titov. "Improving Entity Linking by Modeling Latent Relations
      between Mentions." In Proceedings of the 56th Annual Meeting of the Association
      for Computational Linguistics (Volume 1: Long Papers), pp. 1595-1604. 2018.

      Ganea, Octavian-Eugen, and Thomas Hofmann. "Deep Joint Entity Disambiguation
      with Local Neural Attention." In Proceedings of the 2017 Conference on
      Empirical Methods in Natural Language Processing, pp. 2619-2629. 2017.

1. Instantiate the Linker
#########################

1.1. ``mostpopular``
^^^^^^^^^^^^^^^^^^^^

To use the Linker with the ``mostpopular`` approach, instantiate it as follows:

.. code-block:: python

  from geoparser import linking

  mylinker = linking.Linker(
      method="mostpopular",
      resources_path="resources/",
  )

Description of the parameters:

* ``method``: name of the method, in this case ``mostpopular``.
* ``resources_path``: path to the resources directory.

Note that ``resources_path`` should contain the path to the directory where
the resources are stored.

When using the ``mostpopular`` linking approach, the resources folder should at
least contain the following resources:

::

    T-Res/
      └── resources/
          └── wikidata/
              ├── entity2class.txt
              ├── mentions_to_wikidata.json
              └── wikidata_gazetteer.csv

1.2. ``reldisamb``
^^^^^^^^^^^^^^^^^^

To use the Linker with the ``reldisamb`` approach, instantiate it as follows:

.. code-block:: python

  from geoparser import linking

  with sqlite3.connect("resources/rel_db/embeddings_database.db") as conn:
      cursor = conn.cursor()
      mylinker = linking.Linker(
          method="reldisamb",
          resources_path="resources/",
          rel_params={
              "model_path": "resources/models/disambiguation/",
              "data_path": "experiments/outputs/data/lwm/",
              "training_split": "originalsplit",
              "db_embeddings": cursor,
              "with_publication": True,
              "without_microtoponyms": True,
              "do_test": False,
              "default_publname": "London",
              "default_publwqid": "Q84",
          },
          overwrite_training=False,
      )

Description of the parameters:

* ``method``: name of the method, in this case ``reldisamb``.
* ``resources_path``: path to the resources directory.
* ``overwrite_training``: whether to overwrite the training of the entity
  disambiguation model provided a model with the same path and name already
  exists.
* ``rel_params``: set of parameters specific to the ``reldisamb`` method:

  * ``model_path``: Path to the entity disambiguation model.
  * ``data_path``: Path to the dataset file ``linking_df_split.tsv`` used for
    training a model (see information about the dataset in the ":doc:`resources`"
    page in the documentation).
  * ``training_split``: Column from the ``linking_df_split.tsv`` file that indicates
    which documents are used for training, development, and testing (see more
    information about this in the ":doc:`resources`" page in the documentation).
  * ``db_embeddings``: cursor for the embeddings database (see more
    information about this in the ":doc:`resources`" page in the documentation).
  * ``with_publication``: whether place of publication should be used as a feature
    when disambiguating (by adding it as an already disambiguated entity).
  * ``without_microtoponyms``: whether to filter out microtoponyms or not (i.e.
    filter out all entities that are not ``LOC``).
  * ``do_test``: Whether to train an entity disambiguation model in test mode.
  * ``default_publname``: The default value for the place of publication of
    the texts. For example, "London". This will be the default publication place
    name, but you will be able to override it when using the Linker to do predictions.
    This will be ignored if ``with_publication`` is ``False``.
  * ``default_publwqid``: The wikidata ID of the place of publication. For example,
    ``Q84`` for London. As in ``default_publname``, you will be able to override
    it at inference time, and it will be ignored if ``with_publication`` is ``False``.

In this way, an entity disambiguation model will be trained unless a model trained
using the same characteristics already exists (i.e. same candidate ranker method,
same ``training_split`` column name, and same values for ``with_publication`` and
``without_microtoponyms``).

When using the ``reldisamb`` linking approach, the resources folder should at
least contain the following resources:

::

    T-Res/
      └── resources/
          ├── wikidata/
          |   ├── entity2class.txt
          |   ├── mentions_to_wikidata.json
          |   └── wikidata_gazetteer.csv
          └── rel_db/
              └── embeddings_database.db


2. Load the resources
#####################

The following line of code loads the resources required by the Linker, regardless
of the Linker method.

.. code-block:: python

    mylinker.load_resources()

.. note::

    Note that this step is already taken care of if you use the ``Pipeline``.

1. Train an entity disambiguation model
#######################################

The following line will train an entity disambiguation model, given the arguments
specified when instantiating the Linker.

.. code-block:: python

    mylinker.rel_params["ed_model"] = mylinker.train_load_model(self.myranker)

Note that if the model already exists and ``overwrite_training`` is set to ``False``,
the training will be skipped, even if you call the ``train()`` method. The training
will also be skipped if the Linker is instantiated for ``mostpopular``.

The resulting model will be stored in the location specified when instantiating the
Linker (i.e. ``resources/models/disambiguation/`` in the example) in a new folder
whose name combines information about the ranking and linking arguments used in
training the method.

.. note::

    Note that this step is already taken care of if you use the ``Pipeline``.

`back to top <#top-tour>`_
