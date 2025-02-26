# Recogniser

The Recogniser performs toponym recognition (i.e. geographic named entity recognition), using HuggingFace's `transformers` library. Users can either:

1.  Load an existing model (either directly downloading a model from the HuggingFace hub or loading a locally stored NER model), or
2.  Fine-tune a new model on top of a base model and loading it, or directly load it if it is already pre-trained.

The following notebooks provide examples of both training or loading a NER model using the Recogniser, and using it for detecting entities:

    ./examples/train_use_ner_model.ipynb
    ./examples/load_use_ner_model.ipynb

**TODO:** insert Recogniser class diagram here.

## 1. Instantiate the Recogniser

### Pretrained Recogniser

To load an already trained model (both from HuggingFace or a locally stored pre-trained model), you can just instantiate the recogniser as follows:

```python
from t_res.geoparser import ner

recogniser = ner.PretrainedRecogniser(
    model_name="path-to-model"
)
```

For example, in order to load the [Livingwithmachines/toponym-19thC-en](https://huggingface.co/Livingwithmachines/toponym-19thC-en) NER model from the HuggingFace hub, initialise the Recogniser as follows:

```python
from t_res.geoparser import ner

recogniser = ner.PretrainedRecogniser(
    model_name="Livingwithmachines/toponym-19thC-en"
)
```

You can also load a model that is stored locally in the same way. For example, let's suppose the user has a NER model stored in the relative location `../resources/models/blb_lwm-ner-fine`. The user could load it as follows:

```python
from t_res.geoparser import ner

recogniser = ner.PretrainedRecogniser(
    model_name="resources/models/blb_lwm-ner-fine"
)
```

### Custom Recogniser

Alternatively, you can use the Recogniser to train a new model (and load it, once it's trained). The model will be trained using HuggingFace's `transformers` library. To instantiate the Recogniser for training a new model and loading it once it's trained, you can do it as in the example (see the description of each parameter below):

```python
from t_res.geoparser import ner

recogniser = ner.CustomRecogniser(
    model_name="blb_lwm-ner-fine",
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
)
```

Description of the parameters:

-   `overwrite_training`: it indicates whether a model should be re-trained, even if there already is a model with the same name in the pre-specified output folder. If `load_from_hub` is set to `False` and `overwrite_training` is also set to `False`, then the Recogniser will be prepared to first try to load the model and---if it does not exist---to train it. If `overwrite_training` is set to `True`, it will prepare the Recogniser to train a model, even if a model with the same name already exists.
-   `base_model`: the path to the model that will be used as base to train our NER model. This can be the path to a HuggingFace model (for example, we are using [Livingwithmachines/bert_1760_1900](https://huggingface.co/Livingwithmachines/bert_1760_1900), a BERT model trained on nineteenth-century texts) or the path to a pre-trained model from a local folder.
-   `train_dataset` and `test_dataset`: the path to the train and test data sets necessary for training the NER model. You can find more information about the format of this data in the "[Resources and file structure](../../getting-started/resources.md)" page in the documentation.
-   `model_path`: the path folder where the Recogniser will store the model (and try to load it from).
-   `model`: the name of the NER model.
-   `training_args`: the training arguments: the user can change the learning rate, batch size, number of training epochs, and weight decay.
-   `do_test`: it allows the user to train a mock model and then load it (note that the suffix `_test` will be added to the model name).

## 2. Train the NER model

!!! title "Note"

    Note that this step is already taken care of if you use the default [T-Res Pipeline](./index.md).

Once the Recogniser has been initialised, you can train the model by running:

```python
recogniser.train()
```

Note that if `load_to_hub` is set to `True` or the model already exists (and `overwrite_training` is set to `False`), the training will be skipped, even if you call the `train()` method.

## 3. Resolve toponyms in a sentence of text.

In order to use the Recogniser to resolve toponyms from text, follow the example. The Recogniser's `run` method takes as input a sentence (of type `str`):
```python
sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
sentence_mentions = recogniser.run(sentence)
print(sentence_mentions)
```
The `run` method returns an instance of the `SentenceMentions` dataclass, containing all of the toponym mentions found in the given sentence. To obtain a list of `Mentions` instance, access the `mentions` attribute:
```python
list_of_mentions = sentence_mentions.mentions
```

&nbsp;
