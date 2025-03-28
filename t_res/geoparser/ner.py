import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Pipeline,
    Trainer,
    TrainingArguments,
    pipeline,
)

from ..utils import ner_utils
from ..utils.dataclasses import Mention, Sentence, SentenceMentions

class Recogniser:
    """
    The Recogniser class provides methods for named entity recognition
    applied to toponyms.

    Arguments:
        model_name (str): The name of the NER model.
        device (str, optional): GPU device name (default: ``None``).

    Note:
        This base class should not be instatiated directly. Instead use a subclass
            constructor.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str]=None,
    ):
        """
        Initialises a Recogniser object.
        """
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __str__(self) -> str:
        """
        Returns a string representation of the Recogniser object.

        Returns:
            str: String representation of the Recogniser object.
        """
        s = "\n>>> Toponym recogniser:\n"
        s += f"    * Model: {self.model()}\n"
        return s

    def new(**kwargs) -> 'Recogniser':
        """
        Static constructor.

        Args:
            kwargs (dict): A dictionary of keyword arguments matching the
                arguments to a subclass __init__ constructor, plus a
                `method_name` argument to specify the desired subclass.

        Returns:
            A Recogniser (subclass) instance.

        """
        if not 'method_name' in kwargs.keys():
            raise ValueError("Expected `method_name` keyword argument.")
        method_name = kwargs['method_name']
        del kwargs['method_name']
        if method_name == 'pretrained':
            return PretrainedRecogniser(**kwargs)
        if method_name == 'custom':
            return CustomRecogniser(**kwargs)
        raise ValueError(f"Invalid NER method: {method_name}")

    def model(self) -> str:
        """
        Returns the ``model`` parameter to be passed to the Pipeline factory 
        method in the ``transformers`` package.

        Returns:
            str: The ``model`` parameter
        """
        raise NotImplementedError("Subclass implementation required.")

    def load(self):
        """
        Creates a Named Entity Recognition (NER) pipeline and assigns it
        to the ``pipe`` attribute.

        Note:
            This method creates and loads a NER pipeline for performing named
            entity recognition tasks. The created pipeline is stored in the 
            ``pipe`` attribute of the ``Recogniser`` instance.
        """

        print("*** Creating and loading a NER pipeline.")
        self.pipe = pipeline("ner", model=self.model(), ignore_labels=[], device=self.device)

    # The run method combines `ner_predict` with the `aggregate_mentions`
    # function from `ner_utils.py` (eventually making those redundant).
    def run(self, sentence: str) -> SentenceMentions:
        """
        Identifies named entities in a given sentence using the NER pipeline.

        Arguments:
            sentence (str): The input sentence.

        Returns:
            SentenceMentions: An instance of the SentenceMentions dataclass, containing
                a list of toponym mentions found in the given sentence.

        Note:
            Any n-dash characters (``—``) in the provided sentence are
            replaced with a comma (``,``) to handle parsing issues related to
            the n-dash in OCR from historical newspapers.
        """
        sentence = str(sentence)
        if len(sentence) <= 1:
            return SentenceMentions(Sentence(sentence), [])

        # The n-dash is a very frequent character in historical newspapers,
        # but the NER pipeline does not process it well: Plymouth—Kingston
        # is parsed as "Plymouth (B-LOC), — (B-LOC), Kingston (B-LOC)", instead
        # of the n-dash being interpreted as a word separator. Therefore, we
        # replace it by a comma, except when the n-dash occurs in the opening
        # position of a sentence.
        sentence = sentence[0] + sentence[1:].replace("—", ",")

        # Run the NER pipeline to predict mentions:
        if not hasattr(self, 'pipe'):
            raise ValueError("Missing NER pipeline. Try calling the load() method.")
        ner_preds = self.pipe(sentence)
        return self.post_process(ner_preds, sentence)

    def post_process(self, ner_predictions, sentence: str) -> SentenceMentions:

        sentence = str(sentence)
        if len(sentence) <= 1:
            return SentenceMentions(Sentence(sentence), [])

        # Post-process the predictions, fixing potential grouping errors:
        lEntities = []
        predictions = []
        for pred_ent in ner_predictions:
            pred_ent["score"] = float(pred_ent["score"])
            pred_ent["entity"] = pred_ent["entity"]
            pred_ent = ner_utils.fix_capitalization(pred_ent, sentence)
            predictions = ner_utils.aggregate_entities(pred_ent, lEntities)

        if len(predictions) > 0:
            predictions = ner_utils.fix_hyphens(predictions)
            predictions = ner_utils.fix_nested(predictions)
            predictions = ner_utils.fix_startEntity(predictions)

        # Process predictions (moved from pipeline.py::run_sentence_recognition):
        procpreds = [
            [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
            for x in predictions
        ]

        # Aggregate mentions:
        mentions = ner_utils.aggregate_mentions(procpreds, "pred")

        mentions = [Mention.from_dict(m) for m in mentions]
        return SentenceMentions(Sentence(sentence), mentions=mentions)

    # Deprecated: use the `run` method instead.
    def ner_predict(self, sentence: str) -> List[dict]:
        """
        Predicts named entities in a given sentence using the NER pipeline.

        Arguments:
            sentence (str): The input sentence.

        Returns:
            A list of dictionaries representing the predicted named
                entities. Each dictionary contains the keys ``"word"``,
                ``"entity"``, ``"score"``, ``"start"`` , and ``"end"``
                representing the entity text, entity label, confidence
                score and start and end character position of the text
                respectively. For example:

                ```json
                {
                    "word": "From",
                    "entity": "O",
                    "score": 0.99975187,
                    "start": 0,
                    "end": 4
                }
                ```

        Note:
            This method takes a sentence as input and uses the NER pipeline to
            predict named entities in the sentence.

            Any n-dash characters (``—``) in the provided sentence are
            replaced with a comma (``,``) to handle parsing issues related to
            the n-dash in OCR from historical newspapers.
        """
        # Error if the sentence is too short.
        if len(sentence) <= 1:
            return []

        # The n-dash is a very frequent character in historical newspapers,
        # but the NER pipeline does not process it well: Plymouth—Kingston
        # is parsed as "Plymouth (B-LOC), — (B-LOC), Kingston (B-LOC)", instead
        # of the n-dash being interpreted as a word separator. Therefore, we
        # replace it by a comma, except when the n-dash occurs in the opening
        # position of a sentence.
        sentence = sentence[0] + sentence[1:].replace("—", ",")

        # Run the NER pipeline to predict mentions:
        ner_preds = self.pipe(sentence)

        # Post-process the predictions, fixing potential grouping errors:
        lEntities = []
        predictions = []
        for pred_ent in ner_preds:
            pred_ent["score"] = float(pred_ent["score"])
            pred_ent["entity"] = pred_ent["entity"]
            pred_ent = ner_utils.fix_capitalization(pred_ent, sentence)
            predictions = ner_utils.aggregate_entities(pred_ent, lEntities)

        if len(predictions) > 0:
            predictions = ner_utils.fix_hyphens(predictions)
            predictions = ner_utils.fix_nested(predictions)
            predictions = ner_utils.fix_startEntity(predictions)

        return predictions

class PretrainedRecogniser(Recogniser):
    """
    A pretrained toponym recogniser loaded from HuggingFace.

    Example:
        ```
        # Create an instance of the PretrainedRecogniser class
        recogniser = PretrainedRecogniser(
            model_name="Livingwithmachines/toponym-19thC-en",
        )

        # Create and load the NER pipeline
        recogniser.load()

        # Predict named entities in a sentence
        sentence = "I live in London."
        predictions = recogniser.ner_predict(sentence)
        print(predictions)
        ```
    """

    def model(self) -> str:
        """
        Returns the name of the model loaded from HuggingFace.

        Returns:
            The name of the pretrained HuggingFace model.
        """
        return self.model_name

class CustomRecogniser(Recogniser):
    """
    A toponym recogniser with data and parameters for custom training.

    Arguments:
        model_name (str): The name of the NER model.
        train_dataset (str, optional): Path to the training dataset
            (default: ``""``).
        test_dataset (str, optional): Path to the testing dataset
            (default: ``""``).
        pipe (transformers.Pipeline, optional): A pre-loaded NER pipeline
            (default: ``None``).
        base_model (str, optional): The name of the base model, for
            fine-tuning (default: ``""``)
        model_path (str, optional): Path to store the trained model
            (default: ``""``).
        training_args (dict, optional): Additional fine-tuning training
            arguments (default: {"batch_size": 8, "num_train_epochs": 10,
            "learning_rate": 0.00005, "weight_decay": 0.0}``, a dictionary).
        overwrite_training (bool, optional):  Whether to overwrite an existing
            trained model (default: ``False``).
        do_test (bool, optional): Whether to train in test mode
            (default: ``False``).

    Example:
        ```
        # Create an instance of the CustomRecogniser class
        recogniser = CustomRecogniser(
            model_name="ner-model",
            train_dataset="train.json",
            test_dataset="test.json",
            base_model="bert-base-uncased",
            model_path="/path/to/model/",
            training_args={
                "batch_size": 8,
                "num_train_epochs": 10,
                "learning_rate": 0.00005,
                "weight_decay": 0.0,
                },
            overwrite_training=False,
            do_test=False,
        )

        # Create and load the NER pipeline
        recogniser.load()

        # Predict named entities in a sentence
        sentence = "I live in London."
        predictions = recogniser.ner_predict(sentence)
        print(predictions)
        ```
    """

    def __init__(
        self,
        model_name: str,
        train_dataset: Optional[str] = "",
        test_dataset: Optional[str] = "",
        pipe: Optional[Pipeline] = None,
        base_model: Optional[str] = "",
        model_path: Optional[str] = "",
        training_args: Optional[dict] = {
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training: Optional[bool] = False,
        do_test: Optional[bool] = False,
        device: Optional[str]=None,
    ):
        """
        Initialises a Recogniser object.
        """
        super().__init__(model_name, device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.pipe = pipe
        self.base_model = base_model
        self.model_path = model_path
        self.training_args = training_args
        self.overwrite_training = overwrite_training
        self.do_test = do_test

        # Add "_test" to the model name if do_test is True.
        if self.do_test:
            self.model_name += "_test"

    def __str__(self) -> str:
        """
        Returns a string representation of the Recogniser object.

        Returns:
            A string representation of the Recogniser object.
        """
        s = super().__str__()
        s += f"    * Base model: {self.base_model}\n"
        s += f"    * Overwrite model if exists: {self.overwrite_training}\n"
        s += f"    * Train in test mode: {self.do_test}\n"
        s += f"    * Training args: {self.training_args}\n"
        return s

    def model(self) -> str:
        """
        Returns the path and filename of the trained model.

        Returns:
            The path and filename of the trained model
        """
        return os.path.join(self.model_path, f"{self.model_name}.model")

    # Override the load method to train the model if necessary.
    def load(self):
        """
        Creates a Named Entity Recognition (NER) pipeline and assigns it
        to the ``pipe`` attribute.

        Note:
            This method creates and loads a NER pipeline for performing named
            entity recognition tasks. Unless a trained model already exists and
            overwrite_training is False, it calls the ``train`` method to 
            train a custom model and saves it using the specified model name 
            and model path. It then creates the pipeline from that model.
            The created pipeline is stored in the ``pipe`` attribute of the
            ``Recogniser`` object.
        """

        if Path(self.model()).exists() and not self.overwrite_training:
            s = "\n** Note: Model "
            s += f"{self.model()} is already trained.\n"
            s += "Set overwrite_training to True if needed.\n"
            print(s)
        else:
            self.train()

        super().load()

    def train(self):
        """
        Trains an NER model and saves it under the model path.

        Note:
            Training process is executed, including the
            loading of datasets, model, and tokenizer, tokenization and
            alignment of labels, computation of evaluation metrics,
            training using the Trainer object, evaluation, and saving the
            trained model.

            The training will be run on test mode if ``do_test`` was set to
            True when the Recogniser object was initiated.

        Credit:
            This function is adapted from a [HuggingFace tutorial](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb).
        """

        print("*** Training the toponym recognition model...")

        # Create a path to store the model if it does not exist:
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        # Use the "seqeval" metric to evaluate the predictions during training:
        metric = load_metric("seqeval")

        # Load train and test sets:
        # Note: From https://huggingface.co/docs/datasets/loading: "A dataset
        # without a loading script by default loads all the data into the train
        # split."
        if self.do_test == True:
            # If test is True, train on a portion of the train and test sets:
            lwm_train = load_dataset(
                "json", data_files=self.train_dataset, split="train[:10]"
            )
            lwm_test = load_dataset(
                "json", data_files=self.test_dataset, split="train[:10]"
            )
        else:
            lwm_train = load_dataset(
                "json", data_files=self.train_dataset, split="train"
            )
            lwm_test = load_dataset("json", data_files=self.test_dataset, split="train")

        print("Train:", len(lwm_train))
        print("Test:", len(lwm_test))

        # Obtain unique list of labels:
        df_tmp = lwm_train.to_pandas()
        label_list = sorted(
            list(set([tag for tags in df_tmp["ner_tags"] for tag in tags]))
        )

        # Create mapping between labels and ids:
        id2label = dict()
        for i in range(len(label_list)):
            id2label[i] = label_list[i]
        label2id = {v: k for k, v in id2label.items()}

        # Load model and tokenizer:
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Align tokens and labels when training:
        lwm_train_tok = lwm_train.map(
            partial(
                ner_utils.training_tokenize_and_align_labels,
                tokenizer=tokenizer,
                label_encoding_dict=label2id,
            ),
            batched=True,
        )
        lwm_test_tok = lwm_test.map(
            partial(
                ner_utils.training_tokenize_and_align_labels,
                tokenizer=tokenizer,
                label_encoding_dict=label2id,
            ),
            batched=True,
        )

        # Compute metrics when training:
        def compute_metrics(p: Tuple[list, list]) -> dict:
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(
                predictions=true_predictions, references=true_labels
            )
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        training_args = TrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="epoch",
            logging_dir=os.path.join(self.model_path,"runs/",self.model_name),
            learning_rate=self.training_args["learning_rate"],
            per_device_train_batch_size=self.training_args["batch_size"],
            per_device_eval_batch_size=self.training_args["batch_size"],
            num_train_epochs=self.training_args["num_train_epochs"],
            weight_decay=self.training_args["weight_decay"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lwm_train_tok,
            eval_dataset=lwm_test_tok,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train the model:
        trainer.train()

        # Evaluate the training:
        trainer.evaluate()

        # Save the model:
        trainer.save_model(self.model())
