import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
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

from ..utils import ner


class Recogniser:
    """
    A class for training and using a toponym recogniser with the specified
    parameters.

    Arguments:
        model (str): The name of the NER model.
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
        load_from_hub (bool, optional): Whether to load the model from
            HuggingFace model hub or locally (default: ``False``).

    Example:
        >>> # Create an instance of the Recogniser class
        >>> recogniser = Recogniser(
                model="ner-model",
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
                load_from_hub=False
            )

        >>> # Create and load the NER pipeline
        >>> pipeline = recogniser.create_pipeline()

        >>> # Train the model
        >>> recogniser.train()

        >>> # Predict named entities in a sentence
        >>> sentence = "I live in London."
        >>> predictions = recogniser.ner_predict(sentence)
        >>> print(predictions)
    """

    def __init__(
        self,
        model: str,
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
        load_from_hub: Optional[bool] = False,
    ):
        """
        Initialises a Recogniser object.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.pipe = pipe
        self.base_model = base_model
        self.model_path = model_path
        self.training_args = training_args
        self.overwrite_training = overwrite_training
        self.do_test = do_test
        self.load_from_hub = load_from_hub

        # Add "_test" to the model name if do_test is True, unless
        # the model is downloaded from Huggingface, in which case
        # we keep the name inputed by the user.
        if self.do_test == True and self.load_from_hub == False:
            self.model += "_test"

    # -------------------------------------------------------------
    def __str__(self) -> str:
        """
        Returns a string representation of the Recogniser object.

        Returns:
            str: String representation of the Recogniser object.
        """
        s = "\n>>> Toponym recogniser:\n"
        s += f"    * Model path: {self.model_path}\n"
        s += f"    * Model name: {self.model}\n"
        s += f"    * Base model: {self.base_model}\n"
        s += f"    * Overwrite model if exists: {self.overwrite_training}\n"
        s += f"    * Train in test mode: {self.do_test}\n"
        s += f"    * Load from hub: {self.load_from_hub}\n"
        s += f"    * Training args: {self.training_args}\n"
        return s

    # -------------------------------------------------------------
    def train(self) -> None:
        """
        Trains a NER model.

        Returns:
            None.

        Note:
            If the model is obtained from the HuggingFace model hub
            (``load_from_hub=True``) or if the model already exists at the
            specified model path and ``overwrite_training`` is False,
            training is skipped.

            Otherwise, the training process is executed, including the
            loading of datasets, model, and tokenizer, tokenization and
            alignment of labels, computation of evaluation metrics,
            training using the Trainer object, evaluation, and saving the
            trained model.

            The training will be run on test mode if ``do_test`` was set to
            True when the Recogniser object was initiated.

        Credit:
            This function is adapted from `a HuggingFace tutorial <https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb>`_.
        """

        # Skip training if the model is obtained from the hub:
        if self.load_from_hub == True:
            return None

        # If model exists and overwrite is set to False, skip training:
        model_path = os.path.join(self.model_path,f"{self.model}.model")
        if Path(model_path).exists() and self.overwrite_training == False:
            s = "\n** Note: Model "
            s += f"{model_path} is already trained.\n"
            s += "Set overwrite to True if needed.\n"
            print(s)
            return None

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
                ner.training_tokenize_and_align_labels,
                tokenizer=tokenizer,
                label_encoding_dict=label2id,
            ),
            batched=True,
        )
        lwm_test_tok = lwm_test.map(
            partial(
                ner.training_tokenize_and_align_labels,
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
            logging_dir=os.path.join(self.model_path,"runs/",self.model),
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
        trainer.save_model(os.path.join(self.model_path,f"{self.model}.model"))

    # -------------------------------------------------------------
    def create_pipeline(self) -> Pipeline:
        """
        Creates and loads a Named Entity Recognition (NER) pipeline.

        Returns:
            geoparser.pipeline.Pipeline: The created NER pipeline.

        Note:
            This method creates and loads a NER pipeline for performing named
            entity recognition tasks. It uses the specified model name and
            model path (if the model is not obtained from the HuggingFace
            model hub or from a local path) to initialise the pipeline.
            The created pipeline is stored in the ``pipe`` attribute of the
            ``Recogniser`` object. It is also returned by the method.
        """

        print("*** Creating and loading a NER pipeline.")

        # Path to NER Model:
        model_name = self.model

        # If the model is local (has not been obtained from the hub),
        # pre-append the model path and the extension of the model
        # to obtain the model name.
        if self.load_from_hub == False:
            model_name = os.path.join(self.model_path, f"{self.model}.model")

        # Load a NER pipeline:
        self.pipe = pipeline("ner", model=model_name, ignore_labels=[], device_map="auto")
        return self.pipe

    # -------------------------------------------------------------
    def ner_predict(self, sentence: str) -> List[dict]:
        """
        Predicts named entities in a given sentence using the NER pipeline.

        Arguments:
            sentence (str): The input sentence.

        Returns:
            List[dict]:
                A list of dictionaries representing the predicted named
                entities. Each dictionary contains the keys ``"word"``,
                ``"entity"``, ``"score"``, ``"start"`` , and ``"end"``
                representing the entity text, entity label, confidence
                score and start and end character position of the text
                respectively. For example:

                .. code-block:: json

                    {
                        "word": "From",
                        "entity": "O",
                        "score": 0.99975187,
                        "start": 0,
                        "end": 4
                    }

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
            pred_ent = ner.fix_capitalization(pred_ent, sentence)
            predictions = ner.aggregate_entities(pred_ent, lEntities)

        if len(predictions) > 0:
            predictions = ner.fix_hyphens(predictions)
            predictions = ner.fix_nested(predictions)
            predictions = ner.fix_startEntity(predictions)

        return predictions
