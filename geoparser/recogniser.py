import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import ner


class Recogniser:
    def __init__(
        self,
        model_name,
        model,
        pipe,
        base_model,
        train_dataset,
        test_dataset,
        output_model_path,
        training_args,
        overwrite_training,
        do_test,
        training_tagset,
    ):
        self.model_name = model_name  # NER model name prefix
        self.model = model  # We'll store the NER model here:
        self.pipe = pipe  # We'll store the NER pipeline here
        self.base_model = base_model  # Path to base model to fine-tune
        self.train_dataset = train_dataset  # Path to training dataset
        self.test_dataset = test_dataset  # Path to test dataset
        self.output_path = output_model_path  # Path to output folder
        self.training_args = training_args  # Dictionary of fine-tuning args
        self.overwrite_training = overwrite_training  # Bool: True to overwrite training
        self.do_test = do_test  # Bool: True to run it on test mode
        self.training_tagset = training_tagset  # Use fine or coarse tagset
        self.model_name = self.model_name + "-" + self.training_tagset  # Rename model

    # -------------------------------------------------------------
    def __str__(self):
        """
        Print the string representation of the Recogniser object.
        """
        s = (
            "\n>>> Toponym recogniser:\n"
            "    * Model name: {0}\n"
            "    * Base model: {1}\n"
            "    * Overwrite model if exists: {2}\n"
            "    * Train in test mode: {3}\n"
            "    * Training args: {4}\n"
            "    * Training tagset: {5}\n"
        ).format(
            self.model_name,
            self.base_model,
            str(self.overwrite_training),
            str(self.do_test),
            str(self.training_args),
            self.training_tagset,
        )
        return s

    # -------------------------------------------------------------
    def train(self):
        """
        Train a NER model. The training will be skipped if the model already
        exists and self.overwrite_training it set to False. The training will
        be run on test mode if self.do_test is set to True.

        Returns:
            A trained NER model.

        Code adapted from HuggingFace tutorial: https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb.
        """

        if self.overwrite_training == False:
            print("\nThe NER model is already trained!\n")
            return None

        print("*** Training the toponym recognition model...")

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        metric = load_metric("seqeval")

        # Load train and test sets:
        if self.do_test == True:
            # If test is True, train on a portion of the train and test sets, and add "_test" to the model name.
            self.model_name = self.model_name + "_test"
            lwm_train = load_dataset("json", data_files=self.train_dataset, split="train[:10]")
            lwm_test = load_dataset("json", data_files=self.train_dataset, split="train[:10]")
        else:
            lwm_train = load_dataset("json", data_files=self.train_dataset, split="train")
            lwm_test = load_dataset("json", data_files=self.train_dataset, split="train")

        # If model exists and overwrite is set to False, skip training:
        if (
            Path(self.output_path + self.model_name + ".model").exists()
            and self.overwrite_training == False
        ):
            print(
                "\n** Note: Model "
                + self.output_path
                + self.model_name
                + ".model is already trained. Set overwrite to True if needed.\n"
            )
            return None

        # Map tags to labels to predict:
        label_encoding_dict = ner.encode_dict(self.training_tagset)
        label_list = list(label_encoding_dict.keys())

        # Load model and tokenizer:
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model, num_labels=len(label_list)
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Align tokens and labels when training:
        lwm_train_tok = lwm_train.map(
            partial(
                ner.training_tokenize_and_align_labels,
                tokenizer=tokenizer,
                label_encoding_dict=label_encoding_dict,
            ),
            batched=True,
        )
        lwm_test_tok = lwm_test.map(
            partial(
                ner.training_tokenize_and_align_labels,
                tokenizer=tokenizer,
                label_encoding_dict=label_encoding_dict,
            ),
            batched=True,
        )

        # Compute metrics when training:
        def compute_metrics(p):
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

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        training_args = TrainingArguments(
            output_dir=self.output_path,
            evaluation_strategy="epoch",
            logging_dir=self.output_path + "runs/" + self.model_name,
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
        trainer.save_model(self.output_path + self.model_name + ".model")

    # -------------------------------------------------------------
    def create_pipeline(self):
        """
        Create a pipeline for performing NER given a NER model.

        Returns:
            self.model (str): the model name.
            self.pipe (Pipeline): a pipeline object which performs
                named entity recognition given a model.
        """
        print("*** Creating and loading a NER pipeline.")
        # Path to NER Model:
        self.model = self.output_path + self.model_name + ".model"
        self.pipe = pipeline("ner", model=self.model)
        return self.model, self.pipe

    # -------------------------------------------------------------
    def ner_predict(self, sentence):
        """
        Given a sentence, recognise its mentioned entities.

        Arguments:
            sentence (str): a sentence.

        Returns:
            predictions (list): a list of dictionaries, one per recognised
            token, e.g.: {'entity': 'O', 'score': 0.99975187, 'word': 'From',
            'start': 0, 'end': 4}
        """

        # Use our BERT-based toponym recogniser to detect entities, and
        # postprocess the output to fix easily fixable misassignments.
        # The result, 'predictions', is a list of dictionaries, each
        # corresponding to an entity.
        mapped_label = ner.map_tag_label(self.training_tagset)
        ner_preds = self.pipe(sentence)
        lEntities = []
        predictions = []
        for pred_ent in ner_preds:
            prev_tok = pred_ent["word"]
            pred_ent["score"] = float(pred_ent["score"])
            pred_ent["entity"] = mapped_label[pred_ent["entity"]]
            pred_ent = ner.fix_capitalization(pred_ent, sentence)
            if prev_tok.lower() != pred_ent["word"].lower():
                print("Token processing error.")
            predictions = ner.aggregate_entities(pred_ent, lEntities)
        if len(predictions) > 0:
            predictions = ner.fix_hyphens(predictions)
            predictions = ner.fix_nested(predictions)
            predictions = ner.fix_startEntity(predictions)
        return predictions
