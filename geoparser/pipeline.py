import os
import sys
from pathlib import Path

from sentence_splitter import split_text_into_sentences

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking, ranking, recogniser
from utils import ner, rel_utils


class Pipeline:
    """
    The Pipeline class geoparses a text.
    """

    def __init__(self, myner=None, myranker=None, mylinker=None):
        """
        Arguments:
            myner (recogniser.Recogniser): a Recogniser object.
            myranker (ranking.Ranker): a Ranker object.
            mylinker (linking.Linker): a Linker object.
        """

        self.myner = myner
        self.myranker = myranker
        self.mylinker = mylinker

        if not self.myner:
            self.myner = recogniser.Recogniser(
                model="blb_lwm-ner-fine",  # NER model name prefix (will have suffixes appended)
                pipe=None,  # We'll store the NER pipeline here
                base_model="khosseini/bert_1760_1900",  # Base model to fine-tune (from huggingface)
                train_dataset="../experiments/outputs/data/lwm/ner_fine_train.json",  # Training set (part of overall training set)
                test_dataset="../experiments/outputs/data/lwm/ner_fine_dev.json",  # Test set (part of overall training set)
                model_path="../resources/models/",  # Path where the NER model is or will be stored
                training_args={
                    "learning_rate": 5e-5,
                    "batch_size": 16,
                    "num_train_epochs": 4,
                    "weight_decay": 0.01,
                },
                overwrite_training=False,  # Set to True if you want to overwrite model if existing
                do_test=False,  # Set to True if you want to train on test mode
                load_from_hub=False,
            )

        if not self.myranker:
            self.myranker = ranking.Ranker(
                method="perfectmatch",
                resources_path="../resources/wikidata/",
                mentions_to_wikidata=dict(),
                wikidata_to_mentions=dict(),
                strvar_parameters={
                    # Parameters to create the string pair dataset:
                    "ocr_threshold": 60,
                    "top_threshold": 85,
                    "min_len": 5,
                    "max_len": 15,
                    "w2v_ocr_path": str(Path("../resources/models/w2v/").resolve()),
                    "w2v_ocr_model": "w2v_*_news",
                    "overwrite_dataset": False,
                },
                deezy_parameters={
                    # Paths and filenames of DeezyMatch models and data:
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

        if not self.mylinker:
            self.mylinker = linking.Linker(
                method="mostpopular",
                resources_path="../resources/",
                linking_resources=dict(),
                rel_params={
                    "model_path": "../resources/models/disambiguation/",
                    "data_path": "../experiments/outputs/data/lwm/",
                    "training_split": "originalsplit",
                    "context_length": 100,
                    "topn_candidates": 10,
                    "db_embeddings": "../resources/rel_db/embedding_database.db",
                    "with_publication": False,
                    "with_microtoponyms": False,
                    "do_test": True,
                },
                overwrite_training=True,
            )

        # -----------------------------------------
        # NER training and creating pipeline:
        # Train the NER models if needed:
        self.myner.train()
        # Load the NER pipeline:
        self.myner.pipe = self.myner.create_pipeline()

        # -----------------------------------------
        # Ranker loading resources and training a model:
        # Load the resources:
        self.myranker.mentions_to_wikidata = self.myranker.load_resources()
        # Train a DeezyMatch model if needed:
        self.myranker.train()

        # -----------------------------------------
        # Linker loading resources:
        # Load linking resources:
        self.mylinker.linking_resources = self.mylinker.load_resources()
        # Train a linking model if needed (it requires myranker to generate potential
        # candidates to the training set):
        self.mylinker.rel_params["ed_model"] = self.mylinker.train_load_model(
            self.myranker
        )

    def run_sentence(
        self,
        sentence,
        sent_idx=0,
        context=("", ""),
        place="",
        place_wqid="",
        postprocess_output=True,
    ):
        """
        This function takes a sentence as input and performs the whole pipeline:
        first identifies toponyms, then finds relevant candidates and ranks them,
        and finally links them to the Wikidata Id.

        Arguments:
            sentence (str): The target sentence.
            sent_idx (int): Position of the target sentence in a larger text.
            context (two-element tuple): A two-elements tuple: the first is the
                previous context of the target sentence, the second is the following
                context of the target sentence.
            place (str): The place of publication, in text (e.g. "London").
            place_wqid (str): The Wikidata id of the place of publication (e.g. "Q84")
            postprocess_output (bool): Whether to postprocess the output, adding
                geographic coordinates. This will be false for running the experiments,
                and True otherwise.
        """
        sentence = sentence.replace("—", ";")
        # Get predictions:
        predictions = self.myner.ner_predict(sentence)
        # Process predictions:
        procpreds = [
            [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
            for x in predictions
        ]
        # Aggretate mentions:
        mentions = ner.aggregate_mentions(procpreds, "pred")

        # Perform candidate ranking:
        wk_cands, self.myranker.already_collected_cands = self.myranker.find_candidates(
            mentions
        )

        mentions_dataset = dict()
        mentions_dataset["linking"] = []
        for m in mentions:
            prediction = dict()
            prediction["mention"] = m["mention"]
            prediction["context"] = context
            prediction["candidates"] = []
            prediction["gold"] = ["NONE"]
            prediction["ner_score"] = m["ner_score"]
            prediction["pos"] = m["start_char"]
            prediction["sent_idx"] = sent_idx
            prediction["end_pos"] = m["end_char"]
            prediction["ngram"] = m["mention"]
            prediction["conf_md"] = m["ner_score"]
            prediction["tag"] = m["ner_label"]
            prediction["sentence"] = sentence
            prediction["candidates"] = wk_cands.get(m["mention"], dict())
            prediction["place"] = place
            prediction["place_wqid"] = place_wqid
            mentions_dataset["linking"].append(prediction)

        if self.mylinker.method == "reldisamb":
            mentions_dataset = rel_utils.rank_candidates(
                mentions_dataset,
                wk_cands,
                self.mylinker.linking_resources["mentions_to_wikidata"],
            )
            predicted = self.mylinker.rel_params["ed_model"].predict(mentions_dataset)
            for i in range(len(mentions_dataset["linking"])):
                mentions_dataset["linking"][i]["prediction"] = predicted["linking"][i][
                    "prediction"
                ]
                mentions_dataset["linking"][i]["ed_score"] = predicted["linking"][i][
                    "conf_ed"
                ]

        if self.mylinker.method in ["mostpopular", "bydistance"]:
            for i in range(len(mentions_dataset["linking"])):
                mention = mentions_dataset["linking"][i]
                # Run entity linking per mention:
                selected_cand = self.mylinker.run(
                    {
                        "candidates": wk_cands[mention["mention"]],
                        "place_wqid": place_wqid,
                    }
                )
                mentions_dataset["linking"][i]["prediction"] = selected_cand[0]
                mentions_dataset["linking"][i]["ed_score"] = round(selected_cand[1], 3)
                mentions_dataset["linking"][i]["candidates"] = {
                    x: round(y, 3) for x, y in selected_cand[2].items()
                }

        if not postprocess_output:
            return mentions_dataset

        if postprocess_output:
            # Process output:
            keys = [
                "sent_idx",
                "mention",
                "pos",
                "end_pos",
                "tag",
                "prediction",
                "ner_score",
                "ed_score",
                "sentence",
                "candidates",
            ]
            sentence_dataset = []
            for md in mentions_dataset["linking"]:
                md = dict((k, md[k]) for k in md if k in keys)
                md["latlon"] = self.mylinker.linking_resources["wqid_to_coords"].get(
                    md["prediction"]
                )
                md["wkdt_class"] = self.mylinker.linking_resources["entity2class"].get(
                    md["prediction"]
                )
                sentence_dataset.append(md)
            return sentence_dataset

    def run_text(self, text, place="", place_wqid="", postprocess_output=True):
        """
        This function takes a text as input and splits it into sentences,
        each of which will be parsed with the T-Res pipeline.
        """
        sentences = split_text_into_sentences(text, language="en")
        document_dataset = []
        for i in range(len(sentences)):
            sentence_dataset = self.run_sentence(
                sentences[i],
                sent_idx=i,
                context=("", ""),  # TODO CHANGE THIS!!!
                place=place,
                place_wqid=place_wqid,
                postprocess_output=True,
            )
            for sd in sentence_dataset:
                document_dataset.append(sd)
        return document_dataset
