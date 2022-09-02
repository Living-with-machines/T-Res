import os
import sys
from pathlib import Path
from sentence_splitter import split_text_into_sentences

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import recogniser, ranking, linking
from utils import ner, pipeline_utils


class Pipeline:
    """
    The Pipeline class geoparses a text.
    """

    def __init__(
        self,
        myner=None,
        myranker=None,
        mylinker=None,
    ):
        """
        Arguments:
            text (str): the text to geoparse.
            place_publication (str): place of publication.
            year (int): year of publication.
            myner (recogniser.Recogniser): a Recogniser object.
            myranker (ranking.Ranker): a Ranker object.
            mylinker (linking.Linker): a Linker object.
        """

        self.myner = myner
        self.myranker = myranker
        self.mylinker = mylinker

        if not self.myner:
            self.myner = recogniser.Recogniser(
                model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
                model=None,  # We'll store the NER model here
                pipe=None,  # We'll store the NER pipeline here
                base_model="/resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
                train_dataset="outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
                test_dataset="outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
                output_model_path="outputs/models/",  # Path where the NER model is or will be stored
                training_args={
                    "learning_rate": 5e-5,
                    "batch_size": 16,
                    "num_train_epochs": 4,
                    "weight_decay": 0.01,
                },
                overwrite_training=False,  # Set to True if you want to overwrite model if existing
                do_test=False,  # Set to True if you want to train on test mode
                training_tagset="fine",  # Options are: "coarse" or "fine"
            )

        if not self.myranker:
            self.myranker = ranking.Ranker(
                method="deezymatch",
                resources_path="/resources/wikidata/",
                mentions_to_wikidata=dict(),
                wikidata_to_mentions=dict(),
                wiki_filtering={
                    "top_mentions": 3,  # Filter mentions to top N mentions
                    "minimum_relv": 0.03,  # Filter mentions with more than X relv
                },
                strvar_parameters={
                    # Parameters to create the string pair dataset:
                    "ocr_threshold": 60,
                    "top_threshold": 85,
                    "min_len": 5,
                    "max_len": 15,
                },
                deezy_parameters={
                    # Paths and filenames of DeezyMatch models and data:
                    "dm_path": str(Path("outputs/deezymatch/").resolve()),
                    "dm_cands": "wkdtalts",
                    "dm_model": "w2v_ocr",
                    "dm_output": "deezymatch_on_the_fly",
                    # Ranking measures:
                    "ranking_metric": "faiss",
                    "selection_threshold": 50,
                    "num_candidates": 3,
                    "search_size": 3,
                    "use_predict": False,
                    "verbose": False,
                    # DeezyMatch training:
                    "overwrite_training": False,
                    "w2v_ocr_path": str(Path("outputs/models/").resolve()),
                    "w2v_ocr_model": "w2v_*_news",
                    "do_test": False,
                },
            )

        if not self.mylinker:
            self.mylinker = linking.Linker(
                method="mostpopular",
                resources_path="/resources/wikidata/",
                linking_resources=dict(),
                base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
                rel_params={
                    "base_path": "/resources/rel_db/",
                    "wiki_version": "wiki_2019/",
                },
                overwrite_training=False,
            )

        # -----------------------------------------
        # NER training and creating pipeline:
        # Train the NER models if needed:
        self.myner.train()
        # Load the NER pipeline:
        self.myner.model, self.myner.pipe = self.myner.create_pipeline()

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
        # TODO Add as part of the pipeline:
        # # Train a linking model if needed:
        # self.mylinker.train()
        experiment_name = pipeline_utils.get_experiment_name(self, "withouttest")
        if self.mylinker.method == "reldisamb":
            (
                self.mylinker.rel_params["mention_detection"],
                self.mylinker.rel_params["model"],
            ) = self.mylinker.disambiguation_setup(experiment_name)

    def run_sentence(self, sentence, sent_idx=0, context=("", "")):

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

        # Linking settings
        if self.mylinker.method == "reldisamb":
            mention_prediction = self.mylinker.rel_params["mention_detection"]
            cand_selection = "relcs" if self.myranker.method == "relcs" else "lwmcs"
            linking_model = self.mylinker.rel_params["model"]

        mentions_dataset = dict()
        mentions_dataset["linking"] = []
        for m in mentions:
            prediction = dict()
            prediction["mention"] = m["mention"]
            prediction["context"] = context
            prediction["candidates"] = []
            prediction["gold"] = ["NONE"]
            prediction["pos"] = m["start_char"]
            prediction["sent_idx"] = sent_idx
            prediction["end_pos"] = m["end_char"]
            prediction["ngram"] = m["mention"]
            prediction["conf_md"] = 0.0
            prediction["tag"] = m["ner_label"]
            prediction["sentence"] = sentence
            prediction["candidates"] = wk_cands.get(m["mention"], dict())
            mentions_dataset["linking"].append(prediction)
            if self.mylinker.method == "reldisamb":
                prediction["candidates"] = mention_prediction.get_candidates(
                    prediction["mention"], cand_selection, prediction["candidates"]
                )

        if self.mylinker.method == "reldisamb":
            predictions, timing = linking_model.predict(mentions_dataset)
            for i in range(len(mentions_dataset["linking"])):
                mention_dataset = mentions_dataset["linking"][i]
                prediction = predictions["linking"][i]
                if mention_dataset["mention"] == prediction["mention"]:
                    mentions_dataset["linking"][i]["prediction"] = prediction[
                        "prediction"
                    ]
                    mentions_dataset["linking"][i]["ed_score"] = round(
                        prediction["conf_ed"], 3
                    )
            mentions_dataset["linking"] = self.mylinker.format_linking_dataset(
                mentions_dataset["linking"]
            )
        if self.mylinker.method in ["mostpopular", "bydistance"]:
            for i in range(len(mentions_dataset["linking"])):
                mention = mentions_dataset["linking"][i]
                # Run entity linking per mention:
                selected_cand = self.mylinker.run(
                    {"candidates": wk_cands[mention["mention"]]}
                )
                mentions_dataset["linking"][i]["prediction"] = selected_cand[0]
                mentions_dataset["linking"][i]["ed_score"] = round(selected_cand[1], 3)

        # Process output:
        keys = [
            "sent_idx",
            "mention",
            "pos",
            "end_pos",
            "tag",
            "prediction",
            "ed_score",
            "sentence",
        ]
        sentence_dataset = []
        for md in mentions_dataset["linking"]:
            md = dict((k, md[k]) for k in md if k in keys)
            sentence_dataset.append(md)
        return sentence_dataset

    def run_text(self, text, place_publication="", year=""):
        sentences = split_text_into_sentences(text, language="en")
        document_dataset = []
        for i in range(len(sentences)):
            sentence_dataset = self.run_sentence(sentences[i], sent_idx=i)
            for sd in sentence_dataset:
                document_dataset.append(sd)
        return document_dataset
