import os
import sys
from pathlib import Path

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import recogniser, ranking, linking
from utils import ner


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
                method="perfectmatch",
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
                    "selection_threshold": 1000,
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

    def run_text(self, text, place_publication="", year=""):
        predictions = self.myner.ner_predict(text)
        procpreds = [
            [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
            for x in predictions
        ]
        mentions = ner.aggregate_mentions(
            procpreds, "pred"
        )  # Perform candidate ranking:
        wk_cands, self.myranker.already_collected_cands = self.myranker.find_candidates(
            mentions
        )

        # Perform entity linking:
        linked_mentions = []
        for mention in mentions:
            linked_mention = dict()
            linked_mention["mention"] = mention["mention"]
            linked_mention["ner_label"] = mention["ner_label"]
            # Run entity linking per mention:
            selected_cand = self.mylinker.most_popular(
                {"candidates": wk_cands[mention["mention"]]}
            )
            linked_mention["wqid"] = selected_cand[0]
            linked_mention["wqid_score"] = selected_cand[1]
            linked_mentions.append(linked_mention)
        return linked_mentions

    def run_list(self, list_toponyms, place_publication="", year=""):
        mentions = []
        for toponym in list_toponyms:
            mentions.append(
                {
                    "mention": toponym,
                    "ner_label": "[given]",
                }
            )

        # Perform candidate ranking:
        wk_cands, self.myranker.already_collected_cands = self.myranker.find_candidates(
            mentions
        )

        # Perform entity linking:
        linked_mentions = []
        for mention in mentions:
            linked_mention = dict()
            linked_mention["mention"] = mention["mention"]
            linked_mention["ner_label"] = mention["ner_label"]
            # Run entity linking per mention:
            selected_cand = self.mylinker.most_popular(
                {"candidates": wk_cands[mention["mention"]]}
            )
            linked_mention["wqid"] = selected_cand[0]
            linked_mention["wqid_score"] = selected_cand[1]
            linked_mentions.append(linked_mention)
        return linked_mentions


# To run, from experiments folder:
"""
>>> from geoparser import pipeline
>>> geoparser = pipeline.Pipeline()
>>> resolved = geoparser.run_text("A remarkable case of rattening has just occurred in the building trade at Sheffield.")
>>> for t in resolved:
...     print(t)
... 
{'mention': 'Sheffield', 'ner_label': 'LOC', 'wqid': 'Q42448', 'wqid_score': 0.9060773480662984}
>>> resolved = geoparser.run_list(["Dudley", "Walsall", "Wednesbury", "Bilston", "West Bromwich", "Smethwick", "South Staffordshire"])
>>> for t in resolved:
...     print(t)
... 
{'mention': 'Dudley', 'ner_label': '[given]', 'wqid': 'Q213832', 'wqid_score': 0.7914252607184241}
{'mention': 'Walsall', 'ner_label': '[given]', 'wqid': 'Q504530', 'wqid_score': 0.9022118742724098}
{'mention': 'Wednesbury', 'ner_label': '[given]', 'wqid': 'Q501023', 'wqid_score': 0.992619926199262}
{'mention': 'Bilston', 'ner_label': '[given]', 'wqid': 'Q1015967', 'wqid_score': 0.946078431372549}
{'mention': 'West Bromwich', 'ner_label': '[given]', 'wqid': 'Q212826', 'wqid_score': 0.992619926199262}
{'mention': 'Smethwick', 'ner_label': '[given]', 'wqid': 'Q1018565', 'wqid_score': 0.9131652661064426}
{'mention': 'South Staffordshire', 'ner_label': '[given]', 'wqid': 'Q2116379', 'wqid_score': 0.855}
"""
