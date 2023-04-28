import os
import sys
from pathlib import Path
import sqlite3

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
        Instantiates a Pipeline object.

        Arguments:
            myner (recogniser.Recogniser): a Recogniser object.
            myranker (ranking.Ranker): a Ranker object.
            mylinker (linking.Linker): a Linker object.
        """

        self.myner = myner
        self.myranker = myranker
        self.mylinker = mylinker

        if not self.myner:
            # If myner is None, instantiate the default Recogniser.
            self.myner = recogniser.Recogniser(
                model="blb_lwm-ner-fine",
                pipe=None,
                base_model="khosseini/bert_1760_1900",
                train_dataset="../experiments/outputs/data/lwm/ner_fine_train.json",
                test_dataset="../experiments/outputs/data/lwm/ner_fine_dev.json",
                model_path="../resources/models/",
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

        if not self.myranker:
            # If myranker is None, instantiate the default Ranker.
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
            # If mylinker is None, instantiate the default Linker.
            with sqlite3.connect("../resources/rel_db/embedding_database.db") as conn:
                cursor = conn.cursor()
                self.mylinker = linking.Linker(
                    method="mostpopular",
                    resources_path="../resources/",
                    linking_resources=dict(),
                    rel_params={
                        "model_path": "../resources/models/disambiguation/",
                        "data_path": "../experiments/outputs/data/lwm/",
                        "training_split": "originalsplit",
                        "context_length": 100,
                        "db_embeddings": cursor,
                        "with_publication": True,
                        "without_microtoponyms": True,
                        "do_test": False,
                        "default_publname": "United Kingdom",
                        "default_publwqid": "Q145",
                    },
                    overwrite_training=False,
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
        self.mylinker.rel_params["ed_model"] = self.mylinker.train_load_model(self.myranker)

    def run_sentence(
        self,
        sentence,
        sent_idx=0,
        context=("", ""),
        place="",
        place_wqid="",
        postprocess_output=True,
        without_microtoponyms=False,
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

        Returns either (depending on the value of `postprocess_output`):
            mentions_dataset (list): a list of dictionaries, where each dictionary
                is a resolved entity, without the coordinates, or
            sentence_dataset (list): a list of dictionaries, where each dictionary
                is a resolved entity, with the coordinates and type of location.
        """
        # Get predictions:
        predictions = self.myner.ner_predict(sentence)
        # Process predictions:
        procpreds = [
            [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]] for x in predictions
        ]
        # Aggretate mentions:
        mentions = ner.aggregate_mentions(procpreds, "pred")

        # List of mentions for the ranker:
        rmentions = []
        if without_microtoponyms:
            rmentions = [{"mention": y["mention"]} for y in mentions if y["ner_label"] == "LOC"]
        else:
            rmentions = [{"mention": y["mention"]} for y in mentions]

        # Perform candidate ranking:
        wk_cands, self.myranker.already_collected_cands = self.myranker.find_candidates(rmentions)

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
            # If the linking method is "reldisamb", rank and format candidates,
            # and produce a prediction:
            mentions_dataset = rel_utils.rank_candidates(
                mentions_dataset,
                wk_cands,
                self.mylinker.linking_resources["mentions_to_wikidata"],
            )
            if self.mylinker.rel_params["with_publication"]:
                if place_wqid == "" or place == "":
                    place_wqid = self.mylinker.rel_params["default_publwqid"]
                    place = self.mylinker.rel_params["default_publname"]
                # If "publ", add an artificial publication entry:
                mentions_dataset = rel_utils.add_publication(
                    mentions_dataset,
                    place,
                    place_wqid,
                )
            predicted = self.mylinker.rel_params["ed_model"].predict(mentions_dataset)
            if self.mylinker.rel_params["with_publication"]:
                # ... and if "publ", now remove the artificial publication entry!
                mentions_dataset["linking"].pop()
            for i in range(len(mentions_dataset["linking"])):
                mentions_dataset["linking"][i]["prediction"] = predicted["linking"][i][
                    "prediction"
                ]
                mentions_dataset["linking"][i]["ed_score"] = predicted["linking"][i]["conf_ed"]

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
            # Process output, add coordinates and wikidata class from prediction:
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

        Arguments:
            text (str): The text to geoparse.
            place (str): The place of publication (in text).
            place_wqid (str): The Wikidata ID of the place of publication.

        Returns:
            document_dataset (list): list of geoparsed sentences.
        """
        # Split the text into its sentences:
        sentences = split_text_into_sentences(text, language="en")
        document_dataset = []
        for i in range(len(sentences)):
            # Get context (prev and next sentence)
            context = ["", ""]
            if i - 1 >= 0:
                context[0] = sentences[i - 1]
            if i + 1 < len(sentences):
                context[1] = sentences[i + 1]
            # Run pipeline on sentence:
            sentence_dataset = self.run_sentence(
                sentences[i],
                sent_idx=i,
                context=context,
                place=place,
                place_wqid=place_wqid,
                postprocess_output=postprocess_output,
                without_microtoponyms=self.mylinker.rel_params.get("without_microtoponyms", False),
            )
            # Collect results from all sentences:
            for sd in sentence_dataset:
                document_dataset.append(sd)
        return document_dataset
