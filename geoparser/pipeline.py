import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

from sentence_splitter import split_text_into_sentences

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking, ranking, recogniser
from utils import ner, rel_utils


class Pipeline:
    """
    Represents a pipeline for processing a text using natural language
    processing, including Named Entity Recognition (NER), Ranking, and Linking,
    to geoparse any entities in the text.

    Arguments:
        myner (recogniser.Recogniser, optional): The NER (Named Entity
            Recogniser) object to use in the pipeline. If None, a default
            ``Recogniser`` will be instantiated. For the default settings, see
            Notes below.
        myranker (ranking.Ranker, optional): The ``Ranker`` object to use in
            the pipeline. If None, the default ``Ranker`` will be instantiated.
            For the default settings, see Notes below.
        mylinker (linking.Linker, optional): The ``Linker`` object to use in
            the pipeline. If None, the default ``Linker`` will be instantiated.
            For the default settings, see Notes below.

    Example:
        >>> # Instantiate the Pipeline object with a default setup
        >>> pipeline = Pipeline()

        >>> # Now you can use the pipeline for processing text or sentences
        >>> text = "I visited Paris and New York City last summer."
        >>> processed_data = pipeline.run_text(text)

        >>> # Access the processed mentions in the document
        >>> for mention in processed_data:
        >>>     print(mention)

    Note:
        * The default settings for the ``Recogniser``:

          .. code-block:: python

            recogniser.Recogniser(
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

        * The default settings for the ``Ranker``:

          .. code-block:: python

            ranking.Ranker(
                method="perfectmatch",
                resources_path="../resources/wikidata/",
                mentions_to_wikidata=dict(),
                wikidata_to_mentions=dict(),
                strvar_parameters={
                    "ocr_threshold": 60,
                    "top_threshold": 85,
                    "min_len": 5,
                    "max_len": 15,
                    "w2v_ocr_path": str(Path("../resources/models/w2v/").resolve()),
                    "w2v_ocr_model": "w2v_*_news",
                    "overwrite_dataset": False,
                },
                deezy_parameters={
                    "dm_path": str(Path("../resources/deezymatch/").resolve()),
                    "dm_cands": "wkdtalts",
                    "dm_model": "w2v_ocr",
                    "dm_output": "deezymatch_on_the_fly",
                    "ranking_metric": "faiss",
                    "selection_threshold": 25,
                    "num_candidates": 3,
                    "search_size": 3,
                    "verbose": False,
                    "overwrite_training": False,
                    "do_test": False,
                },
            )

        * The default settings for the ``Linker``:

          .. code-block:: python

            linking.Linker(
                method="mostpopular",
                resources_path="../resources/",
                linking_resources=dict(),
                rel_params={},
                overwrite_training=False,
            )
    """

    def __init__(
        self,
        myner: Optional[recogniser.Recogniser] = None,
        myranker: Optional[ranking.Ranker] = None,
        mylinker: Optional[linking.Linker] = None,
    ):
        """
        Instantiates a Pipeline object.
        """

        self.myner = myner
        self.myranker = myranker
        self.mylinker = mylinker

        # If myner is None, instantiate the default Recogniser.
        if not self.myner:
            dataset_path = "../experiments/outputs/data/lwm"
            self.myner = recogniser.Recogniser(
                model="blb_lwm-ner-fine",
                pipe=None,
                base_model="khosseini/bert_1760_1900",
                train_dataset=f"{dataset_path}/ner_fine_train.json",
                test_dataset=f"{dataset_path}/ner_fine_dev.json",
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

        # If myranker is None, instantiate the default Ranker.
        if not self.myranker:
            # Parameters to create the string pair dataset:
            ocr_path = str(Path("../resources/models/w2v/").resolve())
            strvar_parameters = {
                "ocr_threshold": 60,
                "top_threshold": 85,
                "min_len": 5,
                "max_len": 15,
                "w2v_ocr_path": ocr_path,
                "w2v_ocr_model": "w2v_*_news",
                "overwrite_dataset": False,
            }

            # Paths and filenames of DeezyMatch models and data:
            deezy_parameters = {
                "dm_path": str(Path("../resources/deezymatch/").resolve()),
                "dm_cands": "wkdtalts",
                "dm_model": "w2v_ocr",
                "dm_output": "deezymatch_on_the_fly",
                "ranking_metric": "faiss",  # Ranking measures
                "selection_threshold": 25,
                "num_candidates": 3,
                "search_size": 3,
                "verbose": False,
                "overwrite_training": False,  # DeezyMatch training
                "do_test": False,
            }
            self.myranker = ranking.Ranker(
                method="perfectmatch",
                resources_path="../resources/wikidata/",
                mentions_to_wikidata=dict(),
                wikidata_to_mentions=dict(),
                strvar_parameters=strvar_parameters,
                deezy_parameters=deezy_parameters,
            )

        # If mylinker is None, instantiate the default Linker.
        if not self.mylinker:
            self.mylinker = linking.Linker(
                method="mostpopular",
                resources_path="../resources/",
                linking_resources=dict(),
                rel_params={},
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

        # Train a linking model if needed (it requires myranker to generate
        # potential candidates to the training set):
        self.mylinker.rel_params["ed_model"] = self.mylinker.train_load_model(
            self.myranker
        )

    def run_sentence(
        self,
        sentence: str,
        sent_idx: Optional[int] = 0,
        context: Optional[Tuple[str, str]] = ("", ""),
        place: Optional[str] = "",
        place_wqid: Optional[str] = "",
        postprocess_output: Optional[bool] = True,
        without_microtoponyms: Optional[bool] = False,
    ) -> List[dict]:
        """
        Runs the pipeline on a single sentence.

        Arguments:
            sentence (str): The input sentence to process.
            sent_idx (int, optional): Index position of the target sentence in
                a larger text. Defaults to ``0``.
            context (tuple, optional): A tuple containing the previous and
                next sentences as context. Defaults to ``("", "")``.
            place (str, optional): The place of publication associated with
                the sentence as a human-legible string (e.g. "London").
                Defaults to ``""``.
            place_wqid (str, optional): The Wikidata ID of the place of
                publication provided in ``place`` (e.g. "Q84"). Defaults to
                ``""``.
            postprocess_output (bool, optional): Whether to postprocess the
                output, adding geographic coordinates. Defaults to ``True``.
            without_microtoponyms (bool, optional): Specifies whether to
                exclude microtoponyms during processing. Defaults to ``False``.

        Returns:
            List[dict]:
                A list of dictionaries representing the processed identified
                and linked toponyms in the sentence. Each dictionary contains
                the following keys:

                - ``sent_idx`` (int): The index of the sentence.
                - ``mention`` (str): The mention text.
                - ``pos`` (int): The starting position of the mention in the
                  sentence.
                - ``end_pos`` (int): The ending position of the mention in the
                  sentence.
                - ``tag`` (str): The NER label of the mention.
                - prediction`` (str): The predicted entity linking result.
                - ner_score`` (float): The NER score of the mention.
                - ed_score`` (float): The entity disambiguation score.
                - sentence`` (str): The input sentence.
                - prior_cand_score`` (dict): A dictionary of candidate
                  entities and their string matching confidence scores.
                - ``cross_cand_score`` (dict): A dictionary of candidate
                  entities and their cross-candidate confidence scores.

                If ``postprocess_output`` is set to True, the dictionaries
                will also contain the following two keys:

                - ``latlon`` (tuple): The latitude and longitude coordinates of
                  the predicted entity.
                - ``wkdt_class`` (str): The Wikidata class of the predicted
                  entity.

        Note:
            The ``run_sentence`` method processes a single sentence through the
            pipeline, performing tasks such as Named Entity Recognition (NER),
            ranking, and linking. It takes the input sentence along with
            optional parameters like the sentence index, context, the place of
            publication and its related Wikidata ID. By default, the method
            performs post-processing on the output.

            It first identifies toponyms in the sentence, then finds relevant
            candidates and ranks them, and finally links them to the Wikidata
            ID.
        """
        # Get predictions:
        predictions = self.myner.ner_predict(sentence)

        # Process predictions:
        procpreds = [
            [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
            for x in predictions
        ]

        # Aggregate mentions:
        mentions = ner.aggregate_mentions(procpreds, "pred")

        # List of mentions for the ranker:
        rmentions = []
        if without_microtoponyms:
            rmentions = [
                {"mention": y["mention"]} for y in mentions if y["ner_label"] == "LOC"
            ]
        else:
            rmentions = [{"mention": y["mention"]} for y in mentions]

        # Perform candidate ranking:
        wk_cands, self.myranker.already_collected_cands = self.myranker.find_candidates(
            rmentions
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

        # If the linking method is "reldisamb", rank and format candidates,
        # and produce a prediction:
        if self.mylinker.method == "reldisamb":
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
                mentions_dataset["linking"][i]["ed_score"] = round(
                    predicted["linking"][i]["conf_ed"], 3
                )

                # Get cross-candidate confidence scores per candidate:
                mentions_dataset["linking"][i]["cross_cand_score"] = {
                    cand: score
                    for cand, score in zip(
                        predicted["linking"][i]["candidates"],
                        predicted["linking"][i]["scores"],
                    )
                    if cand != "#UNK#"
                }

                # Sort candidates and round scores:
                mentions_dataset["linking"][i]["cross_cand_score"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        mentions_dataset["linking"][i]["cross_cand_score"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                # Get string matching confidence scores per candidate:
                mentions_dataset["linking"][i]["prior_cand_score"] = {
                    cand: score
                    for cand, score in mentions_dataset["linking"][i]["candidates"]
                    if cand in mentions_dataset["linking"][i]["cross_cand_score"]
                }

                # Sort candidates and round scores:
                mentions_dataset["linking"][i]["prior_cand_score"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        mentions_dataset["linking"][i]["prior_cand_score"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

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
                mentions_dataset["linking"][i]["prior_cand_score"] = dict()

                # Return candidates scores for top n=7 candidates
                # (same returned by REL):
                tmp_cands = {k: round(selected_cand[2][k], 3) for k in selected_cand[2]}
                mentions_dataset["linking"][i]["cross_cand_score"] = dict(
                    sorted(tmp_cands.items(), key=lambda x: x[1], reverse=True)[:7]
                )

        if not postprocess_output:
            return mentions_dataset

        if postprocess_output:
            # Process output, add coordinates and wikidata class from
            # prediction:
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
                "prior_cand_score",
                "cross_cand_score",
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

    def run_text(
        self,
        text: str,
        place: Optional[str] = "",
        place_wqid: Optional[str] = "",
        postprocess_output: Optional[bool] = True,
    ) -> List[dict]:
        """
        Runs the pipeline on a text document.

        Arguments:
            text (str): The input text document to process.
            place (str, optional): The place of publication associated with
                the text document as a human-legible string (e.g.
                ``"London"``). Defaults to ``""``.
            place_wqid (str, optional): The Wikidata ID of the place of
                publication provided in ``place`` (e.g. ``"Q84"``). Defaults
                to ``""``.
            postprocess_output (bool, optional): Whether to postprocess the
                output, adding geographic coordinates. Defaults to ``True``.

        Returns:
            List[dict]:
                A list of dictionaries representing the processed identified
                and linked toponyms in the sentence. Each dictionary contains
                the following keys:

                * "sent_idx" (int): The index of the sentence.
                * "mention" (str): The mention text.
                * "pos" (int): The starting position of the mention in the
                  sentence.
                * "end_pos" (int): The ending position of the mention in the
                  sentence.
                * "tag" (str): The NER label of the mention.
                * "prediction" (str): The predicted entity linking result.
                * "ner_score" (float): The NER score of the mention.
                * "ed_score" (float): The entity disambiguation score.
                * "sentence" (str): The input sentence.
                * "prior_cand_score" (dict): A dictionary of candidate
                  entities and their string matching confidence scores.
                * "cross_cand_score" (dict): A dictionary of candidate
                  entities and their cross-candidate confidence scores.

                If ``postprocess_output`` is set to True, the dictionaries
                will also contain the following two keys:

                * "latlon" (tuple): The latitude and longitude coordinates of
                  the predicted entity.
                * "wkdt_class" (str): The Wikidata class of the predicted
                  entity.

        Note:
            The ``run_text`` method processes an entire text through the
            pipeline, after splitting it into sentences, performing tasks such
            as Named Entity Recognition (NER), ranking, and linking. It takes
            the input text document along with optional parameters like the
            place of publication and its related Wikidata ID. By default, the
            method performs post-processing on the output.

            It first identifies toponyms in each of the text document's
            sentences, then finds relevant candidates and ranks them, and
            finally links them to the Wikidata ID.

            This method runs the
            :py:meth:`~geoparser.pipeline.Pipeline.run_sentence` method for
            each of the document's sentences. The ``without_microtoponyms``
            keyword, passed to ``run_sentence`` comes from the ``Linker``'s
            (passed when initialising the
            :py:meth:`~geoparser.pipeline.Pipeline` object) ``rel_params``
            parameter. See :py:class:`geoparser.linking.Linker` for
            instructions on how to set that up.

        """
        # Split the text into its sentences:
        sentences = split_text_into_sentences(text, language="en")

        document_dataset = []
        for idx, sentence in enumerate(sentences):
            # Get context (prev and next sentence)
            context = ["", ""]
            if idx - 1 >= 0:
                context[0] = sentences[idx - 1]
            if idx + 1 < len(sentences):
                context[1] = sentences[idx + 1]

            # Run pipeline on sentence:
            sentence_dataset = self.run_sentence(
                sentence,
                sent_idx=idx,
                context=context,
                place=place,
                place_wqid=place_wqid,
                postprocess_output=postprocess_output,
                without_microtoponyms=self.mylinker.rel_params.get(
                    "without_microtoponyms", False
                ),
            )

            # Collect results from all sentences:
            for sd in sentence_dataset:
                document_dataset.append(sd)

        return document_dataset
