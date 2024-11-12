import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from sentence_splitter import split_text_into_sentences

from ..utils import ner_utils, rel_utils
from . import linking, ranking, recogniser

class Pipeline:
    """
    Represents a pipeline for processing a text using natural language
    processing, including Named Entity Recognition (NER), Ranking, and Linking,
    to geoparse any entities in the text.

    Arguments:
        ner (recogniser.Recogniser, optional): The NER (Named Entity
            Recogniser) object to use in the pipeline. If None, a default
            ``Recogniser`` will be instantiated. For the default settings, see
            Notes below.
        ranker (ranking.Ranker, optional): The ``Ranker`` object to use in
            the pipeline. If None, the default ``Ranker`` will be instantiated.
            For the default settings, see Notes below.
        linker (linking.Linker, optional): The ``Linker`` object to use in
            the pipeline. If None, the default ``Linker`` will be instantiated.
            For the default settings, see Notes below.
        resources_path (str, optional): The path to your resources directory.
        experiments_path (str, optional): The path to the experiments directory.
            Default is "../experiments".

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

            recogniser.PretrainedRecogniser(
                model="Livingwithmachines/toponym-19thC-en",
            )

        * The default settings for the ``Ranker``:

          .. code-block:: python

            ranking.Ranker(
                method="perfectmatch",
                resources_path=resources_path,
            )

        * The default settings for the ``Linker``:

          .. code-block:: python

            linking.Linker(
                method="mostpopular",
                resources_path=resources_path,
            )
    """

    def __init__(
        self,
        ner: Optional[recogniser.Recogniser] = None,
        ranker: Optional[ranking.Ranker] = None,
        linker: Optional[linking.Linker] = None,
        resources_path: Optional[str] = None,
        experiments_path: Optional[str] = "../experiments",
    ):
        """
        Instantiates a Pipeline object.
        """

        self.ner = ner
        self.ranker = ranker
        self.linker = linker

        # If ner is None, instantiate the default Recogniser.
        if not self.ner:
            self.ner = recogniser.PretrainedRecogniser(
                model_name="Livingwithmachines/toponym-19thC-en",
            )

        # If ranker is None, instantiate the default Ranker.
        if not self.ranker:
            if not resources_path:
                raise ValueError("[ERROR] Please specify path to resources directory.")
            self.ranker = ranking.PerfectMatchRanker(
                resources_path=resources_path,
            )

        # If linker is None, instantiate the default Linker.
        if not self.linker:
            if not resources_path:
                raise ValueError("[ERROR] Please specify path to resources directory.")
            self.linker = linking.MostPopularLinker(
                resources_path=resources_path,
                experiments_path=experiments_path,
            )

        # -----------------------------------------
        # NER training and creating pipeline:
        self.ner.create_pipeline()

        # -----------------------------------------
        # Ranker loading resources and training a model:

        # Load the resources (and train a DeezyMatch model if needed):
        self.ranker.load_resources()

        # -----------------------------------------
        # Linker loading resources:

        # Load linking resources:
        self.linker.load_resources()

        # Train a linking model if needed (it requires ranker to generate
        # potential candidates to the training set):
        if self.linker.method_name() == "reldisamb":
            self.linker.rel_params["ed_model"] = self.linker.train_load_model(
                self.ranker
            )

    def run_sentence(
        self,
        sentence: str,
        sent_idx: Optional[int] = 0,
        context: Optional[Tuple[str, str]] = ("", ""),
        place: Optional[str] = "",
        place_wqid: Optional[str] = "",
        postprocess_output: Optional[bool] = True,
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

        mentions = self.run_sentence_recognition(sentence)

        # List of mentions for the ranker:
        rmentions = []
        without_microtoponyms = self.linker.method_name() == "reldisamb" \
            and self.linker.rel_params.get("without_microtoponyms", False)
        if without_microtoponyms:
            rmentions = [
                {"mention": y["mention"]} for y in mentions if y["ner_label"] == "LOC"
            ]
        else:
            rmentions = [{"mention": y["mention"]} for y in mentions]

        # Perform candidate ranking:
        # NOTE: `rmentions` simply takes the microtoponyms flag into account - apart from
        # that its the same as `mentions`. TODO: handle this in a `Mentions` dataclass.
        # NOTE: we only need the string mention part of the `mention` for the call 
        # to the ranker, but the rest of the data in the `mention` is needed in the 
        # call to format_prediction below.
        wk_cands = {d["mention"] : self.ranker.run(d["mention"]) for d in rmentions}

        print("wk_cands.keys():")
        print(wk_cands.keys())

        mentions_dataset = dict()
        mentions_dataset["linking"] = []
        for m in mentions:
            prediction = self.format_prediction(
                m,
                sentence,
                wk_cands=wk_cands.get(m["mention"], None),
                context=context,
                sent_idx=sent_idx,
                place=place,
                place_wqid=place_wqid,
            )
            mentions_dataset["linking"].append(prediction)

        # If the linking method is "reldisamb", rank and format candidates,
        # and produce a prediction:
        if self.linker.method_name() == "reldisamb":
            mentions_dataset = rel_utils.rank_candidates(
                mentions_dataset,
                wk_cands,
                self.linker.linking_resources["mentions_to_wikidata"],
            )

            if self.linker.rel_params["with_publication"]:
                if place_wqid == "" or place == "":
                    place_wqid = self.linker.rel_params["default_publwqid"]
                    place = self.linker.rel_params["default_publname"]

                # If "publ", add an artificial publication entry:
                mentions_dataset = rel_utils.add_publication(
                    mentions_dataset,
                    place,
                    place_wqid,
                )

            predicted = self.linker.rel_params["ed_model"].predict(mentions_dataset)

            if self.linker.rel_params["with_publication"]:
                # ... and if "publ", now remove the artificial publication entry!
                mentions_dataset["linking"].pop()

            # TODO: fix this brittle iteration with shared index i:
            for i in range(len(mentions_dataset["linking"])):
                mention = mentions_dataset["linking"][i]

                mention["prediction"] = predicted["linking"][i][
                    "prediction"
                ]
                mention["ed_score"] = round(
                    predicted["linking"][i]["conf_ed"], 3
                )

                # Get cross-candidate confidence scores per candidate:
                mention["cross_cand_score"] = {
                    cand: score
                    for cand, score in zip(
                        predicted["linking"][i]["candidates"],
                        predicted["linking"][i]["scores"],
                    )
                    if cand != "#UNK#"
                }

                # Sort candidates and round scores:
                mention["cross_cand_score"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        mention["cross_cand_score"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                # If there are no candidates for this mention, skip the rest.
                if not mention["mention"] in wk_cands.keys():
                    continue

                # Get string matching confidence scores per candidate:
                dCs = mention["string_match_candidates"]

                # TODO: some duplication here and below (but all this linking logic belongs elsewhere).
                mention["string_match_score"] = {
                    candidate_match.variation: (
                        round(candidate_match.string_similarity, 3),
                        [wqc for wqc in candidate_match.wikidata_matches],
                    )
                    for candidate_match in dCs.matches
                }

                # Get linking prior confidence scores per candidate:
                mention["prior_cand_score"] = {
                    cand: score
                    for cand, score in mention["candidates"]
                    if cand in mention["cross_cand_score"]
                }

                # Sort candidates and round scores:
                mention["prior_cand_score"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        mention["prior_cand_score"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

        if self.linker.method_name() in ["mostpopular", "bydistance"]:
            # TODO: fix this brittle iteration with shared index i:
            for i in range(len(mentions_dataset["linking"])):
                mention = mentions_dataset["linking"][i]

                # Run entity linking per mention:
                selected_cand = self.linker.run(
                    {
                        "candidates": wk_cands[mention["mention"]],
                        "place_wqid": place_wqid,
                    }
                )

                # If there are no candidates for this mention, skip the rest.
                if not mention["mention"] in wk_cands.keys():
                    continue

                mention["prediction"] = selected_cand.best_wqid()

                # Set the entity disambiguation score as the highest Wikidata relative frequency.
                # TODO: in general this needs to be handled inside the CandidateMatch dataclass,
                # e.g. with a method named `ed_score` that takes the linking method as an argument.
                mention["ed_score"] = round(selected_cand.best_match().relative_frequencies()[0], 3)

                dCs = mention["string_match_candidates"]
                mention["string_match_score"] = {
                    candidate_match.variation: (
                        round(candidate_match.string_similarity, 3),
                        [wqc for wqc in candidate_match.wikidata_matches],
                    )
                    for candidate_match in dCs.matches
                }

                mention["prior_cand_score"] = dict()

                # Return candidates scores for top n=7 candidates
                # (same returned by REL):
                # NOTE: these are the relative frequencies when linking is `most_popular`.
                # and tmp_cands is just used as an inbetween stop to populate `cross_cand_score`
                mention["cross_cand_score"] = selected_cand.best_match().cross_cand_score()

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
                "string_match_score",
                "prior_cand_score",
                "cross_cand_score",
            ]
            sentence_dataset = []
            for md in mentions_dataset["linking"]:

                md = dict((k, md[k]) for k in md if k in keys)
                md["latlon"] = self.linker.linking_resources["wqid_to_coords"].get(
                    md["prediction"]
                )
                md["wkdt_class"] = self.linker.linking_resources["entity2class"].get(
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
                A list of dictionaries representing the identified and linked
                toponyms in the sentence. Each dictionary contains the following
                keys:

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
            pipeline, after splitting it into sentences, performing the tasks
            of Named Entity Recognition (NER), ranking, and linking. It takes
            the input text document along with optional parameters like the
            place of publication and its related Wikidata ID and splits it
            into sentences. By default, the method performs post-processing
            on the output.

            It first identifies toponyms in each of the text document's
            sentences, then finds relevant candidates and ranks them, and
            finally links them to the Wikidata ID.

            This method runs the
            :py:meth:`~geoparser.pipeline.Pipeline.run_sentence` method for
            each of the document's sentences. Uses the ``without_microtoponyms``
            keyword from the ``Linker`` (assigned when initialising the
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
            )

            # Collect results from all sentences:
            for sd in sentence_dataset:
                document_dataset.append(sd)

        return document_dataset

    def run_sentence_recognition(self, sentence) -> List[dict]:
        # Get predictions:
        predictions = self.ner.ner_predict(sentence)

        # Process predictions:
        procpreds = [
            [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
            for x in predictions
        ]

        # Aggregate mentions:
        mentions = ner_utils.aggregate_mentions(procpreds, "pred")
        return mentions

    def format_prediction(
        self,
        mention,
        sentence: str,
        wk_cands: Optional[ranking.Candidates],
        context: Optional[Tuple[str, str]] = ("", ""),
        sent_idx: Optional[int] = 0,
        place: Optional[str] = "",
        place_wqid: Optional[str] = "",
    ) -> dict:
        prediction = dict()

        prediction["mention"] = mention["mention"]
        prediction["context"] = context
        prediction["candidates"] = []
        prediction["gold"] = ["NONE"]
        prediction["ner_score"] = mention["ner_score"]
        prediction["pos"] = mention["start_char"]
        prediction["sent_idx"] = sent_idx
        prediction["end_pos"] = mention["end_char"]
        prediction["ngram"] = mention["mention"]
        prediction["conf_md"] = mention["ner_score"]
        prediction["tag"] = mention["ner_label"]
        prediction["sentence"] = sentence
        prediction["place"] = place
        prediction["place_wqid"] = place_wqid
        if wk_cands:
            prediction["string_match_candidates"] = wk_cands
            prediction["candidates"] = wk_cands
        return prediction

    def run_text_recognition(
        self,
        text: str,
        place: Optional[str] = "",
        place_wqid: Optional[str] = "",
    ) -> List[dict]:
        """
        Runs the NER on a text document and returns the recognised entities
        in the format required by future steps: candidate selection and
        entity disambiguation.

        Arguments:
            text (str): The input text document to process.
            place (str, optional): The place of publication associated with
                the text document as a human-legible string (e.g.
                ``"London"``). Defaults to ``""``.
            place_wqid (str, optional): The Wikidata ID of the place of
                publication provided in ``place`` (e.g. ``"Q84"``). Defaults
                to ``""``.

        Returns:
            List[dict]:
                A list of dictionaries representing the identified toponyms
                in the sentence, in the format required by the following
                steps in the pipeline: candidate selection and entity
                disambiguation. Each dictionary contains the following keys:

                - ``mention`` (str): The mention text.
                - ``context`` (list): List of two strings corresponding to
                  the context (i.e. previous and next sentence).
                - ``candidates`` (list): List of candidates, which at this
                  point will be empty.
                - ``gold`` (list): List containing the gold standard entity,
                  which is and will remain ``['NONE']``.
                - ``ner_score`` (float): The NER score of the mention.
                - ``pos`` (int): The starting position of the mention in the
                  sentence.
                - ``sent_idx`` (int): The index of the sentence.
                - ``end_pos`` (int): The ending position of the mention in the
                  sentence.
                - ``ngram`` (str): The mention text (redundant).
                - ``conf_md`` (str): The NER score of the mention (redundant).
                - ``tag`` (str): The NER label of the mention.
                - ``prediction`` (str): The predicted entity linking result.
                - ``sentence`` (str): The input sentence.

        Note:
            The ``run_text_recognition`` method runs Named Entity Recognition
            (NER) on a full text, one sentence at a time. It takes the input text
            (along with optional parameters like the place of publication
            and its related Wikidata ID) and splits it into sentences, and
            after that finds mentions for each sentence.
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

            mentions = self.run_sentence_recognition(sentence)

            mentions_dataset = []
            for m in mentions:
                prediction = self.format_prediction(
                    m,
                    sentence,
                    wk_cands=None,
                    context=context,
                    sent_idx=idx,
                    place=place,
                    place_wqid=place_wqid,
                )
                # mentions_dataset["linking"].append(prediction)
                if not len(m["mention"]) == 1 and not m["mention"].islower():
                    mentions_dataset.append(prediction)

            # Collect results from all sentences:
            for sd in mentions_dataset:
                document_dataset.append(sd)

        return document_dataset

    def run_candidate_selection(self, document_dataset: List[dict]) -> dict:
        """
        Performs candidate selection on already identified toponyms,
        resulting from the ``run_text_recognition`` method. Given a
        list of dictionaries corresponding to mentions, this method
        first extracts the subset of mentions for which to try to find
        candidates and then runs the ``find_candidates`` function from
        the Ranker object. This method returns a dictionary of all
        mentions and their candidates, with a similarity score.

        Arguments:
            document_dataset (List[dict]): The list of mentions identified,
                formatted as dictionaries.

        Returns:
            dict:
                A three-level nested dictionary, as show in the example
                in the Note below. The outermost key is the mention as
                has been identified in the text, the first-level nested
                keys are candidate mentions found in Wikidata (i.e. potential
                matches for the original mention). The second-level nested
                keys are the match confidence score and the Wikidata entities
                that correspond to the candidate mentions, each with its
                associated normalised mention-to-wikidata relevance score.

        Note:

          .. code-block:: python

            {'Salop': {
                'Salop': {
                    'Score': 1.0,
                    'Candidates': {
                        'Q201970': 0.0006031363088057901,
                        'Q23103': 0.0075279261777561925
                        }
                    }
                }
            }

        """

        # Get without_microtoponyms value (whether to resolve microtoponyms or not):
        without_microtoponyms = self.linker.method_name() == "reldisamb" \
            and self.linker.rel_params.get("without_microtoponyms", False)

        # List of mentions for the ranker:
        rmentions = []
        if without_microtoponyms:
            rmentions = [y["mention"] for y in document_dataset if y["tag"] == "LOC"]
        else:
            rmentions = [y["mention"] for y in document_dataset]

        # Make list of mentions unique:
        mentions = list(set(rmentions))

        # # Prepare list of mentions as required by candidate selection and ranking:
        # mentions = [{"mention": m} for m in mentions]

        # Perform candidate ranking:
        wk_cands = [self.ranker.run(mention) for mention in mentions]
        return wk_cands

    def run_disambiguation(
        self,
        dataset,
        wk_cands, # TODO: udpate docstring: this is now List[Candidates]
        place: Optional[str] = "",
        place_wqid: Optional[str] = "",
    ):
        """
        Performs entity disambiguation given a list of already identified
        toponyms and selected candidates.

        Arguments:
            dataset (List[dict]): The list of mentions identified,
                formatted as dictionaries.
            wk_cands (dict): A three-level nested dictionary mapping
                mentions to potential Wikidata entities.
            place (str, optional): The place of publication associated with
                the text document as a human-legible string (e.g.
                ``"London"``). Defaults to ``""``.
            place_wqid (str, optional): The Wikidata ID of the place of
                publication provided in ``place`` (e.g. ``"Q84"``). Defaults
                to ``""``.

        Returns:
            List[dict]:
                A list of dictionaries representing the identified and linked
                toponyms in the sentence. Each dictionary contains the following
                keys:

                * "mention" (str): The mention text.
                * "ner_score" (float): The NER score of the mention.
                * "pos" (int): The starting position of the mention in the
                  sentence.
                * "sent_idx" (int): The index of the sentence.
                * "end_pos" (int): The ending position of the mention in the
                  sentence.
                * "tag" (str): The NER label of the mention.
                * "sentence" (str): The input sentence.
                * "prediction" (str): The predicted entity linking result.
                * "ed_score" (float): The entity disambiguation score.
                * "string_match_score" (dict): A dictionary of candidate
                  entities and their string matching confidence scores.
                * "prior_cand_score" (dict): A dictionary of candidate
                  entities and their prior confidence scores.
                * "cross_cand_score" (dict): A dictionary of candidate
                  entities and their cross-candidate confidence scores.
                * "latlon" (tuple): The latitude and longitude coordinates of
                  the predicted entity.
                * "wkdt_class" (str): The Wikidata class of the predicted
                  entity.
        """

        mentions_dataset = dict()
        mentions_dataset["linking"] = []
        for prediction in dataset:

            # TODO: Temp fix (pending dataclass for sentence candidates collection, 
            # or better just pass in the right candidates): 
            prediction["candidates"] = wk_cands[prediction["mention"]]

            prediction["string_match_candidates"] = prediction["candidates"]
            mentions_dataset["linking"].append(prediction)

        # TODO: Duplicated code from run_sentence (above).

        # If the linking method is "reldisamb", rank and format candidates,
        # and produce a prediction:
        if self.linker.method_name() == "reldisamb":
            mentions_dataset = rel_utils.rank_candidates(
                mentions_dataset,
                wk_cands,
                self.linker.linking_resources["mentions_to_wikidata"],
            )

            if self.linker.rel_params["with_publication"]:
                if place_wqid == "" or place == "":
                    place_wqid = self.linker.rel_params["default_publwqid"]
                    place = self.linker.rel_params["default_publname"]

                # If "publ", add an artificial publication entry:
                mentions_dataset = rel_utils.add_publication(
                    mentions_dataset,
                    place,
                    place_wqid,
                )

            predicted = self.linker.rel_params["ed_model"].predict(mentions_dataset)

            if self.linker.rel_params["with_publication"]:
                # ... and if "publ", now remove the artificial publication entry!
                mentions_dataset["linking"].pop()

            for i in range(len(mentions_dataset["linking"])):
                mention = mentions_dataset["linking"][i]

                mention["prediction"] = predicted["linking"][i][
                    "prediction"
                ]
                mention["ed_score"] = round(
                    predicted["linking"][i]["conf_ed"], 3
                )

                # Get cross-candidate confidence scores per candidate:
                mention["cross_cand_score"] = {
                    cand: score
                    for cand, score in zip(
                        predicted["linking"][i]["candidates"],
                        predicted["linking"][i]["scores"],
                    )
                    if cand != "#UNK#"
                }

                # Sort candidates and round scores:
                mention["cross_cand_score"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        mention["cross_cand_score"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                # Get string matching confidence scores per candidate:
                dCs = mention["string_match_candidates"]

                mention["string_match_score"] = {
                    candidate_match.variation: (
                        round(candidate_match.string_similarity, 3),
                        [wqc for wqc in candidate_match.wikidata_matches],
                    )
                    for candidate_match in dCs.matches
                }

                # Get linking prior confidence scores per candidate:
                mention["prior_cand_score"] = {
                    cand: score
                    for cand, score in mention["candidates"]
                    if cand in mention["cross_cand_score"]
                }

                # Sort candidates and round scores:
                mention["prior_cand_score"] = {
                    k: round(v, 3)
                    for k, v in sorted(
                        mention["prior_cand_score"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

        if self.linker.method_name() in ["mostpopular", "bydistance"]:
            for i in range(len(mentions_dataset["linking"])):
                mention = mentions_dataset["linking"][i]

                # Run entity linking per mention:
                selected_cand = self.linker.run(
                    {
                        "candidates": wk_cands[mention["mention"]],
                        "place_wqid": place_wqid,
                    }
                )

                mention["prediction"] = selected_cand.best_wqid()

                # Set the entity disambiguation score as the highest Wikidata relative frequency.
                # TODO: in general this needs to be handled inside the CandidateMatch dataclass,
                # e.g. with a method named `ed_score` that takes the linking method as an argument.
                mention["ed_score"] = round(selected_cand.best_match().relative_frequencies()[0], 3)

                dCs = mention["string_match_candidates"]

                mention["string_match_score"] = {
                    candidate_match.variation: (
                        round(candidate_match.string_similarity, 3),
                        [wqc for wqc in candidate_match.wikidata_matches],
                    )
                    for candidate_match in dCs.matches
                }

                mention["prior_cand_score"] = dict()

                # Return candidates scores for top n=7 candidates
                mention["cross_cand_score"] = selected_cand.best_match().cross_cand_score()

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
            "string_match_score",
            "prior_cand_score",
            "cross_cand_score",
        ]
        sentence_dataset = []
        for md in mentions_dataset["linking"]:
            md = dict((k, md[k]) for k in md if k in keys)
            md["latlon"] = self.linker.linking_resources["wqid_to_coords"].get(
                md["prediction"]
            )
            md["wkdt_class"] = self.linker.linking_resources["entity2class"].get(
                md["prediction"]
            )
            sentence_dataset.append(md)
        return sentence_dataset
