import json
import os
import sys
from ast import literal_eval
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from typing import Optional, Any, Tuple, List, TYPE_CHECKING, Literal

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import ner

if TYPE_CHECKING:
    from geoparser import recogniser
    from experiments import experiment


def eval_with_exception(str2parse: str, in_case: Optional[Any] = "") -> Any:
    """
    Evaluate a string expression using :py:func:`ast.literal_eval`. If
    the evaluation succeeds, the result is returned. If a ``ValueError``
    occurs during evaluation, the provided ``in_case`` value is returned
    instead.

    Arguments:
        str2parse (str): The string expression to be evaluated.
        in_case (Any, optional): The value to return in case of a
            ``ValueError``. Defaults to ``""``.

    Returns:
        Any:
            The evaluated result if successful, or the ``in_case`` value if an
            error occurs.

    Example:
        >>> eval_with_exception("2 + 2")
        4
        >>> eval_with_exception("hello")
        ''
        >>> eval_with_exception("[1, 2, 3]", [])
        [1, 2, 3]
    """
    try:
        return literal_eval(str2parse)
    except ValueError:
        return in_case


def prepare_sents(df: pd.DataFrame) -> Tuple[dict, dict, dict]:
    """
    Prepares annotated data and metadata on a sentence basis.

    Returns:
        Tuple[dict, dict, dict]: A tuple consisting of three dictionaries:

            #. ``dSentences``: A dictionary in which we keep, for each article/
               sentence (expressed as e.g. ``"10732214_1"``, where
               ``"10732214"`` is the article_id and ``"1"`` is the order of
               the sentence in the article), the full original unprocessed
               sentence.
            #. ``dAnnotated``: A dictionary in which we keep, for each article/
               sentence, an inner dictionary mapping the position of an
               annotated named entity (i.e. its start and end character, as a
               tuple, as the key) and another tuple as its value, which
               consists of: the type of named entity (such as ``LOC`` or
               ``BUILDING``, the mention, and its annotated link), all
               extracted from the gold standard.
            #. ``dMetadata``: A dictionary in which we keep, for each article/
               sentence, its metadata: ``place`` (of publication), ``year``,
               ``ocr_quality_mean``, ``ocr_quality_sd``, ``publication_title``,
               ``publication_code``, and ``place_wqid`` (Wikidata ID of the
               place of publication).
    """

    dSentences = dict()
    dAnnotated = dict()
    dMetadata = dict()
    for _, row in df.iterrows():
        sentences = eval_with_exception(row["sentences"], [])
        annotations = eval_with_exception(row["annotations"], [])

        for s in sentences:
            # Sentence position:
            s_pos = s["sentence_pos"]
            # Article-sentence pair unique identifier:
            artsent_id = str(row["article_id"]) + "_" + str(s_pos)
            # Sentence text:
            dSentences[artsent_id] = s["sentence_text"]
            # Annotations in NER-required format:
            for a in annotations:
                if a["sent_pos"] == s_pos:
                    position = (int(a["mention_start"]), int(a["mention_end"]))
                    wqlink = a["wkdt_qid"]
                    if not isinstance(wqlink, str):
                        wqlink = "NIL"
                    elif wqlink == "*":
                        wqlink = "NIL"
                    if artsent_id in dAnnotated:
                        dAnnotated[artsent_id][position] = (
                            a["entity_type"],
                            a["mention"],
                            wqlink,
                        )
                    else:
                        dAnnotated[artsent_id] = {
                            position: (a["entity_type"], a["mention"], wqlink)
                        }

            # Keep metadata:
            dMetadata[artsent_id] = dict()
            dMetadata[artsent_id]["place"] = row["place"]
            dMetadata[artsent_id]["year"] = row["year"]
            dMetadata[artsent_id]["ocr_quality_mean"] = row["ocr_quality_mean"]
            dMetadata[artsent_id]["ocr_quality_sd"] = row["ocr_quality_sd"]
            dMetadata[artsent_id]["publication_title"] = row["publication_title"]
            dMetadata[artsent_id]["publication_code"] = row["publication_code"]
            dMetadata[artsent_id]["place_wqid"] = row["place_wqid"]

    # Now add also an empty annotations dictionary where no mentions have been
    # annotated:
    for artsent_id in dSentences:
        if not artsent_id in dAnnotated:
            dAnnotated[artsent_id] = dict()

    # Now add also the metadata of sentences where no mentions have been
    # annotated:
    for artsent_id in dSentences:
        if not artsent_id in dMetadata:
            dMetadata[artsent_id] = dict()

    return dAnnotated, dSentences, dMetadata


def align_gold(predictions: List[dict], annotations: dict) -> List[dict]:
    """
    Aligns the predictions of a BERT NER model with the gold standard labels
    by aligning their tokenization.

    The gold standard tokenization is not aligned with the tokenization
    produced by the BERT model, as it uses its own tokenizer.

    This function aligns the two tokenizations based on the start and end
    positions of each predicted token.

    Predicted tokens are assigned the ``"O"`` label by default unless their
    position overlaps with an annotated entity, in which case they are
    relabeled according to the corresponding gold standard label.

    Arguments:
        predictions (List[dict]): A list of dictionaries representing the
            predicted mentions. Each dictionary contains the following keys:

            - ``start`` (int): The start position of the predicted token.
            - ``end`` (int): The end position of the predicted token.
            - ``entity`` (str): The predicted entity label (initially set to
              ``"O"``).
            - ``link`` (str): The predicted entity link (initially set to
              ``"O"``).
        annotations (dict): A dictionary where the keys are tuples
            representing the start and end positions of gold standard
            detections in a sentence, and the values are tuples containing the
            label type (e.g. ``"LOC"``), mention (e.g. ``"Point Petre"``), and
            link of the corresponding gold standard entity (e.g.
            ``"Q335322"``).

    Returns:
        List[dict]:
            A list of dictionaries representing the aligned gold standard
            labels. Each dictionary contains the same keys as the predictions:

            - ``start`` (int): The start position of the aligned token.
            - ``end`` (int): The end position of the aligned token.
            - ``entity`` (str): The aligned entity label.
            - ``link`` (str): The aligned entity link.
            - ``score`` (float): The score for the aligned entity (set to
              ``1.0`` as it is manually annotated).
    """

    gold_standard = []
    for pred_ent in predictions:
        gs_for_eval = pred_ent.copy()
        # This has been manually annotated, so perfect score
        gs_for_eval["score"] = 1.0
        # We instantiate the entity class as "O" ("outside", i.e. not a NE)
        gs_for_eval["entity"] = "O"
        gs_for_eval["link"] = "O"
        # It's prefixed as "B-" if the token is the first in a sequence,
        # otherwise it's prefixed as "I-"
        for gse in annotations:
            if pred_ent["start"] == gse[0] and pred_ent["end"] <= gse[1]:
                gs_for_eval["entity"] = "B-" + annotations[gse][0].upper()
                gs_for_eval["link"] = "B-" + annotations[gse][2]
            elif pred_ent["start"] > gse[0] and pred_ent["end"] <= gse[1]:
                gs_for_eval["entity"] = "I-" + annotations[gse][0].upper()
                gs_for_eval["link"] = "I-" + annotations[gse][2]
        gold_standard.append(gs_for_eval)

    return gold_standard


def postprocess_predictions(predictions: List[dict], gold_positions) -> dict:
    """
    Postprocess predictions to be used later in the pipeline.

    Arguments:
        predictions (list): the output of the
            :py:meth:`geoparser.recogniser.Recogniser.ner_predict` method,
            where, given a sentence, a list of dictionaries is returned, where
            each dictionary corresponds to a recognised token, e.g.:

            .. code-block:: json

                {
                    "entity": "O",
                    "score": 0.99975187,
                    "word": "From",
                    "start": 0,
                    "end": 4
                }

        gold_positions (list): the output of the
            :py:func:`utils.process_data.align_gold` function, which
            aligns the gold standard text to the tokenisation performed by the
            named entity recogniser, to enable assessing the performance of
            the NER and linking steps.

    Returns:
        dict: A dictionary with three key-value pairs:

            #. ``sentence_preds`` is mapped to the list of lists
               representation of ``predictions``,
            #. ``sentence_trues`` is mapped to the list of lists
               representation of 'gold_positions', and
            #. ``sentence_skys`` is the same as ``sentence_trues``, but with
               empty link.
    """
    postprocessed_sentence = dict()
    sentence_preds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], float(x["score"])]
        for x in predictions
    ]
    sentence_trues = [
        [x["word"], x["entity"], x["link"], x["start"], x["end"]]
        for x in gold_positions
    ]
    sentence_skys = [
        [x["word"], x["entity"], "O", x["start"], x["end"]] for x in predictions
    ]

    postprocessed_sentence["sentence_preds"] = sentence_preds
    postprocessed_sentence["sentence_trues"] = sentence_trues
    postprocessed_sentence["sentence_skys"] = sentence_skys

    return postprocessed_sentence


# TODO/typing: set ``myner: recogniser.Recogniser`` here, but creates problem with Sphinx currently
def ner_and_process(
    dSentences: dict, dAnnotated: dict, myner
) -> Tuple[dict, dict, dict, dict, dict]:
    """
    Perform named entity recognition in the LwM way, and postprocess the
    output to prepare it for the experiments.

    Arguments:
        dSentences (dict): dictionary in which we keep, for each article/
            sentence (expressed as e.g. ``"10732214_1"``, where ``"10732214"``
            is the article_id and ``"1"`` is the order of the sentence in the
            article), the full original unprocessed sentence.
        dAnnotated (dict): dictionary in which we keep, for each article/
            sentence, an inner dictionary mapping the position of an annotated
            named entity (i.e. its start and end character, as a tuple, as the
            key) and another tuple as its value, which consists of: the type
            of named entity (such as ``LOC`` or ``BUILDING``, the mention, and
            its annotated link), all extracted from the gold standard.
        myner (recogniser.Recogniser): a Recogniser object, for NER.

    Returns:
        Tuple[dict, dict, dict, dict, dict]:
            A tuple consisting of five dictionaries:

            #. **dPreds**: A dictionary where the NER predictions are stored,
               where the key is the sentence_id (i.e. ``article_id + "_" +
               sentence_pos``) and the value is a list of lists, where each
               element corresponds to one token in a sentence, for example:

               .. code-block:: json

                   ["From", "O", "O", 0, 4, 0.999826967716217]

               ...where the the elements by their position are:

               #. the token,
               #. the NER tag,
               #. the link to wikidata, set to ``"O"`` for now because we haven't
                  performed linking yet,
               #. the starting character of the token,
               #. the end character of the token, and
               #. the NER prediction score.

               This dictionary is stored as a JSON file in the ``outputs/data``
               folder, with the suffix ``_ner_predictions.json``.

            #. **dTrues**: A dictionary where the gold standard named entities
               are stored, which has the same format as **dPreds** above, but
               with the manually annotated data instead of the predictions.

               This dictionary is stored as a JSON file in the ``outputs/data``
               folder, with the suffix ``_gold_standard.json``.

            #. **dSkys**: A dictionary where the skyline will be stored, for
               the linking experiments. At this point, it will be the same as
               **dPreds**, without the NER prediction score. During linking, it
               will be filled with the gold standard entities when these have
               been retrieved using candidates.

               This dictionary is stored as a JSON file in the ``outputs/data``
               folder, with the suffix ``_ner_skyline.json``.

            #. **gold_tokenization**: A dictionary where the gold standard
               entities are stored, and keys represent ``sentence_id`` (i.e.
               ``article_id + "_" + sentence_pos``) and the values are lists of
               dictionaries, each looking like this:

               .. code-block:: json

                   {
                       "entity": "B-LOC",
                       "score": 1.0,
                       "word": "Unitec",
                       "start": 193,
                       "end": 199,
                       "link": "B-Q30"
                   }

               This dictionary is stored as a JSON file in the ``outputs/data``
               folder, with the suffix ``_gold_positions.json``.

            #. **dMentionsPred**: A dictionary of detected mentions but not
               yet linked mentions, for example:

               .. code-block:: json

                   {
                       "sn83030483-1790-03-03-a-i0001_9": [
                           {
                               "mention": "Unitec ? States",
                               "start_offset": 38,
                               "end_offset": 40,
                               "start_char": 193,
                               "end_char": 206,
                               "ner_score": 0.79,
                               "ner_label": "LOC",
                               "entity_link": "O"
                           }
                       ],
                   }

               This dictionary is stored as a JSON file in the ``outputs/data``
               folder, with the suffix ``_pred_mentions.json``.

            #. **dMentionsGold**: A dictionary consisting of gold standard
               mentions, analogous to the dictionary of detected mentions, but
               with the gold standard ``ner_label`` and ``entity_link``.
    """
    gold_tokenization = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()
    dMentionsPred = dict()  # Dictionary of detected mentions
    dMentionsGold = dict()  # Dictionary of gold standard mentions
    for sent_id in tqdm(list(dSentences.keys())):
        sent = dSentences[sent_id]
        annotations = dAnnotated[sent_id]
        predictions = myner.ner_predict(sent)
        gold_positions = align_gold(predictions, annotations)
        sentence_postprocessing = postprocess_predictions(predictions, gold_positions)
        dPreds[sent_id] = sentence_postprocessing["sentence_preds"]
        dTrues[sent_id] = sentence_postprocessing["sentence_trues"]
        dSkys[sent_id] = sentence_postprocessing["sentence_skys"]
        gold_tokenization[sent_id] = gold_positions
        dMentionsPred[sent_id] = ner.aggregate_mentions(
            sentence_postprocessing["sentence_preds"], "pred"
        )
        dMentionsGold[sent_id] = ner.aggregate_mentions(
            sentence_postprocessing["sentence_trues"], "gold"
        )

    return dPreds, dTrues, dSkys, gold_tokenization, dMentionsPred, dMentionsGold


def update_with_linking(ner_predictions: dict, link_predictions: pd.Series) -> dict:
    """
    Updates the NER predictions by incorporating linking results.

    Arguments:
        ner_predictions (dict): A dictionary containing NER predictions
            (token-per-token) for a given sentence.
        link_predictions (pd.Series): A pandas series corresponding to one row
            of the test_df, representing one mention.

    Returns:
        dict:
            A dictionary similar to `ner_predictions`, but with the added
            Wikidata link for each predicted entity.
    """
    resulting_preds = ner_predictions
    link_predictions = link_predictions.to_dict(orient="index")
    for lp in link_predictions:
        for x in range(
            link_predictions[lp]["token_start"], link_predictions[lp]["token_end"] + 1
        ):
            position_ner = resulting_preds[x][1][:2]
            resulting_preds[x][2] = position_ner + link_predictions[lp]["pred_wqid"]
    return resulting_preds


def update_with_skyline(ner_predictions: dict, link_predictions: pd.Series) -> dict:
    """
    Updates the NER predictions with the skyline link from entity linking.

    Arguments:
        ner_predictions (dict): A dictionary containing NER predictions
            (token-per-token) for a given sentence.
        link_predictions (pd.Series): A pandas series corresponding to one row
            of the test_df, representing one mention.

    Returns:
        dict:
            A dictionary similar to ``ner_predictions``, but with the added
            skyline link to Wikidata. The skyline link represents the gold
            standard candidate if it has been retrieved through candidate
            ranking, otherwise it is set to ``"O"``.
    """
    resulting_preds = ner_predictions
    link_predictions = link_predictions.to_dict(orient="index")
    for lp in link_predictions:
        all_candidates = [
            list(link_predictions[lp]["candidates"][x]["Candidates"].keys())
            for x in link_predictions[lp]["candidates"]
        ]
        all_candidates = [item for sublist in all_candidates for item in sublist]
        for x in range(
            link_predictions[lp]["token_start"], link_predictions[lp]["token_end"] + 1
        ):
            position_ner = resulting_preds[x][1][:2]
            if link_predictions[lp]["gold_entity_link"] in all_candidates:
                resulting_preds[x][2] = (
                    position_ner + link_predictions[lp]["gold_entity_link"]
                )
            else:
                resulting_preds[x][2] = "O"
    return resulting_preds


def prepare_storing_links(
    processed_data: dict, all_test: List[str], test_df: pd.DataFrame
) -> dict:
    """
    Updates the processed data dictionaries with predicted links for "preds"
    and the skyline for "skys". The skyline represents the maximum achievable
    result by choosing the gold standard entity among the candidates.

    Arguments:
        processed_data (dict): A dictionary containing all processed data.
        all_test (List[str]): A list of article IDs in the current data split
            used for testing.
        test_df (pd.DataFrame): A DataFrame with one mention per row that will
            be used for testing in the current experiment.

    Returns:
        dict
            The updated processed data dictionary with "preds" and "skys"
            incorporating the predicted links and skyline information.
    """
    for sent_id in processed_data["preds"]:
        article_id = sent_id.split("_")[0]
        # First: is sentence in the current dev/test set?
        if article_id in all_test:
            # Update predictions with linking:
            # >> Step 1. If there is no mention in the sentence, it will
            #            not be in the per-mention dataframe. However,
            #            it should still be in the results file.
            if not sent_id in test_df["sentence_id"].unique():
                processed_data["preds"][sent_id] = processed_data["preds"][sent_id]
            if not sent_id in test_df["sentence_id"].unique():
                processed_data["skys"][sent_id] = processed_data["skys"][sent_id]
            # >> Step 2: If there is a mention in the sentence, update the link:
            else:
                processed_data["preds"][sent_id] = update_with_linking(
                    processed_data["preds"][sent_id],  # NER predictions
                    test_df[test_df["sentence_id"] == sent_id],  # Processed df
                )
                processed_data["skys"][sent_id] = update_with_skyline(
                    processed_data["skys"][sent_id],  # NER predictions
                    test_df[test_df["sentence_id"] == sent_id],  # Processed df
                )
    return processed_data


def load_processed_data(experiment: experiment.Experiment) -> dict:
    """
    Loads the data already processed in a previous run of the code, using
    the same parameters.

    Arguments:
        experiment (experiment.Experiment): an Experiment object.

    Returns:
        dict: A dictionary where the processed data is stored.
    """

    output_path = os.path.join(
        experiment.data_path, experiment.dataset, experiment.myner.model
    )

    # Add the candidate experiment info to the path:
    cand_approach = experiment.myranker.method
    if experiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["selection_threshold"]
        )

    output_processed_data = dict()
    try:
        with open(output_path + "_ner_predictions.json") as fr:
            output_processed_data["preds"] = json.load(fr)
        with open(output_path + "_gold_standard.json") as fr:
            output_processed_data["trues"] = json.load(fr)
        with open(output_path + "_ner_skyline.json") as fr:
            output_processed_data["skys"] = json.load(fr)
        with open(output_path + "_gold_positions.json") as fr:
            output_processed_data["gold_tok"] = json.load(fr)
        with open(output_path + "_dict_sentences.json") as fr:
            output_processed_data["dSentences"] = json.load(fr)
        with open(output_path + "_dict_metadata.json") as fr:
            output_processed_data["dMetadata"] = json.load(fr)
        with open(output_path + "_pred_mentions.json") as fr:
            output_processed_data["dMentionsPred"] = json.load(fr)
        with open(output_path + "_gold_mentions.json") as fr:
            output_processed_data["dMentionsGold"] = json.load(fr)
        with open(output_path + "_candidates_" + cand_approach + ".json") as fr:
            output_processed_data["dCandidates"] = json.load(fr)

        return output_processed_data
    except FileNotFoundError:
        print("File not found, process data.")
        return dict()


def store_processed_data(
    experiment: experiment.Experiment,
    preds: dict,
    trues: dict,
    skys: dict,
    gold_tok: dict,
    dSentences: dict,
    dMetadata: dict,
    dMentionsPred: dict,
    dMentionsGold: dict,
    dCandidates: dict,
) -> dict:
    """
    Stores all the postprocessed data as JSON files and returns a dictionary
    containing all processed data.

    Arguments:
        experiment (experiment.Experiment): An experiment object.
        preds (dict): A dictionary of tokens with predictions per sentence.
        trues (dict): A dictionary of tokens with gold standard annotations per
            sentence.
        skys (dict): A dictionary of tokens representing the skyline per
            sentence.
        gold_tok (dict): A dictionary of tokens with gold standard annotations
            as dictionaries per sentence.
        dSentences (dict): A dictionary mapping a sentence ID to the
            corresponding text.
        dMetadata (dict): A dictionary mapping a sentence ID to associated
            metadata.
        dMentionsPred (dict): A dictionary of predicted mentions per sentence.
        dMentionsGold (dict): A dictionary of gold standard mentions per
            sentence.
        dCandidates (dict): A dictionary of candidates per mention in a
            sentence.

    Returns:
        dict:
            A dictionary containing all processed data (predictions, gold
            standard, skyline, candidates) in one place.

    Note:
        This function also creates one JSON file per dictionary, stored in
        ``outputs/data``.
    """
    data_path = experiment.data_path
    dataset = experiment.dataset
    model_name = experiment.myner.model
    output_path = data_path + dataset + "/" + model_name

    cand_approach = experiment.myranker.method
    if experiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["selection_threshold"]
        )

    # Store NER predictions using a specific NER model:
    with open(output_path + "_ner_predictions.json", "w") as fw:
        json.dump(preds, fw)

    # Store gold standard:
    with open(output_path + "_gold_standard.json", "w") as fw:
        json.dump(trues, fw)

    # Store NER skyline:
    with open(output_path + "_ner_skyline.json", "w") as fw:
        json.dump(skys, fw)

    # Store gold tokenisation positions:
    with open(output_path + "_gold_positions.json", "w") as fw:
        json.dump(gold_tok, fw)

    # Store the dictionary of sentences:
    with open(output_path + "_dict_sentences.json", "w") as fw:
        json.dump(dSentences, fw)

    # Store the dictionary of metadata per sentence:
    with open(output_path + "_dict_metadata.json", "w") as fw:
        json.dump(dMetadata, fw)

    # Store the dictionary of predicted results:
    with open(output_path + "_pred_mentions.json", "w") as fw:
        json.dump(dMentionsPred, fw)

    # Store the dictionary of gold standard:
    with open(output_path + "_gold_mentions.json", "w") as fw:
        json.dump(dMentionsGold, fw)

    # Store the dictionary of gold standard:
    with open(output_path + "_candidates_" + cand_approach + ".json", "w") as fw:
        json.dump(dCandidates, fw)

    dict_processed_data = dict()
    dict_processed_data["preds"] = preds
    dict_processed_data["trues"] = trues
    dict_processed_data["skys"] = skys
    dict_processed_data["gold_tok"] = gold_tok
    dict_processed_data["dSentences"] = dSentences
    dict_processed_data["dMetadata"] = dMetadata
    dict_processed_data["dMentionsPred"] = dMentionsPred
    dict_processed_data["dMentionsGold"] = dMentionsGold
    dict_processed_data["dCandidates"] = dCandidates
    return dict_processed_data


def create_mentions_df(experiment: experiment.Experiment) -> pd.DataFrame:
    """
    Create a dataframe for the linking experiment, with one mention per row.

    Arguments:
        experiment (experiment.Experiment): An experiment object.

    Returns:
        pandas.DataFrame:
            A dataframe with one mention per row, and containing all relevant
            information for subsequent steps (i.e. for linking).

    Note:
        This function also creates a TSV file in the
        ``outputs/data/[dataset]/`` folder.
    """
    dMentions = experiment.processed_data["dMentionsPred"]
    dGoldSt = experiment.processed_data["dMentionsGold"]
    dSentences = experiment.processed_data["dSentences"]
    dMetadata = experiment.processed_data["dMetadata"]
    dCandidates = experiment.processed_data["dCandidates"]

    cand_approach = experiment.myranker.method
    if experiment.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            experiment.myranker.deezy_parameters["selection_threshold"]
        )

    rows = []
    for sentence_id in dMentions:
        for mention in dMentions[sentence_id]:
            if mention:
                article_id = sentence_id.split("_")[0]
                sentence_pos = sentence_id.split("_")[1]
                sentence = dSentences[sentence_id]
                token_start = mention["start_offset"]
                token_end = mention["end_offset"]
                char_start = mention["start_char"]
                char_end = mention["end_char"]
                ner_score = round(mention["ner_score"], 3)
                pred_mention = mention["mention"]
                entity_type = mention["ner_label"]
                place = dMetadata[sentence_id]["place"]
                year = dMetadata[sentence_id]["year"]
                publication = dMetadata[sentence_id]["publication_code"]
                place_wqid = dMetadata[sentence_id]["place_wqid"]
                # Match predicted mention with gold standard mention (will just be used for training):
                max_tok_overlap = 0
                gold_standard_link = "NIL"
                gold_standard_ner = "O"
                gold_mention = ""
                for gs in dGoldSt[sentence_id]:
                    pred_token_range = range(token_start, token_end + 1)
                    gs_token_range = range(gs["start_offset"], gs["end_offset"] + 1)
                    overlap = len(list(set(pred_token_range) & set(gs_token_range)))
                    if overlap > max_tok_overlap:
                        max_tok_overlap = overlap
                        gold_mention = gs["mention"]
                        gold_standard_link = gs["entity_link"]
                        gold_standard_ner = gs["ner_label"]
                candidates = dCandidates[sentence_id].get(mention["mention"], dict())

                rows.append(
                    [
                        sentence_id,
                        article_id,
                        sentence_pos,
                        sentence,
                        token_start,
                        token_end,
                        char_start,
                        char_end,
                        ner_score,
                        pred_mention,
                        entity_type,
                        place,
                        year,
                        publication,
                        place_wqid,
                        gold_mention,
                        gold_standard_link,
                        gold_standard_ner,
                        candidates,
                    ]
                )

    processed_df = pd.DataFrame(
        columns=[
            "sentence_id",
            "article_id",
            "sentence_pos",
            "sentence",
            "token_start",
            "token_end",
            "char_start",
            "char_end",
            "ner_score",
            "pred_mention",
            "pred_ner_label",
            "place",
            "year",
            "publication",
            "place_wqid",
            "gold_mention",
            "gold_entity_link",
            "gold_ner_label",
            "candidates",
        ],
        data=rows,
    )

    output_path = (
        experiment.data_path
        + experiment.dataset
        + "/"
        + experiment.myner.model
        + "_"
        + cand_approach
    )

    # List of columns to merge (i.e. columns where we have indicated
    # out data splits), and "article_id", the columns on which we
    # will merge the data:
    keep_columns = [
        "article_id",
        "apply",
        "originalsplit",
        "withouttest",
        "Ashton1860",
        "Dorchester1820",
        "Dorchester1830",
        "Dorchester1860",
        "Manchester1780",
        "Manchester1800",
        "Manchester1820",
        "Manchester1830",
        "Manchester1860",
        "Poole1860",
    ]

    # Add data splits from original dataframe:
    df = experiment.dataset_df[
        [c for c in keep_columns if c in experiment.dataset_df.columns]
    ]

    # Convert article_id to string (it's read as an int):
    df = df.assign(article_id=lambda d: d["article_id"].astype(str))
    processed_df = processed_df.assign(article_id=lambda d: d["article_id"].astype(str))
    processed_df = pd.merge(processed_df, df, on=["article_id"], how="left")

    # Store mentions dataframe:
    processed_df.to_csv(output_path + "_mentions.tsv", sep="\t")

    return processed_df


def store_for_scorer(
    hipe_scorer_results_path: str,
    scenario_name: str,
    dresults: dict,
    articles_test: List[str],
) -> None:
    """
    Stores the results in the required format for evaluation using the CLEF-HIPE scorer.

    Arguments:
        hipe_scorer_results_path (str): The first part of the output file path.
        scenario_name (str): The second part of the output file path.
        dresults (dict): A dictionary containing the results.
        articles_test (list): A list of sentences that are part of the split used
            for evaluating the performance in the provided experiment.

    Returns:
        None.

    Note:
        The function also creates a TSV file with the results in the Conll
        format required by the scorer.

        For more information about the CLEF-HIPE scorer, see
        https://github.com/impresso/CLEF-HIPE-2020-scorer.

    Example:
        Assuming the CLEF-HIPE scorer is stored in
        ``../CLEF-HIPE-2020-scorer/``, you can run scorer as follows:

        For NER:

        .. code-block:: bash

            $ python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nerc_coarse --outdir outputs/results/

        For EL:

        .. code-block:: bash

            $ python ../CLEF-HIPE-2020-scorer/clef_evaluation.py --ref outputs/results/lwm-true_bundle2_en_1.tsv --pred outputs/results/lwm-pred_bundle2_en_1.tsv --task nel --outdir outputs/results/

    """
    # Bundle 2 associated tasks: NERC-coarse and NEL
    with open(
        os.path.join(hipe_scorer_results_path, scenario_name + ".tsv"),
        "w",
    ) as fw:
        fw.write(
            "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n"
        )
        for sent_id in dresults:
            # Filter by article in test:
            if sent_id.split("_")[0] in articles_test:
                fw.write("# sentence_id = " + sent_id + "\n")
                for t in dresults[sent_id]:
                    elink = t[2]
                    if t[2].startswith("B-"):
                        elink = t[2].replace("B-", "")
                    elif t[2].startswith("I-"):
                        elink = t[2].replace("I-", "")
                    elif t[1] != "O":
                        elink = "NIL"
                    fw.write(
                        t[0]
                        + "\t"
                        + t[1]
                        + "\t"
                        + t[1]
                        + "\tO\tO\tO\tO\t"
                        + elink
                        + "\t"
                        + elink
                        + "\tO\n"
                    )
                fw.write("\n")


# TODO: which_split doesn't seem to be used, so can be removed here?
def store_results(
    experiment: experiment.Experiment,
    task: Literal["ner", "linking"],
    how_split: str,
    which_split,
) -> None:
    """
    Function which stores the results of an experiment in the format required
    by the HIPE 2020 evaluation scorer.

    Arguments:
        experiment (experiment.Experiment): the current experiment.
        task (Literal["ner", "linking"]): either "ner" or "linking". Store the
            results for just NER or with links as well.
        how_split (str): which way of splitting the data are we using?
            It could be the ``"originalsplit"`` or ``"Ashton1860"``, for
            example, which would mean that ``"Ashton1860"`` is left out for
            test only.
        which_split (str): on which split are we testing our experiments?
            It can be ``"dev"`` while we're developing the code, or ``"test"``
            when we run it in the final experiments.

    Returns:
        None.
    """
    hipe_scorer_results_path = os.path.join(experiment.results_path, experiment.dataset)
    Path(hipe_scorer_results_path).mkdir(parents=True, exist_ok=True)

    scenario_name = ""
    if task == "ner":
        scenario_name += task + "_" + experiment.myner.model + "_"
    if task == "linking":
        scenario_name += task + "_" + experiment.myner.model + "_"
        cand_approach = experiment.myranker.method
        if experiment.myranker.method == "deezymatch":
            cand_approach += "+" + str(
                experiment.myranker.deezy_parameters["num_candidates"]
            )
            cand_approach += "+" + str(
                experiment.myranker.deezy_parameters["selection_threshold"]
            )

        scenario_name += cand_approach + "_" + how_split + "_"

    link_approach = experiment.mylinker.method
    if experiment.mylinker.method == "reldisamb":
        if experiment.mylinker.rel_params["with_publication"]:
            link_approach += "+wpubl"
        if experiment.mylinker.rel_params["without_microtoponyms"]:
            link_approach += "+wmtops"
        if experiment.mylinker.rel_params["do_test"]:
            link_approach += "_test"

    # Find article ids of the corresponding test set (e.g. 'dev' of the original split,
    # 'test' of the Ashton1860 split, etc):
    all = experiment.dataset_df
    test_articles = list(all[all[how_split] == "test"].article_id.unique())
    test_articles = [str(art) for art in test_articles]

    # Store predictions results formatted for CLEF-HIPE scorer:
    preds_name = link_approach if task == "linking" else "preds"
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + preds_name,
        experiment.processed_data["preds"],
        test_articles,
    )

    # Store gold standard results formatted for CLEF-HIPE scorer:
    store_for_scorer(
        hipe_scorer_results_path,
        scenario_name + "trues",
        experiment.processed_data["trues"],
        test_articles,
    )

    if task == "linking":
        # If task is "linking", store the skyline results (but not for the
        # ranking method of REL):
        store_for_scorer(
            hipe_scorer_results_path,
            scenario_name + "skys",
            experiment.processed_data["skys"],
            test_articles,
        )
