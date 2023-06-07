import json
import os
import sys
from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import ner

if TYPE_CHECKING:
    from geoparser import recogniser


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
        >>> eval_with_exception("[1, 2, 3]")
        [1, 2, 3]
        >>> process_data.eval_with_exception(None, [])
        []
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
