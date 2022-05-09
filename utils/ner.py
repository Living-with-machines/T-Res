from ast import literal_eval
from collections import namedtuple

# Dictionary mapping NER model label with GS label:
label_dict = dict()
label_dict["lwm"] = {
    "LABEL_0": "O",
    "LABEL_1": "B-LOC",
    "LABEL_2": "I-LOC",
    "LABEL_3": "B-STREET",
    "LABEL_4": "I-STREET",
    "LABEL_5": "B-BUILDING",
    "LABEL_6": "I-BUILDING",
    "LABEL_7": "B-OTHER",
    "LABEL_8": "I-OTHER",
    "LABEL_9": "B-FICTION",
    "LABEL_10": "I-FICTION",
}
label_dict["hmd"] = {
    "LABEL_0": "O",
    "LABEL_1": "B-LOC",
    "LABEL_2": "I-LOC",
    "LABEL_3": "B-STREET",
    "LABEL_4": "I-STREET",
    "LABEL_5": "B-BUILDING",
    "LABEL_6": "I-BUILDING",
    "LABEL_7": "B-OTHER",
    "LABEL_8": "I-OTHER",
    "LABEL_9": "B-FICTION",
    "LABEL_10": "I-FICTION",
}
label_dict["hipe"] = {
    "LABEL_0": "O",
    "LABEL_1": "B-LOC",
    "LABEL_2": "I-LOC",
    "LABEL_3": "B-LOC",
    "LABEL_4": "I-LOC",
    "LABEL_5": "B-LOC",
    "LABEL_6": "I-LOC",
    "LABEL_7": "B-LOC",
    "LABEL_8": "I-LOC",
    "LABEL_9": "B-LOC",
    "LABEL_10": "I-LOC",
}


# ----------------------------------------------
# LABEL GROUPING
# There are some consistent errors when grouping
# what constitutes B- or I-. The following functions
# take care of them:
# * fix_capitalization
# * fix_hyphens
# * fix_nested
# * fix_startEntity
# * aggregate_entities
# ----------------------------------------------


def fix_capitalization(entity, sentence):
    """
    These entities are the output of the NER prediction, which returns
    the processed word (uncapitalized, for example). We replace this
    processed word by the true surface form in our original dataset
    (using the character position information).
    """
    newEntity = entity
    if entity["word"].startswith("##"):
        newEntity = {
            "entity": entity["entity"],
            "score": entity["score"],
            # To have "word" with the true capitalization, get token from source sentence:
            "word": "##" + sentence[entity["start"] : entity["end"]],
            "start": entity["start"],
            "end": entity["end"],
        }
    else:
        newEntity = {
            "entity": entity["entity"],
            "score": entity["score"],
            # To have "word" with the true capitalization, get token from source sentence:
            "word": sentence[entity["start"] : entity["end"]],
            "start": entity["start"],
            "end": entity["end"],
        }
    return newEntity


def fix_hyphens(lEntities):
    """
    Fix B- and I- prefix assignment errors in hyphenated entities.
    * Description: There is problem with grouping when there are hyphens in
    words, e.g. "Ashton-under-Lyne" (["Ashton", "-", "under", "-", "Lyne"])
    is grouped as ["B-LOC", "B-LOC", "B-LOC", "B-LOC", "B-LOC"], when
    it should be grouped as ["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"].
    * Solution: if the current token or the previous token is a hyphen,
    and the entity type of both previous and current token is the same
    and not "O", then change the current's entity preffix to "I-".
    """
    numbers = [str(x) for x in range(0, 10)]
    connectors = [
        "-",
        ",",
        ".",
        "’",
        "'",
    ] + numbers  # Numbers and punctuation are common OCR errors
    hyphEntities = []
    hyphEntities.append(lEntities[0])
    for i in range(1, len(lEntities)):
        prevEntity = hyphEntities[i - 1]
        currEntity = lEntities[i]
        if (
            (prevEntity["word"] in connectors or currEntity["word"] in connectors)
            and (
                prevEntity["entity"][2:]
                == currEntity["entity"][2:]  # Either the labels match...
                or currEntity["word"][
                    0
                ].islower()  # ... or the second token is not capitalised...
                or currEntity["word"]
                in numbers  # ... or the second token is a number...
                or prevEntity["end"]
                == currEntity[
                    "start"
                ]  # ... or there's no space between prev and curr tokens
            )
            and prevEntity["entity"] != "O"
            and currEntity["entity"] != "O"
        ):
            newEntity = {
                "entity": "I-" + prevEntity["entity"][2:],
                "score": currEntity["score"],
                "word": currEntity["word"],
                "start": currEntity["start"],
                "end": currEntity["end"],
            }
            hyphEntities.append(newEntity)
        else:
            hyphEntities.append(currEntity)

    return hyphEntities


def fix_nested(lEntities):
    """
    Fix B- and I- prefix assignment errors in nested entities.
    * Description: There is problem with grouping in nested entities,
    e.g. "Island of Terceira" (["Island", "of", "Terceira"])
    is grouped as ["B-LOC", "I-LOC", "B-LOC"], when it should
    be grouped as ["B-LOC", "I-LOC", "I-LOC"], as we consider
    it one entity.
    * Solution: if the current token or the previous token is a hyphen,
    and the entity type of both previous and current token is  not "O",
    then change the current's entity preffix to "I-".
    """
    nestEntities = []
    nestEntities.append(lEntities[0])
    for i in range(1, len(lEntities)):
        prevEntity = nestEntities[i - 1]
        currEntity = lEntities[i]
        if (
            prevEntity["word"].lower() == "of"
            and prevEntity["entity"] != "O"
            and currEntity["entity"] != "O"
        ):
            newEntity = {
                "entity": "I-" + prevEntity["entity"][2:],
                "score": currEntity["score"],
                "word": currEntity["word"],
                "start": currEntity["start"],
                "end": currEntity["end"],
            }
            nestEntities.append(newEntity)
        else:
            nestEntities.append(currEntity)

    return nestEntities


def fix_startEntity(lEntities):
    """
    Fix B- and I- prefix assignment errors:
    * Case 1: The first token of a sentence can only be either
              O (i.e. not an entity) or B- (beginning of an
              entity). There's no way it should be I-. Fix
              those.
    * Case 2: If the first token of a grouped entity is assigned
              the prefix I-, change to B-. We know it's the first
              token in a grouped entity if the entity type of the
              previous token is different.
    """
    fixEntities = []

    # Case 1: If necessary, fix first entity
    currEntity = lEntities[0]
    if currEntity["entity"].startswith("I-"):
        fixEntities.append(
            {
                "entity": "B-" + currEntity["entity"][2:],
                "score": currEntity["score"],
                "word": currEntity["word"],
                "start": currEntity["start"],
                "end": currEntity["end"],
            }
        )
    else:
        fixEntities.append(currEntity)

    # Fix subsequent entities:
    for i in range(1, len(lEntities)):
        prevEntity = fixEntities[i - 1]
        currEntity = lEntities[i]
        # E.g. If a grouped entity begins with "I-", change to "B-".
        if (
            prevEntity["entity"] == "O"
            or (prevEntity["entity"][2:] != currEntity["entity"][2:])
        ) and currEntity["entity"].startswith("I-"):
            newEntity = {
                "entity": "B-" + currEntity["entity"][2:],
                "score": currEntity["score"],
                "word": currEntity["word"],
                "start": currEntity["start"],
                "end": currEntity["end"],
            }
            fixEntities.append(newEntity)
        else:
            fixEntities.append(currEntity)

    return fixEntities


def aggregate_entities(entity, lEntities):
    newEntity = entity
    # We remove the word index because we're altering it (by joining suffixes)
    newEntity.pop("index", None)
    # If word starts with ##, then this is a suffix, join with previous detected entity
    if entity["word"].startswith("##"):
        prevEntity = lEntities.pop()
        newEntity = {
            "entity": prevEntity["entity"],
            "score": ((prevEntity["score"] + entity["score"]) / 2.0),
            "word": prevEntity["word"] + entity["word"].replace("##", ""),
            "start": prevEntity["start"],
            "end": entity["end"],
        }

    lEntities.append(newEntity)
    return lEntities


def ner_predict(sentence, annotations, ner_pipe, dataset):
    """
    This function reads a dataset dataframe and the NER pipeline and returns
    two dictionaries:
    * dPredictions: The dPredictions dictionary keeps the results of the BERT NER
                    as a list of dictionaries (value) for each article/sentence
                    pair (key).
    * dGoldStandard: The dGoldStandard contains the gold standard labels (aligned
                     to the BERT NER tokenisation).
    """

    # The dPredictions dictionary keeps the results of the BERT NER
    # as a list of dictionaries (value) for each article/sentence pair (key).
    ner_preds = ner_pipe(sentence)
    lEntities = []
    for pred_ent in ner_preds:
        prev_tok = pred_ent["word"]
        pred_ent["entity"] = label_dict[dataset][pred_ent["entity"]]
        pred_ent = fix_capitalization(pred_ent, sentence)
        if prev_tok.lower() != pred_ent["word"].lower():
            print("Token processing error.")
        predictions = aggregate_entities(pred_ent, lEntities)
    predictions = fix_hyphens(predictions)
    predictions = fix_nested(predictions)
    predictions = fix_startEntity(predictions)

    # The dGoldStandard dictionary is an alignment between the output
    # of BERT NER (as it uses its own tokenizer) and the gold standard
    # labels. It does so based on the start and end position of each
    # predicted token. By default, a predicted token is assigned the
    # "O" label, unless its position overlaps with the position an
    # annotated entity, in which case we relabel it according to this
    # label.
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
                gs_for_eval["entity"] = "B-" + annotations[gse][0]
                gs_for_eval["link"] = "B-" + annotations[gse][2]
            elif pred_ent["start"] > gse[0] and pred_ent["end"] <= gse[1]:
                gs_for_eval["entity"] = "I-" + annotations[gse][0]
                gs_for_eval["link"] = "I-" + annotations[gse][2]
        gold_standard.append(gs_for_eval)

    return gold_standard, predictions


def aggregate_mentions(predictions, accepted_labels):
    mentions = collect_named_entities(predictions, accepted_labels)
    sent_mentions = []
    for mention in mentions:
        text_mention = " ".join(
            [
                predictions[r][0]
                for r in range(mention.start_offset, mention.end_offset + 1)
            ]
        )

        ner_score = [
            predictions[r][-1]
            for r in range(mention.start_offset, mention.end_offset + 1)
        ]
        ner_score = sum(ner_score) / len(ner_score)
        ner_label = [
            predictions[r][1]
            for r in range(mention.start_offset, mention.end_offset + 1)
        ]
        ner_label = list(
            set([label.split("-")[1] if "-" in label else label for label in ner_label])
        )[0]
        sent_mentions.append(
            {
                "mention": text_mention,
                "start_offset": mention.start_offset,
                "end_offset": mention.end_offset,
                "start_char": mention.start_char,
                "end_char": mention.end_char,
                "ner_score": ner_score,
                "ner_label": ner_label,
            }
        )
    return sent_mentions


def collect_named_entities(tokens, accepted_labels):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    Entity = namedtuple("Entity", "e_type start_offset end_offset start_char end_char")
    dict_tokens = dict(enumerate(tokens))

    for offset, annotation in enumerate(tokens):
        token_tag = annotation[1]

        if not token_tag.lower() in accepted_labels:
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(
                    Entity(
                        ent_type,
                        start_offset,
                        end_offset,
                        dict_tokens[start_offset][3],
                        dict_tokens[end_offset][4],
                    )
                )
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (
            ent_type == token_tag[2:] and token_tag[:1] == "B"
        ):

            end_offset = offset - 1
            named_entities.append(
                Entity(
                    ent_type,
                    start_offset,
                    end_offset,
                    dict_tokens[start_offset][3],
                    dict_tokens[end_offset][4],
                )
            )

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(
            Entity(
                ent_type,
                start_offset,
                len(tokens) - 1,
                dict_tokens[start_offset][3],
                dict_tokens[len(tokens) - 1][4],
            )
        )

    return named_entities


def format_for_ner(df):

    # In the dAnnotatedClasses dictionary, we keep, for each article/sentence,
    # a dictionary that maps the position of an annotated named entity (i.e.
    # its start and end character, as a tuple, as the key of the inner dictionary)
    # and another tuple as its value, with the class of named entity (such as LOC
    # or BUILDING, and its annotated link).
    dAnnotated = dict()
    dSentences = dict()
    dMetadata = dict()
    for i, row in df.iterrows():

        sentences = literal_eval(row["sentences"])
        annotations = literal_eval(row["annotations"])

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

            dMetadata[artsent_id] = dict()
            dMetadata[artsent_id]["place"] = row["place"]
            dMetadata[artsent_id]["year"] = row["year"]
            dMetadata[artsent_id]["ocr_quality_mean"] = row["ocr_quality_mean"]
            dMetadata[artsent_id]["ocr_quality_sd"] = row["ocr_quality_sd"]
            dMetadata[artsent_id]["publication_title"] = row["publication_title"]

    for artsent_id in dSentences:
        if not artsent_id in dAnnotated:
            dAnnotated[artsent_id] = dict()

    for artsent_id in dSentences:
        if not artsent_id in dMetadata:
            dMetadata[artsent_id] = dict()

    return dAnnotated, dSentences, dMetadata
