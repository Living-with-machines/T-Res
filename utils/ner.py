from collections import namedtuple
from typing import List, Literal, NamedTuple, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def training_tokenize_and_align_labels(
    examples: dict,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    label_encoding_dict: dict,
):
    """
    Tokenize and align labels during training.

    This function takes a training instance, consisting of tokens and named
    entity recognition (NER) tags, and aligns the tokens with their
    corresponding labels. It uses a transformers tokenizer object to tokenize
    the input tokens and then maps the NER tags to label IDs based on the
    provided label encoding dictionary.

    Arguments:
        examples (Dict): A dictionary representing a single training instance
            with three keys: ``id`` (instance ID), ``tokens`` (list of tokens),
            and ``ner_tags`` (list of NER tags).
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): A
            transformers tokenizer object, which is the tokenizer of the base
            model.
        label_encoding_dict (Dict): A dictionary mapping NER labels to label
            IDs, from ``label2id`` in
            :py:meth:`~geoparser.recogniser.Recogniser.train`.

    Returns:
        transformers.tokenization_utils_base.BatchEncoding:
            The tokenized inputs with aligned labels.

    Credit:
        This function is adapted from `HuggingFace <https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py>`_.
    """
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        list(examples["tokens"]), truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to
            # -100 so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the
            # current label or -100, depending on the label_all_tokens flag.
            else:
                label_ids.append(
                    label_encoding_dict[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def collect_named_entities(
    tokens: List[Tuple[str, str, str, int, int]]
) -> List[NamedTuple]:
    """
    Collect named entities from a list of tokens and return a list of named
    tuples representing the entities.

    This function iterates over the tokens and identifies named entities based
    on their entity type (``entity_type``), keeping the tokens that are not
    tagged as ``"O"``. Each token is represented as a tuple with the following
    format: ``(token, entity_type, link, start_char, end_char)``.

    Arguments:
        tokens (List[tuple]): A list of tokens, where each token is
            represented as a tuple containing the following elements:

            - ``token`` (str): The token text.
            - ``entity_type`` (str): The entity type (e.g., ``"B-LOC"``,
            ``"I-LOC"``, ``"O"``).
            - ``link`` (str): Empty string reserved for the entity link.
            - ``start_char`` (int): The start character offset of the token.
            - ``end_char`` (int): The end character offset of the token.

    Returns:
        List[NamedTuple]:
            A list of named tuples (called ``Entity``) representing the named
            entities. Each named tuple contains the following fields:

            - ``e_type`` (str): The entity type.
            - ``link`` (str): Empty string reserved for the entity link.
            - ``start_offset`` (int): The start offset of the entity (token
              position).
            - ``end_offset`` (int): The end offset of the entity (token
              position).
            - ``start_char`` (int): The start character offset of the entity.
            - ``end_char`` (int): The end character offset of the entity.
    """
    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None
    link = None

    Entity = namedtuple(
        "Entity", "e_type link start_offset end_offset start_char end_char"
    )
    dict_tokens = dict(enumerate(tokens))

    for offset, annotation in enumerate(tokens):
        token_tag = annotation[1]
        token_link = annotation[2]

        if token_tag == "O":
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(
                    Entity(
                        ent_type,
                        link,
                        start_offset,
                        end_offset,
                        dict_tokens[start_offset][3],
                        dict_tokens[end_offset][4],
                    )
                )
                start_offset = None
                end_offset = None
                ent_type = None
                link = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            link = token_link[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (
            ent_type == token_tag[2:] and token_tag[:1] == "B"
        ):
            end_offset = offset - 1
            named_entities.append(
                Entity(
                    ent_type,
                    link,
                    start_offset,
                    end_offset,
                    dict_tokens[start_offset][3],
                    dict_tokens[end_offset][4],
                )
            )

            # start of a new entity
            ent_type = token_tag[2:]
            link = token_link[2:]
            start_offset = offset
            end_offset = None

    # Catches an entity that goes up until the last token
    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(
            Entity(
                ent_type,
                link,
                start_offset,
                len(tokens) - 1,
                dict_tokens[start_offset][3],
                dict_tokens[len(tokens) - 1][4],
            )
        )

    return named_entities


def aggregate_mentions(
    predictions: List[List[Tuple[str, str, str, int, int]]],
    setting: Literal["pred", "gold"],
) -> List[dict]:
    """
    Aggregate predicted or gold mentions into a consolidated format.

    This function takes a list of predicted or gold mentions and aggregates
    them into a consolidated format. It reconstructs the text of each mention
    by combining the tokens and their corresponding white spaces. It also
    consolidates the NER label, NER score, and entity link for each mention.

    Arguments:
        predictions (List[List]): A list of token predictions, where each
            token prediction is represented as a list of values. For details
            on each of those tuples, see
            :py:meth:`~utils.ner.collect_named_entities`.
        setting (Literal["pred", "gold"]): The setting for aggregation:

            - If set to ``"pred"``, the function aggregates predicted mentions.
              Entity links will be set to ``"O"`` (because we haven't performed
              linking yet).
            - If set to ``"gold"``, the function aggregates gold mentions. NER
              score will be set to ``1.0`` as it is manually detected.

    Returns:
        List[dict]:
            A list of dictionaries representing the aggregated mentions, where
            each dictionary contains the following keys:

            - ``mention``: The text of the mention.
            - ``start_offset``: The start offset of the mention (token position).
            - ``end_offset``: The end offset of the mention (token position).
            - ``start_char``: The start character index of the mention.
            - ``end_char``: The end character index of the mention.
            - ``ner_score``: The consolidated NER score of the mention (``0.0``
              for predicted mentions, ``1.0`` for gold mentions).
            - ``ner_label``: The consolidated NER label of the mention.
            - ``entity_link``: The consolidated entity link of the mention
              (empty string ``"O"`` for predicted mentions, entity label for
              gold mentions).
    """
    mentions = collect_named_entities(predictions)

    sent_mentions = []
    for mention in mentions:
        # Reconstruct the text of the mention:
        text_mention = ""
        mention_token_range = range(mention.start_offset, mention.end_offset + 1)
        for r in mention_token_range:
            add_whitespaces = ""
            # Add white spaces between tokens according to token's char starts
            # and ends:
            if r - 1 in mention_token_range:
                prev_end_char = predictions[r - 1][4]
                curr_start_char = predictions[r][3]
                add_whitespaces = (curr_start_char - prev_end_char) * " "
            text_mention += add_whitespaces + predictions[r][0]

        ner_score = 0.0
        entity_link = ""
        ner_label = ""

        # Consolidate the NER label:
        ner_label = [
            predictions[r][1]
            for r in range(mention.start_offset, mention.end_offset + 1)
        ]
        ner_label = list(
            set([label.split("-")[1] if "-" in label else label for label in ner_label])
        )[0]

        if setting == "pred":
            # Consolidate the NER score
            ner_score = [
                predictions[r][-1]
                for r in range(mention.start_offset, mention.end_offset + 1)
            ]
            ner_score = round(sum(ner_score) / len(ner_score), 3)

            # Link is at the moment not filled:
            entity_link = "O"

        elif setting == "gold":
            ner_score = 1.0

            # Consolidate the enity link:
            entity_link = [
                predictions[r][2]
                for r in range(mention.start_offset, mention.end_offset + 1)
            ]
            entity_link = list(
                set(
                    [
                        label.split("-")[1] if "-" in label else label
                        for label in entity_link
                    ]
                )
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
                "entity_link": entity_link,
            }
        )
    return sent_mentions


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


def fix_capitalization(entity: dict, sentence: str) -> dict:
    """
    Correct capitalization errors in entities.

    This function corrects capitalization errors in entities that occur as a
    result of the NER prediction. The NER prediction may return processed
    words with incorrect capitalization. This function replaces the processed
    word in the entity with the true surface form from the original dataset,
    using the character position information.

    Arguments:
        entity (dict): A dictionary containing the prediction of one token.
        sentence (str): The original sentence.

    Returns:
        dict:
            The corrected entity dictionary with the appropriate
            capitalization.
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


def fix_hyphens(lEntities: List[dict]) -> List[dict]:
    """
    Fix prefix assignment errors in hyphenated entities.

    This function corrects prefix assignment errors that occur in some
    hyphenated entities, where multiple tokens connected by hyphens form a
    single entity but are incorrectly assigned different prefixes (i.e.
    ``B-`` and ``I-``). It specifically addresses the issue of grouping
    in hyphenated entities, where a sequence of tokens connected by hyphens
    should be grouped as a single entity.

    Arguments:
        lEntities (list): A list of dictionaries corresponding to predicted
            tokens.

    Returns:
        list:
            A list of dictionaries with corrected predictions regarding
            hyphenation.

    Note:
        **Description**: There is a problem with grouping when there are
        hyphens in words. For example, the phrase "Ashton-under-Lyne"
        (``["Ashton", "-", "under", "-", "Lyne"]``) is incorrectly grouped
        as ``["B-LOC", "B-LOC", "B-LOC", "B-LOC", "B-LOC"]``, when it should be
        grouped as ``["B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC"]``.

        **Solution**: If the current token or the previous token is a hyphen,
        and the entity type of both the previous and current tokens is the
        same and not ``"O"``, the current entity's prefix is changed to
        ``"I-"`` to maintain the correct grouping.
    """
    numbers = [str(x) for x in range(0, 10)]
    connectors = [
        "-",
        ",",
        ".",
        "’",
        "'",
        "?",
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


def fix_nested(lEntities: List[dict]) -> List[dict]:
    """
    Fix prefix assignment errors in nested entities.

    This function corrects prefix assignment errors that occur in nested
    entities, where multiple tokens are part of the same entity but are
    incorrectly assigned different prefixes. It specifically addresses the
    issue of grouping in nested entities, where a sequence of tokens that form
    a single entity are assigned incorrect prefixes.

    Arguments:
        lEntities (list): A list of dictionaries corresponding to predicted
            tokens.

    Returns:
        list:
            A list of dictionaries with corrected predictions regarding nested
            entities.

    Note:
        **Description**: There is a problem with grouping in some nested
        entities. For example, the phrase "Island of Terceira"
        (``["Island", "of", "Terceira"]``) is incorrectly grouped as
        ``["B-LOC", "I-LOC", "B-LOC"]``, when it should be
        ``["B-LOC", "I-LOC", "I-LOC"]`` as we consider it as one entity.

        **Solution**: If the current token is preposition ``"of"`` and
        the previous and current entity types are not ``"O"``, the current
        entity's prefix is changed to ``"I-"`` to maintain the correct grouping.
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


def fix_startEntity(lEntities: List[dict]) -> List[dict]:
    """
    Fix prefix assignment errors in entity labeling.

    This function corrects two different cases of prefix assignment errors in
    entity labeling:

    1. The first token of a sentence can only be either ``"O"`` (not an entity)
       or ``"B-"`` (beginning of an entity). If it is incorrectly assigned the
       prefix ``"I-"``, this case is fixed by changing it to ``"B-"``.
    2. If the first token of a grouped entity is assigned the prefix ``"I-"``,
       but the entity type of the previous token is different, it should be
       ``"B-"`` instead. This case is fixed by changing the prefix to ``"B-"``.

    Arguments:
        lEntities (list): A list of dictionaries corresponding to predicted
            tokens.

    Returns:
        list:
            A list of dictionaries with corrected predictions regarding the
            grouping of labels.
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


def aggregate_entities(entity: dict, lEntities: List[dict]) -> List[dict]:
    """
    Aggregates entities by joining split tokens.

    This function aggregates entities by joining split tokens that start with
    ``"##"`` with the previous detected entity. It takes the current entity
    and the list of all predicted entities as input and returns a new list of
    dictionaries with corrected predictions regarding split tokens.

    Arguments:
        entity (dict): The current entity (predicted token) as a dictionary.
        lEntities (list): The list of dictionaries corresponding to all
            predicted tokens.

    Returns:
        list:
            A list of dictionaries with the corrected predictions regarding
            split tokens.
    """
    newEntity = entity

    # We remove the word index because we're altering it (by joining suffixes)
    newEntity.pop("index", None)

    # If word starts with ##, then this is a suffix, join with previous
    # detected entity
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
