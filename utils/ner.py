from collections import namedtuple


def training_tokenize_and_align_labels(examples, tokenizer, label_encoding_dict):
    """
    During training, aligns tokens with labels.

    Arguments:
        examples (dict): One training instance (a dictioary with three keys: id, tokens, and ner_tags)
        tokenizer (Tokenizer object): A transformers tokenizer object (the tokeniser of the base model).
        label_encoding_dict (dict): The label2id mapping dictionary.

    Returns:
        tokenized_inputs (Dataset object): The tokenized input.

    Notes:
        Credit: This function is adapted from
        https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py.
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
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_encoding_dict[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity
    type and the start and end offsets of the entity.

    Arguments:
        tokens (list): a list of tags.

    Returns:
        named_entities (list): a list of Entity named-tuples
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


def aggregate_mentions(predictions, setting):
    """
    Aggregates mentions (NER outputs separate tokens) and finds
    mention position in sentence.

    Arguments:
        predictions: a list of lists (the outer list representing
            a sentence, the inner list representing a token) representation
            the NER predictions.
        setting: either "pred" or "gold". If "pred", set entity_link
            to "O" (because we haven't performed linking yet) and
            perform average of the NER score of all the tokens that
            belong to the same entity. If "gold", set ner_score to
            1.0 (it's manually detected) and consolidate the link of the
            mention subtokens.

    Returns:
        sent_mentions (list): a list of dictionaries, there the list
            corresponds to the sentence, and the inner dictionaries
            correspond to the different (multi-token) mentions that
            have been identified (in case of "pred") or that were
            annotated (in case of "gold").
    """

    mentions = collect_named_entities(predictions)

    sent_mentions = []
    for mention in mentions:
        # Reconstruct the text of the mention:
        text_mention = ""
        mention_token_range = range(mention.start_offset, mention.end_offset + 1)
        for r in mention_token_range:
            add_whitespaces = ""
            # Add white spaces between tokens according to token's char starts and ends:
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


def fix_capitalization(entity, sentence):
    """
    These entities are the output of the NER prediction, which returns
    the processed word (uncapitalized, for example). We replace this
    processed word by the true surface form in our original dataset
    (using the character position information).

    Arguments:
        entity (dict): A dictionary containing the prediction of one token.
        sentence (str): The original sentence.

    Returns
        newEntity (dict): The input dictionary, corrected re capitalisation.
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

    Arguments:
        lEntities (list): List of dictionaries (corresponding to predicted tokens).

    Returns:
        hyphEntities (list): List of dictionaries with the corrected predictions
            regarding hyphenation.
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

    Arguments:
        lEntities (list): List of dictionaries (corresponding to predicted tokens).

    Returns:
        nestEntities (list): List of dictionaries with the corrected predictions
            regarding nested entities.
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

    Arguments:
        lEntities (list): List of dictionaries (corresponding to predicted tokens).

    Returns:
        fixEntities (list): List of dictionaries with the corrected predictions
            regarding grouping of labels.
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
    """
    If a word starts with ##, then this is a suffix, and word
    should therefore be joined with previous detected entity.

    Arguments:
        entity (dict): A dictionary containing the current entity (predicted token).
        lEntities (list): List of dictionaries (corresponding to all predicted tokens).

    Returns:
        lEntities (list): List of dictionaries with the corrected predictions
            regarding split tokens.
    """
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
