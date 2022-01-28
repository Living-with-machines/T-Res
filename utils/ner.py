from utils.eval import collect_named_entities, Entity

# Labels 2, 4, 6, 8, and 10 ar I- labels, they follow B-labels:
sequential_groups = dict()
sequential_groups["LABEL_2"] = "LABEL_1"
sequential_groups["LABEL_4"] = "LABEL_3"
sequential_groups["LABEL_6"] = "LABEL_5"
sequential_groups["LABEL_8"] = "LABEL_7"
sequential_groups["LABEL_10"] = "LABEL_9"


# Dictionary mapping NER model label with our label:
label_dict = {"LABEL_0": "UNKNOWN",
              "LABEL_1": "LOC",
              "LABEL_2": "LOC",
              "LABEL_3": "STREET",
              "LABEL_4": "STREET",
              "LABEL_5": "BUILDING",
              "LABEL_6": "BUILDING",
              "LABEL_7": "OTHER",
              "LABEL_8": "OTHER",
              "LABEL_9": "FICTION",
              "LABEL_10": "FICTION"}


def aggregateEntities(entity, lEntities):
    # Group entities
    prevEntity = lEntities.pop()
    newEntity = dict()
    # If word starts with ##, then this is a suffix, join with previous detected entity
    if entity["word"].startswith("##"):
        newEntity = {'entity_group': prevEntity["entity_group"], 
                     'score': ((prevEntity["score"] + entity["score"]) / 2.0), 
                     'word': prevEntity["word"] + entity["word"].replace("##", ""), 
                     'start': prevEntity["start"],
                     'end': entity["end"]}
        
    # If label is a I-label and prev label is its corresponding B-label, then join
    # with previous detected entity.
    else:
        newEntity = {'entity_group': prevEntity["entity_group"], 
                     'score': ((prevEntity["score"] + entity["score"]) / 2.0), 
                     'word': prevEntity["word"] + " " + entity["word"], 
                     'start': prevEntity["start"],
                     'end': entity["end"]}
    lEntities.append(newEntity)
    return lEntities

from transformers import pipeline

def ner_predict(texts, ner_model):
    # Check different aggregation strategies: https://huggingface.co/transformers/v4.10.1/_modules/transformers/pipelines/token_classification.html
    ner_pipe = pipeline("ner", model=ner_model, aggregation_strategy="none", use_fast=True)
    ner_results = [ner_pipe(text) for text in texts]
    return ner_results

def aggregate_mentions(preds):
    pred_ner_labels = [[x[1] for x in x] for x in preds]

    found_mentions = []

    for p in range(len(preds)):
        pred = preds[p]
        mentions = collect_named_entities(pred_ner_labels[p])
        sent_mentions = []
        for mention in mentions:
            text_mention = " ".join([pred[r][0] for r in range(mention.start_offset, mention.end_offset+1)])
            sent_mentions.append({"mention":text_mention,"start_offset":mention.start_offset,"end_offset":mention.end_offset})
        found_mentions.append(sent_mentions)

    return found_mentions


def collect_named_entities(tokens):
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

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities