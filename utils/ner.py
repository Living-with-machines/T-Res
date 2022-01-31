from utils.eval import collect_named_entities, Entity
from transformers import pipeline


# Dictionary mapping NER model label with GS label:
label_dict = {"LABEL_0": "O",
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
              

def aggregate_entities(entity, lEntities):
    newEntity = entity
    # We remove the word index because we're altering it (by joining suffixes)
    newEntity.pop('index', None)
    # If word starts with ##, then this is a suffix, join with previous detected entity
    if entity["word"].startswith("##"):
        prevEntity = lEntities.pop()
        newEntity = {'entity': prevEntity["entity"], 
                     'score': ((prevEntity["score"] + entity["score"]) / 2.0), 
                     'word': prevEntity["word"] + entity["word"].replace("##", ""), 
                     'start': prevEntity["start"],
                     'end': entity["end"]}
        
    lEntities.append(newEntity)
    return lEntities


def ner_predict(df, ner_model):
    # Load NER pipeline, aggregate grouped entities with "average":
    ner_pipe = pipeline("ner", model=ner_model)

    
    # In the dAnnotatedClasses dictionary, we keep, for each article/sentence,
    # a dictionary that maps the position of an annotated named entity (i.e.
    # its start and end character, as a tuple, as the key of the inner dictionary)
    # with the class of named entity (such as LOC or BUILDING, which is
    # the value of the inner dictioary).
    dAnnotatedClasses = dict()
    for i, row in df.iterrows():
        # sent_id is the unique identifier for the article/sentence pair
        sent_id = str(row["article_id"]) + "_" + str(row["sent_id"])
        position = (int(row["start"]), int(row["end"]))
        if sent_id in dAnnotatedClasses:
            dAnnotatedClasses[sent_id][position] = row["place_class"]
        else:
            dAnnotatedClasses[sent_id] = {position: row["place_class"]}

    
    # In the dAnnotatedLinks dictionary, we keep, for each article/sentence,
    # a dictionary that maps the position of an annotated named entity (i.e.
    # its start and end character, as a tuple, as the key of the inner dictionary)
    # with the class of named entity (such as LOC or BUILDING, which is
    # the value of the inner dictioary).
    dAnnotatedLinks = dict()
    for i, row in df.iterrows():
        # sent_id is the unique identifier for the article/sentence pair
        sent_id = str(row["article_id"]) + "_" + str(row["sent_id"])
        position = (int(row["start"]), int(row["end"]))
        wqlink = row["place_wqid"]
        if not isinstance(wqlink, str):
            wqlink = "*"
        if sent_id in dAnnotatedLinks:
            dAnnotatedLinks[sent_id][position] = wqlink
        else:
            dAnnotatedLinks[sent_id] = {position: wqlink}


    # The dPredictions dictionary keeps the results of the BERT NER
    # as a list of dictionaries (value) for each article/sentence pair (key).
    dPredictions = dict()
    for i, row in df.iterrows():
        sent_id = str(row["article_id"]) + "_" + str(row["sent_id"])
        ner_preds = ner_pipe(row["current_sentence"])
        lEntities = []
        for pred_ent in ner_preds:
            pred_ent["entity"] = label_dict[pred_ent["entity"]]
            dPredictions[sent_id] = aggregate_entities(pred_ent, lEntities)


    # The dGoldStandard dictionary is an alignment between the output
    # of BERT NER (as it uses its own tokenizer) and the gold standard
    # labels. It does so based on the start and end position of each
    # predicted token. By default, a predicted token is assigned the
    # "O" label, unless its position overlaps with the position an
    # annotated entity, in which case we relabel it according to this
    # label.
    dGoldStandard = dict()
    for sent_id in dPredictions:
        ner_preds = dPredictions[sent_id]
        dGoldStandard[sent_id] = []
        for pred_ent in ner_preds:
            gs_for_eval = pred_ent.copy()
            # This has been manually annotated, so perfect score
            gs_for_eval["score"] = 1.0
            # We instantiate the entity class as "O" ("outside", i.e. not a NE)
            gs_for_eval["entity"] = "O"
            gs_for_eval["link"] = "O"
            # It's prefixed as "B-" if the token is the first in a sequence,
            # otherwise it's prefixed as "I-"
            for gse in dAnnotatedClasses[sent_id]:
                if pred_ent["start"] == gse[0] and pred_ent["end"] <= gse[1]:
                    gs_for_eval["entity"] = dAnnotatedClasses[sent_id][gse]
                    gs_for_eval["link"] = dAnnotatedLinks[sent_id][gse]
                elif pred_ent["start"] > gse[0] and pred_ent["end"] <= gse[1]:
                    gs_for_eval["entity"] = dAnnotatedClasses[sent_id][gse]
                    gs_for_eval["link"] = dAnnotatedLinks[sent_id][gse]
            dGoldStandard[sent_id].append(gs_for_eval)
            

    return dGoldStandard, dPredictions


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