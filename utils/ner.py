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


def find_grouped_entities(sentence, ner_pipe):
    lEntities = []
    sentence = sentence.replace(".—", ". ") # Short en dash is not recognized as special character, replace
    sentence = sentence.replace("—.", ". ") # Short en dash is not recognized as special character, replace
    sentence = sentence.replace("—", ".") # Short en dash is not recognized as special character, replace
    prev_label = "LABEL_0"
    prev_entity = dict()
    for entity in ner_pipe(sentence):
        if entity["entity_group"] != "LABEL_0":
            
            # If prev_label is 1 (B-LOC) and current label is 2 (I-LOC), group entities:
            if prev_label == sequential_groups.get(entity["entity_group"]):
                lEntities = aggregateEntities(entity, lEntities)
                
            # If prev_label is 1 (B-LOC) and current label is 1 (B-LOC) and current word
            # starts with ## (meaning that it is a suffix, e.g. "over" and "##ton" is
            # actually "overton"), then group entities:
            elif entity["word"].startswith("##"):
                lEntities = aggregateEntities(entity, lEntities)
            
            # Otherwise, just append the entity to the list:
            else:
                lEntities.append(entity)
                
        prev_label = entity["entity_group"]
        prev_word = entity["word"]
    
    # List of entities:
    lEntitiesInText = []
    for entity in lEntities:
        # Find toponym as it appears in text:
        entity["toponym"] = sentence[entity["start"]:entity["end"]]
        
        # Map entity type to our labels:
        entity["place_class"] = label_dict[entity["entity_group"]]
        
        # Round the score to three decimals:
        entity["score"] = round(entity["score"], 3)
        
        # Remove not needed keys:
        del entity["entity_group"]
        del entity["word"]
        del entity["start"]
        del entity["end"]
        
        lEntitiesInText.append(entity)
    
    return lEntitiesInText