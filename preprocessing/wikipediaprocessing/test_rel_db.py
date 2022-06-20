#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner import Cmns, load_flair_ner
from REL.utils import process_results

base_url = "/resources/wikipedia/rel_db/"
wiki_version = "lwm_rel_filtered"


def example_preprocessing():
    # user does some stuff, which results in the format below.
    text = "BRAMSHAW. OM mile from Brook and Cetilitre, four from Lyndhurst, and six from Romsey."
    processed = {"example_lwm": [text, []]}
    return processed


input_text = example_preprocessing()

mention_detection = MentionDetection(base_url, wiki_version)
tagger_ner = load_flair_ner("ner-fast")
mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)

config = {
    "mode": "eval",
    "model_path": base_url + "/" + wiki_version + "/generated/model",
}

model = EntityDisambiguation(base_url, wiki_version, config)
predictions, timing = model.predict(mentions_dataset)

result = process_results(mentions_dataset, predictions, input_text)

print(predictions, timing, result)
