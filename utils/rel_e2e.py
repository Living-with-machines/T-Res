import json
import os
import sys
from pathlib import Path

import pandas as pd

# Import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data
from utils.REL.entity_disambiguation import EntityDisambiguation

# REL imports
from utils.REL.mention_detection import MentionDetection
from utils.REL.ner import load_flair_ner


def run_rel_experiments(self):

    # Continue only is flag is True:
    if self.rel_experiments == False:
        return

    print("--------------------------\n")
    print("Start the REL experiments.\n")
    self.processed_data = self.load_data()

    # -------------------------------------------
    # 0. LIST EVALUATION SCENARIOS
    # -------------------------------------------
    dict_splits = dict()
    dict_splits["dev"] = ["originalsplit"]
    if self.dataset == "hipe":
        dict_splits["test"] = ["originalsplit"]
    elif self.dataset == "lwm":
        dict_splits["test"] = [
            "originalsplit",
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

    # -------------------------------------------
    # REL: SUMMARY OF APPROACHES
    # -------------------------------------------
    dict_rel_approaches = dict()

    # -------------------------------------------
    # 1. END TO END FROM API
    # -------------------------------------------
    # Run REL end-to-end, as well
    # Note: Do not move the next block of code,
    # as REL relies on the tokenisation performed
    # by the previous method, so it needs to be
    # run after ther our method.
    print("* REL: Approach 1")
    rel_approach_name = "rel_end_to_end_api"
    Path(self.results_path + self.dataset).mkdir(parents=True, exist_ok=True)
    rel_end2end_path = self.results_path + self.dataset + "/rel_e2d_from_api.json"
    process_data.get_rel_from_api(self.processed_data["dSentences"], rel_end2end_path)
    with open(rel_end2end_path) as f:
        rel_preds = json.load(f)
    dREL = process_data.postprocess_rel(
        rel_preds,
        self.processed_data["dSentences"],
        self.processed_data["gold_tok"],
    )
    dict_rel_approaches[rel_approach_name] = {"results": dREL}

    # # -------------------------------------------
    # # 2. END TO END USING 2019 WIKI DUMP, TRAINED ON AIDA
    # # -------------------------------------------
    # print("* REL: Approach 2")

    # rel_approach_name = "rel_wiki2019_aida"
    # base_path = "/resources/rel_db"
    # wiki_version = "wiki_2019"
    # model_name = "ed-wiki-2019"
    # config = {
    #     "mode": "eval",
    #     "model_path": "{}/{}/model".format(base_path, model_name),
    # }
    # mention_detection = MentionDetection(base_path, wiki_version)
    # tagger_ner = load_flair_ner("ner-fast")
    # linking_model = EntityDisambiguation(base_path, wiki_version, config)
    # rel_preds = process_data.get_rel_locally(
    #     self.processed_data["dSentences"],
    #     mention_detection,
    #     tagger_ner,
    #     linking_model,
    # )
    # dREL = process_data.postprocess_rel(
    #     rel_preds,
    #     self.processed_data["dSentences"],
    #     self.processed_data["gold_tok"],
    # )

    # dict_rel_approaches[rel_approach_name] = {
    #     "base_path": base_path,
    #     "wiki_version": wiki_version,
    #     "model_name": model_name,
    #     "results": dREL,
    # }

    # -------------------------------------------
    # N. STORE RESULTS PER EVAL SCENARIO
    # -------------------------------------------

    # Store results for each split
    for rel_approach_name in dict_rel_approaches:
        for test_split in dict_splits:
            for split in dict_splits[test_split]:

                # Process REL results:
                process_data.store_rel(
                    self,
                    dict_rel_approaches[rel_approach_name]["results"],
                    approach=rel_approach_name,
                    how_split=split,
                    which_split=test_split,
                )

    print("... done!\n")
