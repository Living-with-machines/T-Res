import json
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_data, process_wikipedia


def rel_end_to_end(sent):
    """
    REL end-to-end entity linking using the API.

    Arguments:
        sent (str): a sentence in plain text.

    Returns:
        el_result (dict): the output from REL end-to-end API
            for the input sentence.
    """
    API_URL = "https://rel.cs.ru.nl/api"
    el_result = requests.post(API_URL, json={"text": sent, "spans": []}).json()
    return el_result


def get_rel_from_api(dSentences, rel_end2end_path):
    """
    Uses the REL API to do end-to-end entity linking.

    Arguments:
        dSentences (dict): dictionary of sentences, where the
            key is the article-sent identifier and the value
            is the full text of the sentence.
        rel_end2end_path (str): the path of the file where the
            REL results will be stored.

    Returns:
        A JSON file with the REL results.
    """
    # Dictionary to store REL predictions:
    rel_preds = dict()
    if Path(rel_end2end_path).exists():
        with open(rel_end2end_path) as f:
            rel_preds = json.load(f)
    print("\nObtain REL linking from API (unless already stored):")
    for s in tqdm(dSentences):
        if not s in rel_preds:
            rel_preds[s] = rel_end_to_end(dSentences[s])
            # Append per processed sentence in case of API limit:
            with open(rel_end2end_path, "w") as fp:
                json.dump(rel_preds, fp)
            with open(rel_end2end_path) as f:
                rel_preds = json.load(f)


def match_wikipedia_to_wikidata(wiki_title):
    """
    Get the Wikidata ID from a Wikipedia title.

    Arguments:
        wiki_title (str): a Wikipedia title, underscore-separated.

    Returns:
        a string, either the Wikidata QID corresponding entity, or NIL.
    """
    wqid = process_wikipedia.title_to_id(
        wiki_title,
        lower=False,
        path_to_db="../resources/wikipedia/index_enwiki-latest.db",
    )
    if not wqid:
        wqid = "NIL"
    return wqid


def match_ent(pred_ents, start, end, prev_ann, gazetteer_ids):
    """
    Function that, given the position in a sentence of a
    specific gold standard token, finds the corresponding
    string and prediction information returned by REL.

    Arguments:
        pred_ents (list): a list of lists, each inner list
            corresponds to a token.
        start (int): start character of a token in the gold standard.
        end (int): end character of a token in the gold standard.
        prev_ann (str): entity type of the previous token.

    Returns:
        A tuple with three elements: (1) the entity type, (2) the
        entity link and (3) the entity type of the previous token.
    """
    for ent in pred_ents:
        wqid = match_wikipedia_to_wikidata(ent[3])
        # If entity is a LOC or linked entity is in our KB:
        if ent[-1] == "LOC" or wqid in gazetteer_ids:
            # Any place with coordinates is considered a location
            # throughout our experiments:
            ent_type = "LOC"
            st_ent = ent[0]
            len_ent = ent[1]
            if start >= st_ent and end <= (st_ent + len_ent):
                if prev_ann == ent_type:
                    ent_pos = "I-"
                else:
                    ent_pos = "B-"
                    prev_ann = ent_type

                n = ent_pos + ent_type
                try:
                    el = ent_pos + match_wikipedia_to_wikidata(ent[3])
                except Exception as e:
                    print(e)
                    # to be checked but it seems some Wikipedia pages are not in our Wikidata
                    # see for instance Zante%2C%20California
                    return n, "O", ""
                return n, el, prev_ann
    return "O", "O", ""


def postprocess_rel(rel_preds, dSentences, gold_tokenization, wikigaz_ids):
    """
    For each sentence, retokenizes the REL output to match the gold
    standard tokenization.

    Arguments:
        rel_preds (dict): dictionary containing the predictions using REL.
        dSentences (dict): dictionary that maps a sentence id to the text.
        gold_tokenization (dict): dictionary that contains the tokenized
            sentence with gold standard annotations of entity type and
            link, per sentence.
        wikigaz_ids (set): set of Wikidata IDs of entities in the gazetteer.

    Returns:
        dREL (dict): dictionary that maps a sentence id with the REL predictions,
            retokenized as in the gold standard.
    """
    dREL = dict()
    for sent_id in tqdm(list(dSentences.keys())):
        sentence_preds = []
        prev_ann = ""
        for token in gold_tokenization[sent_id]:
            start = token["start"]
            end = token["end"]
            word = token["word"]
            current_preds = rel_preds.get(sent_id, [])
            n, el, prev_ann = match_ent(
                current_preds, start, end, prev_ann, wikigaz_ids
            )
            sentence_preds.append([word, n, el])
        dREL[sent_id] = sentence_preds
    return dREL


def store_rel(experiment, dREL, approach, how_split):
    """
    Prepare the data to be stored in the format required by the HIPE scorer.

    Arguments:
        experiment (Experiment object): object for the current experiment.
        dREL (dict): dictionary with the results using the REL approach.
        approach (str): name of the REL approach (only rel_end_to_end_api available).
        how_split (str): data split to store the results for.

    Returns:
        A tsv with the results in the Conll format required by the scorer.

    """
    hipe_scorer_results_path = os.path.join(experiment.results_path, experiment.dataset)
    scenario_name = (
        approach
        + "_"
        + experiment.myner.model  # The model name is needed due to tokenization
        + "_"
        + how_split
    )

    # Find article ids of the corresponding test set (e.g. 'dev' of the original split,
    # 'test' of the Ashton1860 split, etc):
    all = experiment.dataset_df
    test_articles = list(all[all[how_split] == "test"].article_id.unique())
    test_articles = [str(art) for art in test_articles]

    # Store REL results formatted for CLEF-HIPE scorer:
    process_data.store_for_scorer(
        hipe_scorer_results_path,
        scenario_name,
        dREL,
        test_articles,
    )


def run_rel_experiments(self):
    """
    Function that runs the end-to-end experiments using REL.
    """
    # Continue only if flag is True:
    if self.rel_experiments == False:
        return

    print("--------------------------\n")
    print("Start the REL experiments.\n")
    self.processed_data = self.load_data()

    # List of evaluation scenarios
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

    dict_rel_approaches = dict()

    # Run REL end-to-end:
    rel_approach_name = "rel_end_to_end_api"
    Path(self.results_path + self.dataset).mkdir(parents=True, exist_ok=True)
    rel_end2end_path = self.results_path + self.dataset + "/rel_e2d_from_api.json"
    get_rel_from_api(self.processed_data["dSentences"], rel_end2end_path)
    with open(rel_end2end_path) as f:
        rel_preds = json.load(f)
    dREL = postprocess_rel(
        rel_preds,
        self.processed_data["dSentences"],
        self.processed_data["gold_tok"],
        self.mylinker.linking_resources["wikidata_locs"],
    )
    dict_rel_approaches[rel_approach_name] = {"results": dREL}

    # Store results for each split
    for rel_approach_name in dict_rel_approaches:
        for test_split in dict_splits:
            for split in dict_splits[test_split]:
                # Process REL results:
                store_rel(
                    self,
                    dict_rel_approaches[rel_approach_name]["results"],
                    approach=rel_approach_name,
                    how_split=split,
                )

    print("... done!\n")
