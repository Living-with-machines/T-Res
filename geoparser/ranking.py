import os
import sys
import json

import pandas as pd
from pathlib import Path
from DeezyMatch import candidate_ranker
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import deezy_processing


class Ranker:
    """
    A class to create a ranker object.
    """

    def __init__(
        self,
        method,
        resources_path,
        mentions_to_wikidata,
        wikidata_to_mentions,
        strvar_parameters=dict(),
        deezy_parameters=dict(),
        already_collected_cands=dict(),
    ):
        """
        Initialize the ranker.
        """
        self.method = method
        self.resources_path = resources_path
        self.mentions_to_wikidata = mentions_to_wikidata
        self.wikidata_to_mentions = wikidata_to_mentions
        self.strvar_parameters = strvar_parameters
        self.deezy_parameters = deezy_parameters
        self.already_collected_cands = already_collected_cands

    def __str__(self):
        """
        Print the ranker method name.
        """
        s = (">>> Candidate selection:\n" "    * Method: {0}\n").format(self.method)
        if self.method == "deezymatch":
            s += "    * DeezyMatch model: {0}\n".format(
                self.deezy_parameters["dm_model"]
            )
            s += "    * DeezyMatch ranking metric: {0}\n".format(
                self.deezy_parameters["ranking_metric"]
            )
            s += "    * DeezyMatch selection threshold: {0}\n".format(
                self.deezy_parameters["selection_threshold"]
            )
            s += "    * DeezyMatch num candidates: {0}\n".format(
                self.deezy_parameters["num_candidates"]
            )
            s += "    * DeezyMatch search size: {0}\n".format(
                self.deezy_parameters["search_size"]
            )
            s += "    * DeezyMatch overwrite training: {0}\n".format(
                self.deezy_parameters["overwrite_training"]
            )
            s += "    * DeezyMatch test mode: {0}\n".format(
                self.deezy_parameters["do_test"]
            )
        return s

    def load_resources(self):
        """
        Load resources required for linking, and initializes pandarallel
        if needed by the candidate ranking method.

        Returns:
            self.mentions_to_wikidata (dict): loads the mentions_to_wikidata.json
                dictionary, which maps a mention (e.g. "London") to the Wikidata
                entities that are referred to by this mention on Wikipedia (e.g.
                Q84, Q2477346) and mapped each entity with their relevance (i.e.
                number of inlinks) on Wikipedia.
        """
        print("*** Loading the ranker resources.")

        # Load Wikidata mentions-to-wqid:
        with open(
            self.resources_path + "mentions_to_wikidata_normalized.json", "r"
        ) as f:
            self.mentions_to_wikidata = json.load(f)

        # Load Wikidata wqid-to-mentions:
        with open(
            self.resources_path + "wikidata_to_mentions_normalized.json", "r"
        ) as f:
            self.wikidata_to_mentions = json.load(f)

        # Filter mentions to remove noise:
        wikidata_to_mentions_filtered = dict()
        mentions_to_wikidata_filtered = dict()
        for wk in self.wikidata_to_mentions:
            wikipedia_mentions = self.wikidata_to_mentions.get(wk)
            wikipedia_mentions_stripped = dict(
                [
                    (x, wikipedia_mentions[x])
                    for x in wikipedia_mentions
                    if not ", " in x and not " (" in x
                ]
            )
            if wikipedia_mentions_stripped:
                wikipedia_mentions = wikipedia_mentions_stripped
            wikidata_to_mentions_filtered[wk] = dict(
                [(x, wikipedia_mentions[x]) for x in wikipedia_mentions]
            )
            for m in wikidata_to_mentions_filtered[wk]:
                if m in mentions_to_wikidata_filtered:
                    mentions_to_wikidata_filtered[m][
                        wk
                    ] = wikidata_to_mentions_filtered[wk][m]
                else:
                    mentions_to_wikidata_filtered[m] = {
                        wk: wikidata_to_mentions_filtered[wk][m]
                    }
        self.mentions_to_wikidata = mentions_to_wikidata_filtered
        self.wikidata_to_mentions = wikidata_to_mentions_filtered
        del mentions_to_wikidata_filtered
        del wikidata_to_mentions_filtered

        # Parallelize if ranking method is one of the following:
        if self.method in ["partialmatch", "levenshtein"]:
            pandarallel.initialize(nb_workers=10)
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

        return self.mentions_to_wikidata

    def train(self):

        """
        Training a DeezyMatch model. The training will be skipped if the model already
        exists and self.overwrite_training it set to False. The training will
        be run on test mode if self.do_test is set to True.

        Returns:
            A trained DeezyMatch model.
            The DeezyMatch candidate vectors.
        """

        if self.method == "deezymatch":
            Path(self.deezy_parameters["dm_path"]).mkdir(parents=True, exist_ok=True)
            if self.deezy_parameters["do_test"] == True:
                self.deezy_parameters["dm_model"] += "_test"
                self.deezy_parameters["dm_cands"] += "_test"
            deezy_processing.create_training_set(self)
            deezy_processing.train_deezy_model(self)
            deezy_processing.generate_candidates(self)

    def perfect_match(self, queries):
        """
        Perform perfect match between the string and the KB altnames.
        """
        candidates = {}
        for query in queries:
            if query in self.already_collected_cands:
                candidates[query] = self.already_collected_cands[query]
            else:
                if query in self.mentions_to_wikidata:
                    candidates[query] = {query: 1.0}
                    self.already_collected_cands[query] = {query: 1.0}
                else:
                    candidates[query] = {}
                    self.already_collected_cands[query] = {}
        return candidates, self.already_collected_cands

    def damlev_dist(self, query: str, row: pd.Series) -> float:
        """
        Compute damerau levenshtein distance between query and Series

        Args:
            query (str): the mention identified in text
            row (Series): the row corresponding to a mention in the KB

        Returns:
            float: the similarity score, between 1.0 and 0.0
        """
        return 1.0 - normalized_damerau_levenshtein_distance(
            query.lower(), row["mentions"].lower()
        )

    def check_if_contained(self, query: str, row: pd.Series) -> float:
        """
        Takes a query and a Series and return the amount of overlap, if any

        Args:
            query (str): the mention identified in text
            row (Series): the row corresponding to a mention in the KB

        Returns:
            float: the size of overlap between query and mention, max 1.0 (perfect match)
        """
        s1 = query.lower()
        s2 = row["mentions"].lower()
        # E.g. query is 'Dorset' and candidate mention is 'County of Dorset'
        if s1 in s2:
            return len(query) / len(row["mentions"])
        # E.g. query is 'County of Dorset' and candidate mention is 'Dorset'
        if s2 in s1:
            return len(row["mentions"]) / len(query)

    def partial_match(self, queries: list, damlev: bool) -> tuple[dict, dict]:
        """
        Given a list of queries return a dict of partial matches for each of them
        and enrich another overall dictionary of candidatees

        Args:
            queries (list): list of mentions identified in a given sentence
            already_collected_cands (dict): dictionary of already processed mentions
            damlev (bool): either damerau_levenshtein or simple overlap

        Returns:
            tuple: two dictionaries: a (partial) match between queries->mentions
                                    an enriched version of already_collected_cands
        """

        candidates, self.already_collected_cands = self.perfect_match(queries)

        # the rest go through
        remainers = [x for x, y in candidates.items() if len(y) == 0]

        for query in remainers:
            mention_df = pd.DataFrame({"mentions": self.mentions_to_wikidata.keys()})
            if damlev:
                mention_df["score"] = mention_df.parallel_apply(
                    lambda row: self.damlev_dist(query, row), axis=1
                )
            else:
                mention_df["score"] = mention_df.parallel_apply(
                    lambda row: self.check_if_contained(query, row), axis=1
                )
            mention_df = mention_df.dropna()
            # currently hardcoded cutoff
            top_scores = sorted(
                list(set(list(mention_df["score"].unique()))), reverse=True
            )[:1]
            mention_df = mention_df[mention_df["score"].isin(top_scores)]
            mention_df = mention_df.set_index("mentions").to_dict()["score"]
            candidates[query] = mention_df
            self.already_collected_cands[query] = mention_df
        return candidates, self.already_collected_cands

    def deezy_on_the_fly(self, queries):
        """
        Given a list of queries return a dict of fuzzy matches for each of them,
        using DeezyMatch (with DeezyMatch parameters specified when initiating
        the Ranker object).

        Args:
            queries (list): list of mentions identified in a given sentence

        Returns:
            cands_dict: a (fuzzy) match between mentions and mentions in the KB
            self.already_collected_cands (dict): an enriched version of already_collected_cands
        """

        dm_path = self.deezy_parameters["dm_path"]
        dm_cands = self.deezy_parameters["dm_cands"]
        dm_model = self.deezy_parameters["dm_model"]
        dm_output = self.deezy_parameters["dm_output"]

        # first we fill in the perfect matches and already collected queries
        cands_dict, self.already_collected_cands = self.perfect_match(queries)

        # the rest go through
        remainers = [x for x, y in cands_dict.items() if len(y) == 0]

        if remainers:
            try:
                candidates = candidate_ranker(
                    candidate_scenario=os.path.join(
                        dm_path, "combined", dm_cands + "_" + dm_model
                    ),
                    query=remainers,
                    ranking_metric=self.deezy_parameters["ranking_metric"],
                    selection_threshold=self.deezy_parameters["selection_threshold"],
                    num_candidates=self.deezy_parameters["num_candidates"],
                    search_size=self.deezy_parameters["search_size"],
                    verbose=self.deezy_parameters["verbose"],
                    output_path=os.path.join(dm_path, "ranking", dm_output),
                    pretrained_model_path=os.path.join(
                        f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".model"
                    ),
                    pretrained_vocab_path=os.path.join(
                        f"{dm_path}", "models", f"{dm_model}", f"{dm_model}" + ".vocab"
                    ),
                )

                for idx, row in candidates.iterrows():
                    # Reverse cosine distance to cosine similarity:
                    returned_cands = dict()
                    if self.deezy_parameters["ranking_metric"] == "faiss":
                        returned_cands = row["faiss_distance"]
                        returned_cands = {
                            k: (
                                self.deezy_parameters["selection_threshold"]
                                - returned_cands[k]
                            )
                            / self.deezy_parameters["selection_threshold"]
                            for k in returned_cands
                        }
                    else:
                        returned_cands = row["cosine_dist"]
                        returned_cands = {
                            k: 1 - returned_cands[k] for k in returned_cands
                        }
                    cands_dict[row["query"]] = returned_cands
                    self.already_collected_cands[row["query"]] = returned_cands
            except TypeError:
                pass

        return cands_dict, self.already_collected_cands

    def run(self, queries):
        """
        Overall select ranking method.

        Arguments:
            queries (list): the list of mentions as they appear in the text,
                for a given sentence.

        Returns:
            A dictionary, resulting from running a certain rainking method.
        """
        if self.method == "perfectmatch":
            return self.perfect_match(queries)
        if self.method == "partialmatch":
            return self.partial_match(queries, damlev=False)
        if self.method == "levenshtein":
            return self.partial_match(queries, damlev=True)
        if self.method == "deezymatch":
            return self.deezy_on_the_fly(queries)

    def find_candidates(self, mentions):
        """
        Method that obtains potential candidates given the mentions
        detected in a sentence.

        Arguments:
            mentions (list): a list of predicted mentions as dictionaries.

        Returns:
            wk_cands (dict): a dictionary where we keep the original detected mention,
                the variation found by the candidate ranker in the knowledge base, and
                for each variation, we keep the candidate ranking score and the candidates
                in Wikidata. E.g. for mention "Guadaloupe" in sentence "sn83030483-
                1790-03-31-a-i0004_1, we store the candidates as follows:
                {'Guadaloupe': {'Score': 1.0, 'Candidates': {'Q17012': 10, 'Q3153836': 2}}}
        """
        # If method is relcs, candidates are collected in the ED step:
        if self.method == "relcs":
            return dict(), dict()
        # Otherwise:
        queries = list(set([mention["mention"] for mention in mentions]))
        cands, self.already_collected_cands = self.run(queries)
        wk_cands = dict()
        for original_mention in cands:
            wk_cands[original_mention] = dict()
            for variation in cands[original_mention]:
                match_score = cands[original_mention][variation]
                # Find Wikidata ID and relv.
                found_cands = self.mentions_to_wikidata.get(variation, dict())
                if not variation in wk_cands[original_mention]:
                    wk_cands[original_mention][variation] = {"Score": match_score}
                    wk_cands[original_mention][variation]["Candidates"] = found_cands
        return wk_cands, self.already_collected_cands
