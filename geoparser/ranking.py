import os
import json

import pandas as pd
from DeezyMatch import candidate_ranker
from pandarallel import pandarallel
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance


class Ranker:
    """
    A class to create a ranker object.
    """

    def __init__(
        self,
        method,
        resources_path,
        mentions_to_wikidata,
        deezy_parameters=dict(),
        already_collected_cands=dict(),
    ):
        """
        Initialize the ranker.
        """
        self.method = method
        self.resources_path = resources_path
        self.mentions_to_wikidata = mentions_to_wikidata
        self.deezy_parameters = deezy_parameters
        self.already_collected_cands = already_collected_cands

    def __str__(self):
        """
        Print the ranker method name.
        """
        s = "Candidate selection:\n* Method: {0}\n".format(self.method)
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
        # Load Wikidata mentions-to-wqid:
        with open(self.resources_path + "mentions_to_wikidata.json", "r") as f:
            self.mentions_to_wikidata = json.load(f)

        # Parallelize if ranking method is one of the following:
        if self.method in ["partialmatch", "levenshtein"]:
            pandarallel.initialize(nb_workers=10)
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

        return self.mentions_to_wikidata

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
                    candidate_scenario=dm_path
                    + "combined/"
                    + dm_cands
                    + "_"
                    + dm_model,
                    query=remainers,
                    ranking_metric=self.deezy_parameters["ranking_metric"],
                    selection_threshold=self.deezy_parameters["selection_threshold"],
                    num_candidates=self.deezy_parameters["num_candidates"],
                    search_size=self.deezy_parameters["search_size"],
                    use_predict=self.deezy_parameters["use_predict"],
                    verbose=self.deezy_parameters["verbose"],
                    output_path=dm_path + "ranking/" + dm_output,
                    pretrained_model_path=dm_path
                    + "models/"
                    + dm_model
                    + "/"
                    + dm_model
                    + ".model",
                    pretrained_vocab_path=dm_path
                    + "models/"
                    + dm_model
                    + "/"
                    + dm_model
                    + ".vocab",
                )

                for idx, row in candidates.iterrows():
                    # Reverse cosine distance to cosine similarity:
                    returned_cands = row["cosine_dist"]
                    returned_cands = {k: 1 - returned_cands[k] for k in returned_cands}
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
