import os
import re
import sys
import urllib

from haversine import haversine

from utils.REL.db.generic import GenericLookup
from utils.REL.utils import modify_uppercase_phrase, split_in_words

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import process_wikipedia


class MentionDetectionBase:
    def __init__(self, base_url, wiki_version, mylinker=None):
        self.wiki_db = GenericLookup(
            "entity_word_embedding", os.path.join(base_url, wiki_version, "generated")
        )
        self.mylinker = mylinker

    def get_ctxt(self, start, end, idx_sent, sentence, sentences_doc):
        """
        Retrieves context surrounding a given mention up to 100 words from both sides.

        :return: left and right context
        """

        # Iteratively add words up until we have 100
        left_ctxt = split_in_words(sentence[:start])
        if idx_sent > 0:
            i = idx_sent - 1
            while (i >= 0) and (len(left_ctxt) <= 100):
                left_ctxt = split_in_words(sentences_doc[i]) + left_ctxt
                i -= 1
        left_ctxt = left_ctxt[-100:]
        left_ctxt = " ".join(left_ctxt)

        right_ctxt = split_in_words(sentence[end:])
        if idx_sent < len(sentences_doc):
            i = idx_sent + 1
            while (i < len(sentences_doc)) and (len(right_ctxt) <= 100):
                right_ctxt = right_ctxt + split_in_words(sentences_doc[i])
                i += 1
        right_ctxt = right_ctxt[:100]
        right_ctxt = " ".join(right_ctxt)

        return left_ctxt, right_ctxt

    def get_candidates(self, mention, cand_selection="relcs", lwm_cands=None):
        """
        Retrieves a maximum of 100 candidates from the sqlite3 database for a given mention.

        :return: set of candidates
        """

        #### CANDIDATE SELECTION FROM REL
        if cand_selection == "relcs":
            cands = self.wiki_db.wiki(mention, "wiki")
            if cands:
                cands = cands[:100]
                # Keep only candidates that are locations:
                cands = [
                    c
                    for c in cands
                    if process_wikipedia.title_to_id(c[0], lower=False)
                    in self.mylinker.linking_resources["wikidata_locs"]
                ]
                return cands
        if cand_selection == "lwmcs":
            cands = []
            tmp_cands = []
            max_cand_freq = 0
            for c in lwm_cands:
                for qc in lwm_cands[c]["Candidates"]:
                    qcrlv = self.mylinker.linking_resources["mentions_to_wikidata"][c][
                        qc
                    ]
                    if qcrlv > max_cand_freq:
                        max_cand_freq = qcrlv
                    # Wikidata entity to Wikipedia
                    qc_wikipedia = ""
                    gold_ids = process_wikipedia.id_to_title(qc)
                    if gold_ids:
                        qc_wikipedia = gold_ids[0]
                    tmp_cands.append((qc_wikipedia, qcrlv))
            # Append candidate and normalized score weighted by candidate selection conf:
            for cand in tmp_cands:
                qc_wikipedia = cand[0]
                qc_score = round(cand[1] / max_cand_freq, 3)
                cands.append([qc_wikipedia, qc_score])
            return cands
            # print(self.mylinker.most_popular(lwm_mention))
            # pass
            # print(lwm_mention)
            # print("==", self.mylinker.most_popular(lwm_mention))
            # print(lwm_mention)
            # else:
            #     print("HERE:", lwm_mention)
            #     cands = []
            #     tmp_cands = []
            #     max_cand_freq = 0
            #     for c in lwm_cands:
            #         for qc in lwm_cands[c]["Candidates"]:
            #             # Mention-to-entity releavance:
            #             qcrlv = self.mylinker.linking_resources["mentions_to_wikidata"][
            #                 c
            #             ][qc]
            #             if qcrlv > max_cand_freq:
            #                 max_cand_freq = qcrlv
            #             # Wikidata entity to Wikipedia:
            #             gold_ids = self.mylinker.linking_resources[
            #                 "wikidata2wikipedia"
            #             ].get(qc)
            #             qc_wikipedia = ""
            #             max_freq = 0
            #             if gold_ids:
            #                 for k in gold_ids:
            #                     if k["freq"] > max_freq:
            #                         max_freq = k["freq"]
            #                         qc_wikipedia = k["title"]
            #             tmp_cands.append((qc_wikipedia, qcrlv))
            #     # Append candidate and normalized score weighted by candidate selection conf:
            #     for cand in tmp_cands:
            #         qc_wikipedia = urllib.parse.unquote(cand[0]).replace(" ", "_")
            #         qc_score = round(cand[1] / max_cand_freq, 3)
            #         cands.append([qc_wikipedia, qc_score])
            #     return cands

            # elif cs_method == "lwmcs":

            #     cs_ranking = self.mylinker.rel_params["ranking"]

            #     ### CANDIDATE RANKING: Based on distance from publication
            #     if cs_ranking == "dist":
            #         cands = []
            #         tmp_cands = []
            #         max_dist = 0
            #         for c in lwm_cands:
            #             for qc in lwm_cands[c]["Candidates"]:
            #                 lat_publ = self.mylinker.linking_resources["dict_wqid_to_lat"][
            #                     publication
            #                 ]
            #                 lon_publ = self.mylinker.linking_resources["dict_wqid_to_lon"][
            #                     publication
            #                 ]
            #                 lat_cand = self.mylinker.linking_resources["dict_wqid_to_lat"][
            #                     qc
            #                 ]
            #                 lon_cand = self.mylinker.linking_resources["dict_wqid_to_lon"][
            #                     qc
            #                 ]
            #                 # Distance between place of publication and candidate:
            #                 qcdist = haversine((lat_publ, lon_publ), (lat_cand, lon_cand))
            #                 # Keep max distance for later normalizing:
            #                 if qcdist > max_dist:
            #                     max_dist = qcdist
            #                 # Wikidata entity to Wikipedia:
            #                 gold_ids = self.mylinker.linking_resources[
            #                     "wikidata2wikipedia"
            #                 ].get(qc)
            #                 qc_wikipedia = ""
            #                 max_freq = 0
            #                 if gold_ids:
            #                     for k in gold_ids:
            #                         if k["freq"] > max_freq:
            #                             max_freq = k["freq"]
            #                             qc_wikipedia = k["title"]
            #                 tmp_cands.append((qc_wikipedia, qcdist))
            #         # Append candidate and normalized score weighted by candidate selection conf:
            #         for cand in tmp_cands:
            #             qc_wikipedia = urllib.parse.unquote(cand[0]).replace(" ", "_")
            #             qc_score = round(1 - (cand[1] / max_dist), 3)
            #             cands.append([qc_wikipedia, qc_score])
            #         return cands

            #     ### CANDIDATE RANKING: Based on wikipedia mention2entity relevance (covers all other cases apart from dist)
            #     else:
            #         cands = []
            #         tmp_cands = []
            #         max_cand_freq = 0
            #         for c in lwm_cands:
            #             for qc in lwm_cands[c]["Candidates"]:
            #                 # Mention-to-entity releavance:
            #                 qcrlv = self.mylinker.linking_resources["mentions_to_wikidata"][
            #                     c
            #                 ][qc]
            #                 if qcrlv > max_cand_freq:
            #                     max_cand_freq = qcrlv
            #                 # Wikidata entity to Wikipedia:
            #                 gold_ids = self.mylinker.linking_resources[
            #                     "wikidata2wikipedia"
            #                 ].get(qc)
            #                 qc_wikipedia = ""
            #                 max_freq = 0
            #                 if gold_ids:
            #                     for k in gold_ids:
            #                         if k["freq"] > max_freq:
            #                             max_freq = k["freq"]
            #                             qc_wikipedia = k["title"]
            #                 tmp_cands.append((qc_wikipedia, qcrlv))
            #         # Append candidate and normalized score weighted by candidate selection conf:
            #         for cand in tmp_cands:
            #             qc_wikipedia = urllib.parse.unquote(cand[0]).replace(" ", "_")
            #             qc_score = round(cand[1] / max_cand_freq, 3)
            #             cands.append([qc_wikipedia, qc_score])
            #         return cands
            # else:
            #     print(
            #         "Candidate selection method not currently covered, returning empty cands list"
            #     )
        return []

    def preprocess_mention(self, m):
        """
        Responsible for preprocessing a mention and making sure we find a set of matching candidates
        in our database.

        :return: mention
        """

        # TODO: This can be optimised (less db calls required).
        cur_m = modify_uppercase_phrase(m)
        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")

        if not freq_lookup_cur_m:
            cur_m = m

        freq_lookup_m = self.wiki_db.wiki(m, "wiki", "freq")
        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")

        if freq_lookup_m and (freq_lookup_m > freq_lookup_cur_m):
            # Cases like 'U.S.' are handed badly by modify_uppercase_phrase
            cur_m = m

        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")
        # If we cannot find the exact mention in our index, we try our luck to
        # find it in a case insensitive index.
        if not freq_lookup_cur_m:
            # cur_m and m both not found, verify if lower-case version can be found.
            find_lower = self.wiki_db.wiki(m.lower(), "wiki", "lower")

            if find_lower:
                cur_m = find_lower

        freq_lookup_cur_m = self.wiki_db.wiki(cur_m, "wiki", "freq")
        # Try and remove first or last characters (e.g. 'Washington,' to 'Washington')
        # To be error prone, we only try this if no match was found thus far, else
        # this might get in the way of 'U.S.' converting to 'US'.
        # Could do this recursively, interesting to explore in future work.
        if not freq_lookup_cur_m:
            temp = re.sub(r"[\(.|,|!|')]", "", m).strip()
            simple_lookup = self.wiki_db.wiki(temp, "wiki", "freq")

            if simple_lookup:
                cur_m = temp

        return cur_m
