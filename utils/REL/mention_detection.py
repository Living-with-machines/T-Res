from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single
from ast import literal_eval
import urllib

from REL.REL.mention_detection_base import MentionDetectionBase

"""
Class responsible for mention detection.
"""


class MentionDetection(MentionDetectionBase):
    def __init__(self, base_url, wiki_version, mylinker=None):
        self.cnt_exact = 0
        self.cnt_partial = 0
        self.cnt_total = 0

        super().__init__(base_url, wiki_version, mylinker)

    def format_detected_spans(self, test_df, original_df, mylinker=None):
        """
        Responsible for formatting given spans into dataset for the ED step. More specifically,
        it returns the mention, its left/right context and a set of candidates.

        :return: Dictionary with mentions per document.
        """
        results = dict()
        total_ment = 0
        dict_sentences = dict()
        # Collect document sentences:
        for i, row in original_df.iterrows():
            article_id = row["article_id"]
            for sentence in literal_eval(row["sentences"]):
                dict_sentences[
                    str(article_id) + "_" + str(sentence["sentence_pos"])
                ] = sentence["sentence_text"]

        # Get REL candidates for our recognized toponyms:
        for i, prediction in test_df.iterrows():
            article_id = prediction["article_id"]
            dict_mention = dict()
            dict_mention["mention"] = prediction["pred_mention"]
            sent_idx = int(prediction["sentence_pos"])
            dict_mention["sent_idx"] = sent_idx
            dict_mention["sentence"] = dict_sentences[
                str(article_id) + "_" + str(sent_idx)
            ]
            dict_mention["ngram"] = prediction["pred_mention"]
            dict_mention["context"] = ["", ""]
            if str(article_id) + "_" + str(sent_idx - 1) in dict_sentences:
                dict_mention["context"][0] = dict_sentences[
                    str(article_id) + "_" + str(sent_idx - 1)
                ]
            if str(article_id) + "_" + str(sent_idx + 1) in dict_sentences:
                dict_mention["context"][1] = dict_sentences[
                    str(article_id) + "_" + str(sent_idx + 1)
                ]
            dict_mention["pos"] = prediction["char_start"]
            dict_mention["end_pos"] = prediction["char_end"]
            if "reldisamb:relcs" in mylinker.method:
                dict_mention["candidates"] = self.get_candidates(
                    dict_mention["mention"]
                )
            # Use LwM candidates weighted by mention2wikidata relevance and candselection conf:
            # TODO Actually this should happen in the get_candidates function, so it's in the training as well.
            # TODO Convert to Wikipedia......
            if "reldisamb:lwmcs" in mylinker.method:
                dict_mention["candidates"] = self.get_candidates(
                    dict_mention["mention"],
                    prediction["candidates"],
                    prediction["place_wqid"],
                )
            dict_mention["gold"] = ["NONE"]
            dict_mention["tag"] = prediction["pred_ner_label"]
            dict_mention["conf_md"] = prediction["ner_score"]
            total_ment += 1

            if article_id in results:
                if sent_idx in results[article_id]:
                    results[article_id][sent_idx].append(dict_mention)
                else:
                    results[article_id][sent_idx] = [dict_mention]
            else:
                results[article_id] = {sent_idx: [dict_mention]}

            if "publ" in self.mylinker.method:
                # Add place of publication as an entity in each sentence:
                # Wikipedia title of place of publication QID:
                wiki_gold = "NIL"
                gold_ids = self.mylinker.linking_resources["wikidata2wikipedia"].get(
                    prediction["place_wqid"]
                )
                max_freq = 0
                if gold_ids:
                    for k in gold_ids:
                        if k["freq"] > max_freq:
                            max_freq = k["freq"]
                            wiki_gold = k["title"]
                wiki_gold = urllib.parse.unquote(wiki_gold).replace(" ", "_")
                sent2 = (
                    dict_mention["sentence"] + " Published in " + prediction["place"]
                )
                pos2 = len(dict_mention["sentence"]) + len(" Published in ")
                end_pos2 = pos2 + len(prediction["place"])
                dict_publ = {
                    "mention": "publication",
                    "sent_idx": dict_mention["sent_idx"],
                    "sentence": sent2,
                    "gold": [wiki_gold],
                    "ngram": "publication",
                    "context": dict_mention["context"],
                    "pos": pos2,
                    "end_pos": end_pos2,
                    "candidates": [[wiki_gold, 1.0]],
                }
                results[article_id][sent_idx].append(dict_publ)

        return results, total_ment

    def format_spans(self, dataset):
        """
        Responsible for formatting given spans into dataset for the ED step. More specifically,
        it returns the mention, its left/right context and a set of candidates.

        :return: Dictionary with mentions per document.
        """

        dataset, _, _ = self.split_text(dataset)
        results = {}
        total_ment = 0

        for doc in dataset:
            contents = dataset[doc]
            sentences_doc = [v[0] for v in contents.values()]

            results_doc = []
            for idx_sent, (sentence, spans) in contents.items():
                for ngram, start_pos, end_pos in spans:
                    total_ment += 1

                    # end_pos = start_pos + length
                    # ngram = text[start_pos:end_pos]
                    mention = self.preprocess_mention(ngram)
                    left_ctxt, right_ctxt = self.get_ctxt(
                        start_pos, end_pos, idx_sent, sentence, sentences_doc
                    )

                    chosen_cands = self.get_candidates(mention)
                    res = {
                        "mention": mention,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": chosen_cands,
                        "gold": ["NONE"],
                        "pos": start_pos,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end_pos,
                        "sentence": sentence,
                    }

                    results_doc.append(res)
            results[doc] = results_doc
        return results, total_ment

    def split_text(self, dataset, is_flair=False):
        """
        Splits text into sentences with optional spans (format is a requirement for GERBIL usage).
        This behavior is required for the default NER-tagger, which during experiments was experienced
        to achieve higher performance.

        :return: dictionary with sentences and optional given spans per sentence.
        """

        res = {}
        splits = [0]
        processed_sentences = []
        for doc in dataset:
            text, spans = dataset[doc]
            sentences = split_single(text)
            res[doc] = {}

            i = 0
            pos_end = 0  # Added  (issue #49)
            for sent in sentences:
                if len(sent.strip()) == 0:
                    continue
                # Match gt to sentence.
                # pos_start = text.find(sent) # Commented out (issue #49)
                pos_start = text.find(sent, pos_end)  # Added  (issue #49)
                pos_end = pos_start + len(sent)

                # ngram, start_pos, end_pos
                spans_sent = [
                    [text[x[0] : x[0] + x[1]], x[0], x[0] + x[1]]
                    for x in spans
                    if pos_start <= x[0] < pos_end
                ]
                res[doc][i] = [sent, spans_sent]
                if len(spans) == 0:
                    processed_sentences.append(
                        Sentence(sent, use_tokenizer=True) if is_flair else sent
                    )
                i += 1
            splits.append(splits[-1] + i)
        return res, processed_sentences, splits

    def find_mentions(self, dataset, tagger=None):
        """
        Responsible for finding mentions given a set of documents in a batch-wise manner. More specifically,
        it returns the mention, its left/right context and a set of candidates.
        :return: Dictionary with mentions per document.
        """
        if tagger is None:
            raise Exception(
                "No NER tagger is set, but you are attempting to perform Mention Detection.."
            )
        # Verify if Flair, else ngram or custom.
        is_flair = isinstance(tagger, SequenceTagger)
        dataset_sentences_raw, processed_sentences, splits = self.split_text(
            dataset, is_flair
        )
        results = {}
        total_ment = 0
        if is_flair:
            tagger.predict(processed_sentences)
        for i, doc in enumerate(dataset_sentences_raw):
            contents = dataset_sentences_raw[doc]
            raw_text = dataset[doc][0]
            sentences_doc = [v[0] for v in contents.values()]
            sentences = processed_sentences[splits[i] : splits[i + 1]]
            result_doc = []
            cum_sent_length = 0
            offset = 0
            for (idx_sent, (sentence, ground_truth_sentence)), snt in zip(
                contents.items(), sentences
            ):

                # Only include offset if using Flair.
                if is_flair:
                    offset = raw_text.find(sentence, cum_sent_length)

                for entity in (
                    snt.get_spans("ner")
                    if is_flair
                    else tagger.predict(snt, processed_sentences)
                ):
                    text, start_pos, end_pos, conf, tag = (
                        entity.text,
                        entity.start_pos,
                        entity.end_pos,
                        entity.score,
                        entity.tag,
                    )
                    total_ment += 1
                    m = self.preprocess_mention(text)
                    cands = self.get_candidates(m)
                    if len(cands) == 0:
                        continue
                    # Re-create ngram as 'text' is at times changed by Flair (e.g. double spaces are removed).
                    ngram = sentence[start_pos:end_pos]
                    left_ctxt, right_ctxt = self.get_ctxt(
                        start_pos, end_pos, idx_sent, sentence, sentences_doc
                    )
                    res = {
                        "mention": m,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": cands,
                        "gold": ["NONE"],
                        "pos": start_pos + offset,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end_pos + offset,
                        "sentence": sentence,
                        "conf_md": conf,
                        "tag": tag,
                    }
                    result_doc.append(res)
                cum_sent_length += len(sentence) + (offset - cum_sent_length)
            results[doc] = result_doc
        return results, total_ment
