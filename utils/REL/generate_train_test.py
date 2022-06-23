import os
import pickle
import re
from xml.etree import ElementTree

import pandas as pd
import urllib
import json
from ast import literal_eval

from REL.REL.mention_detection_base import MentionDetectionBase
from REL.REL.utils import modify_uppercase_phrase, split_in_words_mention

"""
Class responsible for formatting WNED and AIDA datasets that are required for ED local evaluation and training.
Inherits overlapping functions from the Mention Detection class.
"""


class GenTrainingTest(MentionDetectionBase):
    def __init__(self, base_url, wiki_version, wikipedia, mylinker=None):
        # self.wned_path = os.path.join(base_url, "generic/test_datasets/wned-datasets/")
        self.aida_path = os.path.join(base_url, "generic/test_datasets/AIDA/")
        self.wikipedia = wikipedia
        self.base_url = base_url
        self.wiki_version = wiki_version
        self.mylinker = mylinker
        super().__init__(base_url, wiki_version, mylinker)

    def __format(self, dataset):
        """
        Formats given ground truth spans and entities for local ED datasets.

        :return: wned dataset with respective ground truth values
        """

        results = {}

        for doc in dataset:
            contents = dataset[doc]
            sentences_doc = [v[0] for v in contents.values()]
            result_doc = []

            for idx_sent, (sentence, ground_truth_sentence) in contents.items():
                for m, gt, start, ngram in ground_truth_sentence:
                    end = start + len(ngram)
                    left_ctxt, right_ctxt = self.get_ctxt(
                        start, end, idx_sent, sentence, sentences_doc
                    )
                    cands = self.get_candidates(m)

                    res = {
                        "mention": m,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": cands,
                        "gold": [gt.replace(" ", "_")],
                        "pos": start,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end,
                        "sentence": sentence,
                    }

                    result_doc.append(res)

            results[doc] = result_doc

        return results

    def process_wned(self, dataset):
        """
        Preprocesses wned into format such that it can be used for evaluation the local ED model.

        :return: wned dataset with respective ground truth values
        """
        split = "\n"
        annotations_xml = os.path.join(self.wned_path, dataset, f"{dataset}.xml")
        tree = ElementTree.parse(annotations_xml)
        root = tree.getroot()

        contents = {}
        exist_doc_names = []
        for doc in root:
            doc_name = doc.attrib["docName"].replace("&amp;", "&")
            if doc_name in exist_doc_names:
                print(
                    "Duplicate document found, will be removed later in the process: {}".format(
                        doc_name
                    )
                )
                continue
            exist_doc_names.append(doc_name)
            doc_path = os.path.join(
                self.wned_path, "{}/RawText/{}".format(dataset, doc_name)
            )
            with open(doc_path, "r", encoding="utf-8") as cf:
                doc_text = " ".join(cf.readlines())
            cf.close()
            doc_text = doc_text.replace("&amp;", "&")
            split_text = re.split(r"{}".format(split), doc_text)

            cnt_replaced = 0
            sentences = {}
            mentions_gt = {}
            total_gt = 0
            for annotation in doc:
                mention_gt = annotation.find("mention").text.replace("&amp;", "&")
                ent_title = annotation.find("wikiName").text
                offset = int(annotation.find("offset").text)

                if not ent_title or ent_title == "NIL":
                    continue

                # Replace ground truth.
                if ent_title not in self.wikipedia.wiki_id_name_map["ent_name_to_id"]:
                    ent_title_temp = self.wikipedia.preprocess_ent_name(ent_title)
                    if (
                        ent_title_temp
                        in self.wikipedia.wiki_id_name_map["ent_name_to_id"]
                    ):
                        ent_title = ent_title_temp
                        cnt_replaced += 1

                offset = max(0, offset - 10)
                pos = doc_text.find(mention_gt, offset)

                find_ment = doc_text[pos : pos + len(mention_gt)]
                assert (
                    find_ment == mention_gt
                ), "Ground truth mention not found: {};{};{}".format(
                    mention_gt, find_ment, pos
                )
                if pos not in mentions_gt:
                    total_gt += 1
                mentions_gt[pos] = [
                    self.preprocess_mention(mention_gt),
                    ent_title,
                    mention_gt,
                ]

            total_characters = 0
            i = 0
            total_assigned = 0
            for t in split_text:
                # Now that our text is split, we can fix it (e.g. remove double spaces)
                if len(t.split()) == 0:
                    total_characters += len(t) + len(split)
                    continue

                # Filter ground truth based on position
                gt_sent = [
                    [v[0], v[1], k - total_characters, v[2]]
                    for k, v in mentions_gt.items()
                    if total_characters
                    <= k
                    <= total_characters + len(t) + len(split) - len(v[2])
                ]
                total_assigned += len(gt_sent)

                for _, _, pos, m in gt_sent:
                    assert (
                        m == t[pos : pos + len(m)]
                    ), "Wrong position mention {};{};{}".format(m, pos, t)

                # Place ground truth in sentence.
                sentences[i] = [t, gt_sent]

                i += 1
                total_characters += len(t) + len(split)
            assert (
                total_gt == total_assigned
            ), "We missed a ground truth.. {};{}".format(total_gt, total_assigned)
            contents[doc_name] = sentences
        print("Replaced {} ground truth entites".format(cnt_replaced))

        self.__save(self.__format(contents), "wned-{}".format(dataset))

    def process_aida(self, dataset):
        """
        Preprocesses AIDA into format such that it can be used for training and evaluation the local ED model.

        :return: AIDA dataset with respective ground truth values. In the case of AIDA-A/B (val and test respectively),
        this function returns both in a dictionary.
        """

        if dataset == "train":
            dataset = "aida_train.txt"
        elif dataset == "test" or dataset == "dev":
            dataset = "testa_testb_aggregate_original"

        file_path = os.path.join(self.aida_path, dataset)
        sentences = {}

        sentence = []
        gt_sent = []
        contents = {}
        i_sent = 0

        total_cnt = 0
        doc_name = None
        prev_doc_name = None
        cnt_replaced = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if "-DOCSTART-" in line:
                    if len(sentence) > 0:
                        sentence_words = " ".join(sentence)
                        for gt in gt_sent:
                            assert (
                                sentence_words[gt[2] : gt[2] + len(gt[3])].lower()
                                == gt[3].lower()
                            ), "AIDA ground-truth incorrect position. {};{};{}".format(
                                sentence_words, gt[2], gt[3]
                            )
                        sentences[i_sent] = [sentence_words, gt_sent]

                        for _, _, pos, ment in gt_sent:
                            find_ment = sentence_words[pos : pos + len(ment)]
                            assert (
                                ment.lower() == find_ment.lower()
                            ), "Mention not found on position.. {}, {}, {}, {}".format(
                                ment, find_ment, pos, sentence_words
                            )

                    if len(sentences) > 0:
                        contents[doc_name] = sentences

                    words = split_in_words_mention(line)
                    for w in words:
                        if ("testa" in w) or ("testb" in w):
                            doc_name = w.replace("(", "").replace(")", "")
                            break
                        else:
                            doc_name = line[12:]

                    if ("testb" in doc_name) and ("testa" in prev_doc_name):
                        self.__save(self.__format(contents), "lwm_dev")
                        contents = {}

                    prev_doc_name = doc_name
                    sentences = {}
                    sentence = []
                    gt_sent = []
                    i_sent = 0
                else:
                    parts = line.split("\t")
                    assert len(parts) in [0, 1, 4, 6, 7], line
                    if len(parts) <= 0:
                        continue

                    if len(parts) in [7, 6] and parts[1] == "B":
                        y = parts[4].find("/wiki/") + len("/wiki/")
                        ent_title = parts[4][y:].replace("_", " ")
                        mention_gt = parts[2]
                        total_cnt += 1

                        if (
                            ent_title
                            not in self.wikipedia.wiki_id_name_map["ent_name_to_id"]
                        ):
                            ent_title_temp = self.wikipedia.preprocess_ent_name(
                                ent_title
                            )
                            if (
                                ent_title_temp
                                in self.wikipedia.wiki_id_name_map["ent_name_to_id"]
                            ):
                                ent_title = ent_title_temp
                                cnt_replaced += 1

                        pos_mention_gt = (
                            len(" ".join(sentence)) + 1 if len(sentence) > 0 else 0
                        )  # + 1 for space between mention and sentence
                        gt_sent.append(
                            [
                                self.preprocess_mention(mention_gt),
                                ent_title,
                                pos_mention_gt,
                                mention_gt,
                            ]
                        )
                        words = mention_gt

                    if len(parts) >= 2 and parts[1] == "B":
                        words = [
                            modify_uppercase_phrase(x)
                            for x in split_in_words_mention(parts[2])
                        ]
                    elif len(parts) >= 2 and parts[1] == "I":
                        # Continuation of mention, which we have added prior
                        # to this iteration, so we skip it.
                        continue
                    else:
                        words = [
                            modify_uppercase_phrase(w)
                            for w in split_in_words_mention(parts[0])
                        ]  # WAS _mention

                    if (parts[0] == ".") and (len(sentence) > 0):
                        # End of sentence, store sentence and additional ground truth mentions.
                        sentence_words = " ".join(sentence)
                        if i_sent in sentences:
                            i_sent += 1
                        sentences[i_sent] = [
                            sentence_words,
                            gt_sent,
                        ]
                        i_sent += 1
                        sentence = []
                        gt_sent = []
                    elif len(words) > 0:
                        sentence += words
        if len(sentence) > 0:
            sentence_words = " ".join(sentence)
            sentences[i_sent] = [sentence_words, gt_sent]
        if len(sentences) > 0:
            contents[doc_name] = sentences

        if "train" in dataset:
            self.__save(self.__format(contents), "lwm_train")
        else:
            self.__save(self.__format(contents), "lwm_dev")
        print("Replaced {} ground truth entites".format(cnt_replaced))

    def __save(self, mentions_dataset, file_name):
        """
        Responsible for saving mentions. This is used to get the processed mentions into the format required
        for training/evaluating the local ED model.

        :return: Dictionary with mentions per document.
        """
        with open(
            os.path.join(
                self.base_url,
                self.wiki_version,
                "generated/test_train_data/",
                f"{file_name}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(mentions_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def process_lwm(self, dataset_split, original_df, processed_df, whichsplit):
        """
        Preprocesses LwM dataset in the format that is necessary for training and evaluating
        the local ED model.
        """

        if dataset_split == "train":
            original_df = original_df[original_df[whichsplit] == "train"]
            processed_df = processed_df[processed_df[whichsplit] == "train"]
        elif dataset_split == "dev":
            original_df = original_df[original_df[whichsplit] == "dev"]
            processed_df = processed_df[processed_df[whichsplit] == "dev"]
        elif dataset_split == "test":
            original_df = original_df[original_df[whichsplit] == "test"]
            processed_df = processed_df[processed_df[whichsplit] == "test"]

        dict_sentences = dict()
        # Collect document sentences:
        for i, row in original_df.iterrows():
            article_id = row["article_id"]
            for sentence in literal_eval(row["sentences"]):
                dict_sentences[
                    str(article_id) + "_" + str(sentence["sentence_pos"])
                ] = sentence["sentence_text"]

        if "reldisamb:lwmcs" in self.mylinker.method:
            dict_articles = dict()
            for i, prediction in processed_df.iterrows():
                article_id = prediction["article_id"]
                dict_mention = dict()
                dict_mention["mention"] = prediction["pred_mention"]
                sent_idx = int(prediction["sentence_pos"])
                dict_mention["sent_idx"] = sent_idx
                dict_mention["sentence"] = dict_sentences[
                    str(article_id) + "_" + str(sent_idx)
                ]
                # Convert the gold standard Wikidata id to Wikipedia id (because
                # of redirections, some times more than one Wikipedia title is
                # assigned to each Wikidata id, we choose the most frequent one):
                dict_mention["gold"] = "NIL"
                gold_ids = self.mylinker.linking_resources["wikidata2wikipedia"].get(
                    prediction["gold_entity_link"]
                )
                max_freq = 0
                if gold_ids:
                    for k in gold_ids:
                        if k["freq"] > max_freq:
                            max_freq = k["freq"]
                            dict_mention["gold"] = k["title"]
                dict_mention["gold"] = [
                    urllib.parse.unquote(dict_mention["gold"]).replace(" ", "_")
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
                dict_mention["candidates"] = self.get_candidates(
                    prediction["pred_mention"],
                    prediction["candidates"],
                    prediction["place_wqid"],  # place of publication (wikidata id)
                )
                if article_id in dict_articles:
                    dict_articles[article_id].append(dict_mention)
                else:
                    dict_articles[article_id] = [dict_mention]

                if "publ" in self.mylinker.method:
                    # Add place of publication as a fake entity in each sentence:
                    # Wikipedia title of place of publication QID:
                    wiki_gold = "NIL"
                    gold_ids = self.mylinker.linking_resources[
                        "wikidata2wikipedia"
                    ].get(prediction["place_wqid"])
                    max_freq = 0
                    if gold_ids:
                        for k in gold_ids:
                            if k["freq"] > max_freq:
                                max_freq = k["freq"]
                                wiki_gold = k["title"]
                    wiki_gold = urllib.parse.unquote(wiki_gold).replace(" ", "_")
                    sent2 = (
                        dict_mention["sentence"]
                        + " Published in "
                        + prediction["place"]
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
                    dict_articles[article_id].append(dict_publ)
        else:
            dict_articles = dict()
            for i, row in original_df.iterrows():
                article_id = row["article_id"]
                dict_articles[article_id] = []
                for annotation in literal_eval(row["annotations"]):
                    dict_mention = dict()
                    dict_mention["mention"] = annotation["mention"]
                    sent_idx = int(annotation["sent_pos"])
                    dict_mention["sent_idx"] = sent_idx
                    dict_mention["sentence"] = dict_sentences[
                        str(article_id) + "_" + str(sent_idx)
                    ]
                    # Convert the gold standard Wikidata id to Wikipedia id (because
                    # of redirections, some times more than one Wikipedia title is
                    # assigned to each Wikidata id, we choose the most frequent one):
                    dict_mention["gold"] = "NIL"
                    gold_ids = self.mylinker.linking_resources[
                        "wikidata2wikipedia"
                    ].get(annotation["wkdt_qid"])
                    max_freq = 0
                    if gold_ids:
                        for k in gold_ids:
                            if k["freq"] > max_freq:
                                max_freq = k["freq"]
                                dict_mention["gold"] = k["title"]
                    dict_mention["gold"] = [
                        urllib.parse.unquote(dict_mention["gold"]).replace(" ", "_")
                    ]
                    dict_mention["ngram"] = annotation["mention"]
                    dict_mention["context"] = ["", ""]
                    if str(article_id) + "_" + str(sent_idx - 1) in dict_sentences:
                        dict_mention["context"][0] = dict_sentences[
                            str(article_id) + "_" + str(sent_idx - 1)
                        ]
                    if str(article_id) + "_" + str(sent_idx + 1) in dict_sentences:
                        dict_mention["context"][1] = dict_sentences[
                            str(article_id) + "_" + str(sent_idx + 1)
                        ]
                    dict_mention["pos"] = annotation["mention_start"]
                    dict_mention["end_pos"] = annotation["mention_end"]
                    dict_mention["candidates"] = self.get_candidates(
                        annotation["mention"]
                    )
                    dict_articles[article_id].append(dict_mention)

        if dataset_split == "train":
            self.__save(dict_articles, "lwm_train")

        if dataset_split == "dev":
            self.__save(dict_articles, "lwm_dev")

        if dataset_split == "test":
            self.__save(dict_articles, "lwm_test")

    # COMBINE AIDA AND LWM
    def process_aidalwm(self, dataset_split, original_df, processed_df, whichsplit):
        """
        Preprocesses AIDA into format such that it can be used for training and evaluation the local ED model.

        :return: AIDA dataset with respective ground truth values. In the case of AIDA-A/B (val and test respectively),
        this function returns both in a dictionary.
        """

        if dataset_split == "train":
            dataset_split = "aida_train.txt"
        elif dataset_split == "test" or dataset_split == "dev":
            dataset_split = "testa_testb_aggregate_original"

        file_path = os.path.join(self.aida_path, dataset_split)
        sentences = {}

        sentence = []
        gt_sent = []
        contents = {}
        i_sent = 0

        total_cnt = 0
        doc_name = None
        prev_doc_name = None
        cnt_replaced = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if "-DOCSTART-" in line:
                    if len(sentence) > 0:
                        sentence_words = " ".join(sentence)
                        for gt in gt_sent:
                            assert (
                                sentence_words[gt[2] : gt[2] + len(gt[3])].lower()
                                == gt[3].lower()
                            ), "AIDA ground-truth incorrect position. {};{};{}".format(
                                sentence_words, gt[2], gt[3]
                            )
                        sentences[i_sent] = [sentence_words, gt_sent]

                        for _, _, pos, ment in gt_sent:
                            find_ment = sentence_words[pos : pos + len(ment)]
                            assert (
                                ment.lower() == find_ment.lower()
                            ), "Mention not found on position.. {}, {}, {}, {}".format(
                                ment, find_ment, pos, sentence_words
                            )

                    if len(sentences) > 0:
                        contents[doc_name] = sentences

                    words = split_in_words_mention(line)
                    for w in words:
                        if ("testa" in w) or ("testb" in w):
                            doc_name = w.replace("(", "").replace(")", "")
                            break
                        else:
                            doc_name = line[12:]

                    if ("testb" in doc_name) and ("testa" in prev_doc_name):
                        self.__save(self.__format(contents), "lwm_dev")
                        contents = {}

                    prev_doc_name = doc_name
                    sentences = {}
                    sentence = []
                    gt_sent = []
                    i_sent = 0
                else:
                    parts = line.split("\t")
                    assert len(parts) in [0, 1, 4, 6, 7], line
                    if len(parts) <= 0:
                        continue

                    if len(parts) in [7, 6] and parts[1] == "B":
                        y = parts[4].find("/wiki/") + len("/wiki/")
                        ent_title = parts[4][y:].replace("_", " ")
                        mention_gt = parts[2]
                        total_cnt += 1

                        if (
                            ent_title
                            not in self.wikipedia.wiki_id_name_map["ent_name_to_id"]
                        ):
                            ent_title_temp = self.wikipedia.preprocess_ent_name(
                                ent_title
                            )
                            if (
                                ent_title_temp
                                in self.wikipedia.wiki_id_name_map["ent_name_to_id"]
                            ):
                                ent_title = ent_title_temp
                                cnt_replaced += 1

                        pos_mention_gt = (
                            len(" ".join(sentence)) + 1 if len(sentence) > 0 else 0
                        )  # + 1 for space between mention and sentence
                        gt_sent.append(
                            [
                                self.preprocess_mention(mention_gt),
                                ent_title,
                                pos_mention_gt,
                                mention_gt,
                            ]
                        )
                        words = mention_gt

                    if len(parts) >= 2 and parts[1] == "B":
                        words = [
                            modify_uppercase_phrase(x)
                            for x in split_in_words_mention(parts[2])
                        ]
                    elif len(parts) >= 2 and parts[1] == "I":
                        # Continuation of mention, which we have added prior
                        # to this iteration, so we skip it.
                        continue
                    else:
                        words = [
                            modify_uppercase_phrase(w)
                            for w in split_in_words_mention(parts[0])
                        ]  # WAS _mention

                    if (parts[0] == ".") and (len(sentence) > 0):
                        # End of sentence, store sentence and additional ground truth mentions.
                        sentence_words = " ".join(sentence)
                        if i_sent in sentences:
                            i_sent += 1
                        sentences[i_sent] = [
                            sentence_words,
                            gt_sent,
                        ]
                        i_sent += 1
                        sentence = []
                        gt_sent = []
                    elif len(words) > 0:
                        sentence += words
        if len(sentence) > 0:
            sentence_words = " ".join(sentence)
            sentences[i_sent] = [sentence_words, gt_sent]
        if len(sentences) > 0:
            contents[doc_name] = sentences

        # ADD LWM
        if dataset_split == "train":
            original_df = original_df[original_df[whichsplit] == "train"]
            processed_df = processed_df[processed_df[whichsplit] == "train"]
        elif dataset_split == "dev":
            original_df = original_df[original_df[whichsplit] == "dev"]
            processed_df = processed_df[processed_df[whichsplit] == "dev"]
        elif dataset_split == "test":
            original_df = original_df[original_df[whichsplit] == "test"]
            processed_df = processed_df[processed_df[whichsplit] == "test"]

        dict_sentences = dict()
        # Collect document sentences:
        for i, row in original_df.iterrows():
            article_id = row["article_id"]
            for sentence in literal_eval(row["sentences"]):
                dict_sentences[
                    str(article_id) + "_" + str(sentence["sentence_pos"])
                ] = sentence["sentence_text"]

        if "reldisamb:lwmcs" in self.mylinker.method:
            dict_articles = dict()
            for i, prediction in processed_df.iterrows():
                article_id = prediction["article_id"]
                dict_mention = dict()
                dict_mention["mention"] = prediction["pred_mention"]
                sent_idx = int(prediction["sentence_pos"])
                dict_mention["sent_idx"] = sent_idx
                dict_mention["sentence"] = dict_sentences[
                    str(article_id) + "_" + str(sent_idx)
                ]
                # Convert the gold standard Wikidata id to Wikipedia id (because
                # of redirections, some times more than one Wikipedia title is
                # assigned to each Wikidata id, we choose the most frequent one):
                dict_mention["gold"] = "NIL"
                gold_ids = self.mylinker.linking_resources["wikidata2wikipedia"].get(
                    prediction["gold_entity_link"]
                )
                max_freq = 0
                if gold_ids:
                    for k in gold_ids:
                        if k["freq"] > max_freq:
                            max_freq = k["freq"]
                            dict_mention["gold"] = k["title"]
                dict_mention["gold"] = [
                    urllib.parse.unquote(dict_mention["gold"]).replace(" ", "_")
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
                dict_mention["candidates"] = self.get_candidates(
                    prediction["pred_mention"],
                    prediction["candidates"],
                    prediction["place_wqid"],  # place of publication (wikidata id)
                )
                if article_id in dict_articles:
                    dict_articles[article_id].append(dict_mention)
                else:
                    dict_articles[article_id] = [dict_mention]

                if "publ" in self.mylinker.method:
                    # Add place of publication as a fake entity in each sentence:
                    # Wikipedia title of place of publication QID:
                    wiki_gold = "NIL"
                    gold_ids = self.mylinker.linking_resources[
                        "wikidata2wikipedia"
                    ].get(prediction["place_wqid"])
                    max_freq = 0
                    if gold_ids:
                        for k in gold_ids:
                            if k["freq"] > max_freq:
                                max_freq = k["freq"]
                                wiki_gold = k["title"]
                    wiki_gold = urllib.parse.unquote(wiki_gold).replace(" ", "_")
                    sent2 = (
                        dict_mention["sentence"]
                        + " Published in "
                        + prediction["place"]
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
                    dict_articles[article_id].append(dict_publ)
        else:
            dict_articles = dict()
            for i, row in original_df.iterrows():
                article_id = row["article_id"]
                dict_articles[article_id] = []
                for annotation in literal_eval(row["annotations"]):
                    dict_mention = dict()
                    dict_mention["mention"] = annotation["mention"]
                    sent_idx = int(annotation["sent_pos"])
                    dict_mention["sent_idx"] = sent_idx
                    dict_mention["sentence"] = dict_sentences[
                        str(article_id) + "_" + str(sent_idx)
                    ]
                    # Convert the gold standard Wikidata id to Wikipedia id (because
                    # of redirections, some times more than one Wikipedia title is
                    # assigned to each Wikidata id, we choose the most frequent one):
                    dict_mention["gold"] = "NIL"
                    gold_ids = self.mylinker.linking_resources[
                        "wikidata2wikipedia"
                    ].get(annotation["wkdt_qid"])
                    max_freq = 0
                    if gold_ids:
                        for k in gold_ids:
                            if k["freq"] > max_freq:
                                max_freq = k["freq"]
                                dict_mention["gold"] = k["title"]
                    dict_mention["gold"] = [
                        urllib.parse.unquote(dict_mention["gold"]).replace(" ", "_")
                    ]
                    dict_mention["ngram"] = annotation["mention"]
                    dict_mention["context"] = ["", ""]
                    if str(article_id) + "_" + str(sent_idx - 1) in dict_sentences:
                        dict_mention["context"][0] = dict_sentences[
                            str(article_id) + "_" + str(sent_idx - 1)
                        ]
                    if str(article_id) + "_" + str(sent_idx + 1) in dict_sentences:
                        dict_mention["context"][1] = dict_sentences[
                            str(article_id) + "_" + str(sent_idx + 1)
                        ]
                    dict_mention["pos"] = annotation["mention_start"]
                    dict_mention["end_pos"] = annotation["mention_end"]
                    dict_mention["candidates"] = self.get_candidates(
                        annotation["mention"]
                    )
                    dict_articles[article_id].append(dict_mention)

        contents = self.__format(contents)
        contents.update(dict_articles)

        if "train" in dataset_split:
            self.__save(contents, "lwm_train")
        else:
            self.__save(contents, "lwm_dev")
