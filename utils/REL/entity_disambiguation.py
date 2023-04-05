import os
import re
import sys
import json
import time
import torch
import pickle
import numpy as np
from pathlib import Path
from random import shuffle
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import rel_utils
from utils.REL.mulrel_ranker import MulRelRanker, PreRank
from utils.REL.vocabulary import Vocabulary
import utils.REL.utils as utils

"""
Parent Entity Disambiguation class that directs the various subclasses used
for the ED step.
"""


class EntityDisambiguation:
    def __init__(self, db_embs, user_config, reset_embeddings=False):
        self.embeddings = {}
        self.config = self.__get_config(user_config)

        # Use CPU if cuda is not available:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prerank_model = None
        self.model = None
        self.reset_embeddings = reset_embeddings
        self.db_embs = db_embs
        # Test DB embeddings
        test = rel_utils.get_db_emb(self.db_embs, ["Q84"], "entity")[0]
        assert test is not None, "DB embeddings in wrong folder..? Test embedding not found.."

        # Initialise embedding dictionary:
        self.__load_embeddings()
        # Initialise pre-ranking:
        self.prerank_model = PreRank(self.config).to(self.device)
        # Load LR model for confidence.
        if os.path.exists(Path(self.config["model_path"]).parent / "lr_model.pkl"):
            with open(
                Path(self.config["model_path"]).parent / "lr_model.pkl",
                "rb",
            ) as f:
                self.model_lr = pickle.load(f)
        else:
            print("Confidence scores for the ED module will be set to zero.")
            self.model_lr = None

        if self.config["mode"] == "eval":
            print("Loading model from given path: {}".format(self.config["model_path"]))
            self.model = self.__load(self.config["model_path"])
            self.config["mode"] = "eval"
            # # Load model lr:
            # with open(
            #     Path(self.config["model_path"]).parent / "lr_model.pkl",
            #     "rb",
            # ) as f:
            #     self.model_lr = pickle.load(f)
            print(self.config)
        else:
            if reset_embeddings:
                raise Exception("You cannot train a model and reset the embeddings.")
            self.model = MulRelRanker(self.config, self.device).to(self.device)

    def __get_config(self, user_config):
        """
        User configuration that may overwrite default settings.

        :return: configuration used for ED.
        """

        default_config: Dict[str, Any] = {
            "mode": user_config["mode"],
            "model_path": user_config["model_path"],
            "prerank_ctx_window": 50,
            "keep_p_e_m": 4,
            "keep_ctx_ent": 3,
            "ctx_window": 100,
            "tok_top_n": 25,
            "mulrel_type": "ment-norm",
            "n_rels": 3,
            "hid_dims": 100,
            "emb_dims": 300,
            "snd_local_ctx_window": 6,
            "dropout_rate": 0.3,
            "n_epochs": 1000,
            "dev_f1_change_lr": 0.915,
            "n_not_inc": 10,
            "eval_after_n_epochs": 5,
            "learning_rate": 1e-4,
            "margin": 0.01,
            "df": 0.5,
            "n_loops": 10,
            # 'freeze_embs': True,
            "n_cands_before_rank": 30,
            "first_head_uniforn": False,
            "use_pad_ent": True,
            "use_local": True,
            "use_local_only": False,
            "oracle": False,
        }

        config = default_config
        print("Model path:", config["model_path"], config["mode"])

        return config

    def __load_embeddings(self):
        """
        Initialised embedding dictionary and creates #UNK# token for respective embeddings.
        :return: -
        """
        self.__batch_embs = {}

        for name in ["snd", "entity", "word"]:
            # Init entity embeddings.
            self.embeddings["{}_seen".format(name)] = set()
            self.embeddings["{}_voca".format(name)] = Vocabulary()
            self.embeddings["{}_embeddings".format(name)] = None

            # Add #UNK# token.
            self.embeddings["{}_voca".format(name)].add_to_vocab("#UNK#")
            e = rel_utils.get_db_emb(self.db_embs, ["#{}/UNK#".format(name.upper())], name)[0]
            assert e is not None, "#UNK# token not found for {} in db".format(name)
            self.__batch_embs[name] = []
            self.__batch_embs[name].append(torch.tensor(e))

    def train(self, org_train_dataset, org_dev_dataset):
        """
        Responsible for training the ED model.

        :return: -
        """

        train_dataset = self.get_data_items(org_train_dataset, "train", predict=False)
        dev_dataset = self.get_data_items(org_dev_dataset, "dev", predict=True)

        print("Creating optimizer")
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["learning_rate"],
        )
        best_f1 = -1
        best_p = -1
        best_r = -1
        not_better_count = 0
        eval_after_n_epochs = self.config["eval_after_n_epochs"]

        for e in range(self.config["n_epochs"]):
            shuffle(train_dataset)

            total_loss = 0
            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()
                optimizer.zero_grad()

                # convert data items to pytorch inputs
                token_ids = [
                    m["context"][0] + m["context"][1]
                    if len(m["context"][0]) + len(m["context"][1]) > 0
                    else [self.embeddings["word_voca"].unk_id]
                    for m in batch
                ]
                s_ltoken_ids = [m["snd_ctx"][0] for m in batch]
                s_rtoken_ids = [m["snd_ctx"][1] for m in batch]
                s_mtoken_ids = [m["snd_ment"] for m in batch]

                entity_ids = Variable(
                    torch.LongTensor([m["selected_cands"]["cands"] for m in batch]).to(self.device)
                )
                true_pos = Variable(
                    torch.LongTensor([m["selected_cands"]["true_pos"] for m in batch]).to(
                        self.device
                    )
                )
                p_e_m = Variable(
                    torch.FloatTensor([m["selected_cands"]["p_e_m"] for m in batch]).to(
                        self.device
                    )
                )
                entity_mask = Variable(
                    torch.FloatTensor([m["selected_cands"]["mask"] for m in batch]).to(self.device)
                )

                token_ids, token_mask = utils.make_equal_len(
                    token_ids, self.embeddings["word_voca"].unk_id
                )
                s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(
                    s_ltoken_ids, self.embeddings["snd_voca"].unk_id, to_right=False
                )
                s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(
                    s_rtoken_ids, self.embeddings["snd_voca"].unk_id
                )
                s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
                s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
                s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(
                    s_mtoken_ids, self.embeddings["snd_voca"].unk_id
                )

                token_ids = Variable(torch.LongTensor(token_ids).to(self.device))
                token_mask = Variable(torch.FloatTensor(token_mask).to(self.device))

                # too ugly but too lazy to fix it
                self.model.s_ltoken_ids = Variable(torch.LongTensor(s_ltoken_ids).to(self.device))
                self.model.s_ltoken_mask = Variable(
                    torch.FloatTensor(s_ltoken_mask).to(self.device)
                )
                self.model.s_rtoken_ids = Variable(torch.LongTensor(s_rtoken_ids).to(self.device))
                self.model.s_rtoken_mask = Variable(
                    torch.FloatTensor(s_rtoken_mask).to(self.device)
                )
                self.model.s_mtoken_ids = Variable(torch.LongTensor(s_mtoken_ids).to(self.device))
                self.model.s_mtoken_mask = Variable(
                    torch.FloatTensor(s_mtoken_mask).to(self.device)
                )

                scores, ent_scores = self.model.forward(
                    token_ids,
                    token_mask,
                    entity_ids,
                    entity_mask,
                    p_e_m,
                    self.embeddings,
                    gold=true_pos.view(-1, 1),
                )
                loss = self.model.loss(scores, true_pos)
                # loss = self.model.prob_loss(scores, true_pos)
                loss.backward()
                optimizer.step()
                self.model.regularize(max_norm=100)

                loss = loss.cpu().data.numpy()
                total_loss += loss
                print(
                    "epoch",
                    e,
                    "%0.2f%%" % (dc / len(train_dataset) * 100),
                    loss,
                    end="\r",
                )

            print("epoch", e, "total loss", total_loss, total_loss / len(train_dataset))

            if (e + 1) % eval_after_n_epochs == 0:
                predictions = self.__predict(dev_dataset)
                dev_f1, recall, precision, _ = self.__eval(org_dev_dataset, predictions)
                print(
                    "Micro F1: {}, Recall: {}, Precision: {}".format(dev_f1, recall, precision),
                )

                if (
                    self.config["learning_rate"] == 1e-4
                    and dev_f1 >= self.config["dev_f1_change_lr"]
                ):
                    eval_after_n_epochs = 2
                    best_f1 = dev_f1
                    best_p = precision
                    best_r = recall
                    not_better_count = 0

                    self.config["learning_rate"] = 1e-5
                    print("change learning rate to", self.config["learning_rate"])
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.config["learning_rate"]

                if dev_f1 < best_f1:
                    not_better_count += 1
                    print("Not improving", not_better_count)
                else:
                    not_better_count = 0
                    best_f1 = dev_f1
                    best_p = precision
                    best_r = recall

                    print("save model to", self.config["model_path"])
                    self.__save(self.config["model_path"])

                if not_better_count == self.config["n_not_inc"]:
                    break
        self.best_performance = {"f1": best_f1, "p": best_p, "r": best_r}

    def __create_dataset_LR(self, dataset, predictions):
        X = []
        y = []
        meta = []
        for doc, preds in predictions.items():
            gt_doc = [c["gold"][0] for c in dataset[doc]]
            for pred, gt in zip(preds, gt_doc):
                scores = [float(x) for x in pred["scores"]]
                cands = pred["ranking_candidates"]

                # Build classes
                for i, c in enumerate(cands):
                    if c == "#UNK#":
                        continue

                    X.append([scores[i]])
                    meta.append([doc, gt, c])
                    if gt == c:
                        y.append(1.0)
                    else:
                        y.append(0.0)

        return np.array(X), np.array(y), np.array(meta)

    def train_LR(self, train_json, dev_json, model_path_lr):
        """
        Function that applies LR in an attempt to get confidence scores. Recall should be high,
        because if it is low than we would have ignored a corrrect entity.

        :return: -
        """
        print(os.path.join(model_path_lr, "lr_model.pkl"))

        train_dataset = self.get_data_items(train_json, "train", predict=False)
        dev_dataset = self.get_data_items(dev_json, "dev", predict=True)

        model = LogisticRegression()

        predictions = self.__predict(train_dataset, eval_raw=True)
        X, y, meta = self.__create_dataset_LR(train_json, predictions)
        model.fit(X, y)

        predictions = self.__predict(dev_dataset, eval_raw=True)
        X, y, meta = self.__create_dataset_LR(dev_json, predictions)
        preds = model.predict_proba(X)
        preds = np.array([x[1] for x in preds])

        path = os.path.join(model_path_lr, "lr_model.pkl")
        with open(path, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, data):
        """
        Parent function responsible for predicting on any raw text as input. This does not require ground
        truth entities to be present.

        :return: predictions and time taken for the ED step.
        """
        data = self.get_data_items(data, "raw", predict=True)
        predictions, timing = self.__predict(data, include_timing=True, eval_raw=True)

        return predictions

    def normalize_scores(self, scores):
        """
        Normalizes a list of scores between 0 and 1 by rescaling them and computing their ratio over their sum.

        Args:
            scores (list): A list of numerical scores.

        Returns:
            list: A list of normalized scores where each score is the ratio of the rescaled score over their sum.
        """

        min_score = min(scores)
        max_score = max(scores)
        rescaled_scores = [(score - min_score) / (max_score - min_score) for score in scores]

        # calculate sum of rescaled scores
        score_sum = sum(rescaled_scores)

        # normalize each rescaled score
        normalized_scores = [score / score_sum for score in rescaled_scores]

        return normalized_scores

    def __compute_confidence(self, scores):
        """
        This function takes a series of numpy arrays of scores and returns a list of lists of confidence scores.

        Args:
            scores (numpy.ndarray): A numpy array of scores.

        Returns:
            list: A list of lists of confidence scores.
        """

        normalised_scores = [self.normalize_scores(score) for score in scores]
        return normalised_scores

    # def __compute_confidence(self, scores, preds):
    #     """
    #     Uses LR to find confidence scores for given ED outputs.

    #     :return:
    #     """
    #     X = np.array([[score[pred]] for score, pred in zip(scores, preds)])
    #     if self.model_lr:
    #         preds = self.model_lr.predict_proba(X)
    #         confidence_scores = [x[1] for x in preds]
    #     else:
    #         confidence_scores = [0.0 for _ in scores]
    #     return confidence_scores

    def __predict(self, data, include_timing=False, eval_raw=False):
        """
        Uses the trained model to make predictions of individual batches (i.e. documents).

        :return: predictions and time taken for the ED step.
        """

        predictions = {items[0]["doc_name"]: [] for items in data}
        self.model.eval()

        timing = []

        for batch in data:  # each document is a minibatch

            start = time.time()

            token_ids = [
                m["context"][0] + m["context"][1]
                if len(m["context"][0]) + len(m["context"][1]) > 0
                else [self.embeddings["word_voca"].unk_id]
                for m in batch
            ]
            s_ltoken_ids = [m["snd_ctx"][0] for m in batch]
            s_rtoken_ids = [m["snd_ctx"][1] for m in batch]
            s_mtoken_ids = [m["snd_ment"] for m in batch]

            entity_ids = Variable(
                torch.LongTensor([m["selected_cands"]["cands"] for m in batch]).to(self.device)
            )
            p_e_m = Variable(
                torch.FloatTensor([m["selected_cands"]["p_e_m"] for m in batch]).to(self.device)
            )
            entity_mask = Variable(
                torch.FloatTensor([m["selected_cands"]["mask"] for m in batch]).to(self.device)
            )
            true_pos = Variable(
                torch.LongTensor([m["selected_cands"]["true_pos"] for m in batch]).to(self.device)
            )

            token_ids, token_mask = utils.make_equal_len(
                token_ids, self.embeddings["word_voca"].unk_id
            )
            s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(
                s_ltoken_ids, self.embeddings["snd_voca"].unk_id, to_right=False
            )
            s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(
                s_rtoken_ids, self.embeddings["snd_voca"].unk_id
            )
            s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
            s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
            s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(
                s_mtoken_ids, self.embeddings["snd_voca"].unk_id
            )

            token_ids = Variable(torch.LongTensor(token_ids).to(self.device))
            token_mask = Variable(torch.FloatTensor(token_mask).to(self.device))

            self.model.s_ltoken_ids = Variable(torch.LongTensor(s_ltoken_ids).to(self.device))
            self.model.s_ltoken_mask = Variable(torch.FloatTensor(s_ltoken_mask).to(self.device))
            self.model.s_rtoken_ids = Variable(torch.LongTensor(s_rtoken_ids).to(self.device))
            self.model.s_rtoken_mask = Variable(torch.FloatTensor(s_rtoken_mask).to(self.device))
            self.model.s_mtoken_ids = Variable(torch.LongTensor(s_mtoken_ids).to(self.device))
            self.model.s_mtoken_mask = Variable(torch.FloatTensor(s_mtoken_mask).to(self.device))

            scores, ent_scores = self.model.forward(
                token_ids,
                token_mask,
                entity_ids,
                entity_mask,
                p_e_m,
                self.embeddings,
                gold=true_pos.view(-1, 1),
            )
            pred_ids = torch.argmax(scores, axis=1)
            scores = scores.cpu().data.numpy()

            confidence_scores = self.__compute_confidence(scores)
            pred_scores = [max(score) for score in confidence_scores]
            pred_ids = np.argmax(scores, axis=1)

            if not eval_raw:
                pred_entities = [
                    m["selected_cands"]["named_cands"][i]
                    if m["selected_cands"]["mask"][i] == 1
                    else (
                        m["selected_cands"]["named_cands"][0]
                        if m["selected_cands"]["mask"][0] == 1
                        else "NIL"
                    )
                    for (i, m) in zip(pred_ids, batch)
                ]
                doc_names = [m["doc_name"] for m in batch]

                for dname, entity in zip(doc_names, pred_entities):
                    predictions[dname].append({"pred": (entity, 0.0)})

            else:
                pred_entities = [
                    [
                        m["selected_cands"]["named_cands"][i],
                        m["raw"]["mention"],
                        m["selected_cands"]["named_cands"],
                        s,
                        cs,
                        m["selected_cands"]["mask"],
                    ]
                    if m["selected_cands"]["mask"][i] == 1
                    else (
                        [
                            m["selected_cands"]["named_cands"][0],
                            m["raw"]["mention"],
                            m["selected_cands"]["named_cands"],
                            s,
                            cs,
                            m["selected_cands"]["mask"],
                        ]
                        if m["selected_cands"]["mask"][0] == 1
                        else [
                            "NIL",
                            m["raw"]["mention"],
                            m["selected_cands"]["named_cands"],
                            s,
                            cs,
                            m["selected_cands"]["mask"],
                        ]
                    )
                    for (i, m, s, cs) in zip(pred_ids, batch, confidence_scores, pred_scores)
                ]
                doc_names = [m["doc_name"] for m in batch]

                for dname, entity in zip(doc_names, pred_entities):
                    if entity[0] != "NIL":
                        predictions[dname].append(
                            {
                                "mention": entity[1],
                                "prediction": entity[0],
                                "candidates": entity[2],
                                "conf_ed": entity[4],
                                "scores": entity[3],
                            }
                        )

                    else:
                        predictions[dname].append(
                            {
                                "mention": entity[1],
                                "prediction": entity[0],
                                "candidates": entity[2],
                                "conf_ed": 0.0,
                                "scores": [],
                            }
                        )

            timing.append(time.time() - start)
        if include_timing:
            return predictions, timing
        else:
            return predictions

    def prerank(self, dataset, dname, predict=False):
        """
        Responsible for preranking the set of possible candidates using both context and p(e|m) scores.
        :return: dataset with, by default, max 3 + 4 candidates per mention.
        """
        new_dataset = []
        has_gold = 0
        total = 0

        for content in dataset:
            items = []
            if self.config["keep_ctx_ent"] > 0:
                # rank the candidates by ntee scores
                lctx_ids = [
                    m["context"][0][
                        max(
                            len(m["context"][0]) - self.config["prerank_ctx_window"] // 2,
                            0,
                        ) :
                    ]
                    for m in content
                ]
                rctx_ids = [
                    m["context"][1][
                        : min(len(m["context"][1]), self.config["prerank_ctx_window"] // 2)
                    ]
                    for m in content
                ]
                ment_ids = [[] for m in content]
                token_ids = [
                    l + m + r if len(l) + len(r) > 0 else [self.embeddings["word_voca"].unk_id]
                    for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)
                ]

                entity_ids = [m["cands"] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).to(self.device))

                entity_mask = [m["mask"] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).to(self.device))

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_offsets = Variable(torch.LongTensor(token_offsets).to(self.device))
                token_ids = Variable(torch.LongTensor(token_ids).to(self.device))

                log_probs = self.prerank_model.forward(
                    token_ids, token_offsets, entity_ids, self.embeddings
                )

                # Entity mask makes sure that the UNK entities are zero.
                log_probs = (log_probs * entity_mask).add_((entity_mask - 1).mul_(1e10))
                _, top_pos = torch.topk(log_probs, dim=1, k=self.config["keep_ctx_ent"])
                top_pos = top_pos.data.cpu().numpy()
            else:
                top_pos = [[]] * len(content)

            # select candidats: mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content):
                sm = {
                    "cands": [],
                    "named_cands": [],
                    "p_e_m": [],
                    "mask": [],
                    "true_pos": -1,
                }
                m["selected_cands"] = sm

                selected = set(top_pos[i])

                idx = 0
                while len(selected) < self.config["keep_ctx_ent"] + self.config["keep_p_e_m"]:
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1

                selected = sorted(list(selected))

                for idx in selected:
                    sm["cands"].append(m["cands"][idx])
                    sm["named_cands"].append(m["named_cands"][idx])
                    sm["p_e_m"].append(m["p_e_m"][idx])
                    sm["mask"].append(m["mask"][idx])
                    if idx == m["true_pos"]:
                        sm["true_pos"] = len(sm["cands"]) - 1

                if not predict:
                    if sm["true_pos"] == -1:
                        continue

                items.append(m)
                if sm["true_pos"] >= 0:
                    has_gold += 1
                total += 1

                if predict:
                    # only for oracle model, not used for eval
                    if sm["true_pos"] == -1:
                        sm["true_pos"] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        # if total > 0
        if dname != "raw":
            print("Recall for {}: {}".format(dname, has_gold / total))
            print("-----------------------------------------------")

        return new_dataset

    def __update_embeddings(self, emb_name, embs):
        """
        Responsible for updating the dictionaries with their respective word, entity and snd (GloVe) embeddings.

        :return: -
        """

        embs = embs.to(self.device)

        if self.embeddings["{}_embeddings".format(emb_name)]:
            new_weights = torch.cat(
                (self.embeddings["{}_embeddings".format(emb_name)].weight, embs)
            )
        else:
            new_weights = embs

        # Weights are now updated, so we create a new Embedding layer.
        layer = torch.nn.Embedding(
            self.embeddings["{}_voca".format(emb_name)].size(), self.config["emb_dims"]
        )
        layer.weight = torch.nn.Parameter(new_weights)
        layer.grad = False
        self.embeddings["{}_embeddings".format(emb_name)] = layer
        if emb_name == "word":
            layer = torch.nn.EmbeddingBag(
                self.embeddings["{}_voca".format(emb_name)].size(),
                self.config["emb_dims"],
            )
            layer.weight = torch.nn.Parameter(new_weights)

            layer.requires_grad = False
            self.embeddings["{}_embeddings_bag".format(emb_name)] = layer

        del new_weights

    def __embed_words(self, words_filt, name):
        """
        Responsible for retrieving embeddings using the given sqlite3 database.

        :return: -
        """
        embs = rel_utils.get_db_emb(self.db_embs, words_filt, name)

        # Now we go over the embs and see which one is None. Order is preserved.
        for e, c in zip(embs, words_filt):
            self.embeddings["{}_seen".format(name)].add(c)
            if e is not None:
                # Embedding exists, so we add it.
                self.embeddings["{}_voca".format(name)].add_to_vocab(c)
                self.__batch_embs[name].append(torch.tensor(e))

    def get_data_items(self, dataset, dname, predict=False):
        """
        Responsible for formatting dataset. Triggers the preranking function.

        :return: preranking function.
        """
        data = []

        for doc_name, content in dataset.items():
            items = []
            if len(content) == 0:
                continue
            for m in content:
                named_cands = [c[0] for c in m["ranking_candidates"]]
                p_e_m = [min(1.0, max(1e-3, c[1])) for c in m["ranking_candidates"]]

                try:
                    true_pos = named_cands.index(m["gold"][0])
                    p = p_e_m[true_pos]
                except:
                    true_pos = -1

                # Get all words and check for embeddings.
                named_cands = named_cands[
                    : min(self.config["n_cands_before_rank"], len(named_cands))
                ]

                # Candidate list per mention.
                named_cands_filt = set(
                    [item for item in named_cands if item not in self.embeddings["entity_seen"]]
                )

                self.__embed_words(named_cands_filt, "entity")

                # Use re.split() to make sure that special characters are considered.
                lctx = [
                    x for x in re.split("(\W)", m["context"][0].strip()) if x != " "
                ]  # .split()
                rctx = [
                    x for x in re.split("(\W)", m["context"][1].strip()) if x != " "
                ]  # split()

                words_filt = set(
                    [item for item in lctx + rctx if item not in self.embeddings["word_seen"]]
                )

                self.__embed_words(words_filt, "word")

                snd_lctx = m["sentence"][: m["pos"]]
                snd_lctx = [x for x in re.split("(\W)", snd_lctx.strip()) if x != " "]
                snd_lctx = [t for t in snd_lctx[-self.config["snd_local_ctx_window"] // 2 :]]

                snd_rctx = m["sentence"][m["end_pos"] :]
                snd_rctx = [x for x in re.split("(\W)", snd_rctx.strip()) if x != " "]
                snd_rctx = [t for t in snd_rctx[: self.config["snd_local_ctx_window"] // 2]]

                snd_ment = m["ngram"].strip().split()

                words_filt = set(
                    [
                        item
                        for item in snd_lctx + snd_rctx + snd_ment
                        if item not in self.embeddings["snd_seen"]
                    ]
                )

                self.__embed_words(words_filt, "snd")

                p_e_m = p_e_m[: min(self.config["n_cands_before_rank"], len(p_e_m))]

                if true_pos >= len(named_cands):
                    if not predict:
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m["gold"][0]
                    else:
                        true_pos = -1
                cands = [self.embeddings["entity_voca"].get_id(c) for c in named_cands]

                mask = [1.0] * len(cands)
                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.config["n_cands_before_rank"]:
                    cands += [self.embeddings["entity_voca"].unk_id] * (
                        self.config["n_cands_before_rank"] - len(cands)
                    )
                    named_cands += [Vocabulary.unk_token] * (
                        self.config["n_cands_before_rank"] - len(named_cands)
                    )
                    p_e_m += [1e-8] * (self.config["n_cands_before_rank"] - len(p_e_m))
                    mask += [0.0] * (self.config["n_cands_before_rank"] - len(mask))

                lctx_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in lctx
                    if utils.is_important_word(t)
                ]

                lctx_ids = [tid for tid in lctx_ids if tid != self.embeddings["word_voca"].unk_id]
                lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.config["ctx_window"] // 2) :]

                rctx_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in rctx
                    if utils.is_important_word(t)
                ]
                rctx_ids = [tid for tid in rctx_ids if tid != self.embeddings["word_voca"].unk_id]
                rctx_ids = rctx_ids[: min(len(rctx_ids), self.config["ctx_window"] // 2)]

                ment = m["mention"].strip().split()
                ment_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in ment
                    if utils.is_important_word(t)
                ]
                ment_ids = [tid for tid in ment_ids if tid != self.embeddings["word_voca"].unk_id]

                m["sent"] = " ".join(lctx + rctx)

                # Secondary local context.
                snd_lctx = [self.embeddings["snd_voca"].get_id(t) for t in snd_lctx]
                snd_rctx = [self.embeddings["snd_voca"].get_id(t) for t in snd_rctx]
                snd_ment = [self.embeddings["snd_voca"].get_id(t) for t in snd_ment]

                # This is only used for the original embeddings, now they are never empty.
                if len(snd_lctx) == 0:
                    snd_lctx = [self.embeddings["snd_voca"].unk_id]
                if len(snd_rctx) == 0:
                    snd_rctx = [self.embeddings["snd_voca"].unk_id]
                if len(snd_ment) == 0:
                    snd_ment = [self.embeddings["snd_voca"].unk_id]

                items.append(
                    {
                        "context": (lctx_ids, rctx_ids),
                        "snd_ctx": (snd_lctx, snd_rctx),
                        "ment_ids": ment_ids,
                        "snd_ment": snd_ment,
                        "cands": cands,
                        "named_cands": named_cands,
                        "p_e_m": p_e_m,
                        "mask": mask,
                        "true_pos": true_pos,
                        "doc_name": doc_name,
                        "raw": m,
                    }
                )

            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction
                if len(items) > 100:
                    for k in range(0, len(items), 100):
                        data.append(items[k : min(len(items), k + 100)])
                else:
                    data.append(items)

        # Update batch
        for n in ["word", "entity", "snd"]:
            if self.__batch_embs[n]:
                self.__batch_embs[n] = torch.stack(self.__batch_embs[n])
                self.__update_embeddings(n, self.__batch_embs[n])
                self.__batch_embs[n] = []

        return self.prerank(data, dname, predict)

    def __eval(self, testset, system_pred):
        """
        Responsible for evaluating data points, which is solely used for the local ED step.

        :return: F1, Recall, Precision and number of mentions for which we have no valid candidate.
        """
        gold = []
        pred = []

        for doc_name, content in testset.items():
            if len(content) == 0:
                continue
            gold += [c["gold"][0] for c in content]
            pred += [c["pred"][0] for c in system_pred[doc_name]]

        true_pos = 0
        total_nil = 0
        for g, p in zip(gold, pred):
            if p == "NIL":
                total_nil += 1
            if g == p and p != "NIL":
                true_pos += 1

        precision = true_pos / len([p for p in pred if p != "NIL"])
        recall = true_pos / len(gold)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, recall, precision, total_nil

    def __save(self, path):
        """
        Responsible for storing the trained model during optimisation.

        :return: -.
        """
        torch.save(self.model.state_dict(), "{}.state_dict".format(path))
        with open("{}.config".format(path), "w") as f:
            json.dump(self.config, f)

    def __load(self, path):
        """
        Responsible for loading a trained model and its respective config. Note that this config cannot be
        overwritten. If required, this behavior may be modified in future releases.

        :return: model
        """

        if os.path.exists("{}.config".format(path)):
            with open("{}.config".format(path), "r") as f:
                temp = self.config["model_path"]
                self.config = json.load(f)
                self.config["model_path"] = temp
        else:
            print(
                "No configuration file found at {}, default settings will be used.".format(
                    "{}.config".format(path)
                )
            )

        model = MulRelRanker(self.config, self.device).to(self.device)

        if not torch.cuda.is_available():
            model.load_state_dict(
                torch.load(
                    "{}{}".format(self.config["model_path"], ".state_dict"),
                    map_location=torch.device("cpu"),
                )
            )
        else:
            model.load_state_dict(
                torch.load("{}{}".format(self.config["model_path"], ".state_dict"))
            )
        return model
