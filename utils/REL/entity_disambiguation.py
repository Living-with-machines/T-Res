import json
import os
import pickle
import random
import re  # TODO: not used, can be dropped
import sys
import time
from pathlib import Path
from string import punctuation
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score  # TODO: not used, can be dropped
from torch.autograd import Variable

sys.path.insert(0, os.path.abspath(os.path.pardir))
import utils.REL.utils as utils
from utils import rel_utils
from utils.REL.mulrel_ranker import MulRelRanker, PreRank
from utils.REL.vocabulary import Vocabulary


RANDOM_SEED = 42
"""Constant representing the random seed used for generating pseudo-random
numbers.

The `RANDOM_SEED` is a value that initializes the random number generator
algorithm, ensuring that the sequence of random numbers generated remains the
same across different runs of the program. This is useful for achieving
reproducibility in experiments or when consistent random behavior is
desired.

..
    If this docstring is changed, also make sure to edit prepare_data.py,
    linking.py, rel_utils.py.
"""
random.seed(RANDOM_SEED)


class EntityDisambiguation:
    """
    EntityDisambiguation is a class that performs entity disambiguation, which
    is the task of resolving entity mentions in text to their corresponding
    entities in a knowledge base. It uses a trained model to predict the most
    likely entity for each mention based on contextual information and
    pre-computed scores.

    The ``EntityDisambiguation`` class provides methods for training the
    disambiguation model, predicting entity mentions in raw text, evaluating
    the performance of the model, and saving/loading the trained model.

    This class uses a deep learning architecture, specifically the
    :py:class:`~utils.REL.mulrel_ranker.MulRelRanker` model, for entity
    disambiguation. The model takes into account the context of the mention,
    the pre-computed scores, and the candidate entities, and makes predictions
    based on these features.

    The entity disambiguation process involves preranking the candidate
    entities, computing scores, predicting the most likely entity, and
    evaluating the performance of the predictions. The class encapsulates
    these steps and provides a convenient interface for performing entity
    disambiguation tasks.

    Arguments:
        db_embs (TODO/typing: set correct type): The connection to a SQLite
            database containing the word and entity embeddings.
        user_config (dict): A dictionary containing custom configuration
            settings for the model. If not provided, default settings will be
            used.
        reset_embeddings (bool, optional): Specifies whether to reset the
            embeddings even if a pre-trained model is loaded. Defaults to
            ``False``.

    ..
        TODO: Is it correct to say that "If [user_config] is not provided,
        default settings will be used here, as it is a required keyword
        argument?
    """

    # TODO/typing: Set correct typing on db_embs
    def __init__(
        self, db_embs, user_config: dict, reset_embeddings: Optional[bool] = False
    ):
        """
        Initialises an EntityDisambiguation object.
        """
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
        assert (
            test is not None
        ), "DB embeddings in wrong folder..? Test embedding not found.."
        test = rel_utils.get_db_emb(self.db_embs, ["#ENTITY/UNK#"], "entity")[0]
        assert (
            test is not None
        ), "DB embeddings in wrong folder..? Test embedding not found.."
        test = rel_utils.get_db_emb(self.db_embs, ["#WORD/UNK#"], "word")[0]
        assert (
            test is not None
        ), "DB embeddings in wrong folder..? Test embedding not found.."
        test = rel_utils.get_db_emb(self.db_embs, ["#SND/UNK#"], "snd")[0]
        assert (
            test is not None
        ), "DB embeddings in wrong folder..? Test embedding not found.."

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

    def __get_config(self, user_config: dict) -> dict:
        """
        Retrieves the configuration used for entity disambiguation,
        considering user-defined settings.

        This method retrieves the configuration used for entity disambiguation.
        It considers user-defined settings provided through the ``user_config``
        parameter and combines them with default settings. The resulting
        configuration dictionary is returned.

        Arguments:
            user_config: A dictionary containing user-defined configuration
                settings.

        Returns:
            dict:
                The configuration used for entity disambiguation, including
                both user-defined and default settings.
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

    def __load_embeddings(self) -> None:
        """
        Loads the embeddings for different entities.

        This method initializes and loads the embeddings for different
        entities (``snd``, ``entity``, and ``word``). It sets up the necessary
        data structures and initializes the embeddings with pre-trained values.
        It also adds the unknown token to the vocabulary and retrieves the
        corresponding embedding from the database.

        Returns:
            None
        """
        self.__batch_embs = {}

        for name in ["snd", "entity", "word"]:
            # Init entity embeddings.
            self.embeddings["{}_seen".format(name)] = set()
            self.embeddings["{}_voca".format(name)] = Vocabulary()
            self.embeddings["{}_embeddings".format(name)] = None

            # Add #UNK# token.
            self.embeddings["{}_voca".format(name)].add_to_vocab("#UNK#")
            e = rel_utils.get_db_emb(
                self.db_embs, ["#{}/UNK#".format(name.upper())], name
            )[0]
            assert e is not None, "#UNK# token not found for {} in db".format(name)
            self.__batch_embs[name] = []
            self.__batch_embs[name].append(torch.tensor(e))

    # TODO/typing: Set types for org_train_dataset and org_dev_dataset correctly
    def train(self, org_train_dataset: Any, org_dev_dataset: Any) -> None:
        """
        Trains the entity disambiguation model.

        This method is responsible for training the entity disambiguation
        model using the provided training dataset and evaluating its
        performance on the development dataset. It iterates over multiple
        epochs and performs mini-batch training. The model parameters are
        updated using the Adam optimizer, and the training loss is computed
        and monitored during the training process.

        Arguments:
            org_train_dataset: The original training dataset.
            org_dev_dataset: The original development dataset.

        Returns:
            None.
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
            random.shuffle(train_dataset)

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
                    torch.LongTensor([m["selected_cands"]["cands"] for m in batch]).to(
                        self.device
                    )
                )
                true_pos = Variable(
                    torch.LongTensor(
                        [m["selected_cands"]["true_pos"] for m in batch]
                    ).to(self.device)
                )
                p_e_m = Variable(
                    torch.FloatTensor([m["selected_cands"]["p_e_m"] for m in batch]).to(
                        self.device
                    )
                )
                entity_mask = Variable(
                    torch.FloatTensor([m["selected_cands"]["mask"] for m in batch]).to(
                        self.device
                    )
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
                self.model.s_ltoken_ids = Variable(
                    torch.LongTensor(s_ltoken_ids).to(self.device)
                )
                self.model.s_ltoken_mask = Variable(
                    torch.FloatTensor(s_ltoken_mask).to(self.device)
                )
                self.model.s_rtoken_ids = Variable(
                    torch.LongTensor(s_rtoken_ids).to(self.device)
                )
                self.model.s_rtoken_mask = Variable(
                    torch.FloatTensor(s_rtoken_mask).to(self.device)
                )
                self.model.s_mtoken_ids = Variable(
                    torch.LongTensor(s_mtoken_ids).to(self.device)
                )
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
                    "Micro F1: {}, Recall: {}, Precision: {}".format(
                        dev_f1, recall, precision
                    ),
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

    # TODO/typing: Set types for dataset correctly
    def __create_dataset_LR(
        self, dataset: Any, predictions: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a dataset for logistic regression training.

        This method takes as input a dataset and the corresponding predictions
        made by the entity disambiguation model. It constructs a dataset
        suitable for logistic regression training by extracting features,
        labels, and metadata. The features are represented by the prediction
        scores, the labels indicate whether a prediction is correct or not, and
        the metadata contains additional information about each instance,
        including the document, ground truth, and candidate entity.

        Arguments:
            dataset (Any): The input dataset.
            predictions (dict): A dictionary containing the predictions made
                by the entity disambiguation model.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the feature matrix, label array, and
                metadata array, respectively.
        """
        X = []
        y = []
        meta = []
        for doc, preds in predictions.items():
            gt_doc = [c["gold"][0] for c in dataset[doc]]
            for pred, gt in zip(preds, gt_doc):
                scores = [float(x) for x in pred["scores"]]
                cands = pred["candidates"]

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

    # TODO/typing: Set types for train_json and dev_json correctly
    def train_LR(self, train_json: Any, dev_json: Any, model_path_lr: str) -> None:
        """
        Trains a logistic regression model to obtain confidence scores.

        This method applies logistic regression (LR) in an attempt to obtain
        confidence scores for entity disambiguation predictions. It trains an
        LR model using the training dataset and evaluates it on the
        development dataset. The LR model is trained to achieve high recall
        since it aims to avoid ignoring correct entities. The resulting LR
        model is saved to the specified ``model_path_lr``.

        Arguments:
            train_json (Any): The path to the training dataset in JSON format.
            dev_json (Any): The path to the development dataset in JSON format.
            model_path_lr (str): The path to save the trained LR model.

        Returns:
            None
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
        Predicts entity disambiguation on the given raw text data.

        This method is responsible for predicting entity disambiguation on any
        raw text input. It takes the raw text data as input and performs the
        necessary preprocessing steps to prepare it for entity disambiguation.
        The trained model is used to make predictions on the preprocessed data,
        generating entity disambiguation results.

        Arguments:
            data: The raw text data to be processed and disambiguated.

        Returns:
            dict:
                A dictionary containing the predictions for each document in
                the input data. The keys of the dictionary are the document
                names, and the values are lists of prediction objects. Each
                prediction object contains the mention, predicted entity, list
                of candidate entities, confidence score, and other relevant
                information.

        Note:
            The predictions are made using the trained model and do not
            require ground truth entities to be present.

            The confidence scores indicate the level of confidence or
            certainty in the predicted entity.

            The number of predictions and the time taken for the entity
            disambiguation step may vary depending on the input data and the
            complexity of the model.
        """
        data = self.get_data_items(data, "raw", predict=True)
        predictions, timing = self.__predict(data, include_timing=True, eval_raw=True)

        return predictions

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalizes a list of scores between 0 and 1 by rescaling them and
        computing their ratio over their sum.

        Arguments:
            scores (List[float]): A list of numerical scores.

        Returns:
            List[float]:
                A list of normalized scores where each score is the ratio of
                the rescaled score over their sum.
        """
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [0.0] * len(scores)

        rescaled_scores = [
            (score - min_score) / (max_score - min_score) for score in scores
        ]

        # calculate sum of rescaled scores
        score_sum = sum(rescaled_scores)

        # normalize each rescaled score
        normalized_scores = [score / score_sum for score in rescaled_scores]

        return normalized_scores

    def __compute_cross_cand_confidence(self, scores: np.ndarray) -> List[List[float]]:
        """
        This function takes a series of numpy arrays of scores and returns a
        list of lists of confidence scores.

        Arguments:
            scores (numpy.ndarray): A numpy array of scores.

        Returns:
            List[List[float]]: A list of lists of confidence scores.
        """
        normalised_scores = [self.normalize_scores(score) for score in scores]
        return normalised_scores

    # TODO: Set type of preds to Tensor
    def __compute_confidence(self, scores: np.ndarray, preds) -> List[float]:
        """
        Computes confidence scores for the given entity disambiguation outputs
        using logistic regression.

        This method uses logistic regression (LR) to calculate confidence
        scores for the entity disambiguation outputs. It takes the prediction
        scores and corresponding predicted entities as input. The LR model,
        trained to obtain confidence scores, is applied to the input data.
        The resulting confidence scores reflect the level of confidence or
        certainty in the predicted entities.

        Arguments:
            scores: The prediction scores for each candidate entity.
            preds: The predicted entities corresponding to the prediction
                scores.

        Returns:
            List[float]:
                A list of confidence scores for each entity disambiguation
                output.
        """
        X = np.array([[score[pred]] for score, pred in zip(scores, preds)])
        if self.model_lr:
            preds = self.model_lr.predict_proba(X)
            confidence_scores = [x[1] for x in preds]
        else:
            confidence_scores = [0.0 for _ in scores]
        return confidence_scores

    # TODO/typing: What's the correct type to use for data here?
    def __predict(
        self,
        data,
        include_timing: Optional[bool] = False,
        eval_raw: Optional[bool] = False,
    ) -> Union[Tuple[dict, List[float]], dict]:
        """
        Applies the trained model to make predictions on input data.

        This method utilizes the trained model to predict entity
        disambiguation outputs for a given dataset or raw text. It processes
        the input data in batches, performs necessary computations, and
        returns the predicted entities along with optional timing information.

        Arguments:
            data: The input dataset or raw text to be processed and predicted.
            include_timing (bool, optional): A flag indicating whether to
                include timing information for the prediction step. Defaults
                to ``False``.
            eval_raw (bool, optional): A flag indicating whether the
                prediction is being performed for evaluation on raw text.
                Defaults to ``False``.

        Returns:
            Union[Tuple[dict, List[float]], dict]
                If ``include_timing`` is ``True``, the function returns a
                tuple containing a dictionary of predicted entities and
                optional timing information for the prediction step.

                If ``include_timing`` is ``False``, the function simply
                returns a dictionary of predicted entities.

        Note:
            The prediction can be performed on both structured datasets and
            raw text inputs.

            For structured datasets, the input data is typically organized
            into documents and mentions.

            Batches of data are processed sequentially, and the predictions
            are accumulated into a dictionary structure.

            The model is set to evaluation mode (self.model.eval()) before
            making predictions.

            The prediction process involves tokenizing the input data,
            applying the model's forward pass, and extracting the predicted
            entities and their confidence scores.

            The predicted entities can be stored in different formats
            depending on the evaluation mode.

            If ``eval_raw=True``, additional information such as mention,
            candidates, and scores may be included in the predicted entities.

            Timing information can be useful for measuring the performance of
            the prediction step.
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
                torch.LongTensor([m["selected_cands"]["cands"] for m in batch]).to(
                    self.device
                )
            )
            p_e_m = Variable(
                torch.FloatTensor([m["selected_cands"]["p_e_m"] for m in batch]).to(
                    self.device
                )
            )
            entity_mask = Variable(
                torch.FloatTensor([m["selected_cands"]["mask"] for m in batch]).to(
                    self.device
                )
            )
            true_pos = Variable(
                torch.LongTensor([m["selected_cands"]["true_pos"] for m in batch]).to(
                    self.device
                )
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

            self.model.s_ltoken_ids = Variable(
                torch.LongTensor(s_ltoken_ids).to(self.device)
            )
            self.model.s_ltoken_mask = Variable(
                torch.FloatTensor(s_ltoken_mask).to(self.device)
            )
            self.model.s_rtoken_ids = Variable(
                torch.LongTensor(s_rtoken_ids).to(self.device)
            )
            self.model.s_rtoken_mask = Variable(
                torch.FloatTensor(s_rtoken_mask).to(self.device)
            )
            self.model.s_mtoken_ids = Variable(
                torch.LongTensor(s_mtoken_ids).to(self.device)
            )
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
            pred_ids = torch.argmax(scores, axis=1)

            # scores from the pipeline
            scores = scores.cpu().data.numpy()

            # LR derived scores
            confidence_scores = self.__compute_confidence(scores, pred_ids)

            # normalised scores across candidates
            cross_cands_scores = self.__compute_cross_cand_confidence(scores)
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
                # list of mentions
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
                    for (i, m, s, cs) in zip(
                        pred_ids, batch, cross_cands_scores, confidence_scores
                    )
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

    # TODO/typing: Set type for dataset correctly
    def prerank(
        self, dataset: Any, dname: str, predict: Optional[bool] = False
    ) -> List[List[Any]]:
        """
        Preranks the candidate entities in the dataset using context and
        p(e|m) scores.

        This method is responsible for preranking the set of possible
        candidates for entity disambiguation. It utilises both contextual
        information and the p(e|m) scores to rank the candidates for each
        mention. The preranking process helps narrow down the candidate pool,
        making the subsequent disambiguation step more efficient.

        Arguments:
            dataset: The dataset containing the mentions and candidate
                entities to be preranked.
            dname (str): The name of the dataset, specifying its type or
                purpose.
            predict (bool, optional): A flag indicating whether the preranking
                is performed for prediction purposes. Defaults to ``False``.

        Returns:
            List[List[Any]]: The preranked dataset with a reduced (max 3 + 4)
                set of candidate entities for each mention.

        Note:
            The preranking process is applied to each mention in the dataset.

            The candidate entities are ranked based on a combination of their
            contextual relevance (context scores) and the p(e|m) scores.

            The number of candidates to keep during preranking is determined
            by the configuration settings.

            For each mention, the top candidates based on context scores and
            p(e|m) scores are selected.

            The preranked candidates are stored in a modified dataset, where
            each mention is associated with a reduced set of candidate
            entities.

            The preranking step helps improve efficiency by reducing the
            number of candidates for subsequent disambiguation.

            In prediction mode, additional considerations may be taken into
            account, such as including a fake gold candidate if no valid
            candidate is found.
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
                            len(m["context"][0])
                            - self.config["prerank_ctx_window"] // 2,
                            0,
                        ) :
                    ]
                    for m in content
                ]
                rctx_ids = [
                    m["context"][1][
                        : min(
                            len(m["context"][1]), self.config["prerank_ctx_window"] // 2
                        )
                    ]
                    for m in content
                ]
                ment_ids = [[] for m in content]
                token_ids = [
                    l + m + r
                    if len(l) + len(r) > 0
                    else [self.embeddings["word_voca"].unk_id]
                    for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)
                ]

                entity_ids = [m["cands"] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).to(self.device))

                entity_mask = [m["mask"] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).to(self.device))

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_offsets = Variable(
                    torch.LongTensor(token_offsets).to(self.device)
                )
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
                while (
                    len(selected)
                    < self.config["keep_ctx_ent"] + self.config["keep_p_e_m"]
                ):
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
                        sm[
                            "true_pos"
                        ] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        # if total > 0
        if dname != "raw":
            print("Recall for {}: {}".format(dname, has_gold / total))
            print("-----------------------------------------------")

        return new_dataset

    # TODO/typing: Ensure embs is set to the correct type here
    def __update_embeddings(self, emb_name: str, embs: Any) -> None:
        """
        Updates the embeddings with the respective ``word``, ``entity``, or
        ``snd`` (GloVe) embeddings.

        This method is responsible for updating the embedding dictionaries
        with new ``word``, ``entity``, or ``snd`` (GloVe) embeddings. It takes
        the embeddings as input and incorporates them into the existing
        dictionaries.

        Arguments:
            emb_name (str): The name of the embedding type, specifying whether
                it is ``word``, ``entity``, or ``snd`` (GloVe) embeddings.
            embs: The new embeddings to be added to the existing ones.

        Returns:
            None

        Notes:
            The embedding update process involves adding the new embeddings to
            the existing embedding dictionaries.

            The updated embeddings are stored in embedding layers for
            efficient retrieval during the disambiguation process.

            If the embedding layer for the specified embedding type already
            exists, the new embeddings are concatenated with the existing
            embeddings.

            If the embedding layer for the specified embedding type doesn't
            exist, a new embedding layer is created with the new embeddings.

            The size of the new embeddings should match the expected embedding
            dimensions specified in the configuration.

            After updating the embedding dictionaries, the new embeddings are
            made available for disambiguation.

            The __update_embeddings method is typically called after
            retrieving embeddings from an external source.
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

    def __embed_words(self, words_filt: List[str], name: str) -> None:
        """
        Embeds words using the given SQLite3 database.

        This method retrieves embeddings for the specified words from a
        SQLite3 database and updates the corresponding embedding dictionaries.

        Arguments:
            words_filt (List[str]): A list of words for which embeddings need
                to be retrieved.
            name (str): The name of the embedding type, specifying whether it
                is ``word``, ``entity``, or ``snd`` (GloVe) embeddings.

        Returns:
            None.

        Notes:
            The __embed_words method retrieves embeddings for the specified
            words using an SQLite3 database.

            The embedding dictionaries and sets are updated with the new words
            and their embeddings.

            If an embedding for a word is not found in the database, it is
            treated as an out-of-vocabulary (OOV) word.

            The OOV words are added to the respective "seen" sets to keep
            track of the words for which embeddings are not available.

            The embedding vectors retrieved from the database are appended to
            the batch embeddings for later updating of the embedding
            dictionaries.

            The __embed_words method is typically called during the data
            preprocessing stage to embed words, entities, or GloVe
            representations.
        """
        embs = rel_utils.get_db_emb(self.db_embs, words_filt, name)

        # Now we go over the embs and see which one is None. Order is preserved.
        for e, c in zip(embs, words_filt):
            self.embeddings["{}_seen".format(name)].add(c)
            if e is not None:
                # Embedding exists, so we add it.
                self.embeddings["{}_voca".format(name)].add_to_vocab(c)
                self.__batch_embs[name].append(torch.tensor(e))

    # TODO/typing: Add correct type for dataset and output
    def get_data_items(
        self, dataset: Any, dname: str, predict: Optional[bool] = False
    ) -> Any:
        """
        Formats the dataset and triggers the preranking function.

        This method is responsible for formatting the input dataset, preparing
        it for entity disambiguation. It iterates over the dataset items and
        performs various data processing steps such as embedding words,
        extracting context, handling candidate entities, and creating data
        items. It triggers the preranking function to rank the candidate
        entities for each mention. The formatted data is returned for further
        processing.

        Arguments:
            dataset (Any): The input dataset to be formatted.
            dname (str): The name of the dataset, specifying its type or
                purpose.
            predict (bool, optional): A flag indicating whether the data
                formatting is for prediction purposes or not. Defaults to
                ``False``.

        Returns:
            Any:
                The formatted dataset that has undergone various preprocessing
                steps, ready for preranking.
        """

        data = []

        for doc_name, content in dataset.items():
            items = []
            if len(content) == 0:
                continue
            for m in content:
                named_cands = [c[0] for c in m["candidates"]]
                p_e_m = [min(1.0, max(1e-3, c[1])) for c in m["candidates"]]

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
                    [
                        item
                        for item in named_cands
                        if item not in self.embeddings["entity_seen"]
                    ]
                )

                self.__embed_words(named_cands_filt, "entity")

                my_punctuation = punctuation.replace("'", "")
                my_punctuation = my_punctuation.replace("-", "")
                # Use re.split() to make sure that special characters are considered.
                lctx = [
                    x
                    for x in m["context"][0]
                    .translate(str.maketrans("", "", my_punctuation))
                    .split()
                ]
                rctx = [
                    x
                    for x in m["context"][1]
                    .translate(str.maketrans("", "", my_punctuation))
                    .split()
                ]

                words_filt = set(
                    [
                        item
                        for item in lctx + rctx
                        if utils.is_important_word(item)  # Only non-stopwords
                        and item not in self.embeddings["word_seen"]
                    ]
                )

                self.__embed_words(words_filt, "word")

                snd_lctx = m["sentence"][: m["pos"]]
                snd_lctx = [
                    x
                    for x in snd_lctx.translate(
                        str.maketrans("", "", my_punctuation)
                    ).split()
                ]
                snd_lctx = [
                    t for t in snd_lctx[-self.config["snd_local_ctx_window"] // 2 :]
                ]

                snd_rctx = m["sentence"][m["end_pos"] :]
                snd_rctx = [
                    x
                    for x in snd_rctx.translate(
                        str.maketrans("", "", my_punctuation)
                    ).split()
                ]
                snd_rctx = [
                    t for t in snd_rctx[: self.config["snd_local_ctx_window"] // 2]
                ]

                snd_ment = m["ngram"].strip().split()

                words_filt = set(
                    [
                        item
                        for item in snd_lctx + snd_rctx + snd_ment
                        if utils.is_important_word(item)  # Only non-stopwords
                        and item not in self.embeddings["snd_seen"]
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

                lctx_ids = [
                    tid
                    for tid in lctx_ids
                    if tid != self.embeddings["word_voca"].unk_id
                ]
                lctx_ids = lctx_ids[
                    max(0, len(lctx_ids) - self.config["ctx_window"] // 2) :
                ]

                rctx_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in rctx
                    if utils.is_important_word(t)
                ]
                rctx_ids = [
                    tid
                    for tid in rctx_ids
                    if tid != self.embeddings["word_voca"].unk_id
                ]
                rctx_ids = rctx_ids[
                    : min(len(rctx_ids), self.config["ctx_window"] // 2)
                ]

                ment = m["mention"].strip().split()
                ment_ids = [
                    self.embeddings["word_voca"].get_id(t)
                    for t in ment
                    if utils.is_important_word(t)
                ]
                ment_ids = [
                    tid
                    for tid in ment_ids
                    if tid != self.embeddings["word_voca"].unk_id
                ]

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
                # note: this shouldn't affect the order of prediction because
                # we use doc_name to add predicted entities, and we don't
                # shuffle the data for prediction
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

    # TODO/typing: Make sure the types of testset and system_pred are correctly set
    def __eval(self, testset: Any, system_pred: Any) -> Tuple[float, float, float, int]:
        """
        Evaluates the performance of the local entity disambiguation step.

        This method evaluates the performance of the local entity
        disambiguation by comparing the predicted entity labels with the gold
        labels from the testset.

        Arguments:
            testset (Any): The testset containing the ground truth labels for
                evaluation.
            system_pred (Any): The predicted entity labels generated by the
                system.

        Returns:
            Tuple[float, float, float, int]:
                A tuple containing the F1 score, recall, precision, and the
                number of mentions without valid candidate entities.

        Notes:
            The __eval method compares the predicted entity labels with the
            gold labels to compute evaluation metrics.

            The gold labels are extracted from the testset, while the
            predicted labels are obtained from the system predictions.

            The method calculates the true positives, total number of NIL (no
            valid candidate) mentions, precision, recall, and F1 score.

            The precision is the ratio of true positives to the total number
            of predicted labels excluding NIL mentions.

            The recall is the ratio of true positives to the total number of
            gold labels.

            The F1 score is the harmonic mean of precision and recall,
            providing a balanced measure of the model's performance.

            The number of mentions without valid candidate entities indicates
            the cases where the system could not find a suitable entity label.
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

    def __save(self, path: str) -> None:
        """
        Saves the trained model and its configuration during optimization.

        This method saves the state dictionary of the trained model and its
        configuration to the specified ``path``. The trained model's state
        dictionary contains the learned parameters and can be used to restore
        the model for later use or further optimization. The configuration
        file contains the settings and hyperparameters used during training.

        Arguments:
            path (str): The path where the model and configuration files will
                be saved.

        Returns:
            None.
        """
        torch.save(self.model.state_dict(), "{}.state_dict".format(path))
        with open("{}.config".format(path), "w") as f:
            json.dump(self.config, f)

    def __load(self, path: str) -> MulRelRanker:
        """
        Loads a trained model and its configuration from the specified path.

        This method loads a trained model and its corresponding configuration
        from the given ``path``. The model is instantiated based on the loaded
        configuration, and the learned parameters are loaded into the model.
        The configuration file provides the settings and hyperparameters used
        during training.

        If a configuration file is found at the specified path, it will be
        loaded and replace the existing configuration in the ``self.config``
        attribute. If no configuration file is found, default settings will be
        used.

        Arguments:
            path (str): The path where the model and configuration files are
                stored.

        Returns:
            model (utils.REL.mulrel_ranker.MulRelRanker):
                The loaded trained model.

        Note:
            The cofiguration can currently not be overwritten. If required,
            this behavior may be modified in future releases.
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
