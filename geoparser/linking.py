import os
import sys
import json
import hashlib
import urllib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# from transformers import AutoTokenizer, pipeline

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
from utils import training, process_data, gnn_method
from utils.gnn_method import *

from REL.REL.entity_disambiguation import EntityDisambiguation
from REL.REL.mention_detection import MentionDetection


class Linker:
    def __init__(
        self,
        method,
        resources_path,
        linking_resources,
        base_model,
        overwrite_training,
        rel_params,
        gnn_params,
    ):
        self.method = method
        self.resources_path = resources_path
        self.linking_resources = linking_resources
        self.base_model = base_model
        self.overwrite_training = overwrite_training
        self.rel_params = rel_params
        self.gnn_params = gnn_params

    def __str__(self):
        s = ">>> Entity Linking:\n\t* Method: {0}\n\t* Overwrite training: {1}\n\t* Linking resources: {2}\n".format(
            self.method,
            str(self.overwrite_training),
            ",".join(list(self.linking_resources.keys())),
        )
        return s

    def load_resources(self):
        """
        Load resources required for linking.
        Note: different methods will require different resources.

        Returns:
            self.linking_resources (dict): a dictionary storing the resources
                that will be needed for a specific linking method.
        """
        # Load Wikidata gazetteer
        gaz = pd.read_csv(
            self.resources_path + "wikidata_gazetteer.csv", low_memory=False
        )

        # Gazetteer entity classes:
        gaz["instance_of"] = gaz["instance_of"].apply(process_data.eval_with_exception)

        # Gazetteer:
        self.linking_resources["gazetteer"] = gaz

        if self.method in ["gnn"]:
            print("  > Loading wikidata entity and instance embeddings.")
            self.linking_resources["instance_ids"] = (
                open(self.resources_path + "gazetteer_wkdtclass_ids.txt", "r")
                .read()
                .split("\n")
            )
            self.linking_resources["entity_ids"] = (
                open(self.resources_path + "gazetteer_entity_ids.txt", "r")
                .read()
                .split("\n")
            )

            self.linking_resources["instance_embeddings"] = np.load(
                self.resources_path + "gazetteer_wkdtclass_embeddings.npy"
            )
            self.linking_resources["entity_embeddings"] = np.load(
                self.resources_path + "gazetteer_entity_embeddings.npy"
            )

            # create a random embedding for entities without a instance ids
            # this will make sure the embedding is the same in case there is not instance id
            self.linking_resources["random_instance_embedding"] = np.random.uniform(
                low=-1.0, high=1.0, size=(1, 128)
            )
            # for entities for which we do not have an embedding, we will create an
            # a random embedding for each entity
            self.linking_resources["random_entity_embeddings"] = {}

            print("  > Mapping wikidata ids to instance ids.")
            self.linking_resources[
                "wikidata_id2inst_id"
            ] = process_data.get_wikidata_instance_ids(self)

        # Load Wikidata mentions-to-wikidata (with absolute counts) to QID dictionary
        if self.method in [
            "mostpopular",
            "reldisamb:lwmcs:relv",
            "reldisamb:lwmcs:relvpubl",
            "reldisamb:lwmcs:relvdist",
            "gnn",
        ]:
            print("  > Loading mentions to wikidata mapping.")
            with open(self.resources_path + "mentions_to_wikidata.json", "r") as f:
                self.linking_resources["mentions_to_wikidata"] = json.load(f)

        if self.method in ["reldisamb:lwmcs:dist", "reldisamb:lwmcs:relvdist", "gnn"]:
            print("  > Mapping coordinates to wikidata ids.")
            dict_wqid_to_lat = dict(zip(gaz.wikidata_id, gaz.latitude))
            dict_wqid_to_lon = dict(zip(gaz.wikidata_id, gaz.longitude))
            self.linking_resources["dict_wqid_to_lat"] = dict_wqid_to_lat
            self.linking_resources["dict_wqid_to_lon"] = dict_wqid_to_lon

        # REL disambiguates to Wikipedia, not Wikidata:
        if "reldisamb" in self.method:
            # WIkipedia to Wikidata
            with open(
                "/resources/wikipedia/extractedResources/wikipedia2wikidata.json", "r"
            ) as f:
                self.linking_resources["wikipedia2wikidata"] = json.load(f)
            # Wikidata to Wikipedia
            with open(
                "/resources/wikipedia/extractedResources/wikidata2wikipedia.json", "r"
            ) as f:
                self.linking_resources["wikidata2wikipedia"] = json.load(f)

            # Keep only wikipedia entities in the gazetteer:
            wkdt_allcands = set(gaz["wikidata_id"].tolist())
            self.linking_resources["wikipedia_locs"] = set(
                dict(
                    filter(
                        lambda x: x[1] in wkdt_allcands,
                        self.linking_resources["wikipedia2wikidata"].items(),
                    )
                ).keys()
            )
        return self.linking_resources

    # ----------------------------------------------
    def perform_training(self, all_df, processed_df, whichsplit):
        """
        TODO: Here will go the code to perform training, checking
        variable overwrite_training.

        Arguments:
            training_df (pd.DataFrame): a dataframe with a mention per row, for training.

        Returns:
            xxxxxxx
        """
        if "mostpopular" in self.method:
            return None
        if "reldisamb" in self.method:
            self.rel_params["model"] = training.train_rel_ed(
                self, all_df, processed_df, whichsplit
            )
            return self.rel_params
        if "gnn" in self.method:
            processed_df_notest = processed_df[processed_df[whichsplit] != "test"]
            data = gnn_method.network_data(
                self, processed_df_notest, whichsplit=whichsplit
            )
            model = EnhancedGATCN(data.x.shape[1], hidden_channels=64, edge_dim=4)
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 5.0]), reduction="mean"
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            trainer = training.Trainer(
                data,
                model,
                criterion,
                optimizer,
                self.gnn_params["model_path"],
                whichsplit,
            )
            trainer.set_split()
            trainer.print_split()
            print("\n")
            trainer.print_network_statistics()
            trainer.training_routine(1000)
            return self.gnn_params

    # ----------------------------------------------
    def perform_linking(self, test_df, original_df, whichsplit):
        """
        Perform the linking.

        Arguments:
            test_df (pd.DataFrame): a dataframe with a mention per row, for testing.

        Returns:
            test_df_results (pd.DataFrame): the same dataframe, with two additional
                columns: "pred_wqid" for the linked wikidata Id and "pred_wqid_score"
                for the linking score.
        """

        test_df_results = test_df.copy()

        if "mostpopular" in self.method:
            test_df_results[["pred_wqid", "pred_wqid_score"]] = test_df_results.apply(
                lambda row: self.most_popular(row.to_dict()),
                axis=1,
                result_type="expand",
            )

        if "reldisamb" in self.method:
            dRELresults = self.rel_disambiguation(test_df, original_df)
            test_df_results[["pred_wqid", "pred_wqid_score"]] = test_df_results.apply(
                lambda row: dRELresults[row["article_id"]][int(row["sentence_pos"])][
                    row["pred_mention"]
                ],
                axis=1,
                result_type="expand",
            )

        if "gnn" in self.method:
            data = gnn_method.network_data(self, test_df)
            model_folder = (
                Path(self.gnn_params["model_path"]) / f"best_model_{whichsplit}"
            )
            model = EnhancedGATCN(data.x.shape[1], hidden_channels=64, edge_dim=4)
            model.load_state_dict(torch.load(model_folder / "best-model.pt"))
            model.eval()
            self.gnn_params["model"] = model
            test_df_results = gnn_method.get_model_predictions(
                self, test_df, whichsplit
            )

        return test_df_results

    def rel_disambiguation(self, test_df, original_df):
        # Warning: no model has been trained, the latest one that's been trained will be loaded.
        if not "model" in self.rel_params:
            print(
                "WARNING! There is no ED model in memory, we will load the latest model that has been trained."
            )
        base_path = self.rel_params["base_path"]
        wiki_version = self.rel_params["wiki_version"]
        # Instantiate REL mention detection:
        self.rel_params["mention_detection"] = MentionDetection(
            base_path, wiki_version, locs_only=True, mylinker=self
        )
        # Instantiate REL entity disambiguation:
        config = {
            "mode": "eval",
            "model_path": "{}/{}/generated/model".format(base_path, wiki_version),
        }
        self.rel_params["model"] = EntityDisambiguation(base_path, wiki_version, config)

        dRELresults = dict()
        mentions_dataset = dict()
        # Given our mentions, use REL candidate selection module:
        if "reldisamb" in self.method:
            mentions_dataset, n_mentions = self.rel_params[
                "mention_detection"
            ].format_detected_spans(
                test_df,
                original_df,
                mylinker=self,
            )

        # Given the mentions dataset, predict and return linking:
        for mentions_doc in mentions_dataset:
            link_predictions, timing = self.rel_params["model"].predict(
                mentions_dataset[mentions_doc]
            )
            for p in link_predictions:
                mentions_sent = p
                for m in link_predictions[p]:
                    returned_mention = m["mention"]
                    returned_prediction = m["prediction"]

                    # Wikipedia prediction to Wikidata:
                    percent_encoded_title = urllib.parse.quote(
                        returned_prediction.replace("_", " ")
                    )
                    if "/" in percent_encoded_title or len(percent_encoded_title) > 200:
                        percent_encoded_title = hashlib.sha224(
                            percent_encoded_title.encode("utf-8")
                        ).hexdigest()
                    returned_prediction = self.linking_resources[
                        "wikipedia2wikidata"
                    ].get(percent_encoded_title, "NIL")

                    # Disambiguation confidence:
                    returned_confidence = round(m.get("conf_ed", 0.0), 3)

                    if mentions_doc in dRELresults:
                        if mentions_sent in dRELresults[mentions_doc]:
                            dRELresults[mentions_doc][mentions_sent][
                                returned_mention
                            ] = (returned_prediction, returned_confidence)
                        else:
                            dRELresults[mentions_doc][mentions_sent] = {
                                returned_mention: (
                                    returned_prediction,
                                    returned_confidence,
                                )
                            }
                    else:
                        dRELresults[mentions_doc] = {
                            mentions_sent: {
                                returned_mention: (
                                    returned_prediction,
                                    returned_confidence,
                                )
                            }
                        }

        return dRELresults

    # ----------------------------------------------
    # Most popular candidate:
    def most_popular(self, dict_mention):
        """
        The most popular disambiguation method, which is a painfully strong baseline.
        Given a set of candidates for a given mention, returns as a prediction the
        candidate that is more relevant in terms of inlink structure in Wikipedia.

        Arguments:
            dict_mention (dict): dictionary with all the relevant information needed
                to disambiguate a certain mention.

        Returns:
            keep_most_popular (str): the Wikidata ID (e.g. "Q84") or "NIL".
            final_score (float): the confidence of the predicted link.
        """
        cands = dict_mention["candidates"]
        keep_most_popular = "NIL"
        keep_highest_score = 0.0
        total_score = 0.0
        final_score = 0.0
        if cands:
            for variation in cands:
                for candidate in cands[variation]["Candidates"]:
                    score = self.linking_resources["mentions_to_wikidata"][variation][
                        candidate
                    ]
                    total_score += score
                    if score > keep_highest_score:
                        keep_highest_score = score
                        keep_most_popular = candidate
            # we return the predicted and the score (overall the total):
            final_score = keep_highest_score / total_score

        return keep_most_popular, final_score
