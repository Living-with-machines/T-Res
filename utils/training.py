import pandas as pd
from pathlib import Path
from utils import process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import math
import tqdm
from haversine import haversine

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear
import torch.nn.functional as F

from utils import tools_perceptron


def find_geo_distance(wqid, publ_coordinates, mylinker):
    # By default, distance is the approx max world distance between two places on earth:
    dist_from_publ = 40000
    if wqid in mylinker.linking_resources["dict_wqid_to_lat"]:
        lat_cand = mylinker.linking_resources["dict_wqid_to_lat"][wqid]
        lon_cand = mylinker.linking_resources["dict_wqid_to_lon"][wqid]
        dist_from_publ = haversine(publ_coordinates, (lat_cand, lon_cand))
    if dist_from_publ >= 3:
        return 1 / (math.log(dist_from_publ))
    else:
        return 1.0


def create_trainset(train_path, myranker, mylinker):
    """
    Create entity linking training data (i.e. mentions identified and candidates provided),
    necessary for training our resolution methods:
    """
    print("\n*** Creating the entity linking trainset with candidates...")
    training_cands_path = train_path.split(".tsv")[0] + "_" + myranker.method + ".pkl"
    if not Path(training_cands_path).exists():
        training_set = pd.read_csv(train_path, sep="\t")
        training_df = process_data.crate_training_for_el(training_set)

        # # Train on LOC and OTHERs alone:
        # training_df = training_df[training_df["entity_type"].isin(["LOC", "OTHER"])]

        candidates_qids = dict()

        mention_publ_unique = (
            training_df.groupby(["publication_code", "mention"]).size().reset_index()
        )

        for i, row in tqdm.tqdm(mention_publ_unique.iterrows()):
            original_mention = row["mention"]
            publ_code = row["publication_code"]
            publ_latitude = mylinker.linking_resources["publication_metadata"][
                publ_code
            ]["latitude"]
            publ_longitude = mylinker.linking_resources["publication_metadata"][
                publ_code
            ]["longitude"]
            publ_coordinates = (publ_latitude, publ_longitude)

            # For each mention, find its mention matches according to our ranker:
            (
                candidates_matches,
                myranker.already_collected_cands,
            ) = myranker.run([original_mention])

            # For each mention, find its mention matches, the corresponding wikidata
            # entities, and the confidence score.
            wk_cands = dict()
            for found_mention in candidates_matches[original_mention]:
                found_cands = mylinker.get_candidate_wikidata_ids(found_mention)
                if found_cands:
                    for cand in found_cands:
                        log_geo_dist = find_geo_distance(
                            cand, publ_coordinates, mylinker
                        )
                        if not cand in wk_cands:
                            wk_cands[cand] = {
                                "conf_score": candidates_matches[original_mention][
                                    found_mention
                                ],
                                "wkdt_relv": found_cands[cand],
                                "log_dist": log_geo_dist,
                            }

            candidates_qids[original_mention] = wk_cands

        training_df["candidates"] = training_df["mention"].map(candidates_qids)

        print("\n*** Obtaining the vector embeddings of the mention in context.")
        training_df.loc[:, "mention_emb"] = training_df.loc[:, :].apply(
            lambda x: mylinker.get_mention_vector(x, agg=np.mean), axis=1
        )
        print("*** ... vectors obtained!\n")

        training_df.to_pickle(training_cands_path)
        print("... dataset creation completed!")
        return training_df
    else:
        print("*** ... dataset creation skipped, it already exists!\n")

    training_df = pd.read_pickle(training_cands_path)
    return training_df


def create_data_from_row(
    mylinker, j, row, features=["conf_score", "wkdt_relv", "log_dist"]
):

    candidates = {}

    for c, vd in row.candidates.items():
        candidates[c] = [c] + [vd.get(f, 0.0) for f in features]

    candidates = pd.DataFrame.from_dict(
        candidates, orient="index", columns=["wikidata_id"] + features
    )

    if candidates.shape[0]:

        candidates["wkdt_relv"] = (
            candidates["wkdt_relv"] / candidates["wkdt_relv"].max()
        )
        candidates["ext_vector"] = candidates.apply(lambda x: list(x[features]), axis=1)

    candidates["y"] = 0
    candidates.loc[candidates.wikidata_id == row.wkdt_qid, "y"] = 1

    # add vector representation for class instances
    _embeddings = []
    for i, candidate in candidates.iterrows():
        _embeddings.append(
            tools_perceptron.get_candidate_representation(
                row,
                candidate.wikidata_id,
                mylinker.linking_resources["dict_wqid_to_entemb"],
                mylinker.linking_resources["dict_wqid_to_class"],
                mylinker.linking_resources["dict_wqid_to_clssemb"],
            )
        )

    candidates["x"] = _embeddings
    candidates["mention_id"] = [j] * len(_embeddings)
    candidates["split_random"] = np.random.choice(
        ["train", "val", "test"], p=[0.6, 0.2, 0.2]
    )
    return candidates


class EnhancedMLP(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        ext_channels,
        hidden_channels_1,
        hidden_channels_2,
    ):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(input_channels, hidden_channels_1)
        self.lin2 = Linear(hidden_channels_1 + ext_channels, hidden_channels_2)
        self.lin3 = Linear(hidden_channels_2 + ext_channels, 2)

    def forward(self, x, ext_features):
        x = self.lin1(x)
        x = x.relu()
        x = torch.cat([x, ext_features], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = torch.cat([x, ext_features], dim=1)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return x


class WikiCandidates(Dataset):
    def __init__(self, df, split_col):

        self.train_df = df[df[split_col] == "train"].reset_index()
        self.train_df.mention_id = self.train_df.mention_id.astype(np.int32)
        remap = {v: i for i, v in enumerate(self.train_df.mention_id.unique())}
        self.train_df["mention_id"].replace(remap, inplace=True)
        self.train_size = self.train_df["mention_id"].max()

        self.val_df = df[df[split_col] == "val"].reset_index()
        self.val_df.mention_id = self.val_df.mention_id.astype(np.int32)
        remap = {v: i for i, v in enumerate(self.val_df.mention_id.unique())}
        self.val_df["mention_id"].replace(remap, inplace=True)
        self.val_size = self.val_df["mention_id"].max()

        self.test_df = df[df[split_col] == "test"].reset_index()
        self.test_df.mention_id = self.test_df.mention_id.astype(np.int32)
        remap = {v: i for i, v in enumerate(self.test_df.mention_id.unique())}
        self.test_df["mention_id"].replace(remap, inplace=True)
        self.test_size = self.test_df["mention_id"].max()

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    def __getitem__(self, index):
        indices = np.where(self.target_df.mention_id == index)[0]

        return (
            torch.tensor(np.stack(self.target_df.iloc[indices].x.values, axis=0)),
            torch.tensor(
                np.stack(self.target_df.iloc[indices].ext_vector.values, axis=0),
                dtype=torch.float32,
            ),
            torch.tensor(self.target_df.iloc[indices].y.values),
        )

    def set_split(self, split="train"):
        self.target_df, self.target_size = self._lookup_dict[split]

    def __len__(self):
        return self.target_size


def train_perceptron(data_cands_comb):
    dataset = WikiCandidates(data_cands_comb, "split_random")

    model_folder = Path(
        f"/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/models/lwm_perceptron/best_model_split_random/"
    )
    model_folder.mkdir(exist_ok=True)

    input_channels = len(data_cands_comb.iloc[0].x)
    ext_channels = len(data_cands_comb.iloc[0].ext_vector)
    print(input_channels, ext_channels)

    model = EnhancedMLP(
        input_channels, ext_channels, hidden_channels_1=64, hidden_channels_2=24
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 5.0]), reduction="mean"
    )  # Define loss criterion.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0005, weight_decay=5e-4
    )  # Define optimizer.

    f1score = 0.0
    try:
        for epoch in range(1, 50):
            loss_epoch = 0.0
            dataset.set_split("train")
            for i in range(len(dataset)):
                model.train()
                features, ext_features, labels = dataset[i]
                optimizer.zero_grad()  # Clear gradients
                out = model(features, ext_features)  # Perform a single forward pass.
                loss = criterion(
                    out, labels
                )  # Compute the loss solely based on the training nodes.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                loss_epoch += loss

            pred, y_true = [], []
            val_loss_epoch = 0.0
            dataset.set_split("val")
            for i in range(len(dataset)):
                model.eval()
                features, ext_features, labels = dataset[i]
                out = model(features, ext_features)
                val_loss = criterion(out, labels)
                val_loss_epoch += val_loss
                pred.extend(out.argmax(dim=1))
                y_true.extend(labels)

            _f1score = f1_score(pred, y_true)
            if _f1score > f1score:
                f1score = _f1score
                print(f"Saving best model with F1: {f1score:.4f}")
                torch.save(model, model_folder / "mlp_classifier.pt")
            acc = accuracy_score(pred, y_true)
            print(
                f"Epoch: {epoch:03d}, Loss: {loss_epoch:.4f}, Validation F1: {f1score:.4f}, Validation Acc: {acc:.4f}"
            )

    except KeyboardInterrupt:
        print(f"Stopped training at epoch {epoch}")

    print(dataset)
