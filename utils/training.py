import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from utils.REL.entity_disambiguation import EntityDisambiguation
from utils.REL.generate_train_test import GenTrainingTest
from utils.REL.training_datasets import TrainingEvaluationDatasets
from utils.REL.wikipedia import Wikipedia

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))


class Trainer(object):
    def __init__(self, data, model, criterion, optimizer, path, whichsplit):
        self.data = data
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.path = path
        self.whichsplit = whichsplit

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients
        out = self.model(
            self.data.x.float(),
            self.data.x_ext.float(),
            self.data.edge_index,
            self.data.weight.float(),
        )  # Perform a single forward pass.
        loss = self.criterion(
            out[self.data.train_mask].float(), self.data.y[self.data.train_mask]
        )  # Compute the loss solely based on the training nodes.

        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss.item()

    def set_split(self):
        val_mask = np.array([True if s == "dev" else False for l in self.data.split for s in l])
        train_mask = np.array(
            [True if s == "train" else False for l in self.data.split for s in l]
        )
        test_mask = np.array([True if s == "test" else False for l in self.data.split for s in l])
        self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        self.data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    def print_split(self):
        print(
            "\nNumber of training nodes",
            self.data.train_mask.sum(),
            "\nNumber of dev nodes",
            self.data.val_mask.sum(),
            "\nNumber of test nodes",
            self.data.test_mask.sum(),
        )

    def print_network_statistics(self):
        print(f"Number of nodes: {self.data.num_nodes}")
        print(f"Number of edges: {self.data.num_edges}")
        print(f"Average node degree: {self.data.num_edges / self.data.num_nodes:.2f}")
        print(f"Number of training nodes: {self.data.train_mask.sum()}")
        print(
            f"Training node label rate: {int(self.data.train_mask.sum()) / self.data.num_nodes:.2f}"
        )
        print(f"Has isolated nodes: {self.data.has_isolated_nodes()}")
        print(f"Has self-loops: {self.data.has_self_loops()}")
        print(f"Is undirected: {self.data.is_undirected()}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.optimizer.zero_grad()  # Clear gradients
        out = self.model(
            self.data.x.float(),
            self.data.x_ext.float(),
            self.data.edge_index,
            self.data.weight.float(),
        )  # Perform a single forward pass.

        # loss = self.criterion(out[self.data.val_mask].float(), self.data.y[self.data.val_mask])  # Compute the loss solely based on the validation nodes.
        pred = out.argmax(dim=1)
        f1score = f1_score(pred[self.data.val_mask], self.data.y[self.data.val_mask])
        return f1score

    def training_routine(self, epochs=500):

        out_folder = Path(self.path)
        out_folder.mkdir(exist_ok=True)
        model_folder = out_folder / Path(f"best_model_{self.whichsplit}")
        model_folder.mkdir(exist_ok=True)
        print("Saving model in: ", model_folder)

        highest_f1 = 0.0
        try:
            for epoch in range(1, epochs):
                new_f1 = self.evaluate()
                print(
                    f"Epoch {epoch} Training loss : {self.train():.4f} Validation F1 :,{new_f1:.4f}"
                )
                if new_f1 > highest_f1:
                    print(f"Saving new best model with f1={new_f1:.4f}")

                    torch.save(self.model.state_dict(), model_folder / "best-model.pt")
                    highest_f1 = new_f1
        except KeyboardInterrupt:
            print(f"Quiting training loop. Best model {highest_f1:.4f}")


# ==============================================================
# --------------------------------------------------------------
# Train REL entity disambiguation
def train_rel_ed(mylinker, all_df, train_df, whichsplit):
    base_path = mylinker.rel_params["base_path"]
    wiki_version = mylinker.rel_params["wiki_version"]

    Path("{}/{}/generated/test_train_data/".format(base_path, wiki_version)).mkdir(
        parents=True, exist_ok=True
    )
    wikipedia = Wikipedia(base_path, wiki_version)
    data_handler = GenTrainingTest(base_path, wiki_version, wikipedia, mylinker=mylinker)
    for ds in ["train", "dev"]:
        if "edaidalwm" in mylinker.method:
            data_handler.process_aidalwm(ds, all_df, train_df, whichsplit)
        elif "edaida" in mylinker.method:
            data_handler.process_aida(ds)
        elif "edlwm" in mylinker.method:
            data_handler.process_lwm(ds, all_df, train_df, whichsplit)
        else:
            data_handler.process_lwm(ds, all_df, train_df, whichsplit)

    datasets = TrainingEvaluationDatasets(base_path, wiki_version).load()

    config = {
        "mode": "train",
        "model_path": "{}{}generated/model".format(base_path, wiki_version),
    }
    model = EntityDisambiguation(base_path, wiki_version, config)

    # Train the model using lwm_train:
    model.train(
        datasets["lwm_train"],
        {k: v for k, v in datasets.items() if k != "lwm_train"},
    )
    # Train and predict using LR (to obtain confidence scores)
    model_path_lr = "{}/{}/generated/".format(base_path, wiki_version)
    model.train_LR(datasets, model_path_lr)
    return model
