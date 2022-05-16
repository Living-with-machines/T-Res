import json
import os
import sys
from pathlib import Path

from pandarallel import pandarallel

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath(os.path.pardir))
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from transformers import pipeline
from utils import process_data, ner, ranking, linking, training
from utils.resolution_pipeline import ELPipeline

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear
import torch.nn.functional as F


# Datasets:
datasets = ["lwm", "hipe"]

# Named entity recognition approach, options are:
# * rel
# * lwm
ner_model_id = "lwm"

# Candidate selection approach, options are:
# * perfectmatch
# * partialmatch
# * levenshtein
# * deezymatch
cand_select_method = "deezymatch"

# Toponym resolution approach, options are:
# * mostpopular
# * mostpopularnormalised
# * lwmperceptron
top_res_method = "lwmperceptron"

# Perform training if needed:
do_training = True

# Entities considered for linking, options are:
# * all
# * loc
accepted_labels_str = "all"

# Instantiate dictionary of REL outputs
rel_preds = {}

# Initiate the recogniser object:
myner = ner.Recogniser(
    method=ner_model_id,  # NER method (lwm or rel)
    model_name="blb_lwm-ner",  # NER model name
    pipe=None,  # We'll store the NER pipeline here
    model=None,  # We'll store the NER model here
    base_model="/resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
    train_dataset="outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
    test_dataset="outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
    output_model_path="outputs/models/",  # Path where the NER model is or will be stored
    training_args={
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
    },
    overwrite_training=False,  # Set to True if you want to overwrite model if existing
    do_test=False,  # Set to True if you want to train on test mode
    accepted_labels=accepted_labels_str,
)

# Initiate the ranker object:
myranker = ranking.Ranker(
    method=cand_select_method,
    resources_path="/resources/wikidata/",
    mentions_to_wikidata=dict(),
    deezy_parameters={
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/deezymatch/",
        "dm_cands": "wkdtalts",
        "dm_model": "ocr_avgpool",
        "dm_output": "deezymatch_on_the_fly",
        # Ranking measures:
        "ranking_metric": "faiss",
        "selection_threshold": 20,
        "num_candidates": 1,
        "search_size": 1,
        "use_predict": False,
        "verbose": False,
    },
)

# Initiate the linker object:
mylinker = linking.Linker(
    method=top_res_method,
    do_training=do_training,
    training_csv="/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/data/lwm/linking_df_train.tsv",
    resources_path="/resources/wikidata/",
    linking_resources=dict(),
    myranker=myranker,
    bert_base_model="/resources/models/bert/bert_1760_1900/",  # Base model for vector extraction
    bert_tokenizer=None,
    bert_pipeline=None,
    perceptron=dict(),
)


# --------------------------------------------
# End of user input!
# --------------------------------------------

# Check for method inconsistencies:
# # TO DO: add all possible inconsistencies
if myner.method == "rel" and (myranker.method != "rel" or mylinker.method != "rel"):
    print(
        "\n*** Error: NER is '{0}', ranking method is '{1}' and linking method is '{2}'.\n".format(
            myner.method, myranker.method, mylinker.method
        )
    )
    sys.exit(0)

# Train the NER model if needed:
print("*** Training the NER model...")
myner.training()

# Load the ranker and linker resources:
print("*** Loading the resources...")
myner.model, myner.pipe = myner.create_pipeline()
myranker.mentions_to_wikidata = myranker.load_resources()
mylinker.linking_resources = mylinker.load_resources()
print("*** Resources loaded!\n")

# Parallelize if ranking method is one of the following:
if myranker.method in ["partialmatch", "levenshtein"]:
    pandarallel.initialize(nb_workers=10)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Define the model.
# I can't move this to the other scripts, I get an error...
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


# Some methods are unsupervised, set the do_training flag to False:
if mylinker.method in ["rel", "mostpopular", "mostpopularnormalised"]:
    mylinker.do_training = False
elif mylinker.method in ["lwmperceptron"]:
    # Load trained perceptron:
    mylinker.bert_tokenizer, mylinker.bertpipeline = mylinker.create_pipeline()
    mylinker.perceptron[
        "perceptron_path"
    ] = "/resources/develop/mcollardanuy/toponym-resolution/experiments/outputs/models/lwm_perceptron/best_model_split_random/mlp_classifier.pt"
    # Create a dataset for entity linking, with candidates retrieved
    # using the chosen method:
    training_df = training.create_trainset(
        mylinker.training_csv, mylinker.myranker, mylinker
    )
    # Choose the features to use:
    mylinker.perceptron["features"] = ["conf_score", "wkdt_relv", "log_dist"]

    ### In the future uncomment, to skip training if done:
    """
    if Path(mylinker.perceptron["perceptron_path"]).exists():
        perceptron_model = torch.load(mylinker.perceptron["perceptron_path"])
        m = torch.nn.Softmax(dim=1)
        mylinker.perceptron["perceptron_model"] = perceptron_model
        mylinker.perceptron["m"] = m
    else:
        print("Perceptron does not exist, let's train it!")
    """
    if not mylinker.perceptron.get("perceptron_model"):
        mylinker.train(training_df)
        perceptron_model = torch.load(mylinker.perceptron["perceptron_path"])
        m = torch.nn.Softmax(dim=1)
        mylinker.perceptron["perceptron_model"] = perceptron_model
        mylinker.perceptron["m"] = m

# Instantiate gold tokenization dictionary:
gold_tokenisation = {}

# ---------------------------------------------------
# Perform end-to-end toponym resolution
# ---------------------------------------------------

for dataset in datasets:

    # Path to dev dataframe:
    dev = pd.read_csv(
        "outputs/data/" + dataset + "/linking_df_dev.tsv",
        sep="\t",
    )

    # Path where to store gold tokenization:
    gold_path = (
        "outputs/results/"
        + dataset
        + "/lwm_gold_tokenisation_"
        + accepted_labels_str
        + ".json"
    )

    # Path where to store REL API output:
    rel_end_to_end = "outputs/results/" + dataset + "/rel_end_to_end.json"

    if myner.method == "rel":
        if Path(rel_end_to_end).is_file():
            with open(rel_end_to_end) as f:
                rel_preds = json.load(f)
        else:
            rel_preds = {}

        gold_standard = process_data.read_gold_standard(gold_path)
        # currently it's all based on REL
        # but we could for instance use our pipeline and rely on REL only for disambiguation
        myranker.method = "rel"
        mylinker.method = "rel"

    dAnnotated, dSentences, dMetadata = ner.format_for_ner(dev)

    true_mentions_sents = dict()
    dPreds = dict()
    dTrues = dict()
    dSkys = dict()

    # Print the contents fo the ranker and linker objects:
    print(myner)
    print(myranker)
    print(mylinker)

    # Instantiate the entity linking pipeline:
    end_to_end = ELPipeline(
        myner=myner,
        myranker=myranker,
        mylinker=mylinker,
        dataset=dataset,
    )

    for sent_id in tqdm.tqdm(dSentences.keys()):

        if myner.method == "rel":
            if Path(rel_end_to_end).is_file():
                print(sent_id)
                preds = rel_preds[sent_id]
            else:
                pred_ents = linking.rel_end_to_end(dSentences[sent_id])
                rel_preds[sent_id] = pred_ents
            output = end_to_end.run(
                dSentences[sent_id],
                dataset=dataset,
                annotations=rel_preds[sent_id],
                gold_positions=gold_standard[sent_id],
            )

        else:
            output = end_to_end.run(
                dSentences[sent_id],
                dataset=dataset,
                annotations=dAnnotated[sent_id],
                metadata=dMetadata[sent_id],
            )

        dPreds[sent_id] = output["sentence_preds"]
        dTrues[sent_id] = output["sentence_trues"]
        gold_tokenisation[sent_id] = output["gold_positions"]
        dSkys[sent_id] = output["skyline"]

    if myner.method == "lwm":
        process_data.store_results_hipe(dataset, "true_" + accepted_labels_str, dTrues)
        process_data.store_results_hipe(
            dataset,
            "skyline:"
            + myner.method
            + "+"
            + myranker.method
            + "+"
            + accepted_labels_str,
            dSkys,
        )
        with open(gold_path, "w") as fp:
            json.dump(gold_tokenisation, fp)

    process_data.store_results_hipe(
        dataset,
        myner.method
        + "+"
        + myranker.method
        + "+"
        + mylinker.method
        + "+"
        + accepted_labels_str,
        dPreds,
    )

    with open(rel_end_to_end, "w") as fp:
        json.dump(rel_preds, fp)
