import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
from utils import process_data, candidate_selection, linking

from sklearn.neural_network import MLPClassifier


def create_trainset(train_path, cand_select_method, myranker, already_collected_cands):
    """
    Create entity linking training data (i.e. mentions identified and candidates provided),
    necessary for training our resolution methods:
    """
    training_cands_path = train_path.split(".tsv")[0] + "_" + cand_select_method + ".tsv"
    if not Path(training_cands_path).exists():
        training_set = pd.read_csv(train_path, sep="\t")
        training_df = process_data.crate_training_for_el(training_set)
        candidates_qid = []
        for i, row in training_df.iterrows():
            cands, already_collected_cands = candidate_selection.select(
                [row["mention"]], cand_select_method, myranker, already_collected_cands
            )
            if row["mention"] in cands:
                candidates_qid.append(
                    candidate_selection.get_candidate_wikidata_ids(cands[row["mention"]])
                )
            else:
                candidates_qid.append(dict())

        training_df["wkdt_cands"] = candidates_qid
        training_df.to_csv(training_cands_path, sep="\t", index=False)
        return training_df
    else:
        return pd.read_csv(training_cands_path, sep="\t")


def add_linking_columns(train_path, training_df):
    """
    Add the following columns:
    * class_type: closest geotype considering class_emb (town, city, country, continent...)
    * geoscope: whether location is in county, country, or abroad
    * mention_emb: BERT embedding in context of mention
    """
    training_linkcols_path = train_path.split(".tsv")[0] + "_linkcols.tsv"
    if not Path(training_linkcols_path).exists():
        training_df.loc[:,'class_emb'] = training_df.loc[:,:].apply(lambda x: linking.find_avg_node_embedding(x["wkdt_qid"]), axis=1)
        training_df.loc[:,'class_type'] = training_df.loc[:,:].apply(lambda x: linking.assign_closest_class(x["class_emb"]), axis=1)
        training_df.loc[:, 'geoscope'] = training_df.loc[:, :].apply(lambda x: linking.get_geoscope_from_publication(x["place"], x["wkdt_qid"]), axis=1)
        training_df.loc[:, 'mention_emb'] = training_df.loc[:, :].apply(lambda x: linking.get_mention_vector(x, agg=np.mean), axis=1)
        training_df = training_df[training_df['class_type'].notna()]
        training_df = training_df[training_df['mention_emb'].notna()]
        training_df = training_df[training_df['geoscope'].notna()]
        training_df.to_csv(training_linkcols_path, sep="\t", index=False)
        return training_df
    else:
        training_df.loc[:, 'mention_emb'] = training_df.loc[:, :].apply(lambda x: literal_eval(x["mention_emb"]), axis=1)
        return training_df


def mention2type_classifier(training_df):
    # X is the avg embedding of the wikipedia entry class, y is the label:
    X, y = list(training_df["mention_emb"].values), list(training_df["class_type"].values)
    # Start a MLP classifier:
    model = MLPClassifier(validation_fraction=.2, early_stopping=True, random_state=42, verbose=True)
    model.fit(X,y)
    return model


def mention2scope_classifier(training_df):
    # X is the mention embedding, y is the geographic distange:
    X, y = list(training_df["mention_emb"].values), list(training_df["geoscope"].values)
    # Start a MLP classifier:
    model = MLPClassifier(validation_fraction=.2, early_stopping=True, random_state=42, verbose=True)
    model.fit(X,y)
    return model