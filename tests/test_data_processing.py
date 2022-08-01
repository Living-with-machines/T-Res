#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ast import literal_eval
import os
import json
import pandas as pd
from pathlib import Path
from collections import Counter

large_resources = "/resources/"  # path to large resources
small_resources = "resources/"  # path to small resources
processed_path_lwm = "experiments/outputs/data/lwm/"  # path to processed LwM data
processed_path_hipe = "experiments/outputs/data/hipe/"  # path to processed LwM data


def test_publication_metadata_exists():
    """
    Metadata file exists.
    """
    path_metadata = os.path.join(f"{small_resources}", "publication_metadata.json")
    assert Path(path_metadata)
    with open(path_metadata) as jsonfile:
        df_metadata = json.load(jsonfile)
    dict_titles = {k: df_metadata[k]["publication_title"] for k in df_metadata}
    dict_place = {k: df_metadata[k]["publication_place"] for k in df_metadata}
    dict_placewqid = {k: df_metadata[k]["wikidata_qid"] for k in df_metadata}
    assert len(dict_titles) == len(dict_place) == len(dict_placewqid) == 23


def test_lwm_data_exists():
    """
    Manually annotated LwM data exists.
    """
    path_news_data = os.path.join(
        f"{small_resources}", "news_datasets", "topRes19th_v2"
    )
    path_train_metadata = os.path.join(f"{path_news_data}", "train", "metadata.tsv")
    path_test_metadata = os.path.join(f"{path_news_data}", "test", "metadata.tsv")
    assert Path(path_news_data).is_dir()
    assert Path(path_train_metadata).exists()
    assert Path(path_test_metadata).exists()


def test_original_lwm_data():
    """
    Test manually annotated LwM metadata dataframes.
    """
    path_news_data = os.path.join(
        f"{small_resources}", "news_datasets", "topRes19th_v2"
    )
    path_train_metadata = os.path.join(f"{path_news_data}", "train", "metadata.tsv")
    path_test_metadata = os.path.join(f"{path_news_data}", "test", "metadata.tsv")
    train_metadata = pd.read_csv(path_train_metadata, sep="\t")
    test_metadata = pd.read_csv(path_test_metadata, sep="\t")
    # Assert the size of the metadata files:
    assert train_metadata.shape[0] == 343
    assert test_metadata.shape[0] == 112
    assert train_metadata.shape[1] == 10
    assert test_metadata.shape[1] == 10
    # Items in metadata match number of files in directory, for test:
    assert (
        len(
            list(
                Path(
                    os.path.join(f"{path_news_data}", "test", "annotated_tsv")
                ).iterdir()
            )
        )
        == test_metadata.shape[0]
    )
    # Items in metadata match number of files in directory, for train:
    assert (
        len(
            list(
                Path(
                    os.path.join(f"{path_news_data}", "train", "annotated_tsv")
                ).iterdir()
            )
        )
        == train_metadata.shape[0]
    )


def test_lwm_ner_conversion():
    """
    Test process_lwm_for_ner is not missing articles.
    """
    df_ner_train = pd.read_json(
        os.path.join(f"{processed_path_lwm}", "ner_df_train.json"),
        orient="records",
        lines=True,
        dtype={"id": str},
    )
    df_ner_dev = pd.read_json(
        os.path.join(f"{processed_path_lwm}", "ner_df_dev.json"),
        orient="records",
        lines=True,
        dtype={"id": str},
    )
    # Assert size of the train and dev sets:
    assert df_ner_train.shape == (5216, 3)
    assert df_ner_dev.shape == (1304, 3)
    # Assert number of sentences in train and dev (length of list and set should be the same):
    assert (
        len(list(df_ner_train["id"]) + list(df_ner_dev["id"]))
        == len(set(list(df_ner_train["id"]) + list(df_ner_dev["id"])))
        == df_ner_train.shape[0] + df_ner_dev.shape[0]
    )
    # Assert ID is read as string:
    assert type(df_ner_train["id"].iloc[0]) == str
    # Assert number of unique articles:
    train_articles = [x.split("_")[0] for x in list(df_ner_train["id"])]
    dev_articles = [x.split("_")[0] for x in list(df_ner_dev["id"])]
    assert len(set(train_articles + dev_articles)) == 343


def test_lwm_linking_conversion():
    """
    Test process_lwm_for_linking is not missing articles.
    """
    df_linking = pd.read_csv(
        os.path.join(f"{processed_path_lwm}", "linking_df_split.tsv"),
        sep="\t",
    )
    # Assert size of the dataset (i.e. number of articles):
    assert df_linking.shape[0] == 455
    # Assert if place has been filled correctly:
    for x in df_linking.place:
        assert type(x) == str
        assert x != ""
    # Assert if place QID has been filled correctly:
    for x in df_linking.place_wqid:
        assert type(x) == str
        assert x != ""
    for x in df_linking.annotations:
        x = literal_eval(x)
        for ann in x:
            assert ann["wkdt_qid"] == "NIL" or ann["wkdt_qid"].startswith("Q")


def test_hipe_linking_conversion():
    """
    Test process_hipe_for_linking is not missing articles.
    """
    df_linking = pd.read_csv(
        os.path.join(f"{processed_path_hipe}", "linking_df_split.tsv"),
        sep="\t",
    )
    # Assert size of the dataset (i.e. number of articles):
    assert df_linking.shape[0] == 126
    assert df_linking[df_linking["originalsplit"] == "dev"].shape[0] == 80
    assert df_linking[df_linking["originalsplit"] == "test"].shape[0] == 46
    assert df_linking[df_linking["traindevtest"] == "dev"].shape[0] == 40
    assert df_linking[df_linking["traindevtest"] == "test"].shape[0] == 46
    # Assert if place has been filled correctly:
    for x in df_linking.place:
        assert type(x) == str
        assert x != ""
    # Assert if place QID has been filled correctly:
    for x in df_linking.place_wqid:
        assert type(x) == str
        assert x != ""
    # Do HIPE stats match https://github.com/hipe-eval/HIPE-2022-data/blob/main/notebooks/hipe2022-datasets-stats.ipynb
    number_locs = 0
    for x in df_linking.annotations:
        x = literal_eval(x)
        for ann in x:
            assert ann["wkdt_qid"] == "NIL" or ann["wkdt_qid"].startswith("Q")
        for ann in x:
            if ann["entity_type"] == "LOC":
                number_locs += 1
    number_locs_stats = 573  # locs in test+dev, meto locs in test+dev
    assert number_locs == number_locs_stats
