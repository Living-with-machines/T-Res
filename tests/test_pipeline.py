import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))
from geoparser import linking, pipeline, ranking, recogniser


def test_deezy_mostpopular():

    myner = recogniser.Recogniser(
        model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
        model=None,  # We'll store the NER model here
        pipe=None,  # We'll store the NER pipeline here
        base_model="resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
        train_dataset="experiments/outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
        output_model_path="experiments/outputs/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        training_tagset="fine",  # Options are: "coarse" or "fine"
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("experiments/outputs/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 25,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "w2v_ocr_path": str(Path("experiments/outputs/models/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "do_test": False,
        },
    )

    mylinker = linking.Linker(
        method="mostpopular",
        resources_path="resources/",
        linking_resources=dict(),
        base_model="to-be-removed",  # Base model for vector extraction
        rel_params={},
        overwrite_training=False,
    )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefrield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
        place_wqid="Q18125",
    )
    assert resolved[0]["mention"] == "Shefrield"
    assert resolved[0]["candidates"]["Q665346"] == 0.007
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.893
    assert resolved[0]["ner_score"] == 0.994


def test_deezy_rel_withoutpubl():

    myner = recogniser.Recogniser(
        model_name="blb_lwm-ner",  # NER model name prefix (will have suffixes appended)
        model=None,  # We'll store the NER model here
        pipe=None,  # We'll store the NER pipeline here
        base_model="resources/models/bert/bert_1760_1900/",  # Base model to fine-tune
        train_dataset="experiments/outputs/data/lwm/ner_df_train.json",  # Training set (part of overall training set)
        test_dataset="experiments/outputs/data/lwm/ner_df_dev.json",  # Test set (part of overall training set)
        output_model_path="experiments/outputs/models/",  # Path where the NER model is or will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        training_tagset="fine",  # Options are: "coarse" or "fine"
    )

    myranker = ranking.Ranker(
        method="deezymatch",
        resources_path="resources/wikidata/",
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        wiki_filtering={
            "top_mentions": 3,  # Filter mentions to top N mentions
            "minimum_relv": 0.03,  # Filter mentions with more than X relv
        },
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": str(Path("experiments/outputs/deezymatch/").resolve()),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 25,
            "num_candidates": 3,
            "search_size": 3,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "w2v_ocr_path": str(Path("experiments/outputs/models/").resolve()),
            "w2v_ocr_model": "w2v_*_news",
            "do_test": False,
        },
    )

    mylinker = linking.Linker(
        method="reldisamb",
        resources_path="resources/",
        linking_resources=dict(),
        base_model="to-be-removed",  # Base model for vector extraction
        rel_params={
            "base_path": "resources/rel_db/",
            "wiki_version": "wiki_2019/",
            "training_data": "lwm",  # lwm, aida
            "ranking": "relv",  # relv, publ
            "micro_locs": "nil",  # "dist", "nil", ""
        },
        overwrite_training=False,
    )

    geoparser = pipeline.Pipeline(myner=myner, myranker=myranker, mylinker=mylinker)

    resolved = geoparser.run_text(
        "A remarkable case of rattening has just occurred in the building trade at Shefrield, but also in Lancaster. Not in Nottingham though. Not in Ashton either, nor in Salop!",
    )
    assert resolved[0]["mention"] == "Shefrield"
    assert resolved[0]["ed_score"] == resolved[0]["candidates"][resolved[0]["prediction"]]
    assert resolved[0]["candidates"]["Q665346"] == 0.916
    assert resolved[0]["prediction"] == "Q42448"
    assert resolved[0]["ed_score"] == 0.982
    assert resolved[0]["ner_score"] == 0.994
