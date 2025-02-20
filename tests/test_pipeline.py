import os
import sqlite3
from pathlib import Path

import pytest

from t_res.geoparser import ner, ranking, linking, pipeline
from t_res.utils.dataclasses import *

current_dir = Path(__file__).parent.resolve()

# TODO: add a test with & without microtoponyms. 
# Check that the predictions exclude them if configured.

def test_pipeline_constructor():
    resources_path=os.path.join(current_dir, "sample_files/resources")
    geoparser = pipeline.Pipeline(resources_path=resources_path)

    # Check default pipeline components.
    assert isinstance(geoparser.recogniser, ner.PretrainedRecogniser)
    assert geoparser.recogniser.model_name == "Livingwithmachines/toponym-19thC-en"

    assert isinstance(geoparser.ranker, ranking.PerfectMatchRanker)
    assert geoparser.ranker.resources_path == resources_path

    assert isinstance(geoparser.linker, linking.MostPopularLinker)
    assert geoparser.linker.resources_path == resources_path

def test_pipeline_basic():
    resources_path=os.path.join(current_dir, "sample_files/resources")
    geoparser = pipeline.Pipeline(resources_path=resources_path)

    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    predictions = geoparser.run(sentence)

    assert len(predictions.sentence_candidates) == 1
    assert len(predictions.sentence_candidates[0].candidates) == 1
    assert len(predictions.candidates()) == 1
    assert predictions.candidates()[0].best_string_match().variation == "Sheffield"
    assert predictions.candidates()[0].best_string_match().string_similarity == 1.0
    assert predictions.candidates()[0].mention.mention == "Sheffield"
    assert predictions.candidates()[0].mention.ner_score == 1.0
    assert predictions.candidates()[0].best_wqid() == "Q42448"
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.807, abs=1e-3)

def test_pipeline_modular():
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "sample_files/resources"),
    )
    linker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir, "sample_files/resources"),
    )
    geoparser = pipeline.Pipeline(ranker=ranker, linker=linker)
    
    sentence = "A remarkable case of rattening has just occurred in the building trade at Sheffield."
    predictions = geoparser.run(sentence)

    assert len(predictions.sentence_candidates) == 1
    assert len(predictions.sentence_candidates[0].candidates) == 1
    assert len(predictions.candidates()) == 1
    assert predictions.candidates()[0].mention.mention == "Sheffield"
    assert predictions.candidates()[0].mention.ner_score == 1.0
    assert predictions.candidates()[0].best_wqid() == "Q42448"
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.807, abs=1e-3)

@pytest.mark.resources(reason="Needs deezy model")
def test_deezy_mostpopular(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": os.path.join(tmp_path,"resources/models/"),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    linker = linking.MostPopularLinker(
        resources_path=os.path.join(current_dir, "../resources/"),
    )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)
    assert len(geoparser.ranker.mentions_to_wikidata.keys())>0

    text = "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though."
    predictions = geoparser.run(text)

    assert len(predictions.sentence_candidates) == 2
    assert len(predictions.sentence_candidates[0].candidates) == 2
    assert len(predictions.sentence_candidates[1].candidates) == 1
    assert len(predictions.candidates()) == 3
    assert predictions.candidates()[0].mention.mention == "Shefiield"
    assert predictions.candidates()[0].best_match().string_match.variation == "Sheffield"
    assert predictions.candidates()[0].best_string_match().string_similarity == 0.999494
    assert predictions.candidates()[0].best_wqid() == "Q42448"
    assert predictions.candidates()[0].best_match().cross_cand_scores()["Q42448"] == 0.903
    assert predictions.candidates()[0].best_match().best_disambiguation_score() == pytest.approx(0.903, abs=1e-3)
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.903, abs=1e-3)
    assert predictions.candidates()[0].mention.ner_score == 1.0

    # The predictions are Sheffield (Q42448), Leeds (Q39121) and London (Q84).
    assert predictions.best_wqids() == ['Q42448', 'Q39121', 'Q84']
    assert predictions.best_disambiguation_scores() == [
        pytest.approx(0.903, abs=1e-3), 
        pytest.approx(0.913, abs=1e-3), 
        pytest.approx(0.972, abs=1e-3)]

    assert geoparser.run("").is_empty()

    assert geoparser.run(" ").is_empty()

    # asserting behaviour with • character
    text = " • - S G pOllO-P• FERRIS - • - , i "
    assert geoparser.run(text).is_empty()

@pytest.mark.resources(reason="Needs large resources")
def test_deezy_rel_wpubl_wmtops(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "predict_place_of_publication": False,
                "combined_score": False,
                "without_microtoponyms": True,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)

    text = "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though."
    predictions = geoparser.run(text, place_of_pub_wqid="Q42448", place_of_pub="Sheffield")

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.sentence_candidates) == 2
    assert len(predictions.sentence_candidates[0].candidates) == 2
    assert len(predictions.sentence_candidates[1].candidates) == 1
    assert len(predictions.candidates()) == 3
    assert predictions.candidates()[0].mention.mention == "Shefiield"
    assert predictions.candidates()[0].best_match().string_match.variation == "Sheffield"
    assert predictions.candidates()[0].best_match().string_match.string_similarity == 0.999494
    assert predictions.candidates()[0].best_wqid() == "Q42448"
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.766, abs=1e-3)

    # Check the interim disambiguation score (i.e. before applying the REL model).
    # The only difference is in the best_disambiguation_score result.
    # Note: previously the term "prior_cand_score" was used to refer to the interim score.
    assert len(predictions.interim_candidates()) == 3
    assert predictions.interim_candidates()[0].mention.mention == "Shefiield"
    assert predictions.interim_candidates()[0].best_match().string_match.variation == "Sheffield"
    assert predictions.interim_candidates()[0].best_match().string_match.string_similarity == 0.999494
    assert predictions.interim_candidates()[0].best_wqid() == "Q42448"
    assert predictions.interim_candidates()[0].best_disambiguation_score() == pytest.approx(0.891, abs=1e-3)

    # The predictions are Sheffield (Q42448), Leeds (Q39121) and London (Q84).
    assert predictions.best_wqids() == ['Q42448', 'Q39121', 'Q84']
    assert predictions.best_disambiguation_scores() == [
        pytest.approx(0.766, abs=1e-3), 
        pytest.approx(0.755, abs=1e-3), 
        pytest.approx(0.734, abs=1e-3)]

    # Compare with the interim predictions (produced before running the REL model).
    interim_best_wqids = [c.best_wqid() 
                          for scs in predictions.sentence_candidates 
                          for c in scs.candidates]
    interim_best_disambiguation_scores = [c.best_disambiguation_score() 
                                          for scs in predictions.sentence_candidates 
                                          for c in scs.candidates]

    assert interim_best_wqids == ['Q42448', 'Q39121', 'Q84']
    # Note that the interim disambiguation scores are higher, and are still available 
    # after applying the REL model, but the REL scores take precedence (see above).
    assert interim_best_disambiguation_scores == [
        pytest.approx(0.891, abs=1e-3), 
        pytest.approx(0.897, abs=1e-3), 
        pytest.approx(0.895, abs=1e-3)]
    
    assert predictions.candidates()[0].best_match().cross_cand_scores()["Q42448"] == pytest.approx(0.766, abs=1e-3)
    assert predictions.candidates()[0].mention.ner_score == 1.0

@pytest.mark.resources(reason="Needs large resources")
def test_deezy_rel_wpubl(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 10,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir,"../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "predict_place_of_publication": False,
                "combined_score": False,
                "without_microtoponyms": False,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)
    text = "The charming seaside town of Swanage is noted for its Town Hall whose distinctive façade was designed by Edward Jerman, a pupil of Sir Christopher Wren. Also the Grosvenor Hotel, with its clock tower originally erected at the south end of London Bridge as a memorial to the Duke of Wellington."

    # Test with microtoponyms.
    predictions = geoparser.run(text, place_of_pub_wqid="Q203349", place_of_pub="Poole, Dorset")
    assert isinstance(predictions, RelPredictions)

    # When the "without_microtoponyms" parameter set to False, there are four candidates:
    assert len(predictions.candidates()) == 4
    assert predictions.candidates()[0].mention.mention == "Swanage"
    assert predictions.candidates()[1].mention.mention == "Town Hall"
    assert predictions.candidates()[2].mention.mention == "Grosvenor Hotel"
    assert predictions.candidates()[3].mention.mention == "London Bridge"

    # Test without microtoponyms.
    geoparser.linker.rel_params["without_microtoponyms"] = True
    predictions = geoparser.run(text, place_of_pub_wqid="Q203349", place_of_pub="Poole, Dorset")
    assert isinstance(predictions, RelPredictions)
    
    # When the "without_microtoponyms" parameter set to True, only one candidate remains:
    assert len(predictions.candidates()) == 1
    assert predictions.candidates()[0].mention.mention == "Swanage"

@pytest.mark.resources(reason="Needs large resources")
def test_perfect_rel_wpubl_wmtops():
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "without_microtoponyms": True,
                "do_test": True,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)

    predictions = geoparser.run(
        "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though.",
        place_of_pub_wqid="Q42448",
        place_of_pub="Sheffield",
    )

    assert isinstance(predictions, RelPredictions)

    assert len(predictions.candidates(ignore_empty_candidates=True)) == 2

    candidates = predictions.candidates(ignore_empty_candidates=False)
    assert len(candidates) == 3
    assert candidates[0].mention.mention == "Shefiield"
    assert candidates[0].mention.ner_score == 1.0
    assert candidates[0].best_match() is None
    assert candidates[0].best_wqid() is None
    assert candidates[0].best_disambiguation_score() is None
    assert predictions.rel_scores[0].mention == "Shefiield"
    assert predictions.rel_scores[0].confidence == 0.0

    assert candidates[1].mention.mention == "Leeds"
    assert candidates[1].mention.ner_score == 1.0
    assert candidates[1].best_match() is not None
    assert isinstance(candidates[1].best_match(), PredictedLinks)
    assert candidates[1].best_match().best_disambiguation_score() == pytest.approx(0.419, abs=1e-3)
    assert candidates[1].best_wqid() == "Q39121"
    assert candidates[1].best_disambiguation_score() == pytest.approx(0.419, abs=1e-3)
    assert predictions.rel_scores[1].mention == "Leeds"
    assert predictions.rel_scores[1].confidence == pytest.approx(0.168, abs=1e-3)
    assert predictions.rel_scores[1].scores["Q39121"] == pytest.approx(0.419, abs=1e-3)

    assert candidates[2].mention.mention == "London"
    assert candidates[2].mention.ner_score == 0.998
    assert candidates[2].best_match() is not None
    assert isinstance(candidates[1].best_match(), PredictedLinks)
    assert candidates[2].best_match().best_disambiguation_score() == pytest.approx(0.573, abs=1e-3)
    assert candidates[2].best_wqid() == "Q84"
    assert candidates[2].best_disambiguation_score() == pytest.approx(0.573, abs=1e-3)
    assert predictions.rel_scores[2].mention == "London"
    assert predictions.rel_scores[2].confidence == pytest.approx(0.178, abs=1e-3)
    assert predictions.rel_scores[2].scores["Q84"] == pytest.approx(0.573, abs=1e-3)

@pytest.mark.resources(reason="Needs large resources")
def test_perfect_rel_predict_place_of_pub():
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "predict_place_of_publication": False,
                "combined_score": False,
                "without_microtoponyms": True,
                "do_test": True,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)

    predictions = geoparser.run(
        "A remarkable case of rattening has just occurred in the building trade at Stockton, but also in Leeds. Not in London though.",
        place_of_pub_wqid="Q989418",
        place_of_pub="Stockton-on-Tees, Cleveland, England",
    )

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.candidates()) == 3

    # With "predict_place_of_publication" set to False, the wrong Stockton is predicted:
    assert predictions.candidates()[0].best_wqid() != "Q989418"

    geoparser.linker.rel_params["predict_place_of_publication"] = True

    predictions = geoparser.run(
        "A remarkable case of rattening has just occurred in the building trade at Stockton, but also in Leeds. Not in London though.",
        place_of_pub_wqid="Q989418",
        place_of_pub="Stockton-on-Tees, Cleveland, England",
    )

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.candidates()) == 3

    # With "predict_place_of_publication" set to True, the correct Stockton is predicted
    # because the place of publication is the favoured candidate:
    assert predictions.candidates()[0].best_wqid() == "Q989418"

@pytest.mark.resources(reason="Needs large resources")
def test_perfect_rel_combined_score():
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.PerfectMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir, "../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
                "training_split": "originalsplit",
                "db_embeddings": cursor,
                "with_publication": True,
                "predict_place_of_publication": False,
                "combined_score": False,
                "without_microtoponyms": True,
                "do_test": True,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
                "reference_separation": ((49.956739, -8.17751), (60.87, 1.762973)),
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)

    predictions = geoparser.run(
        "A remarkable case of rattening has just occurred in the building trade at Stockton, but also in Leeds.",
        place_of_pub_wqid="Q39121",
        place_of_pub="Leeds, West Yorkshire, England",
    )

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.candidates()) == 2

    # With "combined_score" set to False, the wrong Stockton is predicted:
    assert predictions.candidates()[0].best_wqid() != "Q989418"
    assert predictions.candidates()[0].best_wqid() == "Q49240"
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.225, abs=1e-3)

    geoparser.linker.rel_params["combined_score"] = True

    predictions = geoparser.run(
        "A remarkable case of rattening has just occurred in the building trade at Stockton, but also in Leeds.",
        place_of_pub_wqid="Q39121",
        place_of_pub="Leeds, West Yorkshire, England",
    )

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.candidates()) == 2

    # With "combined_score" set to True, the correct Stockton is predicted
    # because the disambiguation score for the previous best candidate
    # is curtailed by the combined score:
    assert predictions.candidates()[0].best_wqid() == "Q989418"
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.21, abs=1e-3)

@pytest.mark.resources(reason="Needs large resources")
def test_modular_deezy_rel(tmp_path):
    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir, "../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir,"../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "model_path": os.path.join(current_dir,"../resources/models/disambiguation/"),
                "data_path": os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/"),
                "training_split": "apply",
                "db_embeddings": cursor,
                "with_publication": True,
                "predict_place_of_publication": False,
                "combined_score": False,
                "without_microtoponyms": False,
                "do_test": False,
                "default_publname": "United Kingdom",
                "default_publwqid": "Q145",
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)

    text = "STOCKTON AND MIDDLESBROUGH WATER IVARD.  The monthly meeting of the Sr-id:toe and bladtiltwitrough Water Lkerd was held at the Corp.acit:o.i liniklinga, Middlesbrough, on Monday."
    place_of_pub_wqid = "Q989418"
    place_of_pub = "Stockton-on-Tees, Cleveland, England"

    sentence_mentions = geoparser.run_text_recognition(text)

    assert isinstance(sentence_mentions, list)
    # Two sentences:
    assert len(sentence_mentions) == 2
    # Two toponyms identified in the first sentence:
    assert len(sentence_mentions[0].mentions) == 2
    # Three toponyms identified in the second sentence:
    assert len(sentence_mentions[1].mentions) == 3

    cands = geoparser.run_candidate_selection(sentence_mentions, place_of_pub_wqid, place_of_pub)

    assert isinstance(cands, Candidates)
    # The double space between sentences is lost:
    assert cands.text() == ' '.join(text.split())

    assert len(cands.candidates()) == 5
    for c in cands.candidates():
        assert isinstance(c, MentionCandidates)

    predictions = geoparser.run_disambiguation(cands)

    assert isinstance(predictions, Predictions)
    assert predictions.candidates()[0].best_wqid() == "Q989418"
    assert predictions.candidates()[0].best_disambiguation_score() == pytest.approx(0.350, abs=1e-3)
    assert predictions.candidates()[-1].best_wqid() == "Q171866"
    assert predictions.candidates()[-1].best_disambiguation_score() == pytest.approx(0.615, abs=1e-3)

    ### Test on another chunk of text.
    text = """Palmer, labonrce aged Y., costautted usiiide by hanging himeelf at his residence in Whittle's-eard. (lathe's street. lifiddlesbromh.
Re threatened to hap:: himself on Elsitarday morning. and al night was found to have earcied ont hit threat with is piece of rope in his bed-room.,An inuesL was held on the licdy ad the Cleveland Pay Hotel, clegel and-s tree t, iddLesbronet, on :'!oart.ly al taru °oil.
Fos Tint TIeETII AND 'kill ,A few drops of the W 4" FlorIllue" 'dee a wee eolli-bir neh prxi a. a: &pteaa Lai •-h thomv.hly de.ruitas tike tooth from all --Wee. harden% the WM; prevenlw '..esto.the froth e pOtilay itdatmd I had Fri. to the parlour. , Folkestone.
Beech-street, London."""

    place_of_pub_wqid = "Q989418"
    place_of_pub = "Stockton-on-Tees, Cleveland, England"

    sentence_mentions = geoparser.run_text_recognition(text)
    cands = geoparser.run_candidate_selection(sentence_mentions, place_of_pub_wqid, place_of_pub)

    predictions = geoparser.run_disambiguation(cands)

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.candidates(ignore_empty_candidates=False)) == 9
    assert len(predictions.candidates(ignore_empty_candidates=True)) == 8

    # Test without microtoponyms.
    geoparser.linker.rel_params["without_microtoponyms"] = True

    sentence_mentions = geoparser.run_text_recognition(text)
    cands = geoparser.run_candidate_selection(sentence_mentions, place_of_pub_wqid, place_of_pub)
    predictions = geoparser.run_disambiguation(cands)

    assert isinstance(predictions, RelPredictions)

    assert len(predictions.candidates(ignore_empty_candidates=False)) == 5
    assert all([not c.mention.is_microtoponym() for c in predictions.candidates(ignore_empty_candidates=False)])
    assert len(predictions.candidates(ignore_empty_candidates=True)) == 4

@pytest.mark.resources(reason="Needs large resources")
def test_combined_score(tmp_path):

    model_path = os.path.join(current_dir, "../resources/models/")
    assert os.path.isdir(model_path) is True

    recogniser = ner.CustomRecogniser(
        model_name="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=model_path,
        training_args={
            "batch_size": 8,
            "num_train_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.0,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
    )

    # --------------------------------------
    # Instantiate the ranker:
    ranker = ranking.DeezyMatchRanker(
        resources_path=os.path.join(current_dir, "../resources/"),
        mentions_to_wikidata=dict(),
        wikidata_to_mentions=dict(),
        strvar_parameters={
            # Parameters to create the string pair dataset:
            "ocr_threshold": 60,
            "top_threshold": 85,
            "min_len": 5,
            "max_len": 15,
            "w2v_ocr_path": str(tmp_path),
            "w2v_ocr_model": "w2v_1800s_news",
            "overwrite_dataset": False,
        },
        deezy_parameters={
            # Paths and filenames of DeezyMatch models and data:
            "dm_path": os.path.join(current_dir, "../resources/deezymatch/"),
            "dm_cands": "wkdtalts",
            "dm_model": "w2v_ocr",
            "dm_output": "deezymatch_on_the_fly",
            # Ranking measures:
            "ranking_metric": "faiss",
            "selection_threshold": 50,
            "num_candidates": 1,
            "verbose": False,
            # DeezyMatch training:
            "overwrite_training": False,
            "do_test": False,
        },
    )

    with sqlite3.connect(os.path.join(current_dir, "../resources/rel_db/embeddings_database.db")) as conn:
        cursor = conn.cursor()
        linker = linking.RelDisambLinker(
            resources_path=os.path.join(current_dir,"../resources/"),
            ranker=ranker,
            linking_resources=dict(),
            rel_params={
                "db_embeddings": cursor,
                "with_publication": True,
                "predict_place_of_publication": False,
                "combined_score": True,
                "without_microtoponyms": False,
            },
            overwrite_training=False,
        )

    geoparser = pipeline.Pipeline(recogniser=recogniser, ranker=ranker, linker=linker)

    text = """There was very little to choose between the play of the two teams, and why the Penrith forwards did not bang the ball out of the scrummage during the quarter of an hour they had the Aspatria men penned within their "25," and their backs having the assistance of the wind to kick with, was a puzzler to me, and why the backs didn't kick more during the second half was another puzzler."""

    place_of_pub = "Carlisle, Cumbria, England"
    place_of_pub_wqid = "Q192896"

    predictions = geoparser.run(text, place_of_pub_wqid, place_of_pub)

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.rel_scores) == 1
    combined_scores = predictions.rel_scores[0]

    # Check that Penrith, Australia is the REL prediction but Penrith, Cumbria 
    # is the prediction *after* applying the combined score.

    # Penrith, Cumbria is Q798906, latlon (54.6648, -2.7548).
    assert predictions.best_wqids()[0] == 'Q798906'
    assert predictions.best_coords()[0] == (54.6648, -2.7548)

    # Combined scores:
    assert combined_scores.scores['Q798906'] == pytest.approx(0.26184, 1e-4)
    assert combined_scores.scores['Q798906'] == max(combined_scores.scores.values())
    # REL scores:
    assert combined_scores.rel_scores['Q798906'] == pytest.approx(0.26195, 1e-4)
    assert combined_scores.rel_scores['Q798906'] != max(combined_scores.scores.values())

    # Penrith, Australia is Q385155, latlon (-33.751111, 150.694167).
    assert combined_scores.scores['Q385155'] == pytest.approx(0.15684, 1e-4)
    assert combined_scores.scores['Q385155'] != max(combined_scores.scores.values())

    assert combined_scores.rel_scores['Q385155'] == pytest.approx(0.39417, 1e-4)
    assert combined_scores.rel_scores['Q385155'] == max(combined_scores.rel_scores.values())

    # print("combined scores:")
    # for k, v in sorted(combined_scores.scores.items(), key=lambda item: item[1], reverse=True):
    #     print(f'{k}: {v}')
    # print("REL scores:")
    # for k, v in sorted(combined_scores.rel_scores.items(), key=lambda item: item[1], reverse=True):
    #     print(f'{k}: {v}')

    # Re-run the same test but omitting place of publication info, so the default is used (UK).
    predictions = geoparser.run(text)

    assert isinstance(predictions, RelPredictions)
    assert len(predictions.rel_scores) == 1
    combined_scores = predictions.rel_scores[0]

    # Check that Penrith, Australia is the REL prediction but Penrith, Cumbria 
    # is the prediction *after* applying the combined score.

    # Penrith, Cumbria is Q798906, latlon (54.6648, -2.7548).
    assert predictions.best_wqids()[0] == 'Q798906'
    assert predictions.best_coords()[0] == (54.6648, -2.7548)

    # Combined scores:
    assert combined_scores.scores['Q798906'] == pytest.approx(0.29257, 1e-4)
    assert combined_scores.scores['Q798906'] == max(combined_scores.scores.values())
    # REL scores:
    assert combined_scores.rel_scores['Q798906'] == pytest.approx(0.29295, 1e-4)
    assert combined_scores.rel_scores['Q798906'] != max(combined_scores.scores.values())

    # Penrith, Australia is Q385155, latlon (-33.751111, 150.694167).
    assert combined_scores.scores['Q385155'] == pytest.approx(0.12602, 1e-4)
    assert combined_scores.scores['Q385155'] != max(combined_scores.scores.values())

    assert combined_scores.rel_scores['Q385155'] == pytest.approx(0.316710, 1e-4)
    assert combined_scores.rel_scores['Q385155'] == max(combined_scores.rel_scores.values())
