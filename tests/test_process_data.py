import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from t_res.geoparser import recogniser
from t_res.utils import process_data

current_dir = Path(__file__).parent.resolve()

def test_eval_with_exception():
    # test normal behaviour

    str_list_of_dict = "[{'key_1': 1, 'key_2': 2}]"
    list_of_dict = process_data.eval_with_exception(str_list_of_dict)

    assert list_of_dict != str_list_of_dict
    assert isinstance(list_of_dict,list)
    assert isinstance(list_of_dict[0],dict)

    # test that it returns "" if the input is None

    str_list_of_dict = None
    list_of_dict = process_data.eval_with_exception(str_list_of_dict)

    assert list_of_dict == ""

    # test that it raises an error if the syntax is wrong
    str_list_of_dict = "[{'key_1': 1, 'key_2': 2}"

    with pytest.raises(SyntaxError) as cm:
        process_data.eval_with_exception(str_list_of_dict)


@pytest.mark.experiment
def test_prepare_sents():
    dataset_df = pd.read_csv(
        os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/linking_df_split.tsv"),
        sep="\t",
    )

    test_data = process_data.eval_with_exception(dataset_df["annotations"][0])
    test_data[0]["wkdt_qid"] = 42
    test_data = str(test_data)
    dataset_df["annotations"][0] = test_data

    dAnnotated, dSentences, dMetadata = process_data.prepare_sents(dataset_df)

    assert dAnnotated["3580760_2"][(0, 6)] == ('LOC', 'LONDON', 'Q84')

    test_data = process_data.eval_with_exception(dataset_df["annotations"][0])
    test_data[0]["wkdt_qid"] = "*"
    test_data = str(test_data)
    dataset_df["annotations"][0] = test_data

    dAnnotated, dSentences, dMetadata = process_data.prepare_sents(dataset_df)

    assert dAnnotated["3580760_2"][(0, 6)] == ('LOC', 'LONDON', 'Q84')

    assert len(dAnnotated) == len(dSentences) == len(dMetadata)

    # this assert that there are no cases where we were missing metadata
    assert len([x for x, y in dMetadata.items() if len(y) == 0]) == 0


def test_align_gold(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=str(tmp_path),  # Path where the NER model will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myner.train()
    myner.pipe = myner.create_pipeline()

    dataset_df = pd.read_csv(
        os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/linking_df_split.tsv"),
        sep="\t",
    )

    dAnnotated, dSentences, dMetadata = process_data.prepare_sents(dataset_df)
    empty_list = []
    for sent_id in dSentences.keys():
        if "3580760_2" == sent_id:
            sent = dSentences[sent_id]
            annotations = dAnnotated[sent_id]
            predictions = myner.ner_predict(sent)
            gold_positions = process_data.align_gold(predictions, annotations)

            I_elements = [
                x
                for x in range(len(gold_positions))
                if "I-LOC" == gold_positions[x]["entity"]
            ]
            B_elements = [
                x
                for x in range(len(gold_positions))
                if "B-LOC" == gold_positions[x]["entity"]
            ]

            # assert that the previous element of a I-element is either a B- or a I-
            for i in I_elements:
                if (i - 1 in I_elements) or (i - 1 in B_elements):
                    continue
                else:
                    empty_list.append(sent_id)

    assert len(empty_list) == 0


def test_ner_and_process(tmp_path):
    myner = recogniser.Recogniser(
        model="blb_lwm-ner-fine",
        train_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_train.json"),
        test_dataset=os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/ner_fine_dev.json"),
        pipe=None,
        base_model="khosseini/bert_1760_1900",  # Base model to fine-tune
        model_path=str(tmp_path),  # Path where the NER model will be stored
        training_args={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
        },
        overwrite_training=False,  # Set to True if you want to overwrite model if existing
        do_test=False,  # Set to True if you want to train on test mode
        load_from_hub=False,  # Bool: True if model is in HuggingFace hub
    )

    myner.train()
    myner.pipe = myner.create_pipeline()

    dataset_df = pd.read_csv(
        os.path.join(current_dir,"sample_files/experiments/outputs/data/lwm/linking_df_split.tsv"),
        sep="\t",
    )

    dAnnotated, dSentences, dMetadata = process_data.prepare_sents(dataset_df)

    (
        dPreds,
        dTrues,
        dSkys,
        gold_tokenization,
        dMentionsPred,
        dMentionsGold,
    ) = process_data.ner_and_process(dSentences, dAnnotated, myner)

    B_els = [
        [z for z in range(len(y)) if "B-" in y[z]["entity"]]
        for x, y in gold_tokenization.items()
    ]
    I_els = [
        [z for z in range(len(y)) if "I-" in y[z]["entity"]]
        for x, y in gold_tokenization.items()
    ]
    misaligned_labels = []
    for l in range(len(I_els)):
        I_elements = I_els[l]
        B_elements = B_els[l]
        for i in I_elements:
            if (i - 1 in I_elements) or (i - 1 in B_elements):
                continue
            else:
                misaligned_labels.append(l)

    # No sentences should be misaligned:
    assert len(set(misaligned_labels)) == 0
