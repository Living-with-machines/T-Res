import os
import sys

import pandas as pd
import warnings

from regex import E

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # To fix properly in the future

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath("CLEF-HIPE-2020-scorer/"))

import clef_evaluation


# -------------------------------------
# GENERIC NAMED ENTITY RECOGNITION
# -------------------------------------

datasets = ["lwm", "hipe"]
filtering_labels = ["all", "loc"]
approaches = ["preds", "rel"]
granularities = ["fine", "coarse"]
ner_models = ["blb_lwm"]

ne_tag = "ALL"
settings = ["strict", "partial", "exact"]
measures = ["P_micro", "R_micro", "F1_micro"]

df_ner = pd.DataFrame()
overall_results_nerc = dict()

for dataset in datasets:
    for filtering_label in filtering_labels:
        for granularity in granularities:
            for approach in approaches:
                for ner_model in ner_models:
                    if filtering_label == "loc":
                        ne_tag = "LOC"
                    approach_fullname = (
                        dataset
                        + "_"
                        + approach
                        + "-"
                        + ner_model
                        + "-"
                        + granularity
                        + "-"
                        + filtering_label
                    )

                    # String in common in pred and true filenames:
                    filename_common_str = (
                        "../experiments/outputs/results/"
                        + dataset
                        + "/ner_"
                        + ner_model
                        + "-ner-"
                        + granularity
                        + "_"
                        + filtering_label
                    )

                    # Predictions file:
                    pred_file = filename_common_str + "_" + approach + ".tsv"
                    # Gold standard file:
                    true_file = filename_common_str + "_trues.tsv"

                    print(pred_file)
                    print(true_file)

                    try:
                        # Get NER score:
                        ner_score = clef_evaluation.get_results(
                            f_ref=true_file,
                            f_pred=pred_file,
                            task="nerc_coarse",
                            outdir="results/",
                            skip_check=True,
                        )

                        # Produce results table:
                        overall_results_nerc["ner_method"] = [approach_fullname]
                        for setting in settings:
                            for measure in measures:
                                overall_results_nerc[
                                    setting.replace("_", "")
                                    + ":"
                                    + measure.split("_")[0]
                                ] = round(
                                    ner_score["NE-COARSE-LIT"]["TIME-ALL"]["LED-ALL"][
                                        ne_tag
                                    ][setting][measure],
                                    3,
                                )
                        df_ner = df_ner.append(pd.DataFrame(overall_results_nerc))
                    except FileNotFoundError:
                        print("File does not exist.")

print("# --------------------------------")
print("# Toponym recognition:\n")
print(df_ner.to_latex(index=False))

"""
# -------------------------------------
# TAG-SPECIFIC NAMED ENTITY RECOGNITION
# -------------------------------------

datasets = ["lwm"]
accepted_labels = "all"
ner_approaches = [
    {
        "ner_model_id": "lwm",
        "cand_select_method": "perfectmatch",  # This does not really matter here
        "top_res_method": "mostpopular",  # This does not really matter here
    }
]

ne_tags = ["ALL", "LOC", "STREET", "BUILDING"]
settings = ["strict", "ent_type"]
measures = ["P_micro", "R_micro", "F1_micro"]

df_ner = pd.DataFrame()
overall_results_nerc = dict()

for dataset in datasets:
    for approach in ner_approaches:

        ner_method = approach["ner_model_id"]
        cand_select_method = approach["cand_select_method"]
        top_res_method = approach["top_res_method"]

        # Get approach name:
        approach_name = (
            ner_method
            + "+"
            + cand_select_method
            + "+"
            + top_res_method
            + "+"
            + accepted_labels
        )

        # Predictions file:
        pred = (
            "../experiments/outputs/results/"
            + dataset
            + "/"
            + approach_name
            + "_bundle2_en_1.tsv"
        )

        # Gold standard file:
        true = (
            "../experiments/outputs/results/"
            + dataset
            + "/true_"
            + accepted_labels
            + "_bundle2_en_1.tsv"
        )

        # Get NER score:
        ner_score = clef_evaluation.get_results(
            f_ref=true, f_pred=pred, task="nerc_coarse", outdir="results/"
        )

        for ne_tag in ne_tags:
            overall_results_nerc["NE tag"] = [ne_tag]
            for setting in settings:
                for measure in measures:
                    overall_results_nerc[
                        setting.replace("_", "") + ":" + measure.split("_")[0]
                    ] = round(
                        ner_score["NE-COARSE-LIT"]["TIME-ALL"]["LED-ALL"][ne_tag][
                            setting
                        ][measure],
                        3,
                    )
            df_ner = df_ner.append(pd.DataFrame(overall_results_nerc))

print(df_ner.to_latex(index=False))

# -------------------------------------
# ENTITY LINKING
# -------------------------------------

datasets = ["lwm", "hipe"]
approaches = dict()
approaches["rel"] = {"candselect": ["rel"], "topres": ["rel"]}
approaches["lwm"] = {
    "candselect": ["perfectmatch", "deezymatch"],
    "topres": ["skyline", "mostpopular"],
}
accepted_labels = ["all", "loc"]

df_nel = pd.DataFrame()
overall_results_nel = dict()

for al in accepted_labels:
    for dataset in datasets:
        for approach in approaches:
            ner_model_id = approach
            candselect_methods = approaches[approach]["candselect"]
            topres_methods = approaches[approach]["topres"]
            for cand_select_method in candselect_methods:
                for top_res_method in topres_methods:
                    approach_name = (
                        ner_model_id
                        + "+"
                        + cand_select_method
                        + "+"
                        + top_res_method
                        + "+"
                        + al
                    )
                    if al == "loc" and (ner_model_id == "rel" or dataset == "hipe"):
                        continue
                    if top_res_method == "skyline":
                        approach_name = (
                            top_res_method
                            + ":"
                            + ner_model_id
                            + "+"
                            + cand_select_method
                            + "+"
                            + al
                        )
                    print(approach_name, al)
                    pred = (
                        "../experiments/outputs/results/"
                        + dataset
                        + "/"
                        + approach_name
                        + "_bundle2_en_1.tsv"
                    )
                    true = (
                        "../experiments/outputs/results/"
                        + dataset
                        + "/true_"
                        + al
                        + "_bundle2_en_1.tsv"
                    )

                    try:
                        linking_score = clef_evaluation.get_results(
                            f_ref=true, f_pred=pred, task="nel", outdir="results/"
                        )
                    except FileNotFoundError:
                        print("* File does not exist: " + approach_name)
                        continue

                    ne_tags = ["ALL"]
                    settings = ["strict", "partial", "exact"]
                    measures = ["P_micro", "R_micro", "F1_micro"]

                    overall_results_nel["dataset:approach"] = [
                        dataset
                        + ":"
                        + cand_select_method
                        + ":"
                        + top_res_method
                        + ":"
                        + al
                    ]
                    for ne_tag in ne_tags:
                        for setting in settings:
                            for measure in measures:
                                if setting == "strict":
                                    overall_results_nel[
                                        setting.replace("_", "")
                                        + ":"
                                        + measure.split("_")[0]
                                    ] = round(
                                        linking_score[1]["NEL-LIT"]["TIME-ALL"][
                                            "LED-ALL"
                                        ][ne_tag][setting][measure],
                                        3,
                                    )
                    df_nel = df_nel.append(pd.DataFrame(overall_results_nel))

print(df_nel.to_latex(index=False))
"""
