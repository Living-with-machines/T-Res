import os
import sys

import pandas as pd
from pathlib import Path
import warnings

from regex import E

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # To fix properly in the future

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath("CLEF-HIPE-2020-scorer/"))

import clef_evaluation


# -------------------------------------
# NAMED ENTITY RECOGNITION
# -------------------------------------

datasets = ["lwm", "hipe"]
approaches = ["preds", "rel"]
granularities = ["fine", "coarse"]
ner_models = ["blb_lwm"]

ne_tag = "ALL"
settings = ["strict", "partial", "exact"]
measures = ["P_micro", "R_micro", "F1_micro"]

df_ner = pd.DataFrame()
overall_results_nerc = dict()

for dataset in datasets:
    for granularity in granularities:
        for approach in approaches:
            for ner_model in ner_models:
                approach_fullname = (
                    dataset + "_" + approach + "-" + ner_model + "-" + granularity
                )

                # String in common in pred and true filenames:
                filename_common_str = (
                    "../experiments/outputs/results/"
                    + dataset
                    + "/ner_"
                    + ner_model
                    + "-ner-"
                    + granularity
                )

                # Predictions file:
                pred_file = filename_common_str + "_" + approach + ".tsv"
                # Gold standard file:
                true_file = filename_common_str + "_trues.tsv"

                print(pred_file)
                print(true_file)

                if Path(pred_file).exists() and Path(true_file).exists():

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
                                if (
                                    # Using REL, if dataset is LwM, "strict" is not fair,
                                    # because the tagsets are different.
                                    approach == "rel"
                                    and dataset == "lwm"
                                    and setting == "strict"
                                ) or (
                                    # If granularity is "coarse", if dataset is LwM, "strict"
                                    # is not fair, because again the tagsets are different.
                                    granularity == "coarse"
                                    and dataset == "lwm"
                                    and setting == "strict"
                                ):
                                    overall_results_nerc[
                                        setting.replace("_", "")
                                        + ":"
                                        + measure.split("_")[0]
                                    ] = "---"
                                else:
                                    overall_results_nerc[
                                        setting.replace("_", "")
                                        + ":"
                                        + measure.split("_")[0]
                                    ] = round(
                                        ner_score["NE-COARSE-LIT"]["TIME-ALL"][
                                            "LED-ALL"
                                        ][ne_tag][setting][measure],
                                        3,
                                    )
                        df_ner = df_ner.append(pd.DataFrame(overall_results_nerc))
                    except FileNotFoundError:
                        print("File does not exist.")

print("# --------------------------------")
print("# Toponym recognition:\n")
print(df_ner.to_latex(index=False))

# -------------------------------------
# ENTITY LINKING
# -------------------------------------

datasets = ["lwm", "hipe"]
ner_approaches = ["linking_blb_lwm-ner"]
ranking_approaches = [
    "perfectmatch",
    "deezymatch+2+20",
]
linking_approaches = ["mostpopular", "skys", "rel"]
granularities = ["fine", "coarse"]
splits = [
    "originalsplit",
    # "traindevtest",
    # "Ashton1860",
    # "Dorchester1820",
    # "Dorchester1830",
    # "Dorchester1860",
    # "Manchester1780",
    # "Manchester1800",
    # "Manchester1820",
    # "Manchester1830",
    # "Manchester1860",
    # "Poole1860",
]
devtest_list = ["dev"]

df_nel = pd.DataFrame()
overall_results_nel = dict()

for dataset in datasets:
    for ner_approach in ner_approaches:
        for ranking_approach in ranking_approaches:
            for linking_approach in linking_approaches:
                for granularity in granularities:
                    for split in splits:
                        for devtest in devtest_list:
                            pred = (
                                "../experiments/outputs/results/"
                                + dataset
                                + "/"
                                + ner_approach
                                + "-"
                                + granularity
                                + "_"
                                + ranking_approach
                                + "_"
                                + split
                                + "-"
                                + devtest
                                + "_"
                                + linking_approach
                                + ".tsv"
                            )
                            true = (
                                "../experiments/outputs/results/"
                                + dataset
                                + "/"
                                + ner_approach
                                + "-"
                                + granularity
                                + "_"
                                + ranking_approach
                                + "_"
                                + split
                                + "-"
                                + devtest
                                + "_trues.tsv"
                            )

                            if Path(pred).exists() and Path(true).exists():
                                print("File exists:", pred)
                                try:
                                    linking_score = clef_evaluation.get_results(
                                        f_ref=true,
                                        f_pred=pred,
                                        task="nel",
                                        outdir="results/",
                                        skip_check=True,
                                    )

                                    ne_tags = ["ALL"]
                                    settings = ["strict"]
                                    measures = ["P_micro", "R_micro", "F1_micro"]

                                    approach_name = (
                                        dataset
                                        + ":"
                                        + granularity
                                        + "+"
                                        + ranking_approach.replace("match", "")
                                        + "+"
                                        + split
                                        + "+"
                                        + linking_approach
                                    )
                                    overall_results_nel["dataset:approach"] = [
                                        approach_name
                                    ]
                                    for ne_tag in ne_tags:
                                        for setting in settings:
                                            for measure in measures:
                                                overall_results_nel[
                                                    setting.replace("_", "")
                                                    + ":"
                                                    + measure.split("_")[0]
                                                ] = round(
                                                    linking_score[1]["NEL-LIT"][
                                                        "TIME-ALL"
                                                    ]["LED-ALL"][ne_tag][setting][
                                                        measure
                                                    ],
                                                    3,
                                                )
                                    df_nel = df_nel.append(
                                        pd.DataFrame(overall_results_nel)
                                    )

                                except FileNotFoundError:
                                    continue

print(df_nel.to_latex(index=False))
