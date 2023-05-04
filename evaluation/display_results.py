import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # To fix properly in the future

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath("HIPE-scorer/"))
import clef_evaluation

dApprNames = dict()
dApprNames["rel_end_to_end_api"] = "rel-api"
dApprNames["rel_wiki2019_aida"] = "rel-19aida"
dApprNames["rel_wikilwm_lwm_locs"] = "rel-19lwmlocs"


# -------------------------------------
# CREATE TEMPORARY RESULTS FOLDER
# -------------------------------------

Path("results").mkdir(parents=True, exist_ok=True)

# -------------------------------------
# NAMED ENTITY RECOGNITION
# -------------------------------------

datasets = ["lwm", "hipe"]
granularities = ["fine", "coarse"]
ner_models = ["blb_lwm"]

# Get files with results:
pred_files = []
true_files = []
approach_names = []
for dataset in datasets:
    for granularity in granularities:
        if dataset == "hipe" and granularity == "fine":
            continue
        for ner_model in ner_models:
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
            pred_file = filename_common_str + "_preds.tsv"
            # Gold standard file:
            true_file = filename_common_str + "_trues.tsv"

            if Path(pred_file).exists() and Path(true_file).exists():
                pred_files.append(pred_file)
                true_files.append(true_file)
                approach_names.append(
                    dataset
                    + "-"
                    + ner_model.replace("_", "")
                    + "-"
                    + granularity
                    + "-preds"
                )

for dataset in datasets:
    for granularity in granularities:
        for ner_model in ner_models:
            for rel_approach in dApprNames.keys():
                pred_file = (
                    "../experiments/outputs/results/"
                    + dataset
                    + "/"
                    + rel_approach
                    + "_"
                    + ner_model
                    + "-ner-"
                    + granularity
                    + "_"
                    + "originalsplit.tsv"
                )
                true_file = (
                    "../experiments/outputs/results/"
                    + dataset
                    + "/ner_"
                    + ner_model
                    + "-ner-"
                    + granularity
                    + "_trues.tsv"
                )

                if Path(pred_file).exists() and Path(true_file).exists():
                    pred_files.append(pred_file)
                    true_files.append(true_file)
                    approach_names.append(
                        dataset + "-" + granularity + "-" + dApprNames[rel_approach]
                    )


# -------------------------
# Produce results for the selected files:
ne_tag = "ALL"
settings = ["strict", "partial", "exact"]
measures = ["P_micro", "R_micro", "F1_micro"]
df_ner = pd.DataFrame()
overall_results_nerc = dict()

for i in range(len(pred_files)):
    pred_file = pred_files[i]
    true_file = true_files[i]
    approach_fullname = approach_names[i]

    try:
        # Get NER score:
        ner_score = clef_evaluation.get_results(
            f_ref=true_file,
            f_pred=pred_file,
            task="nerc_coarse",
            outdir="results/",
            skip_check=True,
            edition="HIPE-2020",
        )

        # Produce results table:
        overall_results_nerc["ner_method"] = [approach_fullname]
        for setting in settings:
            for measure in measures:
                if (
                    # Using REL, if dataset is LwM, "strict" is not fair,
                    # because the tagsets are different.
                    approach_fullname.split("-")[2] == "rel"
                    and approach_fullname.split("-")[0] == "lwm"
                    and setting == "strict"
                ) or (
                    # If granularity is "coarse", if dataset is LwM, "strict"
                    # is not fair, because again the tagsets are different.
                    "-coarse-" in approach_fullname
                    and approach_fullname.split("-")[0] == "lwm"
                    and setting == "strict"
                ):
                    overall_results_nerc[
                        setting.replace("_", "") + ":" + measure.split("_")[0]
                    ] = "---"
                else:
                    overall_results_nerc[
                        setting.replace("_", "") + ":" + measure.split("_")[0]
                    ] = round(
                        ner_score["NE-COARSE-LIT"]["TIME-ALL"]["LED-ALL"][ne_tag][
                            setting
                        ][measure],
                        3,
                    )
        df_ner = df_ner.append(pd.DataFrame(overall_results_nerc))
    except FileNotFoundError:
        print("File does not exist.")

print("\n# --------------------------------")
print("# Toponym recognition:")
print("# --------------------------------\n")
print(df_ner.to_latex(index=False))


# -------------------------------------
# ENTITY LINKING
# -------------------------------------

datasets = ["lwm", "hipe"]
ner_approaches = ["blb_lwm-ner"]
ranking_approaches = [
    "perfectmatch",
    "deezymatch+3+25",
]
linking_approaches = [
    "skys",
    "mostpopular",
    "bydistance",
    "reldisamb",
    "reldisamb+wmtops",
    "reldisamb+wpubl",
    "reldisamb+wpubl+wmtops_test",
]
granularities = ["fine"]
splits = [
    "originalsplit",
    "withouttest",
    "Ashton1860",
    "Dorchester1820",
    "Dorchester1830",
    "Dorchester1860",
    "Manchester1780",
    "Manchester1800",
    "Manchester1820",
    "Manchester1830",
    "Manchester1860",
    "Poole1860",
]

df_nel = pd.DataFrame()
overall_results_nel = dict()
pred_files = []
true_files = []
approach_names = []

# GET RELEVANT OUR-METHOD FILES:
for dataset in datasets:
    for split in splits:
        for ner_approach in ner_approaches:
            for ranking_approach in ranking_approaches:
                for linking_approach in linking_approaches:
                    for granularity in granularities:
                        pred = (
                            "../experiments/outputs/results/"
                            + dataset
                            + "/linking_"
                            + ner_approach
                            + "-"
                            + granularity
                            + "_"
                            + ranking_approach
                            + "_"
                            + split
                            + "_"
                            + linking_approach
                            + ".tsv"
                        )
                        true = (
                            "../experiments/outputs/results/"
                            + dataset
                            + "/linking_"
                            + ner_approach
                            + "-"
                            + granularity
                            + "_"
                            + ranking_approach
                            + "_"
                            + split
                            + "_trues.tsv"
                        )

                        if Path(pred).exists() and Path(true).exists():
                            pred_files.append(pred)
                            true_files.append(true)
                            approach_names.append(
                                dataset
                                + "+"
                                + split
                                + ":"
                                + granularity
                                + "+"
                                + ranking_approach.replace("match", "")
                                + "+"
                                + linking_approach
                            )

# GET RELEVANT REL FILES:
for dataset in datasets:
    for granularity in granularities:
        for ner_approach in ner_approaches:
            for rel_approach in dApprNames.keys():
                for split in splits:
                    pred_file = (
                        "../experiments/outputs/results/"
                        + dataset
                        + "/"
                        + rel_approach
                        + "_"
                        + ner_approach
                        + "-"
                        + granularity
                        + "_"
                        + split
                        + ".tsv"
                    )
                    true_file = (
                        "../experiments/outputs/results/"
                        + dataset
                        + "/linking_"
                        + ner_approach
                        + "-"
                        + granularity
                        + "_perfectmatch_"
                        + split
                        + "_trues.tsv"
                    )
                    if Path(pred_file).exists() and Path(true_file).exists():
                        pred_files.append(pred_file)
                        true_files.append(true_file)
                        approach_names.append(
                            dataset
                            + "+"
                            + split
                            + ":"
                            + granularity
                            + "+"
                            + dApprNames[rel_approach]
                        )

# RUN SCORER:
for i in range(len(pred_files)):
    pred_file = pred_files[i]
    true_file = true_files[i]
    approach_fullname = approach_names[i]

    try:
        print(str(i) + ">> " + pred_file)
        linking_score = clef_evaluation.get_results(
            f_ref=true_file,
            f_pred=pred_file,
            task="nel",
            outdir="results/",
            skip_check=True,
            edition="HIPE-2020",
        )

        ne_tags = ["ALL"]
        settings = ["ent_type"]  # ["strict", "ent_type"]
        measures = ["P_micro", "R_micro", "F1_micro"]

        overall_results_nel["dataset:approach"] = [str(i) + " > " + approach_fullname]
        for ne_tag in ne_tags:
            for setting in settings:
                for measure in measures:
                    if measure != "Acc":
                        overall_results_nel[
                            setting.replace("_", "") + ":" + measure.split("_")[0]
                        ] = round(
                            linking_score[1]["NEL-LIT"]["TIME-ALL"]["LED-ALL"][ne_tag][
                                setting
                            ][measure],
                            3,
                        )
                    else:
                        correct = linking_score[1]["NEL-LIT"]["TIME-ALL"]["LED-ALL"][
                            ne_tag
                        ][setting]["correct"]
                        incorrect = linking_score[1]["NEL-LIT"]["TIME-ALL"]["LED-ALL"][
                            ne_tag
                        ][setting]["incorrect"]
                        overall_results_nel[
                            setting.replace("_", "") + ":" + measure.split("_")[0]
                        ] = round(
                            correct / (correct + incorrect),
                            3,
                        )

        df_nel = df_nel.append(pd.DataFrame(overall_results_nel))

    except FileNotFoundError:
        continue


print("\n# --------------------------------")
print("# Toponym resolution:")
print("# --------------------------------\n")
print(df_nel.to_latex(index=False))
print("\n")
