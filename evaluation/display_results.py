import os
import sys

import pandas as pd

# Add "../" to path to import utils
sys.path.insert(0, os.path.abspath("CLEF-HIPE-2020-scorer/"))

import clef_evaluation

datasets = ["lwm", "hipe"]
approaches = [
    {
        "ner_model_id": "skyline:lwm",  # LwM baseline
        "cand_select_method": "perfectmatch",
        "top_res_method": "mostpopular",
    },
    {
        "ner_model_id": "lwm",  # LwM baseline
        "cand_select_method": "perfectmatch",
        "top_res_method": "mostpopular",
    },
    {
        "ner_model_id": "rel",  # API end-to-end REL
        "cand_select_method": "rel",
        "top_res_method": "rel",
    },
]

df_ner = pd.DataFrame()
df_nel = pd.DataFrame()
overall_results_nerc = dict()
overall_results_nel = dict()
for dataset in datasets:
    for approach in approaches:

        eval_results = []

        # Approach:
        ner_model_id = approach["ner_model_id"]
        cand_select_method = approach["cand_select_method"]
        top_res_method = approach["top_res_method"]

        approach_name = ner_model_id + "+" + cand_select_method + "+" + top_res_method
        pred = (
            "../experiments/outputs/results/" + dataset + "/" + approach_name + "_bundle2_en_1.tsv"
        )
        true = "../experiments/outputs/results/" + dataset + "/true_bundle2_en_1.tsv"

        ner_score = clef_evaluation.get_results(
            f_ref=true, f_pred=pred, task="nerc_coarse", outdir="results/"
        )
        linking_score = clef_evaluation.get_results(
            f_ref=true, f_pred=pred, task="nel", outdir="results/"
        )

        ne_tags = ["ALL"]
        settings = ["strict", "partial", "exact"]
        measures = ["P_micro", "R_micro", "F1_micro"]

        overall_results_nerc["dataset_approach"] = [dataset + "_" + ner_model_id]
        for ne_tag in ne_tags:
            for setting in settings:
                for measure in measures:
                    overall_results_nerc[setting + measure] = round(
                        ner_score["NE-COARSE-LIT"]["TIME-ALL"]["LED-ALL"][ne_tag][setting][
                            measure
                        ],
                        3,
                    )
        df_ner = df_ner.append(pd.DataFrame(overall_results_nerc))

        overall_results_nel["dataset_approach"] = [dataset + "_" + ner_model_id]
        for ne_tag in ne_tags:
            for setting in settings:
                for measure in measures:
                    if setting == "strict":
                        overall_results_nel[setting + measure] = round(
                            linking_score[1]["NEL-LIT"]["TIME-ALL"]["LED-ALL"][ne_tag][setting][
                                measure
                            ],
                            3,
                        )
        df_nel = df_nel.append(pd.DataFrame(overall_results_nel))

print(df_ner.to_latex(index=False))
print(df_nel.to_latex(index=False))
