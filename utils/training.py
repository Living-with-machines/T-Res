import pandas as pd
from pathlib import Path
from utils import process_data


def create_trainset(train_path, myranker, already_collected_cands=dict()):
    """
    Create entity linking training data (i.e. mentions identified and candidates provided),
    necessary for training our resolution methods:
    """
    training_cands_path = train_path.split(".tsv")[0] + "_" + myranker.method + ".tsv"
    if not Path(training_cands_path).exists():
        training_set = pd.read_csv(train_path, sep="\t")
        training_df = process_data.crate_training_for_el(training_set)
        candidates_qid = []
        for i, row in training_df.iterrows():
            cands, already_collected_cands = myranker.select(
                [row["mention"]], myranker.method, myranker, already_collected_cands
            )
            if row["mention"] in cands:
                candidates_qid.append(
                    myranker.get_candidate_wikidata_ids(cands[row["mention"]])
                )
            else:
                candidates_qid.append(dict())

        training_df["wkdt_cands"] = candidates_qid
        training_df.to_csv(training_cands_path, sep="\t", index=False)
        return training_df
    else:
        return pd.read_csv(training_cands_path, sep="\t")
