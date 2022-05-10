import pandas as pd
from pathlib import Path
from utils import process_data
import numpy as np
import tqdm


def create_trainset(train_path, myranker, mylinker):
    """
    Create entity linking training data (i.e. mentions identified and candidates provided),
    necessary for training our resolution methods:
    """
    print("\n*** Creating the entity linking trainset with candidates...")
    training_cands_path = train_path.split(".tsv")[0] + "_" + myranker.method + ".pkl"
    if not Path(training_cands_path).exists():
        training_set = pd.read_csv(train_path, sep="\t")
        training_df = process_data.crate_training_for_el(training_set)

        # Train on LOC and OTHERs alone:
        training_df = training_df[training_df["entity_type"].isin(["LOC", "OTHER"])]

        candidates_qids = dict()
        for original_mention in tqdm.tqdm(list(training_df.mention.unique())):
            # For each mention, find its mention matches according to our ranker:
            (
                candidates_matches,
                myranker.already_collected_cands,
            ) = myranker.run([original_mention])

            # For each mention, find its mention matches, the corresponding wikidata
            # entities, and the confidence score.
            wk_cands = dict()
            for found_mention in candidates_matches[original_mention]:
                found_cands = mylinker.get_candidate_wikidata_ids(found_mention)
                if found_cands:
                    for cand in found_cands:
                        if not cand in wk_cands:
                            wk_cands[cand] = {
                                "conf_score": candidates_matches[original_mention][
                                    found_mention
                                ],
                                "wkdt_relv": found_cands[cand],
                            }

            candidates_qids[original_mention] = wk_cands

        training_df["candidates"] = training_df["mention"].map(candidates_qids)

        print("\n*** Obtaining the vector embeddings of the mention in context.")
        training_df.loc[:, "mention_emb"] = training_df.loc[:, :].apply(
            lambda x: mylinker.get_mention_vector(x, agg=np.mean), axis=1
        )
        print("*** ... vectors obtained!\n")

        training_df.to_pickle(training_cands_path)
        print("... dataset creation completed!")
        return training_df

    else:
        training_df = pd.read_pickle(training_cands_path)
        print("*** ... dataset creation skipped, it already exists!\n")
        return training_df
