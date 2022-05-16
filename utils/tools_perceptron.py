import numpy as np
from ast import literal_eval


def eval_with_exception(string):
    try:
        return literal_eval(string)
    except ValueError:
        return None


def get_candidate_representation(
    row,
    candidate,
    dict_wqid_to_entemb,
    dict_wqid_to_class,
    dict_wqid_to_clssemb,
):

    wkdt_class = dict_wqid_to_class.get(candidate, "Unknown")
    wkdt_class_emb = dict_wqid_to_clssemb.get(wkdt_class, None)
    wkdt_ent_emb = dict_wqid_to_entemb.get(candidate, None)

    # Generate random vector in case not instance embedding:
    if wkdt_class_emb is None or wkdt_class == "Unknwon":
        # Get the size from class "country" (Q6256):
        size_class_emb = dict_wqid_to_clssemb["Q6256"].shape[0]
        wkdt_class_emb = np.random.uniform(
            low=-1.0, high=1.0, size=(1, size_class_emb)
        )[0]
    else:
        wkdt_class_emb = np.array(wkdt_class_emb)

    # Generate random vector in case not instance embedding:
    if wkdt_ent_emb is None:
        # Get the size from class "country" (Q6256):
        size_ent_emb = dict_wqid_to_entemb["Q84"].shape[0]
        wkdt_ent_emb = np.random.uniform(low=-1.0, high=1.0, size=(1, size_ent_emb))[0]
    else:
        wkdt_ent_emb = np.array(wkdt_ent_emb)

    mention_emb = row["mention_emb"]
    if mention_emb is None:
        mention_emb = np.random.uniform(low=-1.0, high=1.0, size=(1, 768))[0]

    vector = np.concatenate(
        # [instances_vector, candidate_vector, mention_emb], # with candidate embeddings.
        [wkdt_class_emb, wkdt_ent_emb, mention_emb],  # without candidate embeddings.
        axis=None,
        dtype=np.float32,
    )

    return vector
