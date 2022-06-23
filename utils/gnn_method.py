from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from haversine import haversine
from scipy.spatial.distance import cosine
from torch_geometric.nn import GATConv
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class EnhancedGATCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, edge_dim, x_ext_dim=3):
        super().__init__()
        torch.manual_seed(RANDOM_SEED)
        self.conv1 = GATConv(input_channels + x_ext_dim, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        # self.conv2 = GATConv(hidden_channels+x_ext_dim, 2,edge_dim=edge_dim)
        self.lin = torch.nn.Linear(hidden_channels + x_ext_dim, 2)

    def forward(self, x, x_ext, edge_index, edge_weight):
        x = torch.cat([x, x_ext], dim=1)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, x_ext], dim=1)
        x = self.lin(x).relu()
        return x


def get_entity_vector(mylinker, candidate: str) -> np.array:
    """
    Function that obtains the entity embedding given a wikidata id.

    Arguments:
        candidate (str): the wikidata id of the candidate
                for which to extract the entity embedding

    Returns:
        a vector in the shape of numpy.array with dimension 200
        embeddings are extracted from BigGraph
    """
    try:
        # see if candidate appears as entity embedding in the graph
        candidate_vector = mylinker.linking_resources["entity_embeddings"][
            mylinker.linking_resources["entity_ids"].index(candidate)
        ]

    except Exception as e:
        # check if we already encountered the embedding
        if candidate not in mylinker.linking_resources["random_entity_embeddings"]:
            # if entity not found create a random vector
            candidate_vector = np.random.uniform(low=-1.0, high=1.0, size=(200,))
            # and add it to the dictionary that keeps track these randomized entity embedding
            # the idea is that even when we don't have an embedding we want to use the
            # same random embedding for this entity
            mylinker.linking_resources["random_entity_embeddings"][
                candidate
            ] = candidate_vector  # .squeeze()
        else:
            # make sure each entity not in the embedding space has at least the same vector
            candidate_vector = mylinker.linking_resources["random_entity_embeddings"][candidate]

    return candidate_vector


def get_instance_vector(mylinker, candidate: str, use_random=True) -> np.array:
    """
    Function that obtains the instance (or class) embedding given a wikidata id.

    Arguments:
        candidate (str):
    Returns:
        a np.array vector with dimension 128
    """

    instance_id = mylinker.linking_resources["wikidata_id2inst_id"].get(candidate, None)

    if instance_id:
        try:
            instance_vector = mylinker.linking_resources["instance_embeddings"][
                mylinker.linking_resources["instance_ids"].index(instance_id)
            ]
        except Exception as e:
            if use_random:
                instance_vector = mylinker.linking_resources["random_instance_embedding"][0]
            else:
                return None
    else:  # if none of the instance ids found, get random vector
        if use_random:
            instance_vector = mylinker.linking_resources["random_instance_embedding"][0]
        else:
            return None

    return instance_vector


def to_network(
    mylinker,
    processed_df,
    level: str,
    level_id: str,
    max_distance: int,
    similarity_threshold: float,
    whichsplit: str = "originalsplit",
    for_plotting: bool = False,
) -> pd.DataFrame:

    """
    turns a unit of level (mention, sentence, article) into a network
    each candidate is a node, the place of publication is added as node as well
    nodes are represented by their entity embedding, confidence and relevance
    nodes are connected edges are close and / or belong to similar classes
    """

    # values over this threshold will be included
    distance_threshold = 1 / max_distance

    # get all the rows for a given id at specified level
    rows = processed_df[processed_df[level] == level_id].reset_index()

    # create place of publication as anchor point
    publication_wqid = str(rows.iloc[0].place_wqid)
    publication_wqid_lat = mylinker.linking_resources["dict_wqid_to_lat"][publication_wqid]
    publication_wqid_lon = mylinker.linking_resources["dict_wqid_to_lon"][publication_wqid]
    point1 = (publication_wqid_lat, publication_wqid_lon)

    pop_wikidata_id = publication_wqid + "_POP"

    df_candidates = []

    # each row represents one mention in the textual unit
    for query_id, row in rows.iterrows():
        candidatesList = []
        # iterate over the candidates in each row
        for mention, candidate_dict in row.candidates.items():
            # values that are stable across the candidates
            candidates = candidate_dict["Candidates"]
            confidence = candidate_dict["Score"]  # we use
            for c in candidates:
                candidatesList.append(
                    (
                        query_id,
                        row["gold_entity_link"],
                        c,
                        candidates[c],  #
                        mylinker.linking_resources["mentions_to_wikidata"][mention][c],
                        confidence,
                    )
                )

            _df = pd.DataFrame(
                candidatesList,
                columns=[
                    "mention_id",
                    "gold_entity_link",
                    "wikidata_id",
                    "relevance_1",
                    "relevance_2",
                    "confidence",
                ],
            )  # .sort_values('relevance_1',ascending=False)[:10]

            dTypes = {"LOC": 0, "BUILDING": 1, "STREET": 2}
            ## !! Also change line below, integrate choice of split as argument
            ## !! now we only use original split
            _df["split"] = row[whichsplit]
            # _df["entitytype"] = dTypes[row["pred_ner_label"]]
            _df["mention"] = row.matched
            # normalize relevance between 0 and 1
            _df["relevance_2"] = _df["relevance_2"] / _df["relevance_2"].max()
            # IMPORTANT: maybe change this later
            # take only the most relevant candidates
            # possibly change from here ---->
            _df_sel = _df[(_df.relevance_1 > 0.8) | (_df.relevance_2 > 0.8)].reset_index()
            if _df_sel is None:
                _df_sel = _df.sort_values("relevance_2", ascending=False)[:10]
            # <----- change till here
            _df_sel["distance"] = _df_sel.wikidata_id.apply(
                lambda x: haversine(
                    point1,
                    (
                        mylinker.linking_resources["dict_wqid_to_lat"][x],
                        mylinker.linking_resources["dict_wqid_to_lon"][x],
                    ),
                )
            )
            # normalize distance by converting it to proximity
            # which is one over the distance for entities within a 1 km range
            # the proximity will be one, otherwise the values will decrease
            # this ensure that proximity is also between 0 and
            _df_sel["proximity"] = _df_sel.distance.apply(
                lambda x: 1.0 / max(x, 1.0)
            )  # 1 / ( max(_df['distance'],1))
            df_candidates.append(_df_sel)

    if not df_candidates:
        return None

    df_candidates = pd.concat(df_candidates).reset_index()
    df_candidates["y"] = 0
    correct = np.where(df_candidates["wikidata_id"] == df_candidates["gold_entity_link"])[0]
    df_candidates.iloc[correct, -1] = 1
    G = nx.MultiGraph()
    nodes = []

    for i, row in df_candidates.iterrows():
        x = get_entity_vector(mylinker, row.wikidata_id).astype(np.float32)
        x_ext = row[["relevance_1", "relevance_2", "confidence"]].values.astype(np.float32)
        nodes.append(
            (
                row.wikidata_id,
                {
                    "y": row.y,
                    "x": x,
                    "x_ext": x_ext,
                    "split": row["split"],
                    "mention_id": row.mention_id + 1,
                    "mention": row.mention,
                    "correct_id": row.gold_entity_link,
                    "name": row.wikidata_id,
                },
            )
        )

    nodes.append(
        (
            pop_wikidata_id,
            {
                "y": 0,
                "x": np.array([0.0] * len(x)),
                "x_ext": np.array([0.0] * len(x_ext)),
                "correct_id": "NaN",
                "mention_id": 0,
                "mention": "NaN",
                "split": row["split"],
                "name": pop_wikidata_id,
            },
        )
    )
    G.add_nodes_from(nodes)

    # add nodes wrt to distance from place of publication
    for i, row in df_candidates.iterrows():
        if for_plotting:
            G.add_edge(
                row.wikidata_id,
                pop_wikidata_id,
                color="r",
                weight=sum([2, row.proximity, 0.0, 0.0]),
            )
        else:
            G.add_edge(
                row.wikidata_id,
                pop_wikidata_id,
                weight=[2, row.proximity, 0.0, 0.0],
            )

    # we can make this faster using matrix computations
    # i.e. like pdist(candidates[['longitude','latitude']].values, metric=get_distance)
    # possibly change later
    combs = combinations(df_candidates.wikidata_id.unique(), 2)

    for s, e in combs:
        color = "b"
        # only connext nodes across queries
        if (
            df_candidates[df_candidates.wikidata_id == s].mention_id.values[0]
            == df_candidates[df_candidates.wikidata_id == e].mention_id.values[0]
        ):
            weight = [1, 0.0, 0.0, 0.0]
            color = "y"
        else:
            # print(color)
            cand_s_lat = mylinker.linking_resources["dict_wqid_to_lat"][s]
            cand_s_lon = mylinker.linking_resources["dict_wqid_to_lon"][s]
            cand_e_lat = mylinker.linking_resources["dict_wqid_to_lat"][e]
            cand_e_lon = mylinker.linking_resources["dict_wqid_to_lon"][e]
            distance = haversine((cand_s_lat, cand_s_lon), (cand_e_lat, cand_e_lon))
            proximity = 1 / (distance + 0.0001)
            inst_vector_s = get_instance_vector(mylinker, s, use_random=False)
            inst_vector_e = get_instance_vector(mylinker, e, use_random=False)
            if (inst_vector_s is not None) and (inst_vector_e is not None):
                similarity = 1 - cosine(inst_vector_s, inst_vector_e)
            else:
                similarity = 0.0

            weight = [
                0,
                0.0,
                proximity if proximity > distance_threshold else 0.0,
                similarity if similarity > similarity_threshold else 0.0,
            ]

        # only add edges if distance or similarity meet the threshold value
        if sum(weight) > 0.0:
            if for_plotting:
                G.add_edge(s, e, color=color, weight=sum(weight))
            else:
                G.add_edge(s, e, weight=weight)

    return G


def network_data(
    mylinker,
    processed_df,
    whichsplit: str = "originalsplit",
    level: str = "sentence_id",
    max_distance: int = 200,
    similarity_threshold: float = 0.7,
):
    # print(processed_df)
    processed_df["matched"] = processed_df.candidates.apply(lambda x: "_".join(x.keys()))
    # print(processed_df)
    level = mylinker.gnn_params["level"]
    max_distance = mylinker.gnn_params["max_distance"]
    similarity_threshold = mylinker.gnn_params["similarity_threshold"]
    units = processed_df[level].unique()
    graph_data = []
    for u in tqdm(units):
        graph = to_network(
            mylinker,
            processed_df,
            level,
            u,
            max_distance,
            similarity_threshold,
            whichsplit,
        )
        if graph:
            graph_data.append(from_networkx(graph))

    data_batch = pyg_data.Batch().from_data_list(graph_data)
    return data_batch


def get_model_predictions(mylinker, test_df, whichsplit):
    level = mylinker.gnn_params["level"]
    prediction_dict = defaultdict(dict)
    unit_ids = test_df[level].unique()

    for unit_id in tqdm(unit_ids):

        try:
            inputs = from_networkx(
                to_network(
                    mylinker,
                    test_df,
                    level,
                    unit_id,
                    max_distance=mylinker.gnn_params["max_distance"],
                    similarity_threshold=mylinker.gnn_params["similarity_threshold"],
                    whichsplit=whichsplit,
                )
            )

        except AttributeError:
            prediction_dict[unit_id] = {}
            continue

        probs = mylinker.gnn_params["model"](
            inputs.x.float(),
            inputs.x_ext.float(),
            inputs.edge_index,
            inputs.weight.float(),
        ).softmax(dim=1)

        pred = probs.argmax(dim=1)

        pred_idx = np.where(pred > 0)[0]

        for name, mention, conf in list(
            zip(
                np.array(inputs.name)[pred_idx],
                np.array(inputs.mention)[pred_idx],
                probs[pred_idx, 1].detach().numpy(),
            )
        ):
            prediction_dict[unit_id][mention] = (name, conf)

        test_df["pred_wqid"] = "NIL"
        test_df["pred_wqid_score"] = 0.0

    for unit_id, preds in prediction_dict.items():
        for mention, score in preds.items():
            print(mention, score)
            test_df.loc[
                (test_df[level] == unit_id) & (test_df["matched"] == mention),
                "pred_wqid",
            ] = score[0]
            test_df.loc[
                (test_df[level] == unit_id) & (test_df["matched"] == mention),
                "pred_wqid_score",
            ] = score[1]
    return test_df
