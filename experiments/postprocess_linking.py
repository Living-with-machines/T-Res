import re
import glob
import json
import pandas as pd
from pathlib import Path

gaz = pd.read_csv(
    "/resources/wikidata/wikidata_gazetteer.csv",
    low_memory=False,
    usecols=["latitude", "longitude", "wikidata_id"],
)
wqid_to_latitude = dict(zip(gaz.wikidata_id, gaz.latitude))
wqid_to_longitude = dict(zip(gaz.wikidata_id, gaz.longitude))

dict_dataframes = dict()

# TODO CHange this
query = "noquery"
output_path_csv = "../experiments/outputs/newspapers/csvs/"
output_path_resolved = "../experiments/outputs/newspapers/resolved/"
output_path_georesolved = "../experiments/outputs/newspapers/georesolved/"
Path(output_path_georesolved).mkdir(parents=True, exist_ok=True)

folder = output_path_resolved + "*/*.json"
all_publications = set()
for i in glob.glob(folder):
    publication = i.split("/")[-2]
    all_publications.add(publication)

# Generate
for publication in all_publications:
    # TODO CHange this
    if not "randsample" in publication:
        continue
    publication_dfs = []
    for i in glob.glob(folder):
        current_publication = i.split("/")[-2]

        if current_publication == publication:
            if "metadata" in i:
                with open(i) as json_file:
                    metadata_dict = json.load(json_file)

                with open(i.replace("_metadata", "_toponyms")) as json_file:
                    toponyms_data = json.load(json_file)

                rows = []
                for line in toponyms_data:
                    metadata_info = metadata_dict[str(line)]
                    for t in toponyms_data[line]:
                        wqid = t["wqid"]
                        lat = wqid_to_latitude.get(t["wqid"], None)
                        lon = wqid_to_longitude.get(t["wqid"], None)
                        wq_score = t["wqid_score"]

                        # TODO CHange this
                        # # This can be removed after rerunning apply_pipeline:
                        # if not query in sentence_txt.lower():
                        #     continue

                        rows.append(
                            [
                                t["mention"],
                                t["ner_label"],
                                wqid,
                                lat,
                                lon,
                                wq_score,
                                metadata_info["article_id"],
                                metadata_info["nlp_id"],
                                metadata_info["sentence_id"],
                                metadata_info["sentence"],
                                metadata_info["year"],
                                metadata_info["month"],
                                metadata_info["day"],
                                metadata_info["date"],
                            ]
                        )

                df = pd.DataFrame(
                    columns=[
                        "mention",
                        "ner_label",
                        "wkdt_id",
                        "latitude",
                        "longitude",
                        "el_score",
                        "path",
                        "publication",
                        "sentence_order",
                        "sentence",
                        "year",
                        "month",
                        "day",
                        "date",
                    ],
                    data=rows,
                )

                publication_dfs.append(df)

    results_df = pd.concat(publication_dfs)
    results_df.to_csv(output_path_georesolved + publication + ".csv")
