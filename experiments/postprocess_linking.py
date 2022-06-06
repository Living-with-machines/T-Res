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

output_path_csv = "../experiments/outputs/newspapers/csvs/"
output_path_resolved = "../experiments/outputs/newspapers/resolved/"
output_path_georesolved = "../experiments/outputs/newspapers/georesolved/"
Path(output_path_georesolved).mkdir(parents=True, exist_ok=True)

folder = output_path_resolved + "*/*.json"
for i in glob.glob(folder):
    publication = i.split("/")[-2]

    if "metadata" in i:
        metadata_df = pd.read_json(i, orient="index")

        with open(i.replace("_metadata", "_toponyms")) as json_file:
            toponyms_data = json.load(json_file)

        rows = []
        for line in toponyms_data:
            metadata_info = metadata_df.loc[int(line)]
            for t in toponyms_data[line]:
                wqid = t["wqid"]
                lat = wqid_to_latitude.get(t["wqid"], None)
                lon = wqid_to_longitude.get(t["wqid"], None)
                wq_score = t["wqid_score"]

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
                "year",
                "month",
                "day",
                "date",
            ],
            data=rows,
        )

        if publication in dict_dataframes:
            dict_dataframes[publication].append(df)
        else:
            dict_dataframes[publication] = [df]

Path(output_path_georesolved).mkdir(parents=True, exist_ok=True)
for publ in dict_dataframes:
    results_df = pd.concat(dict_dataframes[publ])
    results_df.to_csv(output_path_georesolved + publ + ".csv")
