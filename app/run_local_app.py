import os
import sys
import time
from pathlib import Path
from typing import Union

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

if "toponym-resolution" in __file__:
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    root_path = os.path.dirname(os.path.abspath(__file__))
experiments_path = Path(root_path, "experiments")
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(experiments_path))
os.chdir(experiments_path)

os.environ['APP_CONFIG_NAME'] = "t-res_deezy_reldisamb-wpubl-wmtops"
import importlib
config_mod = importlib.import_module('.t-res_deezy_reldisamb-wpubl-wmtops', 'app.configs')
pipeline_config = config_mod.CONFIG


from geoparser import pipeline

geoparser = pipeline.Pipeline(**pipeline_config)


class APIQuery(BaseModel):
    sentence: str
    place: Union[str, None] = None
    place_wqid: Union[str, None] = None


class FullTextAPIQuery(BaseModel):
    text: str
    place: Union[str, None] = None
    place_wqid: Union[str, None] = None


app_config_name = os.environ["APP_CONFIG_NAME"]
app = FastAPI(title=f"Toponym Resolution Pipeline API ({app_config_name})")


@app.get("/")
async def read_root(request: Request):
    return {
        "Title": request.app.title,
        "request.url": request.url,
        "request.query_params": request.query_params,
        "root_path": request.scope.get("root_path"),
        "request.client": request.client,
        "hostname": os.uname()[1],
        "worker_id": os.getpid(),
    }


@app.get("/test")
async def test_pipeline():
    resolved = geoparser.run_sentence(
        "Harvey, from London;Thomas and Elizabeth, Barnett.",
        place="Manchester",
        place_wqid="Q18125",
    )

    return resolved


@app.get("/toponym_resolution")
async def run_pipeline(api_query: APIQuery, request_id: Union[str, None] = None):
    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    resolved = geoparser.run_sentence(
        api_query.sentence, place=api_query.place, place_wqid=api_query.place_wqid
    )

    return resolved


@app.get("/resolve_full_text")
async def run_text(api_query: FullTextAPIQuery):
    print(api_query)
    print(api_query.text)
    print(api_query.place)
    print(api_query.place_wqid)
    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    resolved = geoparser.run_text(
        api_query.text, place=api_query.place, place_wqid=api_query.place_wqid
    )

    return resolved


@app.get("/ner")
async def run_ner(api_query: APIQuery):
    
    predictions = geoparser.myner.ner_predict(api_query.sentence)

    procpreds = [
        [x["word"], x["entity"], "O", x["start"], x["end"], x["score"]]
        for x in predictions
    ]

    # Aggregate mentions:
    mentions = geoparser.myner.aggregate_mentions(procpreds, "pred")

    return mentions





@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    # poetry run uvicorn app.run_local_app:app --port 8123
    uvicorn.run(app, host="0.0.0.0", port=8123)
