import os
import sys
from pathlib import Path
from typing import Union
import time
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
import asyncio

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
experiments_path = Path(root_path, "experiments")
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(experiments_path))
os.chdir(experiments_path)
from geoparser import pipeline, ranking, linking

myranker = ranking.Ranker(
    method="deezymatch",
    resources_path="../resources/wikidata/",
    mentions_to_wikidata=dict(),
    wikidata_to_mentions=dict(),
    wiki_filtering={
        "top_mentions": 3,  # Filter mentions to top N mentions
        "minimum_relv": 0.03,  # Filter mentions with more than X relv
    },
    strvar_parameters={
        # Parameters to create the string pair dataset:
        "ocr_threshold": 60,
        "top_threshold": 85,
        "min_len": 5,
        "max_len": 15,
    },
    deezy_parameters={
        # Paths and filenames of DeezyMatch models and data:
        "dm_path": str(Path("outputs/deezymatch/").resolve()),
        "dm_cands": "wkdtalts",
        "dm_model": "w2v_ocr",
        "dm_output": "deezymatch_on_the_fly",
        # Ranking measures:
        "ranking_metric": "faiss",
        "selection_threshold": 25,
        "num_candidates": 3,
        "search_size": 3,
        "verbose": False,
        # DeezyMatch training:
        "overwrite_training": False,
        "w2v_ocr_path": str(Path("outputs/models/").resolve()),
        "w2v_ocr_model": "w2v_*_news",
        "do_test": False,
    },
)

mylinker = linking.Linker(
    method="mostpopular",
    resources_path="../resources/wikidata/",
    linking_resources=dict(),
    base_model="to-be-removed",  # Base model for vector extraction
    rel_params={},
    overwrite_training=False,
)

geoparser = pipeline.Pipeline(myranker=myranker, mylinker=mylinker)


class APIQuery(BaseModel):
    sentence: str
    place: Union[str, None] = None
    place_wqid: Union[str, None] = None


app_config_name = "Deezy_MostPopular"
app = FastAPI(title=f"Toponym Resolution Pipeline API ({app_config_name})")

@app.get("/")
async def read_root(request: Request):
    
    return {"Title": request.app.title,
            "request.url": request.url,
            "request.query_params": request.query_params,
            "root_path": request.scope.get("root_path"),
            "request.client": request.client,
            "hostname": os.uname()[1]
            }

@app.get("/test")
async def test_pipeline():

    resolved = geoparser.run_sentence("Harvey, from London;Thomas and Elizabeth, Barnett.", place="Manchester", place_wqid="Q18125")
    
    return resolved

@app.get("/toponym_resolution/")
async def run_pipeline(api_query: APIQuery, request_id: Union[str, None] = None):

    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    resolved = geoparser.run_sentence(api_query.sentence,
                                      place=api_query.place, 
                                      place_wqid=api_query.place_wqid)
    
    return resolved

@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)