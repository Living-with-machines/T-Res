import os
import sys
import time
from pathlib import Path
from typing import Union, Optional, List

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

from config import CONFIG as pipeline_config

from t_res.geoparser import pipeline

geoparser = pipeline.Pipeline(**pipeline_config)


class APIQuery(BaseModel):
    text: str
    place: Optional[Union[str, None]] = None
    place_wqid: Optional[Union[str, None]] = None


class CandidatesAPIQuery(BaseModel):
    toponyms: List[dict]


class DisambiguationAPIQuery(BaseModel):
    dataset: List[dict]
    wk_cands: dict
    place: Optional[Union[str, None]] = None
    place_wqid: Optional[Union[str, None]] = None


app_config_name = os.environ["APP_CONFIG_NAME"]
app = FastAPI(title=f"Toponym Resolution Pipeline API ({app_config_name})")


@app.get("/")
async def read_root(request: Request):
    return {"Welcome to T-Res!": request.app.title}


@app.get("/test")
async def test_pipeline():
    resolved = geoparser.run_sentence(
        "Harvey, from London;Thomas and Elizabeth, Barnett.",
        place="Manchester",
        place_wqid="Q18125",
    )

    return resolved


@app.get("/resolve_sentence")
async def run_sentence(api_query: APIQuery, request_id: Union[str, None] = None):
    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    resolved = geoparser.run_sentence(
        api_query.text, place=place, place_wqid=place_wqid
    )

    return resolved


@app.get("/resolve_full_text")
async def run_text(api_query: APIQuery):

    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    resolved = geoparser.run_text(api_query.text, place=place, place_wqid=place_wqid)

    return resolved


@app.get("/run_ner")
async def run_ner(api_query: APIQuery):

    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    ner_output = geoparser.run_text_recognition(
        api_query.text, place=place, place_wqid=place_wqid
    )

    return ner_output


@app.get("/run_candidate_selection")
async def run_candidate_selection(cand_api_query: CandidatesAPIQuery):

    wk_cands = geoparser.run_candidate_selection(cand_api_query.toponyms)
    return wk_cands


@app.get("/run_disambiguation")
async def run_disambiguation(api_query: DisambiguationAPIQuery):
    place = "" if api_query.place is None else api_query.place
    place_wqid = "" if api_query.place_wqid is None else api_query.place_wqid
    disamb_output = geoparser.run_disambiguation(
        api_query.dataset, api_query.wk_cands, place, place_wqid
    )
    return disamb_output


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
