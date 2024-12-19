import importlib
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from t_res.geoparser import pipeline
from t_res.utils.dataclasses import SentenceMentions, Candidates

os.environ["APP_CONFIG_NAME"] = "t-res_deezy_reldisamb-wpubl-wmtops"

config_mod = importlib.import_module(
    ".t-res_deezy_reldisamb-wpubl-wmtops", "app.configs"
)
pipeline_config = config_mod.CONFIG

geoparser = pipeline.Pipeline(**pipeline_config)


class APIQuery(BaseModel):
    text: str


class CandidatesAPIQuery(BaseModel):
    sentence_mentions: List[dict]
    place_of_pub_wqid: Optional[str] = None
    place_of_pub: Optional[str] = None


class DisambiguationAPIQuery(BaseModel):
    candidates: dict


class PipelineAPIQuery(BaseModel):
    text: str
    place_of_pub_wqid: Optional[str] = None
    place_of_pub: Optional[str] = None


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

@app.get("/run_ner")
async def run_ner(api_query: APIQuery):
    ner_output = geoparser.run_text_recognition(
        api_query.text
    )
    return ner_output

@app.get("/run_candidate_selection")
async def run_candidate_selection(cand_api_query: CandidatesAPIQuery):
    sentence_mentions = SentenceMentions.from_json(cand_api_query.sentence_mentions)
    candidates = geoparser.run_candidate_selection(
        sentence_mentions,
        place_of_pub_wqid=cand_api_query.place_of_pub_wqid,
        place_of_pub=cand_api_query.place_of_pub,
        )
    return candidates

@app.get("/run_disambiguation")
async def run_disambiguation(api_query: DisambiguationAPIQuery):
    candidates = Candidates.from_dict(api_query.candidates)
    predictions = geoparser.run_disambiguation(candidates)
    return predictions

@app.get("/run_pipeline")
async def run_pipeline(api_query: PipelineAPIQuery):
    predictions = geoparser.run(
        text=api_query.text,
        place_of_pub_wqid=api_query.place_of_pub_wqid,
        place_of_pub=api_query.place_of_pub,
    )
    return predictions

@app.get("/test")
async def test_pipeline():
    predictions = geoparser.run(
        "Harvey, from London;Thomas and Elizabeth, Barnett.",
        place_of_pub_wqid="Q18125",
        place_of_pub="Manchester",
    )
    return predictions

@app.get("/health")
async def healthcheck():
    return {"status": "ok"}

if __name__ == "__main__":
    # poetry run uvicorn app.run_local_app:app --port 8123
    uvicorn.run(app, host="0.0.0.0", port=8123)
