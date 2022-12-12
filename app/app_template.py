import os
import sys
from pathlib import Path
from typing import Union
import time
from fastapi import FastAPI, Request
import uvicorn
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
from geoparser import pipeline

geoparser = pipeline.Pipeline(**pipeline_config)


class APIQuery(BaseModel):
    sentence: str
    place: Union[str, None] = None
    place_wqid: Union[str, None] = None


app_config_name = os.environ["APP_CONFIG_NAME"]
app = FastAPI(title=f"Toponym Resolution Pipeline API ({app_config_name})")

@app.get("/")
async def read_root(request: Request):
    
    return {"Title": request.app.title,
            "request.url": request.url,
            "request.query_params": request.query_params,
            "root_path": request.scope.get("root_path"),
            "request.client": request.client,
            "hostname": os.uname()[1],
            "worker_id": os.getpid()
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