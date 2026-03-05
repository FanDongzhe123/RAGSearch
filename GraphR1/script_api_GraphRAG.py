from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import asyncio

from pathlib import Path
import graphrag.api as api
from graphrag.config.load_config import load_config


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--project_dir", type=str, required=True)
parser.add_argument("--community_level", type=int, default=2)
parser.add_argument("--claim_extraction", type=int, default=0)
parser.add_argument("--response_type", type=str, default="Multiple Paragraphs")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

PROJECT_DIRECTORY = args.project_dir
COMMUNITY_LEVEL = args.community_level
CLAIM_EXTRACTION_ENABLED = bool(args.claim_extraction)
RESPONSE_TYPE = args.response_type
SERVER_PORT = args.port


print(f"[Startup] Loading GraphRAG config and data from: {PROJECT_DIRECTORY}")

CONFIG = load_config(Path(PROJECT_DIRECTORY))
ENTITIES = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
COMMUNITIES = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
COMMUNITY_REPORTS = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
TEXT_UNITS = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
RELATIONSHIPS = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")
COVARIATES = (
    pd.read_parquet(f"{PROJECT_DIRECTORY}/output/covariates.parquet")
    if CLAIM_EXTRACTION_ENABLED else None
)

print("[Startup] GraphRAG data loaded.")



app = FastAPI()



class SearchRequest(BaseModel):
    queries: list[str]


@app.post("/search")
def global_batch_search(request: SearchRequest):
    print(f"[Request] Received {len(request.queries)} queries for global batch search.")
    results = []

    for q in request.queries:
        try:
            print(f"[Processing] Query: {q}")
            response, context = asyncio.run(
                api.global_search(
                    config=CONFIG,
                    entities=ENTITIES,
                    communities=COMMUNITIES,
                    community_reports=COMMUNITY_REPORTS,
                    community_level=COMMUNITY_LEVEL,
                    dynamic_community_selection=False,
                    response_type=RESPONSE_TYPE,
                    query=q,
                )
            )
            results.append({"results": response})

        except Exception as e:
            results.append({"results": f"ERROR: {str(e)}"})

    return JSONResponse(results)


@app.get("/status")
def status():
    return {"status": "OK"}


if __name__ == "__main__":
    print(f"Starting GraphRAG API on port {SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)


