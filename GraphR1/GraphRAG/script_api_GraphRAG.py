# import argparse
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.responses import JSONResponse
# from contextlib import asynccontextmanager
# import uvicorn
# from pathlib import Path
# import pandas as pd

# import graphrag.api as api
# from graphrag.config.load_config import load_config
# from utils import process_context_data


# # ======================================================
# # ① 读取命令行参数
# # ======================================================
# parser = argparse.ArgumentParser(description="GraphRAG FastAPI Server")

# parser.add_argument(
#     "--project_dir",
#     type=str,
#     required=True,
#     help="Directory of the GraphRAG project (contains output/*.parquet)"
# )
# parser.add_argument(
#     "--community_level",
#     type=int,
#     default=2,
#     help="Community level for global/local/drift search"
# )
# parser.add_argument(
#     "--claim_extraction",
#     type=int,
#     default=0,
#     help="Enable claim extraction (1=True, 0=False)"
# )
# parser.add_argument(
#     "--response_type",
#     type=str,
#     default="Multiple Paragraphs",
#     help="Response type for GraphRAG (tree/json/markdown/etc.)"
# )
# parser.add_argument(
#     "--port",
#     type=int,
#     default=8000,
#     help="Server port"
# )

# args = parser.parse_args()

# PROJECT_DIRECTORY = args.project_dir
# COMMUNITY_LEVEL = args.community_level
# CLAIM_EXTRACTION_ENABLED = bool(args.claim_extraction)
# RESPONSE_TYPE = args.response_type
# SERVER_PORT = args.port


# # ======================================================
# # ② lifespan: 启动时一次性加载所有 GraphRAG 数据
# # ======================================================
# @asynccontextmanager
# async def lifespan(app: FastAPI):

#     print(f"[Startup] Loading GraphRAG config and parquet data from {PROJECT_DIRECTORY}")

#     app.state.config = load_config(Path(PROJECT_DIRECTORY))

#     app.state.entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
#     app.state.communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
#     app.state.community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
#     app.state.text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
#     app.state.relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")

#     if CLAIM_EXTRACTION_ENABLED:
#         app.state.covariates = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/covariates.parquet")
#     else:
#         app.state.covariates = None

#     print("[Startup] GraphRAG data loaded successfully.")

#     yield

#     print("[Shutdown] FastAPI server shutting down.")


# app = FastAPI(lifespan=lifespan)


# # ======================================================
# # ③ GraphRAG Search APIs
# # ======================================================

# @app.get("/search/global")
# async def global_search(query: str = Query(..., description="Global Search")):
#     try:
#         response, context = await api.global_search(
#             config=app.state.config,
#             entities=app.state.entities,
#             communities=app.state.communities,
#             community_reports=app.state.community_reports,
#             community_level=COMMUNITY_LEVEL,
#             dynamic_community_selection=False,
#             response_type=RESPONSE_TYPE,
#             query=query,
#         )
#         return JSONResponse({
#             "response": response,
#             "context_data": process_context_data(context),
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/search/local")
# async def local_search(query: str = Query(..., description="Local Search")):
#     try:
#         response, context = await api.local_search(
#             config=app.state.config,
#             entities=app.state.entities,
#             communities=app.state.communities,
#             community_reports=app.state.community_reports,
#             text_units=app.state.text_units,
#             relationships=app.state.relationships,
#             covariates=app.state.covariates,
#             community_level=COMMUNITY_LEVEL,
#             response_type=RESPONSE_TYPE,
#             query=query,
#         )
#         return JSONResponse({
#             "response": response,
#             "context_data": process_context_data(context),
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/search/drift")
# async def drift_search(query: str = Query(..., description="DRIFT Search")):
#     try:
#         response, context = await api.drift_search(
#             config=app.state.config,
#             entities=app.state.entities,
#             communities=app.state.communities,
#             community_reports=app.state.community_reports,
#             text_units=app.state.text_units,
#             relationships=app.state.relationships,
#             community_level=COMMUNITY_LEVEL,
#             response_type=RESPONSE_TYPE,
#             query=query,
#         )
#         return JSONResponse({
#             "response": response,
#             "context_data": process_context_data(context),
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/search/basic")
# async def basic_search(query: str = Query(..., description="Basic Search")):
#     try:
#         response, context = await api.basic_search(
#             config=app.state.config,
#             text_units=app.state.text_units,
#             query=query,
#         )
#         return JSONResponse({
#             "response": response,
#             "context_data": process_context_data(context),
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/status")
# async def status():
#     return JSONResponse({"status": "Server is up and running"})


# # ======================================================
# # ④ 主入口：启动 Uvicorn
# # ======================================================
# if __name__ == "__main__":
#     print(f"Starting GraphRAG API Server | Port = {SERVER_PORT}")

#     uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import asyncio

from pathlib import Path
import graphrag.api as api
from graphrag.config.load_config import load_config


# ======================================================
# 1）加载命令行参数（保持不变）
# ======================================================
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


# ======================================================
# 2）⭐ 全局加载 GraphRAG 数据（无 app.state）
# ======================================================
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


# ======================================================
# 3）FastAPI 主应用
# ======================================================
app = FastAPI()


# ======================================================
# 4）Global Batch Search（同步版，稳定）
# ======================================================
class SearchRequest(BaseModel):
    queries: list[str]


@app.post("/search")
def global_batch_search(request: SearchRequest):
    print(f"[Request] Received {len(request.queries)} queries for global batch search.")
    results = []

    for q in request.queries:
        try:
            print(f"[Processing] Query: {q}")
            # 在同步代码中执行 global_search
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


# ======================================================
# 5）健康检查
# ======================================================
@app.get("/status")
def status():
    return {"status": "OK"}


# ======================================================
# 6）启动服务器
# ======================================================
if __name__ == "__main__":
    print(f"Starting GraphRAG API on port {SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)


