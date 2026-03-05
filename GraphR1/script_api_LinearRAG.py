import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import argparse
import os
import threading
import torch
import time
from tqdm import tqdm

# =============== LinearRAG 导入 ===============
import sys
sys.path.append("/scratch/df2362/Graph-R1/LinearRAG")  # 必须包含 src 路径
from LinearRAG.src.LinearRAG import LinearRAG
from LinearRAG.src.config import LinearRAGConfig
from LinearRAG.src.utils import LLM_Model, setup_logging
from sentence_transformers import SentenceTransformer
# ============================================


# =============== Parameter parsing =====================
parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default="hotpotqa_5000_full")
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--node_scale', type=int, default=5000)
parser.add_argument('--spacy_model', default="en_core_web_trf")
parser.add_argument('--embedding_model', default="sentence-transformers/all-mpnet-base-v2")
parser.add_argument('--llm_model', default="deepseek-chat")
parser.add_argument('--max_workers', type=int, default=16)
parser.add_argument('--working_dir', default="./LinearRAG/graphrags")
args = parser.parse_args()
# ==============================================



# =============== Load LinearRAG =================
embedding_model = SentenceTransformer(args.embedding_model, device="cuda")
print("[DEBUG] Embedding Model LOADED")

llm_model = LLM_Model(args.llm_model)

config = LinearRAGConfig(
    dataset_name=args.data_source,
    embedding_model=embedding_model,
    spacy_model=args.spacy_model,
    llm_model=llm_model,
    max_workers=args.max_workers,
    working_dir=args.working_dir
)

rag = LinearRAG(global_config=config)
print("[DEBUG] LinearRAG LOADED")



# ====================== Batch RAG query processing ======================

def process_query_batch(query_list, rag_instance):
    """
    Input: List[str] - Fully compatible with HippoRAG
    """


    results = rag_instance.retrieve(query_list)



    return results


def queries_to_results(queries: List[str]) -> List[str]:
    """
    Output all passage content (no format conversion)
    """
    batch_result = process_query_batch(queries, rag)

    outputs = []

    for i, q in enumerate(queries):
        item = batch_result[i]
        sorted_passages = item["sorted_passage"]

        retrieve_result = ""
        for passage in sorted_passages:
            retrieve_result += f"{passage}\n"

        print(retrieve_result)
        outputs.append(json.dumps({"results": retrieve_result}))

    return outputs


# ======================= FastAPI service =======================

app = FastAPI(
    title="Search API of LinearRAG (Batch Mode)",
    description="API for batch document retrieval using LinearRAG."
)

class SearchRequest(BaseModel):
    queries: List[str]

@app.post("/search")
def search(request: SearchRequest):
    return queries_to_results(request.queries)


if __name__ == "__main__":
    print(f"🚀 Starting LinearRAG Batch API service, port: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
