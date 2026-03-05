import json
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from typing import List
import argparse
import os
import threading
import torch
import time

# ================== Raptor 导入 ==================
from raptor import RetrievalAugmentation
# =====================================================

# ================== Parameter parsing =========================
parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default="HotpotQA", type=str, help='Data source name')
parser.add_argument('--port', type=int, default=8002, help='API service port number')
parser.add_argument('--node_scale', type=int, default=1000)
args = parser.parse_args()

data_source = args.data_source
node_scale = args.node_scale
data_path = f"./raptor/graphrags/{data_source}"
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

print("[DEBUG] Raptor API LOADED")

# =====================================================
# 🔥 Heartbeat mechanism - keep completely consistent with HippoRAG
# =====================================================





# =====================================================
# 🔥 Start heartbeat before loading Raptor
# =====================================================

print("[Load] Initializing Raptor... (this may take time)")
RA = RetrievalAugmentation(tree=data_path)


# =====================================================
# 🔥 Single retrieval + batch interface (completely mimic HippoRAG)
# =====================================================

def process_query_batch(query_list, ra_instance):
    """Process all queries in batch, but Raptor only supports single → call retrieve one by one"""
    start_time = time.time()
    results = []

    for q in query_list:
        context, _ = ra_instance.retrieve(question=q, top_k=2)

        results.append(context)
    end_time = time.time()
    print(f"[DEBUG] Batch retrieval time: {end_time - start_time:.2f} seconds")
    return results


def queries_to_results(queries: List[str]) -> List[str]:
    """Keep output format consistent with HippoRAG"""
    batch_result = process_query_batch(queries, RA)
    results = []

    for item in batch_result:
        results.append(json.dumps({"results": item}))

    return results


# ======================= API service =======================

app = FastAPI(
    title="Search API of Raptor (Batch Style API, Single Query Engine)",
    description="Raptor API that mimics HippoRAG batch API style but uses single-query retrieval."
)

class SearchRequest(BaseModel):
    queries: List[str]   # 🔥 Same as HippoRAG, it's List[str]

@app.post("/search")
def search(request: SearchRequest):
    return queries_to_results(request.queries)


# ======================= Startup entry =======================

if __name__ == "__main__":
    print(f"Starting Raptor API service, listening on port: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
