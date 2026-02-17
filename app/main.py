from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
from .search import vector_search
from .reranker import rerank
from .data import documents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search")
def search(data: dict):
    start_time = time.time()

    query = data["query"]
    k = data.get("k", 10)
    rerank_flag = data.get("rerank", True)
    rerank_k = data.get("rerankK", 6)

    initial_results = vector_search(query, k)

    if rerank_flag:
        final_results = rerank(query, initial_results, rerank_k)
    else:
        final_results = initial_results[:rerank_k]

    results = []

    for doc, score in final_results:
        results.append({
            "id": doc["id"],
            "score": round(float(score), 3),
            "content": doc["content"],
            "metadata": doc["metadata"]
        })

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": results,
        "reranked": rerank_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
