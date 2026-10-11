import os
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import SearchEngineManager
from evaluation import GROUND_TRUTH, calculate_metrics

app = FastAPI(
    title="Academic Paper Information Retrieval API",
    description="Backend API for Boolean, TF-IDF, BM25, and Semantic Search",
    version="1.0.0"
)

# Global dataset and search engine manager
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
current_dataset_name = "sample_data.csv"
DATASET_PATH = os.path.join(BASE_DIR, current_dataset_name)

df = None
engine = None

def load_dataset(filename):
    global current_dataset_name, DATASET_PATH, df, engine
    target_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(target_path):
        target_path = os.path.join(BASE_DIR, "sample_data.csv")
        filename = "sample_data.csv"

    print(f"Loading dataset from: {target_path}")
    current_dataset_name = filename
    DATASET_PATH = target_path
    df = pd.read_csv(DATASET_PATH)
    engine = SearchEngineManager(df)

# Initial load (default to sample_data.csv or arxiv_data.csv if preferred)
initial_file = "sample_data.csv" if os.path.exists(os.path.join(BASE_DIR, "sample_data.csv")) else "arxiv_data.csv"
load_dataset(initial_file)

# Mount frontend directory for static assets
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.post("/api/dataset/switch")
async def switch_dataset(mode: str = Query("sample", description="sample | full")):
    target_file = "sample_data.csv" if mode == "sample" else "arxiv_data.csv"
    load_dataset(target_file)
    return {
        "status": "success",
        "dataset": current_dataset_name,
        "total_papers": len(df)
    }

@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API is running. Frontend files are being configured."}

@app.get("/api/stats")
async def get_stats():
    avg_len = int(df['abstract'].str.split().str.len().mean()) if 'abstract' in df.columns else 0
    return {
        "dataset_name": os.path.basename(DATASET_PATH),
        "total_papers": len(df),
        "avg_abstract_words": avg_len,
        "available_methods": ["boolean", "tfidf", "bm25", "semantic"],
        "status": "ready"
    }

@app.get("/api/papers")
async def get_papers(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    start = (page - 1) * limit
    end = start + limit
    sliced = df.iloc[start:end]
    
    papers = []
    for idx, row in sliced.iterrows():
        papers.append({
            "index": int(idx),
            "id": str(row.get('id', idx + 1)),
            "title": str(row.get('title', '')),
            "abstract": str(row.get('abstract', ''))
        })
        
    return {
        "total": len(df),
        "page": page,
        "limit": limit,
        "papers": papers
    }

@app.get("/api/search")
async def search_papers(
    query: str = Query(..., min_length=1, description="Search query string"),
    method: str = Query("semantic", description="boolean | tfidf | bm25 | semantic | compare_all"),
    top_k: int = Query(10, ge=1, le=50)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    if method == "compare_all":
        all_results = engine.search_all(query, top_k=top_k)
        return {
            "query": query,
            "mode": "compare_all",
            "results": all_results
        }
    else:
        results = engine.search(query, method=method, top_k=top_k)
        return {
            "query": query,
            "method": method,
            "total_matches": len(results),
            "results": results
        }

@app.get("/api/evaluate")
async def get_evaluation():
    try:
        sample_path = os.path.join(BASE_DIR, "sample_data.csv")
        eval_df = pd.read_csv(sample_path)
    except Exception:
        eval_df = df

    eval_engine = SearchEngineManager(eval_df)
    metrics_summary = []

    for model_name, model in eval_engine.models.items():
        total_p1, total_p3, total_r3, total_f1 = 0, 0, 0, 0
        n = len(GROUND_TRUTH)

        for q, relevant_ids in GROUND_TRUTH.items():
            relevant_indices = [i - 1 for i in relevant_ids]
            raw_results = model.search(q)
            retrieved_indices = [idx for idx, _ in raw_results[:3]]

            # P@1
            p1 = 1.0 if (retrieved_indices and retrieved_indices[0] in relevant_indices) else 0.0
            # P@3, R@3, F1
            p3, r3, f1 = calculate_metrics(retrieved_indices, relevant_indices)

            total_p1 += p1
            total_p3 += p3
            total_r3 += r3
            total_f1 += f1

        metrics_summary.append({
            "model": model_name.upper(),
            "precision_1": round(total_p1 / n, 4),
            "precision_3": round(total_p3 / n, 4),
            "recall_3": round(total_r3 / n, 4),
            "f1_score": round(total_f1 / n, 4)
        })

    return {
        "eval_dataset": "sample_data.csv",
        "benchmark_queries_count": len(GROUND_TRUTH),
        "metrics": metrics_summary
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
