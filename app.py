from fastapi import FastAPI
import json
import numpy as np
from json_repair import repair_json
from utils.retrieval import build_faiss_index, get_index
from utils.embeddings import get_embedding

app = FastAPI()


# -----------------------------
# SAFE CATALOG LOADING
# -----------------------------
def load_catalog():
    try:
        with open("data/shl_product_catalog.json", "r", encoding="utf-8") as f:
            raw = f.read()

        # FIX BROKEN JSON AUTOMATICALLY
        fixed = repair_json(raw)

        return json.loads(fixed)

    except Exception as e:
        print("❌ Failed to load catalog even after repair:", e)
        return []

catalog = load_catalog()


# -----------------------------
# BUILD FAISS ONLY IF VALID DATA
# -----------------------------
if catalog and len(catalog) > 0:
    build_faiss_index(catalog)
    print(f"✅ FAISS built with {len(catalog)} items")
else:
    print("⚠️ Catalog empty or invalid. FAISS NOT built.")


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# DEBUG SEARCH ENDPOINT
# -----------------------------
@app.post("/search")
def search(payload: dict):
    query = payload.get("query", "")
    top_k = payload.get("top_k", 5)

    index, metadata = get_index()

    if index is None:
        return {
            "error": "FAISS index not available (check catalog load)"
        }

    query_vec = get_embedding(query).astype("float32").reshape(1, -1)

    scores, ids = index.search(query_vec, top_k)

    results = []

    for i, score in zip(ids[0], scores[0]):
        if i == -1:
            continue

        item = metadata[i]

        results.append({
            "name": item.get("name"),
            "url": item.get("link"),
            "score": float(score),
            "job_levels": item.get("job_levels", []),
            "keys": item.get("keys", [])
        })

    return {
        "query": query,
        "results": results
    }