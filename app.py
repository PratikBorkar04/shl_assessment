from fastapi import FastAPI
import json
import numpy as np
import faiss

from json_repair import repair_json
from sentence_transformers import SentenceTransformer

app = FastAPI()

# =========================
# MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)


# =========================
# GLOBALS
# =========================
index = None
metadata = []


def set_index(faiss_index, meta):
    global index, metadata
    index = faiss_index
    metadata = meta


def get_index():
    return index, metadata


# =========================
# LOAD CATALOG
# =========================
def load_catalog():
    try:
        with open("data/shl_product_catalog.json", "r", encoding="utf-8") as f:
            raw = f.read()

        fixed = repair_json(raw)
        data = json.loads(fixed)

        cleaned = []

        for item in data:
            cleaned.append({
                "name": item.get("name", ""),
                "link": item.get("link", ""),
                "keys": item.get("keys", []),
                "job_levels": item.get("job_levels", []),
                "description": item.get("description", ""),

                "text": " ".join([
                    item.get("name", ""),
                    item.get("description", ""),
                    " ".join(item.get("keys", [])),
                    " ".join(item.get("job_levels", []))
                ]).lower()
            })

        return cleaned

    except Exception as e:
        print("❌ Catalog load error:", e)
        return []


catalog = load_catalog()


# =========================
# BUILD FAISS INDEX
# =========================
def build_index(data):
    vectors = []

    for item in data:
        vectors.append(get_embedding(item["text"]))

    vectors = np.array(vectors).astype("float32")

    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(vectors)

    set_index(faiss_index, data)


if catalog:
    build_index(catalog)
    print(f"✅ FAISS built with {len(catalog)} items")
else:
    print("⚠️ Catalog empty")


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# UTILS
# =========================
def extract_last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# =========================
# INTENT DETECTION
# =========================
def classify_intent(messages):
    last_user = extract_last_user(messages).lower().strip()

    vague_patterns = [
        "need assessment",
        "need an assessment",
        "i need assessment",
        "i need an assessment",
        "looking for assessment",
        "suggest assessment",
        "help me choose",
        "not sure",
        "what assessment"
    ]

    if any(p in last_user for p in vague_patterns):
        return "vague"

    refine_patterns = ["add", "remove", "instead", "change", "include", "exclude"]

    if any(p in last_user for p in refine_patterns):
        return "refine"

    if "vs" in last_user or "compare" in last_user or "difference" in last_user:
        return "compare"

    return "search"


# =========================
# SOFT FILTER
# =========================
def job_level_soft_filter(item, query):
    q = query.lower()
    levels = " ".join(item.get("job_levels", [])).lower()

    if "entry" in q:
        return "entry" in levels

    if "mid" in q:
        return "mid" in levels

    if "senior" in q or "lead" in q:
        return "manager" in levels or "executive" in levels

    return True


# =========================
# KEYWORD OVERLAP SCORE (NEW)
# =========================
def keyword_overlap_score(query, item):
    q = set(query.lower().split())

    text = " ".join([
        item.get("name", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", []))
    ]).lower()

    t = set(text.split())

    if not t:
        return 0

    return len(q.intersection(t)) / len(q)


# =========================
# SEARCH ENDPOINT
# =========================
@app.post("/search")
def search(payload: dict):
    query = payload.get("query", "")
    top_k = payload.get("top_k", 10)

    index, metadata = get_index()

    vec = get_embedding(query).astype("float32").reshape(1, -1)
    _, ids = index.search(vec, top_k * 3)

    candidates = []

    for i in ids[0]:
        if i == -1:
            continue

        item = metadata[i]

        if not job_level_soft_filter(item, query):
            continue

        score = keyword_overlap_score(query, item)
        candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = [
        {
            "name": item["name"],
            "url": item["link"],
            "job_levels": item["job_levels"],
            "keys": item["keys"]
        }
        for score, item in candidates[:top_k]
    ]

    return {
        "query": query,
        "results": results
    }


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
def chat(payload: dict):
    messages = payload.get("messages", [])

    query = extract_last_user(messages)

    if not query:
        return {
            "reply": "Please provide role or assessment requirements.",
            "recommendations": [],
            "end_of_conversation": False
        }

    intent = classify_intent(messages)

    # VAGUE → ASK CLARIFICATION
    if intent == "vague":
        return {
            "reply": "Could you share more details like job role, seniority level, or required skills?",
            "recommendations": [],
            "end_of_conversation": False
        }

    index, metadata = get_index()

    vec = get_embedding(query).astype("float32").reshape(1, -1)
    _, ids = index.search(vec, 25)

    candidates = []

    for i in ids[0]:
        if i == -1:
            continue

        item = metadata[i]

        if not job_level_soft_filter(item, query):
            continue

        score = keyword_overlap_score(query, item)
        candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = [item for score, item in candidates[:10]]

    # REFINE
    if intent == "refine":
        return {
            "reply": "Updated recommendations based on your new requirements.",
            "recommendations": [
                {
                    "name": r["name"],
                    "url": r["link"],
                    "test_type": r["keys"][0] if r["keys"] else "General"
                }
                for r in results
            ],
            "end_of_conversation": False
        }

    # COMPARE
    if intent == "compare":
        return {
            "reply": "Please specify two SHL assessments to compare.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # SEARCH
    return {
        "reply": f"Here are {len(results)} SHL assessments matching your requirement.",
        "recommendations": [
            {
                "name": r["name"],
                "url": r["link"],
                "test_type": r["keys"][0] if r["keys"] else "General"
            }
            for r in results
        ],
        "end_of_conversation": False
    }