from fastapi import FastAPI
import json
import numpy as np
import faiss
from json_repair import repair_json
from sentence_transformers import SentenceTransformer
from google import genai

app = FastAPI()

# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(
    api_key="AIzaSyDubjv8Aq9H8XgCop3rEPj3gniA2lsMHTg"
)

# =========================
# EMBEDDING MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text):
    return model.encode(text)


# =========================
# LOAD + FIX JSON
# =========================
def load_catalog():
    with open("data/shl_product_catalog.json", "r", encoding="utf-8") as f:
        raw = f.read()

    fixed = repair_json(raw)
    data = json.loads(fixed)

    processed = []

    for item in data:
        text = " ".join([
            item.get("name", ""),
            " ".join(item.get("keys", [])),
            item.get("job_levels_raw", "") if isinstance(item.get("job_levels_raw", ""), str) else ""
        ]).lower()

        processed.append({
            "name": item.get("name"),
            "url": item.get("url", item.get("link", "")),
            "keys": item.get("keys", []),
            "job_levels_raw": item.get("job_levels_raw", ""),
            "text": text
        })

    return processed


catalog = load_catalog()

# =========================
# BUILD INDEX
# =========================
vectors = np.array([embed(x["text"]) for x in catalog]).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)


# =========================
# HELPERS
# =========================
def last_user(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def is_vague(query):
    return len(query.split()) < 4


def keyword_score(item, query):
    q = query.lower()
    return sum(1 for w in q.split() if w in item["text"])


# =========================================================
# 🔥 PATCH 1: ROLE-AWARE SCORING (ADDED, NOT REPLACED)
# =========================================================
def role_boost_score(item, query):
    q = query.lower()
    text = item["text"]

    score = keyword_score(item, query)

    # role boosts
    if "senior" in q or "lead" in q:
        if "advanced" in text or "verify" in text or "leadership" in text:
            score += 2

    if "stakeholder" in q:
        if "behavior" in text or "opq" in text or "communication" in text:
            score += 2

    if "java" in q:
        if "java" in text:
            score += 3

    if "developer" in q:
        if "coding" in text or "programming" in text:
            score += 2

    return score


# =========================
# RETRIEVAL (PATCHED ONLY)
# =========================
def retrieve(query):
    vec = embed(query).astype("float32").reshape(1, -1)
    _, ids = index.search(vec, 30)

    results = []

    for i in ids[0]:
        if i == -1:
            continue

        item = catalog[i]

        # 🔥 PATCHED SCORING (REPLACED LOGIC ONLY)
        score = role_boost_score(item, query)

        # 🔥 PATCH 2: HARD FILTER (added)
        if score < 2:
            continue

        results.append((item, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in results[:10]]


# =========================
# GEMINI (EXPLANATION ONLY - UNCHANGED)
# =========================
def explain(query, results):
    top = "\n".join([f"- {r['name']}" for r in results])

    prompt = f"""
You are an SHL assessment expert.

User query:
{query}

Selected assessments:
{top}

Explain briefly why these fit.
DO NOT add new assessments.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# =========================
# API (UNCHANGED LOGIC)
# =========================
@app.post("/chat")
def chat(payload: dict):
    messages = payload.get("messages", [])
    query = last_user(messages)

    # vague check
    if is_vague(query):
        return {
            "reply": "Could you share job role, experience level, or required skills?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # retrieval
    results = retrieve(query)

    if not results:
        return {
            "reply": "I couldn't find relevant assessments. Please refine your query.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # explanation
    reply = explain(query, results)

    return {
        "reply": reply,
        "recommendations": [
            {
                "name": r["name"],
                "url": r["url"],
                "test_type": r.get("test_type", "Knowledge & Skills")
            }
            for r in results
        ],
        "end_of_conversation": False
    }