from fastapi import FastAPI
import json
import numpy as np
import faiss
from json_repair import repair_json
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# =========================
# LOAD ENV
# =========================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
    api_key="API KEY"
)

# =========================
# EMBEDDING MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text):
    return model.encode(text)


# =========================
# LOAD DATA
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
            str(item.get("job_levels_raw", ""))
        ]).lower()

        processed.append({
            "name": item.get("name"),
            "url": item.get("url", ""),
            "text": text
        })

    return processed


catalog = load_catalog()

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


# =========================
# 🔥 UNIVERSAL SKILL EXTRACTION (NO HARD CODING)
# =========================
SKILL_SEED = {
    "backend": ["api", "server", "microservices", "backend", "database"],
    "frontend": ["ui", "frontend", "react", "html", "css"],
    "data": ["data", "analytics", "sql", "etl", "pipeline"],
    "cloud": ["aws", "azure", "cloud", "devops", "kubernetes"],
    "leadership": ["lead", "manager", "stakeholder", "team", "ownership"],
    "testing": ["qa", "testing", "automation", "selenium"]
}


def detect_skill_score(query):
    q = query.lower()
    scores = {}

    for skill, keywords in SKILL_SEED.items():
        scores[skill] = sum(1 for k in keywords if k in q)

    return scores


# =========================
# UNIVERSAL SCORING (CORE FIX)
# =========================
def score_item(item, query):
    q_vec = embed(query)
    item_vec = embed(item["text"])

    semantic_score = float(
        cosine_similarity([q_vec], [item_vec])[0][0]
    )

    skill_scores = detect_skill_score(query)

    # boost based on matching skill hints
    skill_boost = 0
    for skill, val in skill_scores.items():
        if val > 0 and skill in item["text"]:
            skill_boost += 0.05 * val

    return semantic_score + skill_boost


# =========================
# RETRIEVAL (UNIVERSAL)
# =========================
def retrieve(query):
    results = []

    for item in catalog:
        score = score_item(item, query)
        results.append((item, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in results[:10]]


# =========================
# LLM (EXPLANATION ONLY)
# =========================
def explain(query, results):
    top = "\n".join([f"- {r['name']}" for r in results])

    prompt = f"""
You are an SHL assessment expert.

User query:
{query}

Selected assessments:
{top}

Explain briefly why these are relevant.
Do NOT add new assessments.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# =========================
# API
# =========================
@app.post("/chat")
def chat(payload: dict):
    messages = payload.get("messages", [])
    query = last_user(messages)

    if is_vague(query):
        return {
            "reply": "Could you share job role, experience level, or skills required?",
            "recommendations": [],
            "end_of_conversation": False
        }

    results = retrieve(query)

    reply = explain(query, results)

    return {
        "reply": reply,
        "recommendations": [
            {
                "name": r["name"],
                "url": r["url"],
                "test_type": "Knowledge & Skills"
            }
            for r in results
        ],
        "end_of_conversation": False
    }