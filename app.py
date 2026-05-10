from fastapi import FastAPI
import json
import numpy as np
import re
import os

from json_repair import repair_json
from google import genai
from dotenv import load_dotenv

app = FastAPI()

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "running"}


# =========================
# GEMINI EMBEDDING
# =========================
def embed(text):

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )

    return np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )


# =========================
# CLEAN GEMINI OUTPUT
# =========================
def clean_text(text):
    if not text:
        return text

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
# BLOCKED / REFUSAL QUERIES
# =========================
BLOCKED_TOPICS = [
    "legal",
    "lawsuit",
    "fire someone",
    "terminate employee",
    "salary negotiation",
    "court",
    "ignore previous instructions",
    "prompt injection"
]


def is_blocked_query(query):

    q = query.lower()

    return any(
        topic in q
        for topic in BLOCKED_TOPICS
    )


# =========================
# TEST TYPE MAPPING
# =========================
def get_test_type(name):

    n = name.lower()

    if "opq" in n or "personality" in n:
        return "Personality & Behavior"

    if "verify" in n or "reasoning" in n or "ability" in n:
        return "Ability & Aptitude"

    return "Knowledge & Skills"


# =========================
# UNIVERSAL SKILL SIGNALS
# =========================
SKILL_SEED = {
    "backend": [
        "api",
        "server",
        "microservices",
        "backend",
        "database"
    ],

    "frontend": [
        "ui",
        "frontend",
        "react",
        "html",
        "css"
    ],

    "data": [
        "data",
        "analytics",
        "sql",
        "etl",
        "pipeline"
    ],

    "cloud": [
        "aws",
        "azure",
        "cloud",
        "devops",
        "kubernetes"
    ],

    "leadership": [
        "lead",
        "manager",
        "stakeholder",
        "team",
        "ownership"
    ],

    "testing": [
        "qa",
        "testing",
        "automation",
        "selenium"
    ]
}


def detect_skill_score(query):

    q = query.lower()

    scores = {}

    for skill, keywords in SKILL_SEED.items():

        scores[skill] = sum(
            1 for k in keywords if k in q
        )

    return scores


# =========================
# COSINE SIMILARITY
# =========================
def cosine_sim(a, b):

    return float(
        np.dot(a, b) /
        (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )
    )


# =========================
# UNIVERSAL SCORING
# =========================
def score_item(item, query):

    q_vec = embed(query)
    item_vec = embed(item["text"])

    semantic_score = cosine_sim(
        q_vec,
        item_vec
    )

    skill_scores = detect_skill_score(query)

    skill_boost = 0

    for skill, val in skill_scores.items():

        if val > 0 and skill in item["text"]:
            skill_boost += 0.05 * val

    final_score = semantic_score + skill_boost

    return final_score


# =========================
# RETRIEVAL
# =========================
def retrieve(query):

    results = []

    for item in catalog:

        score = score_item(
            item,
            query
        )

        if score < 0.50:
            continue

        results.append((item, score))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [r[0] for r in results[:10]]


# =========================
# GEMINI EXPLANATION
# =========================
def explain(query, results):

    top = "\n".join([
        f"- {r['name']}"
        for r in results
    ])

    prompt = f"""
You are an SHL assessment recommendation assistant.

User Query:
{query}

Selected Assessments:
{top}

Explain briefly why these assessments fit the role.

Rules:
- Do NOT recommend new assessments
- Keep explanation concise
- No markdown
- No bullet nesting
- No legal advice
- No hiring strategy outside SHL assessments
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return clean_text(response.text)


# =========================
# CHAT API
# =========================
@app.post("/chat")
def chat(payload: dict):

    messages = payload.get("messages", [])

    # =========================
    # TURN LIMIT
    # =========================
    user_turns = len([
        m for m in messages
        if m["role"] == "user"
    ])

    if user_turns >= 8:

        return {
            "reply": "We have reached the maximum number of turns for this session. I hope these recommendations were helpful!",
            "recommendations": [],
            "end_of_conversation": True
        }

    query = last_user(messages)

    # =========================
    # BLOCKED QUERIES
    # =========================
    if is_blocked_query(query):

        return {
            "reply": "I can only help with SHL assessment recommendations and related hiring evaluation queries.",
            "recommendations": [],
            "end_of_conversation": True
        }

    # =========================
    # VAGUE QUERY
    # =========================
    if is_vague(query):

        return {
            "reply": "Could you share job role, experience level, or required skills?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # =========================
    # RETRIEVE RESULTS
    # =========================
    results = retrieve(query)

    if not results:

        return {
            "reply": "I could not find relevant SHL assessments for this query. Please refine the role or skills.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # =========================
    # GENERATE EXPLANATION
    # =========================
    reply = explain(
        query,
        results
    )

    # =========================
    # RESPONSE
    # =========================
    return {

        "reply": reply,

        "recommendations": [
            {
                "name": r["name"],
                "url": r["url"],
                "test_type": get_test_type(r["name"])
            }
            for r in results
        ],

        "end_of_conversation": False
    }