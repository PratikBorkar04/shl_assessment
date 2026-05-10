from fastapi import FastAPI
import json
import re
import os

from json_repair import repair_json
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai

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
@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "running"}


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

    with open(
        "data/shl_product_catalog.json",
        "r",
        encoding="utf-8"
    ) as f:

        raw = f.read()

    fixed = repair_json(raw)

    data = json.loads(fixed)

    processed = []

    for item in data:

        text = " ".join([
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            str(item.get("job_levels_raw", ""))
        ]).lower()

        processed.append({
            "name": item.get("name"),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
            "text": text
        })

    return processed


catalog = load_catalog()

# =========================
# TF-IDF SETUP
# =========================
documents = [item["text"] for item in catalog]

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(documents)


# =========================
# HELPERS
# =========================
def full_user_context(messages):

    return " ".join([
        m["content"]
        for m in messages
        if m["role"] == "user"
    ])


def last_user(messages):

    for m in reversed(messages):

        if m["role"] == "user":
            return m["content"]

    return ""


# =========================
# VAGUE QUERY DETECTION
# =========================
VAGUE_TERMS = [
    "assessment",
    "test",
    "job",
    "hiring",
    "candidate"
]


def is_vague(query):

    q = query.lower()

    if len(q.split()) < 4:
        return True

    if q.strip() in VAGUE_TERMS:
        return True

    return False


# =========================
# BLOCKED QUERIES
# =========================
BLOCKED_TOPICS = [
    "legal",
    "lawsuit",
    "fire someone",
    "terminate employee",
    "salary negotiation",
    "court",
    "ignore previous instructions",
    "prompt injection",
    "bypass system",
    "jailbreak"
]


def is_blocked_query(query):

    q = query.lower()

    return any(
        topic in q
        for topic in BLOCKED_TOPICS
    )


# =========================
# COMPARE DETECTION
# =========================
COMPARE_WORDS = [
    "compare",
    "difference",
    "vs",
    "versus"
]


def is_compare_query(query):

    q = query.lower()

    return any(
        word in q
        for word in COMPARE_WORDS
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
# FIND ASSESSMENTS IN QUERY
# =========================
def find_matching_assessments(query):

    q = query.lower()

    matches = []

    for item in catalog:

        if item["name"] and item["name"].lower() in q:
            matches.append(item)

    return matches[:2]


# =========================
# COMPARE RESPONSE
# =========================
def compare_assessments(query):

    matches = find_matching_assessments(query)

    if len(matches) < 2:

        return {
            "reply": "Please specify two SHL assessments you want to compare.",
            "recommendations": [],
            "end_of_conversation": False
        }

    a = matches[0]
    b = matches[1]

    prompt = f"""
You are an SHL assessment assistant.

Compare these two SHL assessments ONLY using the provided catalog information.

Assessment 1:
Name: {a['name']}
Description: {a['description']}

Assessment 2:
Name: {b['name']}
Description: {b['description']}

Rules:
- Use only provided information
- Do not invent details
- Keep comparison concise
- Mention purpose and differences
- No markdown tables
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "reply": clean_text(response.text),

        "recommendations": [
            {
                "name": a["name"],
                "url": a["url"],
                "test_type": get_test_type(a["name"])
            },
            {
                "name": b["name"],
                "url": b["url"],
                "test_type": get_test_type(b["name"])
            }
        ],

        "end_of_conversation": False
    }


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
# RETRIEVAL
# =========================
def retrieve(query):

    query_vec = vectorizer.transform([query.lower()])

    similarities = cosine_similarity(
        query_vec,
        tfidf_matrix
    )[0]

    skill_scores = detect_skill_score(query)

    results = []

    for idx, score in enumerate(similarities):

        item = catalog[idx]

        skill_boost = 0

        for skill, val in skill_scores.items():

            if val > 0 and skill in item["text"]:

                skill_boost += 0.05 * val

        final_score = score + skill_boost

        if final_score < 0.10:
            continue

        results.append(
            (item, final_score)
        )

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
- No legal advice
- Stay strictly within SHL assessments
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

    user_turns = len([
        m for m in messages
        if m["role"] == "user"
    ])

    if user_turns >= 8:

        return {
            "reply": "We have reached the maximum number of turns for this session.",
            "recommendations": [],
            "end_of_conversation": True
        }

    query = full_user_context(messages)

    latest_query = last_user(messages)

    # =========================
    # BLOCKED QUERIES
    # =========================
    if is_blocked_query(latest_query):

        return {
            "reply": "I can only help with SHL assessments and related evaluation queries.",
            "recommendations": [],
            "end_of_conversation": True
        }

    # =========================
    # COMPARE MODE
    # =========================
    if is_compare_query(latest_query):

        return compare_assessments(latest_query)

    # =========================
    # VAGUE QUERY
    # =========================
    if is_vague(latest_query):

        return {
            "reply": "Could you share the role, experience level, or skills required for the assessment recommendation?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # =========================
    # RETRIEVE RESULTS
    # =========================
    results = retrieve(query)

    if not results:

        return {
            "reply": "I could not find relevant SHL assessments for this query. Please refine the role or required skills.",
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