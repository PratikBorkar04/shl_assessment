# utils/embeddings.py

from sentence_transformers import SentenceTransformer

# Load model ONCE (important for FastAPI performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    """
    Convert text → normalized embedding vector
    """
    return model.encode(text, normalize_embeddings=True)