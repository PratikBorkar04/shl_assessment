import faiss
import numpy as np

from utils.embeddings import get_embedding


# global storage
index = None
metadata = []


def build_search_text(item: dict) -> str:
    return f"""
Name: {item.get('name', '')}
Description: {item.get('description', '')}
Job Levels: {', '.join(item.get('job_levels', []))}
Categories: {', '.join(item.get('keys', []))}
Remote: {item.get('remote', '')}
Adaptive: {item.get('adaptive', '')}
"""


def build_faiss_index(catalog: list):
    global index, metadata

    vectors = []
    metadata = []

    for item in catalog:
        text = build_search_text(item)
        vec = get_embedding(text)

        vectors.append(vec)
        metadata.append(item)

    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    print(f"FAISS built with {len(metadata)} items")


def get_index():
    return index, metadata