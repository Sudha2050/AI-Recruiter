import numpy as np
import faiss
from typing import Tuple

def compute_similarities(embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    query = query_embedding.astype(np.float32).reshape(1, -1)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if top_k:
        scores, indices = index.search(query, top_k)
    else:
        scores, indices = index.search(query, embeddings.shape[0])
    return indices[0], scores[0]