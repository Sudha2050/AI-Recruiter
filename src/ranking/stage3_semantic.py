import numpy as np
from typing import Tuple

def semantic_rank(candidate_embeddings: np.ndarray, jd_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    emb = candidate_embeddings.astype(np.float32)
    query = jd_embedding.astype(np.float32)
    scores = np.dot(emb, query)  # cosine since normalized
    scores = np.clip(scores, 0, 1)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]