import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(embeddings: np.ndarray, output_path: Path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(output_path))
    return index