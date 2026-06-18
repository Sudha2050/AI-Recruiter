#!/usr/bin/env python3
"""
Offline precomputation of candidate embeddings.
Run this before the ranking step. Can take 30-60 minutes but runs once.
"""

import sys
from pathlib import Path

# Add project root to sys.path so that 'config' and 'src' are found
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Now these imports will work
from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from src.ingestion.loader import load_candidates
from src.embeddings.builder import aggregate_profile_text


def main():
    print("Loading candidates...")
    # Ensure the data file exists at this path
    candidates_file = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
    candidates = load_candidates(candidates_file)
    print(f"Loaded {len(candidates)} candidates.")

    print("Aggregating profile texts...")
    texts = [aggregate_profile_text(c) for c in tqdm(candidates)]

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings (this may take a while)...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Normalize to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = (embeddings / norms).astype(np.float16)

    output_path = PROJECT_ROOT / "data" / "embeddings" / "candidate_embeddings.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_norm)

    print(f"Saved {len(embeddings_norm)} embeddings (shape {embeddings_norm.shape}) to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()