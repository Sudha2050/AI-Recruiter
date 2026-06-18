#!/usr/bin/env python3
"""
Main ranking pipeline. Expects data/raw/candidates.jsonl.gz and data/raw/job_description.docx.
Precomputed embeddings must exist at data/embeddings/candidate_embeddings.npy.
Usage: python rank.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import settings
from src.ingestion.loader import load_candidates
from src.features.jd_parser import parse_jd_docx
from src.ranking.stage1_filter import honeypot_penalty
from src.ranking.stage2_coarse import coarse_score
from src.ranking.stage3_semantic import semantic_rank
from src.ranking.stage4_fine import fine_score
from src.ranking.reasoner import generate_reasoning


def main():
    start = time.time()

    # --- Load candidates ---
    print("Loading candidates...")
    candidates = load_candidates(settings.CANDIDATES_FILE)
    print(f"Loaded {len(candidates)} candidates.")

    # --- Load precomputed embeddings ---
    emb_path = settings.DATA_EMBEDDINGS / "candidate_embeddings.npy"
    candidate_embs = np.load(emb_path)
    print(f"Loaded embeddings shape: {candidate_embs.shape}")

    # --- Parse JD and embed it ---
    print("Embedding JD...")
    jd_text = parse_jd_docx(settings.JD_FILE)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    jd_emb = model.encode(jd_text, convert_to_numpy=True)
    jd_norm = jd_emb / np.linalg.norm(jd_emb)

    # --- Stage 1: Honeypot filtering (pre-compute for all) ---
    print("Stage 1: Honeypot filtering...")
    for i, cand in enumerate(candidates):
        candidates[i]['_honeypot'] = honeypot_penalty(cand)

    # --- Stage 2: Coarse scoring (fast heuristic, keeps top 5K) ---
    print("Stage 2: Coarse ranking...")
    coarse_scores = [coarse_score(cand) for cand in candidates]
    coarse_indices = np.argsort(coarse_scores)[::-1][:settings.TOP_K_COARSE]
    coarse_candidates = [candidates[i] for i in coarse_indices]
    coarse_embs = candidate_embs[coarse_indices]

    # --- Stage 3: Semantic ranking via precomputed embeddings (top 500) ---
    print("Stage 3: Semantic ranking...")
    sem_indices, sem_scores = semantic_rank(coarse_embs, jd_norm, settings.TOP_K_SEMANTIC)
    sem_candidates = [coarse_candidates[i] for i in sem_indices]
    sem_scores_list = sem_scores.tolist()

    # --- Stage 4: Fine ranking with LightGBM + behavioral signals ---
    print("Stage 4: Fine ranking...")
    final_scores = []
    for idx, cand in enumerate(sem_candidates):
        semantic = sem_scores_list[idx]
        fine = fine_score(cand, semantic)
        final = fine if cand['_honeypot'] > 0.0 else 0.0
        final_scores.append((cand['candidate_id'], final, cand))

    # Convert final to 0–100 scale
    scaled_scores = []
    for cand_id, final, cand in final_scores:
        scaled = final * 100.0
        scaled_scores.append((cand_id, scaled, cand))

    # Sort by scaled score descending, then candidate_id ascending
    scaled_scores.sort(key=lambda x: (-round(x[1], 4), x[0]))
    top100 = scaled_scores[:100]

    # Generate output rows
    rows = []
    for rank, (cand_id, score, cand) in enumerate(top100, 1):
        reasoning = generate_reasoning(cand, rank)
        rows.append({
            'candidate_id': cand_id,
            'rank': rank,
            'score': round(score, 2),   # now 0–100 with 2 decimals
            'reasoning': reasoning
        })

    out_path = Path("outputs") / "submission.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    print(f"Total runtime: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()