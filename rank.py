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

import argparse
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
    parser = argparse.ArgumentParser(description="AI Recruiter CLI Ranker")
    parser.add_argument("--candidates", type=str, default=str(settings.CANDIDATES_FILE), help="Path to candidates file")
    parser.add_argument("--out", type=str, default="outputs/submission.csv", help="Path to output CSV file")
    parser.add_argument("--jd", type=str, default=str(settings.JD_FILE), help="Path to Job Description docx file")
    args = parser.parse_args()

    # --- File existence checks ---
    candidates_path = Path(args.candidates)
    jd_path = Path(args.jd)

    if not candidates_path.exists():
        sys.exit(f"CRITICAL ERROR: Candidates file not found at '{candidates_path}'. Please check your file path.")
    if not jd_path.exists():
        sys.exit(f"CRITICAL ERROR: Job Description file not found at '{jd_path}'. Please check your file path.")

    start = time.time()

    # --- Load candidates ---
    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates.")

    # --- Load or compute candidate embeddings ---
    emb_path = settings.DATA_EMBEDDINGS / "candidate_embeddings.npy"
    model = None
    if emb_path.exists():
        candidate_embs = np.load(emb_path)
        if len(candidate_embs) == len(candidates):
            print(f"Loaded precomputed candidate embeddings of shape: {candidate_embs.shape}")
        else:
            print(f"Precomputed embeddings shape {candidate_embs.shape} does not match candidates count {len(candidates)}.")
            print("Generating candidate embeddings on-the-fly...")
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            from src.embeddings.builder import aggregate_profile_text
            texts = [aggregate_profile_text(c) for c in candidates]
            raw_embs = model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            candidate_embs = (raw_embs / norms).astype(np.float16)
    else:
        print("Precomputed candidate embeddings not found. Generating on-the-fly...")
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        from src.embeddings.builder import aggregate_profile_text
        texts = [aggregate_profile_text(c) for c in candidates]
        raw_embs = model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        candidate_embs = (raw_embs / norms).astype(np.float16)

    # --- Parse JD and embed it ---
    print("Embedding JD...")
    jd_text = parse_jd_docx(Path(args.jd))
    if model is None:
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
    jd_emb = model.encode(jd_text, convert_to_numpy=True)
    jd_norm = jd_emb / np.linalg.norm(jd_emb)

    # --- Stage 1: Honeypot filtering (pre-compute for all and filter) ---
    print("Stage 1: Honeypot filtering...")
    filtered_candidates = []
    keep_indices = []
    for i, cand in enumerate(candidates):
        hp = honeypot_penalty(cand)
        if hp > 0.0:  # Remove hard-disqualified honeypots from candidate pool immediately
            cand['_honeypot'] = hp
            filtered_candidates.append(cand)
            keep_indices.append(i)
    candidates = filtered_candidates
    candidate_embs = candidate_embs[keep_indices]
    print(f"Loaded {len(candidates)} clean candidates and aligned embeddings.")

    # --- Stage 2: Coarse scoring (fast heuristic, keeps top 5K) ---
    print("Stage 2: Coarse ranking...")
    coarse_scores = [coarse_score(cand) for cand in candidates]
    
    # Adjust top K coarse to handle small candidate datasets
    top_k_coarse = min(settings.TOP_K_COARSE, len(candidates))
    coarse_indices = np.argsort(coarse_scores)[::-1][:top_k_coarse]
    coarse_candidates = [candidates[i] for i in coarse_indices]
    coarse_embs = candidate_embs[coarse_indices]

    # --- Stage 3: Semantic ranking via precomputed/computed embeddings (top 500) ---
    print("Stage 3: Semantic ranking...")
    top_k_semantic = min(settings.TOP_K_SEMANTIC, len(coarse_candidates))
    sem_indices, sem_scores = semantic_rank(coarse_embs, jd_norm, top_k_semantic)
    sem_candidates = [coarse_candidates[i] for i in sem_indices]
    sem_scores_list = sem_scores.tolist()

    # --- Stage 4: Fine ranking with dynamic category weights ---
    print("Stage 4: Fine ranking...")
    final_scores = []
    for idx, cand in enumerate(sem_candidates):
        semantic = sem_scores_list[idx]
        fine = fine_score(cand, semantic)
        final = fine  # since hard honeypots are already filtered out, and soft penalties are applied in fine_score
        final_scores.append((cand['candidate_id'], final, cand))

    # Convert final to 0–100 scale (with clamping to [0.0, 100.0] range)
    scaled_scores = []
    for cand_id, final, cand in final_scores:
        scaled = max(0.0, min(100.0, final * 100.0))
        scaled_scores.append((cand_id, scaled, cand))

    # Sort by scaled score descending (using 2-decimal rounded precision), then candidate_id ascending
    scaled_scores.sort(key=lambda x: (-round(x[1], 2), x[0]))
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    print(f"Total runtime: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()