#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config import settings
from src.ingestion.loader import load_candidates
from src.features.jd_parser import parse_jd_docx
from src.features.career_features import company_tier
from src.features.skill_features import skill_depth_score
from src.features.signal_features import compute_behavioral_multiplier
from src.ranking.stage1_filter import honeypot_penalty
from sentence_transformers import SentenceTransformer


def has_complex_projects(career_history: list) -> float:
    complex_keywords = [
        'predictive mechanism', 'supply chain', 'orchestrator', 'orchestrate', 
        'feedback loop', 'sentiment analysis', 'predict sentiment', 'end-to-end',
        'architecture', 'scalable system', 'deployed', 'optimized', 'infrastructure',
        'production ml', 'pipeline design', 'distributed', 'reduced latency', 'saved cost'
    ]
    
    score_bonus = 0.0
    for job in career_history:
        desc = job.get('description', '').lower()
        matches = sum(1 for kw in complex_keywords if kw in desc)
        if matches > 0:
            score_bonus += min(0.15, matches * 0.05)
            
    return min(0.20, score_bonus)


def build_features(candidate, semantic_score):
    """Return feature vector for training/inference."""
    profile = candidate['profile']
    title = profile.get('current_title', '').lower()
    title_score = 0.5
    if any(term in title for term in ['ml', 'ai', 'engineer', 'scientist']):
        title_score = 1.0
    elif any(term in title for term in ['marketing', 'hr', 'operations']):
        title_score = 0.2

    company = profile.get('current_company', '')
    ctype = company_tier(company)
    company_score = {'product':1.0, 'other':0.7, 'consulting':0.4, 'fictional':0.0}.get(ctype, 0.5)

    exp = profile.get('years_of_experience', 0)
    exp_score = 1.0 if 5 <= exp <= 9 else (0.7 if 4 <= exp < 5 else (0.6 if 9 < exp <= 12 else 0.4))

    skill_score = skill_depth_score(candidate.get('skills', []), exp)

    education = candidate.get('education', [])
    tier_w = {'tier_1':1.0, 'tier_2':0.8, 'tier_3':0.6, 'tier_4':0.4, 'unknown':0.5}
    edu_score = max([tier_w.get(e.get('tier', 'unknown'), 0.5) for e in education] + [0.3])

    mult = compute_behavioral_multiplier(candidate.get('redrob_signals', {}))
    hp = honeypot_penalty(candidate)

    features = [
        title_score,
        company_score,
        exp_score,
        skill_score,
        edu_score,
        semantic_score,
        mult,
        hp,
    ]
    return np.array(features)


def main():
    print("Loading candidates...")
    candidates = load_candidates(settings.CANDIDATES_FILE)
    print(f"Loaded {len(candidates)} candidates.")

    print("Loading precomputed embeddings...")
    embeddings = np.load(settings.DATA_EMBEDDINGS / "candidate_embeddings.npy")

    print("Parsing JD and computing JD embedding...")
    jd_text = parse_jd_docx(settings.JD_FILE)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    jd_emb = model.encode(jd_text, convert_to_numpy=True)
    jd_norm = jd_emb / np.linalg.norm(jd_emb)

    # Compute semantic similarity for all candidates
    print("Computing semantic similarities...")
    semantic_scores = np.dot(embeddings.astype(np.float32), jd_norm.astype(np.float32))
    semantic_scores = np.clip(semantic_scores, 0, 1)

    print("Building feature matrix and targets...")
    X = []
    y = []

    for i, cand in enumerate(candidates):
        sem = semantic_scores[i]
        features = build_features(cand, sem)

        # Target = comprehensive score used in inference
        # (0.55*semantic + 0.45*structured) * behavioral_multiplier * honeypot
        structured_part = (
            0.15 * features[0] +   # title
            0.15 * features[1] +   # company
            0.10 * features[2] +   # experience
            0.15 * features[3] +   # skills
            0.05 * features[4]     # education
        )
        
        exp = cand['profile'].get('years_of_experience', 0)
        project_bonus = 0.0
        if 5.0 <= exp <= 9.0:
            project_bonus = has_complex_projects(cand.get('career_history', []))
            
        normalized_structured = structured_part / 0.60
        normalized_structured = min(1.0, normalized_structured + project_bonus)
        structured_part = normalized_structured * 0.60

        base = 0.55 * sem + 0.45 * structured_part
        mult = features[6]
        hp = features[7]
        target = base * mult * hp
        target = max(0.0, min(1.0, target))

        X.append(features)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training LightGBM model...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    # Save model – ensure directory exists
    model_path = settings.DATA_MODELS / "lgb_ranker.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"Model saved to {model_path}")

    y_pred = lgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")


if __name__ == "__main__":
    main()