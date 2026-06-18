import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.ranking.stage1_filter import honeypot_penalty
from src.ranking.stage4_fine import fine_score
from src.ranking.reasoner import generate_reasoning


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_candidates_from_json(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of candidate objects.")

    normalized = []
    for item in data:
        if 'profile' not in item:
            normalized.append({'profile': item})
        else:
            normalized.append(item)
    return normalized


def aggregate_profile_text(candidate: dict) -> str:
    profile = candidate.get('profile', {})
    parts = [
        profile.get('headline', ''),
        profile.get('summary', '')
    ]
    for career in candidate.get('career_history', []):
        parts.append(career.get('title', ''))
        parts.append(career.get('description', ''))
    skills = candidate.get('skills', [])
    for skill in skills:
        if isinstance(skill, dict):
            parts.append(skill.get('name', ''))
        elif isinstance(skill, str):
            parts.append(skill)
    return ' '.join(parts)


def tokenize(text: str):
    return text.lower().split()


# ------------------------------------------------------------
# Ranking pipeline (LLM-powered re-ranking)
# ------------------------------------------------------------
def rank_candidates(jd_text: str, candidates: list, top_k: int = 100) -> pd.DataFrame:
    start = time.time()

    if not candidates:
        raise ValueError("No candidates provided.")

    # 1. BM25 retrieval (fast lexical)
    print("Building BM25 index...")
    corpus = [aggregate_profile_text(c) for c in candidates]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    jd_tokens = tokenize(jd_text)
    bm25_scores = bm25.get_scores(jd_tokens)

    top_k_coarse = min(2000, len(candidates))
    top_indices = np.argsort(bm25_scores)[::-1][:top_k_coarse]
    top_candidates = [candidates[i] for i in top_indices]
    top_texts = [corpus[i] for i in top_indices]

    # 2. LLM Re-ranking using Cross-Encoder
    print("Loading Cross-Encoder (LLM) for re-ranking...")
    cross_encoder = CrossEncoder(
        'cross-encoder/ms-marco-TinyBERT-L-2-v2',
        max_length=256
    )
    pairs = [(jd_text, text) for text in top_texts]
    cross_scores = cross_encoder.predict(
        pairs,
        batch_size=153,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    top_500_count = min(300, len(top_candidates))
    top_500_indices = np.argsort(cross_scores)[::-1][:top_500_count]
    top_500_candidates = [top_candidates[i] for i in top_500_indices]
    top_500_scores = cross_scores[top_500_indices]

    # 3. Final scoring with structured, behavioural, and honeypot
    final_scores = []
    for idx, cand in enumerate(top_500_candidates):
        semantic = float(top_500_scores[idx])

        # Precompute and assign _honeypot since fine_score expects it
        hp = honeypot_penalty(cand)
        cand['_honeypot'] = hp

        # Compute fine score (uses LightGBM if trained, else fallback)
        fine = fine_score(cand, semantic)

        # Apply hard filter for honeypots (guarantee 0.0)
        final = fine if hp > 0.0 else 0.0

        final_scores.append((cand['candidate_id'], final, cand))

    # Convert final to 0–100 scale
    scaled_scores = []
    for cand_id, final, cand in final_scores:
        scaled = final * 100.0
        scaled_scores.append((cand_id, scaled, cand))

    scaled_scores.sort(key=lambda x: (-round(x[1], 4), x[0]))
    top_results = scaled_scores[:top_k]

    rows = []
    for rank, (cand_id, score, cand) in enumerate(top_results, 1):
        reasoning = generate_reasoning(cand, rank)
        rows.append({
            'candidate_id': cand_id,
            'rank': rank,
            'score': round(score, 2),
            'reasoning': reasoning
        })

    df = pd.DataFrame(rows)
    print(f"Ranking completed in {time.time() - start:.2f} seconds")
    return df


# ------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------
def process_inputs(jd_text, candidate_file):
    error_msg = ""
    df = pd.DataFrame()
    csv_path = None

    if not jd_text.strip():
        error_msg = "Please enter a job description."
        return df, error_msg, None

    if candidate_file is None:
        error_msg = "Please upload a candidates JSON file."
        return df, error_msg, None

    filepath = Path(candidate_file.name)
    if filepath.suffix.lower() != '.json':
        error_msg = "Only .json files are allowed."
        return df, error_msg, None

    try:
        candidates = load_candidates_from_json(filepath)
    except Exception as e:
        error_msg = f"Error loading JSON: {e}"
        return df, error_msg, None

    if len(candidates) == 0:
        error_msg = "No candidates found in the JSON array."
        return df, error_msg, None

    required_top = ['profile', 'career_history', 'skills', 'redrob_signals']
    first = candidates[0]
    missing = [f for f in required_top if f not in first]
    if missing:
        error_msg = f"Invalid candidate schema: missing top-level fields: {missing}. Expected format: {required_top}"
        return df, error_msg, None

    try:
        df = rank_candidates(jd_text, candidates)
    except KeyError as e:
        error_msg = f"Missing field in candidate data: {e}. Please check your JSON schema."
        return df, error_msg, None
    except Exception as e:
        import traceback
        error_msg = f"Ranking error: {e}\n{traceback.format_exc()}"
        return df, error_msg, None

    # Save CSV to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    return df, error_msg, csv_path


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
with gr.Blocks(title="Redrob AI Ranker") as demo:
    gr.Markdown("""
    # 🧠 Intelligent Candidate Ranker (LLM‑powered)
    Paste a job description and upload a **JSON array** of candidates to get the top‑100 ranking.
    """)

    with gr.Row():
        jd_input = gr.Textbox(label="Job Description", lines=10,
                               placeholder="Paste the job description here...")

    with gr.Row():
        file_input = gr.File(label="Upload Candidates JSON", file_types=[".json"])

    with gr.Row():
        submit_btn = gr.Button("Rank Candidates", variant="primary")

    with gr.Row():
        error_output = gr.Textbox(label="Status / Error", interactive=False, visible=True)
        output_table = gr.Dataframe(label="Top 100 Candidates", interactive=False)
        download_btn = gr.DownloadButton(label="Download CSV")

    submit_btn.click(
        fn=process_inputs,
        inputs=[jd_input, file_input],
        outputs=[output_table, error_output, download_btn]
    )

    gr.Markdown("""
    ---
    **LLM‑Powered Workflow**
    1. **BM25** – fast lexical retrieval (top 2,000)
    2. **Cross‑Encoder (Transformer) Re‑ranking** – fine‑tuned LLM (top 300)
    3. **Structured Features** – title, company, experience, skills, education
    4. **Behavioral Signals** – recency, response rate, open‑to‑work, GitHub
    5. **Honeypot Penalty** – catches impossible profiles
    """)

if __name__ == "__main__":
    demo.launch(share=True)