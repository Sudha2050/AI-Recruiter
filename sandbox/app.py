import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import gzip
import time
import tempfile
import re
from datetime import datetime
import numpy as np
import pandas as pd
import gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ------------------------------------------------------------
# Constants (same as before)
# ------------------------------------------------------------
FICTIONAL_COMPANIES = {
    "Stark Industries", "Wayne Enterprises", "Pied Piper",
    "Dunder Mifflin", "Initech", "Acme Corp", "Globex Inc", "Hooli"
}
PRODUCT_COMPANIES = {
    'Uber', 'Swiggy', 'Zomato', 'Razorpay', 'CRED', 'Flipkart', 'Ola',
    'Wayne Enterprises', 'Pied Piper', 'Google', 'Microsoft', 'Amazon',
    'Meta', 'Apple', 'Netflix', 'Mad Street Den', 'Redrob AI'
}
CONSULTING_COMPANIES = {
    'TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant', 'HCL',
    'Tech Mahindra', 'Capgemini', 'Deloitte', 'PwC', 'EY', 'KPMG',
    'Mindtree', 'LTI', 'Mphasis'
}
RELEVANT_SKILLS = {
    'sentence-transformers', 'openai embeddings', 'bge', 'e5', 'vector search',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'elasticsearch',
    'opensearch', 'rag', 'llm', 'nlp', 'recommendation systems', 'learning-to-rank',
    'xgboost', 'evaluation frameworks', 'a/b testing', 'production ml',
    'mlops', 'prompt engineering', 'langchain', 'haystack', 'retrieval', 'ranking',
    'information retrieval', 'embeddings', 'fine-tuning', 'peft', 'lora', 'transformers',
    'pytorch', 'tensorflow'
}

REFERENCE_DATE = datetime(2026, 6, 17)


# ------------------------------------------------------------
# File loading
# ------------------------------------------------------------
def load_candidates_from_file(filepath: Path) -> list:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            if filepath.name.endswith('.jsonl.gz'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                content = f.read()
                data = json.loads(content)
    elif suffix == '.jsonl':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
    elif suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .json, .jsonl, or .jsonl.gz")

    if not isinstance(data, list):
        raise ValueError("File must contain a JSON array (or JSONL array of objects).")

    normalized = []
    for item in data:
        if 'profile' not in item:
            normalized.append({'profile': item})
        else:
            normalized.append(item)
    return normalized


# ------------------------------------------------------------
# Text aggregation
# ------------------------------------------------------------
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
# Company tier
# ------------------------------------------------------------
def company_tier(company: str) -> str:
    if company in PRODUCT_COMPANIES:
        return "product"
    if company in CONSULTING_COMPANIES:
        return "consulting"
    if company in FICTIONAL_COMPANIES:
        return "fictional"
    return "other"


# ------------------------------------------------------------
# Skill depth score
# ------------------------------------------------------------
def skill_depth_score(skills: list) -> float:
    total = 0.0
    count = 0
    for s in skills:
        name = s.get('name', '').lower()
        if not any(term in name for term in RELEVANT_SKILLS):
            continue
        prof = {'beginner': 0.4, 'intermediate': 0.7, 'advanced': 1.0, 'expert': 1.2}.get(s.get('proficiency'), 0.5)
        dur = min(1.0, s.get('duration_months', 0) / 24) if s.get('duration_months', 0) > 0 else 0.3
        end = min(1.5, 1 + s.get('endorsements', 0) / 30)
        total += prof * dur * end
        count += 1
    return total / count if count else 0.0


# ------------------------------------------------------------
# Behavioral multiplier (14 signals)
# ------------------------------------------------------------
def compute_behavioral_multiplier(redrob_signals: dict) -> float:
    if not redrob_signals:
        return 1.0

    mult = 1.0

    # Existing signals (7)
    try:
        last_active = datetime.strptime(redrob_signals['last_active_date'], '%Y-%m-%d')
        inactive_days = (REFERENCE_DATE - last_active).days
    except (KeyError, ValueError):
        inactive_days = 0

    if inactive_days > 180:
        mult *= 0.35
    elif inactive_days > 90:
        mult *= 0.55
    elif inactive_days > 30:
        mult *= 0.80

    response_rate = redrob_signals.get('recruiter_response_rate', 0.5)
    if response_rate > 0.7:
        mult *= 1.10
    elif response_rate > 0.4:
        mult *= 1.00
    else:
        mult *= 0.65

    if not redrob_signals.get('open_to_work_flag', False):
        mult *= 0.85

    github_score = redrob_signals.get('github_activity_score', -1)
    if github_score > 50:
        mult *= 1.05
    elif github_score == -1:
        mult *= 0.95

    completeness = redrob_signals.get('profile_completeness_score', 50)
    if completeness < 40:
        mult *= 0.80

    if not redrob_signals.get('verified_email', False):
        mult *= 0.95
    if not redrob_signals.get('verified_phone', False):
        mult *= 0.95

    # High priority signals
    mode = redrob_signals.get('preferred_work_mode', '').lower()
    if mode in ['remote', 'hybrid']:
        mult *= 1.04
    elif mode == 'onsite':
        mult *= 0.96

    if redrob_signals.get('willing_to_relocate', False):
        mult *= 1.03

    # Medium priority signals
    notice = redrob_signals.get('notice_period_days', 90)
    if notice <= 30:
        mult *= 1.04
    elif notice > 90:
        mult *= 0.95

    avg_resp = redrob_signals.get('avg_response_time_hours', 48)
    if avg_resp < 24:
        mult *= 1.03
    elif avg_resp > 72:
        mult *= 0.96

    completion = redrob_signals.get('interview_completion_rate', 0.5)
    if completion > 0.8:
        mult *= 1.04
    elif completion < 0.3:
        mult *= 0.95

    apps = redrob_signals.get('applications_submitted_30d', 3)
    if 3 <= apps <= 5:
        mult *= 1.03
    elif apps == 0:
        mult *= 0.97
    elif apps > 10:
        mult *= 0.95

    assessments = redrob_signals.get('skill_assessment_scores', {})
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments) / 100.0
        if avg_assessment > 0.7:
            mult *= 1.03
        elif avg_assessment < 0.3:
            mult *= 0.97

    return max(0.3, min(1.2, mult))


# ------------------------------------------------------------
# Honeypot penalty
# ------------------------------------------------------------
def honeypot_penalty(candidate: dict) -> float:
    career_history = candidate.get('career_history', [])
    total_companies = len(career_history)
    fictional_count = 0
    for career in career_history:
        if career.get('company') in FICTIONAL_COMPANIES:
            fictional_count += 1

    if total_companies > 0 and (fictional_count == total_companies or fictional_count >= 3):
        return 0.0

    suspect_skills = 0
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') in ('expert', 'advanced'):
            if skill.get('duration_months', 0) <= 0:
                suspect_skills += 1
    if suspect_skills >= 8:
        return 0.0

    red_flags = 0.0
    if fictional_count == 2:
        red_flags += 1.0
    if 4 <= suspect_skills < 8:
        red_flags += 1.0

    title = candidate['profile'].get('current_title', '').lower()
    if any(t in title for t in ['marketing', 'hr', 'operations', 'sales', 'accountant']):
        ai_count = sum(1 for s in candidate.get('skills', [])
                       if any(term in s['name'].lower() for term in ['nlp', 'llm', 'rag', 'vector', 'embedding', 'recommendation']))
        if ai_count >= 4:
            red_flags += 2.0
        elif ai_count >= 2:
            red_flags += 1.0

    return max(0.0, 1.0 - 0.20 * red_flags)


# ------------------------------------------------------------
# Reasoning generator
# ------------------------------------------------------------
def generate_reasoning(candidate: dict, rank: int) -> str:
    profile = candidate['profile']
    signals = candidate.get('redrob_signals', {})
    parts = []
    title = profile.get('current_title', '')
    exp = profile.get('years_of_experience', 0)
    parts.append(f"{title} with {exp:.1f} yrs")
    company = profile.get('current_company', '')
    parts.append(f"at {company}")
    skills = candidate.get('skills', [])
    strong = [s['name'] for s in skills
              if s.get('proficiency') in ('expert', 'advanced')
              and any(term in s['name'].lower() for term in RELEVANT_SKILLS)]
    if strong:
        parts.append(f"skills: {', '.join(strong[:2])}")
    resp = signals.get('recruiter_response_rate', 0)
    if resp > 0.7:
        parts.append("highly responsive")
    elif resp > 0.4:
        parts.append("good engagement")
    return "; ".join(parts[:4])


# ------------------------------------------------------------
# Main ranking pipeline
# ------------------------------------------------------------
def rank_candidates(jd_text: str, candidates: list, top_k: int = 100) -> pd.DataFrame:
    start = time.time()
    if not candidates:
        raise ValueError("No candidates provided.")

    # 1. BM25 retrieval
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

    # 2. Cross‑Encoder re‑ranking
    print("Loading Cross-Encoder for re-ranking...")
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

    # 3. Final scoring
    final_scores = []
    for idx, cand in enumerate(top_500_candidates):
        semantic = float(top_500_scores[idx])

        profile = cand['profile']
        title = profile.get('current_title', '').lower()
        title_score = 0.5
        if any(term in title for term in ['ml', 'ai', 'engineer', 'scientist']):
            title_score = 1.0
        elif any(term in title for term in ['marketing', 'hr', 'operations']):
            title_score = 0.2

        company = profile.get('current_company', '')
        ctype = company_tier(company)
        company_score = {'product': 1.0, 'other': 0.7, 'consulting': 0.4, 'fictional': 0.0}.get(ctype, 0.5)

        exp = profile.get('years_of_experience', 0)
        exp_score = 1.0 if 5 <= exp <= 9 else (0.7 if 4 <= exp < 5 else (0.6 if 9 < exp <= 12 else 0.4))

        skill_score = skill_depth_score(cand.get('skills', []))

        education = cand.get('education', [])
        tier_w = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4, 'unknown': 0.5}
        edu_score = max([tier_w.get(e.get('tier', 'unknown'), 0.5) for e in education] + [0.3])

        mult = compute_behavioral_multiplier(cand.get('redrob_signals', {}))
        hp = honeypot_penalty(cand)

        structured_part = (
            0.15 * title_score +
            0.15 * company_score +
            0.10 * exp_score +
            0.15 * skill_score +
            0.05 * edu_score
        )
        base = 0.40 * semantic + 0.60 * structured_part
        final = base * mult * hp
        final = max(0.0, min(1.0, final))

        final_scores.append((cand['candidate_id'], final, cand))

    # Sort and take top 100
    final_scores.sort(key=lambda x: (-round(x[1], 4), x[0]))
    top_results = final_scores[:top_k]

    rows = []
    for rank, (cand_id, score, cand) in enumerate(top_results, 1):
        reasoning = generate_reasoning(cand, rank)
        rows.append({
            'candidate_id': cand_id,
            'rank': rank,
            'score': round(score * 100, 2),
            'reasoning': reasoning
        })
    df = pd.DataFrame(rows)
    print(f"Ranking completed in {time.time()-start:.2f} seconds")
    return df


# ------------------------------------------------------------
# Chatbot
# ------------------------------------------------------------
def chatbot_response(message, history):
    lower = message.lower()
    if re.search(r'cand[_-]?\d{5,7}', lower):
        cid = re.search(r'cand[_-]?\d{5,7}', lower, re.I).group(0).upper()
        return f"Please run a ranking first, then I can explain candidate {cid}."
    if 'how' in lower or 'work' in lower:
        return "The system uses BM25 → Cross‑Encoder → Structured scoring → Behavioral multiplier → Honeypot penalty. Scores are 0–100."
    if 'hello' in lower or 'hi' in lower:
        return "Hello! Upload a JD and candidates file, then ask me about results."
    return "I can help with: 'Explain CAND-00021', 'How does it work?', 'Compare CAND-00021 and CAND-00123'."


# ------------------------------------------------------------
# Gradio UI (hardcoded 100 candidates)
# ------------------------------------------------------------
def process_inputs(jd_text, candidate_file):
    empty = pd.DataFrame()
    if not jd_text.strip():
        return empty, "⚠️ Please enter a job description.", None
    if candidate_file is None:
        return empty, "⚠️ Please upload a candidates file.", None

    filepath = Path(candidate_file.name)
    try:
        candidates = load_candidates_from_file(filepath)
    except Exception as e:
        return empty, f"❌ Error loading file: {e}", None

    if len(candidates) == 0:
        return empty, "⚠️ No candidates found in the file.", None

    required_top = ['profile', 'career_history', 'skills', 'redrob_signals']
    missing = [f for f in required_top if f not in candidates[0]]
    if missing:
        return empty, f"❌ Invalid schema — missing fields: {missing}", None

    try:
        # Always return exactly 100 candidates
        df = rank_candidates(jd_text, candidates, top_k=100)
    except Exception as e:
        import traceback
        return empty, f"❌ Ranking error: {e}\n{traceback.format_exc()}", None

    temp_dir = tempfile.mkdtemp()
    csv_path = Path(temp_dir) / "team_xxx.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')

    return df, f"✅ Ranking complete — top {len(df)} candidates ready. Filename: team_xxx.csv", str(csv_path)


# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.header-card {
    background: linear-gradient(135deg, #4f46e5 0%, #312e81 100%) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    margin-bottom: 1.5rem !important;
    color: white !important;
}
.header-card h1 { font-size: 2.2rem !important; font-weight: 700 !important; color: white !important; margin: 0 0 0.5rem 0 !important; }
.header-card p { font-size: 1rem !important; color: #e0e7ff !important; margin: 0 !important; }
.primary-btn { background: linear-gradient(135deg, #4f46e5, #4338ca) !important; border: none !important; }
.primary-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(79,70,229,0.4) !important; }
.download-btn { background: linear-gradient(135deg, #0d9488, #0f766e) !important; border: none !important; }
.download-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(13,148,136,0.4) !important; }
.panel-box { border-radius: 16px !important; border: 1px solid #e2e8f0 !important; padding: 1.5rem !important; background: white !important; }
.dark .panel-box { border-color: #334155 !important; background: #1e293b !important; }
"""

# ------------------------------------------------------------
# Gradio UI layout
# ------------------------------------------------------------
with gr.Blocks(
    title="Redrob AI Ranker",
    theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"),
    css=css
) as demo:
    with gr.Column(elem_classes=["header-card"]):
        gr.Markdown("""
        # 🧠 Intelligent Candidate Ranker (LLM‑powered)
        Paste a job description and upload a candidate dataset (**JSON / JSONL / JSONL.GZ**).
        Always returns the **top 100** candidates (scores 0–100).
        """)

    with gr.Tab("Ranking"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel-box"]):
                gr.Markdown("### 📂 Input")
                jd_input = gr.Textbox(label="Job Description", lines=12,
                                       placeholder="Paste the job description here...")
                file_input = gr.File(label="Upload Candidates", file_types=[".json", ".jsonl", ".jsonl.gz"])
                submit_btn = gr.Button("⚡ Rank Candidates", variant="primary", elem_classes=["primary-btn"])

            with gr.Column(scale=1.5, elem_classes=["panel-box"]):
                gr.Markdown("### 📊 Results (Top 100)")
                error_output = gr.Textbox(label="Status", interactive=False)
                output_table = gr.Dataframe(label="Ranked Candidates", interactive=False, wrap=True)
                download_btn = gr.DownloadButton(label="⬇️ Download CSV", elem_classes=["download-btn"])

        submit_btn.click(
            fn=process_inputs,
            inputs=[jd_input, file_input],
            outputs=[output_table, error_output, download_btn]
        )

    with gr.Tab("Chat Assistant"):
        with gr.Column(elem_classes=["panel-box"]):
            gr.Markdown("""
            ### 💬 Ask about the ranking
            **Examples:**  
            - `Explain candidate CAND-00021`  
            - `Compare CAND-00021 and CAND-00123`  
            - `How does the algorithm work?`
            """)
            chatbot = gr.ChatInterface(fn=chatbot_response, title="Ranking Assistant")

# ------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)