import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")

# Silence asyncio selector unraisable exceptions on process shutdown/reload
def custom_unraisable_hook(unraisable):
    if unraisable.exc_type is ValueError and "Invalid file descriptor" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)

sys.unraisablehook = custom_unraisable_hook

# Patch gradio_client schema parser to avoid boolean schema TypeError (unhashable type: 'bool')
try:
    import gradio_client.utils
    original_json_schema_to_python_type = gradio_client.utils._json_schema_to_python_type
    def patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            schema = {}
        return original_json_schema_to_python_type(schema, defs)
    gradio_client.utils._json_schema_to_python_type = patched_json_schema_to_python_type
except Exception:
    pass

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
from sentence_transformers import SentenceTransformer

print("Initializing Bi-Encoder model globally (all-MiniLM-L6-v2)...")
model_general = SentenceTransformer('all-MiniLM-L6-v2')
model_skill = model_general


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
        # Repeat the career description to give it higher importance/weight in the semantic embeddings
        parts.append(career.get('description', ''))
        parts.append(career.get('description', ''))
    skills = candidate.get('skills', [])
    for skill in skills:
        if isinstance(skill, dict):
            parts.append(skill.get('name', ''))
        elif isinstance(skill, str):
            parts.append(skill)
    return ' '.join(parts)


def aggregate_profile_text_general(candidate: dict) -> str:
    profile = candidate.get('profile', {})
    parts = [
        profile.get('headline', ''),
        profile.get('summary', '')
    ]
    for career in candidate.get('career_history', []):
        parts.append(career.get('title', ''))
        parts.append(career.get('description', ''))
    return ' '.join([p for p in parts if p])


def aggregate_profile_text_skills(candidate: dict) -> str:
    skills = candidate.get('skills', [])
    skill_names = []
    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get('name', '')
            if name:
                skill_names.append(name)
        elif isinstance(skill, str) and skill:
            skill_names.append(skill)
    return ', '.join(skill_names)


def aggregate_profile_text_bm25(candidate: dict) -> str:
    profile = candidate.get('profile', {})
    parts = [
        profile.get('headline', ''),
        profile.get('summary', '')
    ]
    for career in candidate.get('career_history', []):
        parts.append(career.get('title', ''))
    skills = candidate.get('skills', [])
    for skill in skills:
        if isinstance(skill, dict):
            parts.append(skill.get('name', ''))
        elif isinstance(skill, str):
            parts.append(skill)
    return ' '.join([p for p in parts if p])


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
ENTRY_LEVEL_SKILLS = {'nltk', 'nlp', 'python', 'basic python', 'programming', 'software engineering'}

def skill_depth_score(skills: list, years_exp: float = 0.0) -> float:
    total = 0.0
    count = 0
    is_experienced_5_to_9 = (5.0 <= years_exp <= 9.0)
    for s in skills:
        name = s.get('name', '').lower()
        if is_experienced_5_to_9:
            if name in ENTRY_LEVEL_SKILLS or any(basic in name for basic in ['nltk', 'basic python']):
                continue
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


def rank_candidates(jd_text: str, candidates: list, top_k: int = 100) -> pd.DataFrame:
    start = time.time()
    if not candidates:
        raise ValueError("No candidates provided.")

    print(f"Total candidates: {len(candidates)}")

    # 1. Coarse Lexical Retrieval (BM25) to select top 2000 candidates
    print("Computing BM25 lexical scores...")
    corpus = [aggregate_profile_text_bm25(c) for c in candidates]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    jd_tokens = tokenize(jd_text)
    bm25_scores = bm25.get_scores(jd_tokens)

    # Pre-filter: keep top 2000 candidates using BM25
    pre_filter_k = min(2000, len(candidates))
    pre_filter_indices = np.argsort(bm25_scores)[::-1][:pre_filter_k]
    
    top_bm25_candidates = [candidates[i] for i in pre_filter_indices]
    print(f"BM25 retrieval: selected top {pre_filter_k} candidates for semantic ensembling")

    # 2. Dual Bi-Encoder Semantic Scoring
    print("Encoding Job Description and candidates using general and skill models...")
    # A. General representation
    jd_emb_gen = model_general.encode(jd_text, convert_to_numpy=True)
    jd_norm_gen = jd_emb_gen / (np.linalg.norm(jd_emb_gen) or 1.0)

    general_texts = [aggregate_profile_text_general(c) for c in top_bm25_candidates]
    gen_embs = model_general.encode(
        general_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    gen_norms = np.linalg.norm(gen_embs, axis=1, keepdims=True)
    gen_norms = np.where(gen_norms == 0, 1.0, gen_norms)
    gen_embs_norm = gen_embs / gen_norms
    general_scores = np.dot(gen_embs_norm, jd_norm_gen)

    # B. Skill representation
    jd_emb_skill = model_skill.encode(jd_text, convert_to_numpy=True)
    jd_norm_skill = jd_emb_skill / (np.linalg.norm(jd_emb_skill) or 1.0)

    skill_texts = [aggregate_profile_text_skills(c) for c in top_bm25_candidates]
    skill_embs = model_skill.encode(
        skill_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    skill_norms = np.linalg.norm(skill_embs, axis=1, keepdims=True)
    skill_norms = np.where(skill_norms == 0, 1.0, skill_norms)
    skill_embs_norm = skill_embs / skill_norms
    skill_scores = np.dot(skill_embs_norm, jd_norm_skill)

    # C. Combine scores (average)
    combined_semantic = (general_scores + skill_scores) / 2
    combined_semantic = np.clip(combined_semantic, 0.0, 1.0)

    # 3. Keep top 300 candidates based on combined semantic score
    top_sem_k = min(300, len(top_bm25_candidates))
    sem_sorted_indices = np.argsort(combined_semantic)[::-1][:top_sem_k]
    
    top_300_candidates = [top_bm25_candidates[i] for i in sem_sorted_indices]
    top_300_semantic_scores = combined_semantic[sem_sorted_indices]
    print(f"Semantic ensembling: selected top {top_sem_k} candidates for final structured and behavioral scoring")

    # 4. Final scoring stage
    final_scores = []
    for idx, cand in enumerate(top_300_candidates):
        semantic = float(top_300_semantic_scores[idx])

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

        skill_score = skill_depth_score(cand.get('skills', []), exp)

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
        
        project_bonus = 0.0
        if 5.0 <= exp <= 9.0:
            project_bonus = has_complex_projects(cand.get('career_history', []))
            
        normalized_structured = structured_part / 0.60
        normalized_structured = min(1.0, normalized_structured + project_bonus)
        structured_part = normalized_structured * 0.60

        base = 0.55 * semantic + 0.45 * structured_part
        final = base * mult * hp
        final = max(0.0, min(1.0, final))

        final_scores.append((cand['candidate_id'], final, cand))

    # Sort and take top 100
    final_scores.sort(key=lambda x: (-x[1], x[0]))
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
        return "The system uses Hybrid (BM25 + Dual Bi-Encoder Ensemble) retrieval → Structured scoring (experience, company tier, skills, education) → Behavioral multiplier → Honeypot penalty. Semantic scoring uses an ensemble of General Text (headline + summary + careers) and Skill-only embeddings. Scores are 0–100."
    if 'hello' in lower or 'hi' in lower:
        return "Hello! Upload a JD and candidates file, then ask me about results."
    return "I can help with: 'Explain CAND-00021', 'How does it work?', 'Compare CAND-00021 and CAND-00123'."


# ------------------------------------------------------------
# Gradio UI (hardcoded 100 candidates)
# ------------------------------------------------------------
def process_inputs(jd_text, candidate_file, team_id):
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

    # Determine filename base
    if team_id and team_id.strip():
        base_name = team_id.strip()
    else:
        import re
        first_line = jd_text.strip().split('\n')[0]
        role_name = re.sub(r'[^a-zA-Z0-9 ]', '', first_line)[:50]
        role_name = role_name.replace(' ', '_')
        if not role_name:
            role_name = "candidate_rankings"
        base_name = role_name

    # Save XLSX
    xlsx_filename = f"{base_name}.xlsx"
    xlsx_path = Path(tempfile.gettempdir()) / xlsx_filename
    df.to_excel(xlsx_path, index=False, sheet_name="Top 100")

    return df, f"✅ Ranking complete — top {len(df)} candidates ready. Filename: {xlsx_filename}", str(xlsx_path)


# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: radial-gradient(circle at 10% 20%, rgb(18, 16, 32) 0%, rgb(7, 5, 14) 90%) !important;
    color: #e2e8f0 !important;
}

/* Custom Header with animated background and glowing border */
.header-card {
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.45) 0%, rgba(30, 27, 75, 0.6) 100%) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 20px !important;
    padding: 2.5rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7), 0 0 15px rgba(139, 92, 246, 0.15) !important;
}

.header-card h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(to right, #c084fc, #6366f1, #38bdf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 0.75rem 0 !important;
    text-shadow: 0 0 40px rgba(139, 92, 246, 0.2) !important;
}

.header-card p {
    font-size: 1.1rem !important;
    color: #cbd5e1 !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

/* Beautiful panels using glassmorphism */
.panel-box {
    background: rgba(17, 24, 39, 0.5) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 20px !important;
    padding: 1.75rem !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.panel-box:hover {
    border-color: rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
}

/* Input Fields styling */
.panel-box textarea, .panel-box input {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.panel-box textarea:focus, .panel-box input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3) !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
}

.primary-btn:active {
    transform: translateY(0) !important;
}

.download-btn {
    background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
    transition: all 0.2s ease !important;
}

.download-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(5, 150, 105, 0.5) !important;
}

/* Results table */
.panel-box table {
    border-collapse: separate !important;
    border-spacing: 0 8px !important;
}

.panel-box tr {
    background: rgba(30, 41, 59, 0.4) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.panel-box tr:hover {
    background: rgba(30, 41, 59, 0.8) !important;
}

.panel-box th {
    background: transparent !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.05) !important;
}

.panel-box td {
    border-bottom: none !important;
}

/* Tabs styling */
.tabs {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    margin-bottom: 1.5rem !important;
}

.tab-nav button {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.1rem !important;
    color: #94a3b8 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* Custom styling for status box */
.status-box {
    border-left: 4px solid #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.05) !important;
}
"""

# ------------------------------------------------------------
# Gradio UI layout
# ------------------------------------------------------------
with gr.Blocks(
    title="Redrob AI Ranker",
    theme=gr.themes.Glass(primary_hue="purple", secondary_hue="indigo", neutral_hue="slate"),
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
                team_id_input = gr.Textbox(
                    label="Participant ID (for filename)",
                    placeholder="e.g., team_123 (optional)"
                )
                jd_input = gr.Textbox(
                    label="Job Description",
                    lines=12,
                    placeholder="Paste the job description here..."
                )
                file_input = gr.File(
                    label="Upload Candidates",
                    file_types=[".json", ".jsonl", ".jsonl.gz"]
                )
                submit_btn = gr.Button(
                    "⚡ Rank Candidates",
                    variant="primary",
                    elem_classes=["primary-btn"]
                )

            with gr.Column(scale=2, elem_classes=["panel-box"]):
                gr.Markdown("### 📊 Results (Top 100)")
                error_output = gr.Textbox(label="Status", interactive=False)
                output_table = gr.Dataframe(
                    label="Ranked Candidates",
                    interactive=False,
                    wrap=True
                )
                download_xlsx_btn = gr.DownloadButton(
                    label="⬇️ Download XLSX",
                    elem_classes=["download-btn"]
                )

        submit_btn.click(
            fn=process_inputs,
            inputs=[jd_input, file_input, team_id_input],
            outputs=[output_table, error_output, download_xlsx_btn]
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
    demo.launch(
        share=True
    )