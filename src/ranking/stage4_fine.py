import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from config import settings
from config.jd_config import TITLE_BONUS
from ..features.career_features import company_tier, career_progression_score
from ..features.skill_features import skill_depth_score
from ..features.signal_features import compute_behavioral_multiplier
from .utils import is_fresher_candidate

# Lazy load LightGBM model
_LGB_MODEL = None

def load_model():
    global _LGB_MODEL
    if _LGB_MODEL is None:
        model_path = settings.DATA_MODELS / "lgb_ranker.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                _LGB_MODEL = pickle.load(f)
        else:
            _LGB_MODEL = None
    return _LGB_MODEL


def build_feature_vector(candidate, semantic_score):
    profile = candidate['profile']
    # Title
    title = profile.get('current_title', '').lower()
    title_score = 0.5
    if any(term in title for term in ['ml', 'ai', 'engineer', 'scientist']):
        title_score = 1.0
    elif any(term in title for term in ['marketing', 'hr', 'operations']):
        title_score = 0.2

    # Company
    company = profile.get('current_company', '')
    ctype = company_tier(company)
    company_score = {'product':1.0, 'other':0.7, 'consulting':0.4, 'fictional':0.0}.get(ctype, 0.5)

    # Experience
    exp = profile.get('years_of_experience', 0)
    exp_score = 1.0 if 5 <= exp <= 9 else (0.7 if 4 <= exp < 5 else (0.6 if 9 < exp <= 12 else 0.4))

    # Skills
    skill_score = skill_depth_score(candidate.get('skills', []), exp)

    # Education
    education = candidate.get('education', [])
    tier_w = {'tier_1':1.0, 'tier_2':0.8, 'tier_3':0.6, 'tier_4':0.4, 'unknown':0.5}
    edu_score = max([tier_w.get(e.get('tier', 'unknown'), 0.5) for e in education] + [0.3])

    # Behavioral multiplier
    signals = candidate.get('redrob_signals', {})
    mult = compute_behavioral_multiplier(signals)

    # Honeypot
    hp = candidate.get('_honeypot', 1.0)

    # Feature vector (order must match training)
    return np.array([
        title_score,
        company_score,
        exp_score,
        skill_score,
        edu_score,
        semantic_score,
        mult,
        hp,
    ])


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


def fine_score(candidate, semantic_score):
    """
    Computes a category-aware dynamic score for candidates.
    - Experienced: Work Experience & Leadership (45%), Technical Assessments (35%), Behavioral (20%)
    - Freshers: Academic, Internships & Foundational (50%), Aptitude & Soft Skills (50%)
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    education = candidate.get('education', [])
    
    # Pre-calculate base features
    # 1. Title Score
    title = profile.get('current_title', '').lower()
    title_score = 0.5
    for key, val in TITLE_BONUS.items():
        if key in title:
            title_score = max(title_score, val)
    if any(term in title for term in ['ml', 'ai', 'engineer', 'scientist']):
        title_score = max(title_score, 1.0)
    elif any(term in title for term in ['marketing', 'hr', 'operations']):
        title_score = min(title_score, 0.2)

    # 2. Company Score
    company = profile.get('current_company', '')
    ctype = company_tier(company)
    company_score = {'product': 1.0, 'other': 0.7, 'consulting': 0.4, 'fictional': 0.0}.get(ctype, 0.5)

    # 3. Experience Score
    exp = profile.get('years_of_experience', 0)
    exp_score = 1.0 if 5 <= exp <= 9 else (0.7 if 4 <= exp < 5 else (0.6 if 9 < exp <= 12 else 0.4))

    # 4. Skill Score
    skill_score = skill_depth_score(skills, exp)

    # 5. Education Score
    tier_w = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4, 'unknown': 0.5}
    edu_base = max([tier_w.get(e.get('tier', 'unknown'), 0.5) for e in education] + [0.3])
    relevant_fields = ["computer science", "machine learning", "artificial intelligence",
                       "data science", "information technology", "statistics", "mathematics"]
    field_relevance = 0.5
    if education and any(any(f in e.get("field_of_study", "").lower() for f in relevant_fields) for e in education):
        field_relevance = 1.0
    academic_score = 0.7 * edu_base + 0.3 * field_relevance

    # 6. Behavioral Multiplier & Signals
    mult = compute_behavioral_multiplier(signals)
    interview_completion = signals.get('interview_completion_rate', 50) / 100.0
    response_rate = signals.get('recruiter_response_rate', 0.5)
    completeness = signals.get('profile_completeness_score', 50) / 100.0

    # 7. Assessments Score
    assessments = signals.get('skill_assessment_scores', {})
    if assessments:
        assessments_score = sum(assessments.values()) / (len(assessments) * 100.0)
    else:
        assessments_score = 0.5

    # 8. Honeypot check
    hp = candidate.get('_honeypot', 1.0)

    # --- Category-Based Weighted Scoring ---
    if is_fresher_candidate(candidate):
        # Category A: Fresher
        # 1. Academic Performance, Internships & Foundational Knowledge (50%)
        internship_months = sum(role.get('duration_months', 0) for role in career)
        internship_score = min(1.0, internship_months / 12.0)
        
        academic_intern_foundational = (
            0.40 * academic_score +
            0.30 * skill_score +
            0.15 * internship_score +
            0.15 * semantic_score
        )
        
        # 2. Aptitude Tests & Soft Skills Evaluations (50%)
        github_raw = signals.get('github_activity_score', 0)
        github_score = min(1.0, github_raw / 100.0) if github_raw > 0 else 0.0
        aptitude = 0.7 * assessments_score + 0.3 * github_score
        
        soft_skills = 0.4 * (mult / 1.2) + 0.3 * response_rate + 0.3 * completeness
        aptitude_soft_skills = 0.5 * aptitude + 0.5 * soft_skills
        
        score = 0.50 * academic_intern_foundational + 0.50 * aptitude_soft_skills
    else:
        # Category B: Experienced
        # 1. Relevant Work Experience, Project Successes, and Leadership (45%)
        prog_score = career_progression_score(career)
        
        project_bonus = 0.0
        if 5.0 <= exp <= 9.0:
            project_bonus = has_complex_projects(career)
            
        experience_leadership = (
            0.20 * exp_score +
            0.35 * semantic_score +
            0.20 * title_score +
            0.15 * company_score +
            0.10 * prog_score
        )
        experience_leadership = min(1.0, experience_leadership + project_bonus)
        
        # 2. Technical Assessments (25%)
        technical = 0.5 * skill_score + 0.5 * assessments_score
        
        # 3. Behavioral Interviews (20%)
        behavioral = 0.6 * (mult / 1.2) + 0.4 * interview_completion
        
        score = 0.55 * experience_leadership + 0.25 * technical + 0.20 * behavioral

    # Apply Honeypot penalty
    final = score * hp
    return max(0.0, min(1.0, final))