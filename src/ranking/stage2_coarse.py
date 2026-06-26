from config.jd_config import TITLE_BONUS
from src.features.career_features import company_tier
from src.features.skill_features import skill_depth_score
from .utils import is_fresher_candidate

def coarse_score(candidate: dict) -> float:
    profile = candidate['profile']
    
    # 1. Title Score
    title = profile.get('current_title', '').lower()
    title_score = 0.5
    for key, val in TITLE_BONUS.items():
        if key in title:
            title_score = max(title_score, val)
            
    # 2. Company Score
    company = profile.get('current_company', '')
    ctype = company_tier(company)
    company_score = {'product': 1.0, 'other': 0.7, 'consulting': 0.4, 'fictional': 0.0}.get(ctype, 0.5)
    
    # 3. Experience Score
    exp = profile.get('years_of_experience', 0)
    exp_score = 1.0 if 5 <= exp <= 9 else (0.7 if 4 <= exp < 5 else (0.6 if 9 < exp <= 12 else 0.4))
    
    # 4. Skill Score
    skill_score = skill_depth_score(candidate.get('skills', []), exp)
    
    # 5. Education Score
    education = candidate.get('education', [])
    tier_w = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4, 'unknown': 0.5}
    edu_score = max([tier_w.get(e.get('tier', 'unknown'), 0.5) for e in education] + [0.3])
    relevant_fields = ["computer science", "machine learning", "artificial intelligence",
                       "data science", "information technology", "statistics", "mathematics"]
    field_relevance = 0.5
    if education and any(any(f in e.get("field_of_study", "").lower() for f in relevant_fields) for e in education):
        field_relevance = 1.0
    edu_score = 0.7 * edu_score + 0.3 * field_relevance

    if is_fresher_candidate(candidate):
        # Freshers: Academics/Education (40%), Foundational Skills (40%), and Early Signals (20%)
        early_signals = 0.5 * title_score + 0.5 * company_score
        return 0.4 * edu_score + 0.4 * skill_score + 0.2 * early_signals
    else:
        # Experienced candidates: Title (30%), Company (30%), Experience (20%), Skills (20%)
        return 0.3 * title_score + 0.3 * company_score + 0.2 * exp_score + 0.2 * skill_score