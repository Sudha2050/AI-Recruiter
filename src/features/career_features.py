from src.utils.constants import PRODUCT_COMPANIES, CONSULTING_COMPANIES, FICTIONAL_COMPANIES
from typing import List, Dict, Any

def company_tier(company: str) -> str:
    if company in PRODUCT_COMPANIES: return "product"
    if company in CONSULTING_COMPANIES: return "consulting"
    if company in FICTIONAL_COMPANIES: return "fictional"
    return "other"

def career_progression_score(career_history: List[Dict]) -> float:
    if not career_history:
        return 0.5
    titles = [c.get('title', '').lower() for c in career_history]
    senior_keywords = ['senior', 'lead', 'principal', 'manager']
    has_senior = any(any(kw in t for kw in senior_keywords) for t in titles)
    return 1.0 if has_senior else 0.5