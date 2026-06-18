# src/ranking/stage1_filter.py
from ..utils.constants import FICTIONAL_COMPANIES

def honeypot_penalty(candidate: dict) -> float:
    """
    Returns a score multiplier (0.0 to 1.0) based on honeypot indicators.
    Hard filters (multiplier = 0.0) are applied for absolute disqualifiers:
      - Any employer in FICTIONAL_COMPANIES.
      - Keyword stuffing: 5+ expert/advanced skills with <=0 months experience.
    Soft penalties (0.0 < multiplier < 1.0) for other suspicious patterns.
    """
    red_flags = 0.0
    hard_filter = False

    # 1. Check for fictional companies (hard filter)
    for career in candidate.get('career_history', []):
        if career.get('company') in FICTIONAL_COMPANIES:
            hard_filter = True
            break

    # 2. Keyword stuffing: expert/advanced skills with 0 months experience (skip for freshers)
    from .utils import is_fresher_candidate
    if not is_fresher_candidate(candidate):
        suspect_skills = 0
        for skill in candidate.get('skills', []):
            if skill.get('proficiency') in ('expert', 'advanced'):
                if skill.get('duration_months', 0) <= 0:
                    suspect_skills += 1
        if suspect_skills >= 5:
            hard_filter = True

    # If hard filter triggered, return 0.0 (disqualify)
    if hard_filter:
        return 0.0

    # Otherwise, apply soft penalties for other suspicious patterns
    title = candidate['profile'].get('current_title', '').lower()
    if any(t in title for t in ['marketing', 'hr', 'operations', 'sales', 'accountant']):
        ai_count = sum(1 for s in candidate.get('skills', [])
                       if any(term in s['name'].lower() for term in ['nlp', 'llm', 'rag', 'vector', 'embedding', 'recommendation']))
        if ai_count >= 4:
            red_flags += 2.0
        elif ai_count >= 2:
            red_flags += 1.0

    # Apply soft penalty: each flag reduces score by 20%
    penalty = max(0.0, 1.0 - 0.20 * red_flags)
    return penalty

if __name__ == "__main__":
    print("This is a module, not a script. Import it from other files.")