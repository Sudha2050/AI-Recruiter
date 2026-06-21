# src/ranking/stage1_filter.py
from ..utils.constants import FICTIONAL_COMPANIES

def honeypot_penalty(candidate: dict) -> float:
    """
    Returns a score multiplier (0.0 to 1.0) based on honeypot indicators.
    - Hard disqualify (0.0) only for extreme cases:
      * All companies are fictional, OR at least 3 fictional companies.
      * 8+ expert/advanced skills with 0 months experience.
    - Otherwise, apply soft penalties (multiply by 0.8 per flag).
    """
    career_history = candidate.get('career_history', [])
    total_companies = len(career_history)
    fictional_count = 0

    for career in career_history:
        if career.get('company') in FICTIONAL_COMPANIES:
            fictional_count += 1

    # Hard disqualify: all fictional OR 3+ fictional
    if total_companies > 0 and (fictional_count == total_companies or fictional_count >= 3):
        return 0.0

    # Hard disqualify: keyword stuffing (8+ expert/advanced with 0 months)
    suspect_skills = 0
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') in ('expert', 'advanced'):
            if skill.get('duration_months', 0) <= 0:
                suspect_skills += 1
    if suspect_skills >= 8:
        return 0.0

    # --- Soft penalties ---
    red_flags = 0.0

    # 2 fictional companies → penalty
    if fictional_count == 2:
        red_flags += 1.0

    # 4–7 suspect skills → penalty
    if 4 <= suspect_skills < 8:
        red_flags += 1.0

    # Non‑tech title with many AI skills (keyword stuffer)
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