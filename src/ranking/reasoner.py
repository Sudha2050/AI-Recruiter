from src.utils.constants import RELEVANT_SKILLS

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