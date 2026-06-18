# Scoring weights (used in stage4_fine.py)
SCORING_WEIGHTS = {
    "semantic": 0.40,
    "title": 0.15,
    "company": 0.15,
    "experience": 0.10,
    "skills": 0.15,
    "education": 0.05,
    "behavioral": 0.10,   # multiplicative modifier
}

# Company type multipliers
COMPANY_TYPE_MULTIPLIER = {
    "product": 1.0,
    "other": 0.7,
    "consulting": 0.4,
    "fictional": 0.0,
}

# Title‑based bonus/penalty
TITLE_BONUS = {
    "ml engineer": 1.0,
    "ai engineer": 1.0,
    "data scientist": 0.9,
    "software engineer": 0.7,
    "backend engineer": 0.6,
    "marketing manager": 0.1,
    "hr manager": 0.1,
}