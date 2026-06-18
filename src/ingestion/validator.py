from typing import Dict, Any

def validate_candidate_schema(candidate: Dict[str, Any]) -> bool:
    required = {"candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"}
    return all(k in candidate for k in required)