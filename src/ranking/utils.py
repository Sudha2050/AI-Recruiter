def is_fresher_candidate(candidate: dict) -> bool:
    """
    Classifies a candidate as a fresher or experienced.
    Returns True if the candidate is classified as a fresher.
    """
    profile = candidate.get('profile', {})
    exp = profile.get('years_of_experience', 0)
    title = profile.get('current_title', '').lower()
    
    # Fresher defined as < 1.5 years of experience, or student/intern/fresher titles
    is_fresher = (exp < 1.5) or any(t in title for t in ['student', 'intern', 'graduate', 'fresher'])
    return is_fresher
