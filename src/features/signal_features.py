from datetime import datetime
from typing import Dict, Any

# Static reference date matching the dataset's timestamp.
# All candidates' last_active_date is relative to this snapshot.
REFERENCE_DATE = datetime(2026, 6, 17)


def behavioral_features(redrob_signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract raw behavioral features from redrob_signals.
    Uses a fixed reference date (REFERENCE_DATE) to avoid system clock leakage.
    """
    if redrob_signals is None:
        redrob_signals = {}
    # Parse last active date from the signals
    raw_date = redrob_signals.get('last_active_date', None)
    try:
        last_active = datetime.strptime(raw_date, '%Y-%m-%d') if raw_date else REFERENCE_DATE
    except (ValueError, TypeError):
        last_active = REFERENCE_DATE
    # Compute inactivity using the static reference date
    inactive_days = (REFERENCE_DATE - last_active).days

    response_rate = redrob_signals.get('recruiter_response_rate', 0.5)
    open_to_work = redrob_signals.get('open_to_work_flag', False)
    github_score = redrob_signals.get('github_activity_score', -1)
    completeness = redrob_signals.get('profile_completeness_score', 50)
    verified_email = redrob_signals.get('verified_email', False)
    verified_phone = redrob_signals.get('verified_phone', False)

    return {
        'inactive_days': inactive_days,
        'response_rate': response_rate,
        'open_to_work': open_to_work,
        'github_score': github_score,
        'completeness': completeness,
        'verified_email': verified_email,
        'verified_phone': verified_phone,
    }


def compute_behavioral_multiplier(redrob_signals: Dict[str, Any]) -> float:
    """
    Compute a single multiplier (0.3 - 1.2) from behavioral signals.
    Uses fixed reference date.
    """
    bf = behavioral_features(redrob_signals)
    mult = 1.0

    # Inactivity penalty – based on static reference date
    if bf['inactive_days'] > 180:
        mult *= 0.35
    elif bf['inactive_days'] > 90:
        mult *= 0.55
    elif bf['inactive_days'] > 30:
        mult *= 0.80

    # Response rate
    if bf['response_rate'] > 0.7:
        mult *= 1.10
    elif bf['response_rate'] > 0.4:
        mult *= 1.00
    else:
        mult *= 0.65

    # Open to work
    if not bf['open_to_work']:
        mult *= 0.85

    # GitHub activity
    if bf['github_score'] > 50:
        mult *= 1.05
    elif bf['github_score'] == -1:
        mult *= 0.95

    # Profile completeness
    if bf['completeness'] < 40:
        mult *= 0.80

    # Verified email/phone
    if not bf['verified_email']:
        mult *= 0.95
    if not bf['verified_phone']:
        mult *= 0.95

    return max(0.3, min(1.2, mult))