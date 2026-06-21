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
    Compute a behavioral multiplier (0.3 - 1.2) using 14 signals.
    Includes all High and Medium priority signals.
    """
    if not redrob_signals:
        return 1.0

    mult = 1.0

    # ---- Existing signals (7) ----
    # 1. Inactivity (last_active_date)
    try:
        last_active = datetime.strptime(redrob_signals['last_active_date'], '%Y-%m-%d')
        inactive_days = (REFERENCE_DATE - last_active).days
    except (KeyError, ValueError):
        inactive_days = 0

    if inactive_days > 180:
        mult *= 0.35
    elif inactive_days > 90:
        mult *= 0.55
    elif inactive_days > 30:
        mult *= 0.80

    # 2. Response rate
    response_rate = redrob_signals.get('recruiter_response_rate', 0.5)
    if response_rate > 0.7:
        mult *= 1.10
    elif response_rate > 0.4:
        mult *= 1.00
    else:
        mult *= 0.65

    # 3. Open to work
    if not redrob_signals.get('open_to_work_flag', False):
        mult *= 0.85

    # 4. GitHub activity
    github_score = redrob_signals.get('github_activity_score', -1)
    if github_score > 50:
        mult *= 1.05
    elif github_score == -1:
        mult *= 0.95

    # 5. Profile completeness
    completeness = redrob_signals.get('profile_completeness_score', 50)
    if completeness < 40:
        mult *= 0.80

    # 6. Verified email
    if not redrob_signals.get('verified_email', False):
        mult *= 0.95

    # 7. Verified phone
    if not redrob_signals.get('verified_phone', False):
        mult *= 0.95

    # ---- New High Priority signals (2) ----
    # 8. Preferred work mode (JD expects hybrid/remote)
    mode = redrob_signals.get('preferred_work_mode', '').lower()
    if mode in ['remote', 'hybrid']:
        mult *= 1.04
    elif mode == 'onsite':
        mult *= 0.96

    # 9. Willing to relocate
    if redrob_signals.get('willing_to_relocate', False):
        mult *= 1.03

    # ---- New Medium Priority signals (5) ----
    # 10. Notice period (JD prefers sub-30, can buy out 30)
    notice = redrob_signals.get('notice_period_days', 90)
    if notice <= 30:
        mult *= 1.04
    elif notice > 90:
        mult *= 0.95

    # 11. Average response time
    avg_resp = redrob_signals.get('avg_response_time_hours', 48)
    if avg_resp < 24:
        mult *= 1.03
    elif avg_resp > 72:
        mult *= 0.96

    # 12. Interview completion rate
    completion = redrob_signals.get('interview_completion_rate', 0.5)
    if completion > 0.8:
        mult *= 1.04
    elif completion < 0.3:
        mult *= 0.95

    # 13. Applications submitted in last 30 days
    apps = redrob_signals.get('applications_submitted_30d', 3)
    if 3 <= apps <= 5:
        mult *= 1.03
    elif apps == 0:
        mult *= 0.97
    elif apps > 10:
        mult *= 0.95

    # 14. Skill assessment scores (average of all assessments)
    assessments = redrob_signals.get('skill_assessment_scores', {})
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments) / 100.0  # 0-1
        if avg_assessment > 0.7:
            mult *= 1.03
        elif avg_assessment < 0.3:
            mult *= 0.97

    # Clip to reasonable range
    return max(0.3, min(1.2, mult))