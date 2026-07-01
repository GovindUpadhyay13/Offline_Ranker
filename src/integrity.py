"""Integrity gate: isolation-forest outliers corroborated by consistency checks.

The forest is fit on the pool's own suspicious-signal subspace, so "outlier" is
defined by the distribution rather than by hardcoded thresholds. A candidate is
excluded only when the forest flags them AND at least one interpretable
consistency check fails, so neither signal alone can punish an unusual-but-real
profile. The -1 sentinels in github_activity_score and offer_acceptance_rate are
missing, never low, and are imputed before fitting.
"""

import json
import os
import numpy as np
from sklearn.ensemble import IsolationForest

import config

_FOUNDING_YEARS = {}
if os.path.exists(config.FOUNDING_YEARS_PATH):
    try:
        with open(config.FOUNDING_YEARS_PATH, "r", encoding="utf-8") as f:
            _FOUNDING_YEARS = json.load(f)
    except Exception:
        pass

# Ordinal encoding of the company-size bands, the one categorical we use.
_SIZE_ORD = {
    "1-10": 0, "11-50": 1, "51-200": 2, "201-500": 3,
    "501-1000": 4, "1001-5000": 5, "5001-10000": 6, "10001+": 7,
}

# Order of the columns produced by feature_vector, kept for readability only.
FEATURES = (
    "profile_completeness", "endorsements", "connections", "profile_views",
    "search_appearance", "saved_by_recruiters", "recruiter_response",
    "interview_completion", "offer_acceptance", "github_activity",
    "avg_response_hours", "mean_assessment", "applications", "company_size",
)


def feature_vector(cand) -> list:
    """Suspicious-signal row for one candidate, NaN where a value is missing."""
    s = cand.signals
    scores = list(s.skill_assessment_scores.values())
    return [
        _f(s.profile_completeness_score),
        _f(s.endorsements_received),
        _f(s.connection_count),
        _f(s.profile_views_received_30d),
        _f(s.search_appearance_30d),
        _f(s.saved_by_recruiters_30d),
        _f(s.recruiter_response_rate),
        _f(s.interview_completion_rate),
        _f(s.offer_acceptance_rate),  # None when -1 sentinel
        _f(s.github_activity_score),  # None when -1 sentinel
        _f(s.avg_response_time_hours),
        float(np.mean(scores)) if scores else np.nan,
        _f(s.applications_submitted_30d),
        float(_SIZE_ORD.get(cand.profile.current_company_size, np.nan)),
    ]


def _f(x):
    return float(x) if x is not None else np.nan


def impute(matrix):
    """Fill missing entries with the column median so missingness is not an outlier."""
    med = np.nanmedian(matrix, axis=0)
    nan = np.isnan(matrix)
    matrix[nan] = np.take(med, np.where(nan)[1])
    return matrix


def outlier_flags(matrix):
    """Fit a subsampled IsolationForest and return a boolean extreme-outlier mask."""
    n = len(matrix)
    clf = IsolationForest(
        n_estimators=config.ISO_ESTIMATORS,
        max_samples=min(config.ISO_SUBSAMPLE, n),
        contamination=config.ISO_CONTAMINATION,
        random_state=config.SEED,
        n_jobs=1,
    )
    clf.fit(matrix)
    return clf.predict(matrix) == -1


def update_company_bounds(bounds, cand):
    """Track the two smallest distinct start dates per company, in one linear pass."""
    for r in cand.career_history:
        if r.start is None:
            continue
        m = bounds.get(r.company)
        if m is None:
            bounds[r.company] = [r.start, None]
        elif r.start < m[0]:
            m[1] = m[0]
            m[0] = r.start
        elif r.start > m[0] and (m[1] is None or r.start < m[1]):
            m[1] = r.start


def _months(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def hard_contradictions(cand, bounds) -> list:
    """Within-profile impossibilities, each sufficient to exclude on its own.

    Names the checks that fail: summed role months exceeding stated experience by
    a clear margin, a role whose duration contradicts its own start/end span, or a
    start predating the company's plausible age (its next-earliest pool appearance).
    """
    v = []
    yoe = cand.profile.years_of_experience
    total = sum(r.duration_months for r in cand.career_history)
    if yoe is not None and total > yoe * 12 + config.HARD_TENURE_MARGIN_MONTHS:
        v.append("tenure_sum_exceeds_experience")

    for r in cand.career_history:
        if r.start is None:
            continue
        span = _months(r.start, r.end or config.REFERENCE_DATE)
        if span < 0 or abs(r.duration_months - span) > config.DURATION_TOLERANCE_MONTHS:
            v.append("role_duration_vs_dates")
            break

    for r in cand.career_history:
        if r.start is None:
            continue
        m = bounds.get(r.company)
        if m and r.start == m[0] and m[1] is not None and _months(m[0], m[1]) > config.COMPANY_PREDATE_MONTHS:
            v.append("predates_company_cohort")
            break

    # Honeypot check: check if candidate tenure implies start date earlier than founding year
    for r in cand.career_history:
        if r.company and r.start:
            founding_year = _FOUNDING_YEARS.get(r.company)
            if founding_year is not None and r.start.year < founding_year:
                v.append("predates_company_founding")
                break

    return v


def soft_anomalies(cand) -> list:
    """Suspicious but not impossible; excluded only with forest corroboration.

    Phantom experience: years_of_experience far exceeding the total documented
    career. Undocumented early roles make this possible for a real senior, so it
    needs the outlier signal before it floors anyone.
    """
    v = []
    yoe = cand.profile.years_of_experience
    total = sum(r.duration_months for r in cand.career_history)
    if yoe is not None and yoe * 12 - total > config.PHANTOM_EXPERIENCE_MARGIN_MONTHS:
        v.append("experience_exceeds_documented")
    return v
