"""Combine cross-encoder relevance with data-derived modifiers, then order.

Relevance from the cross-encoder is the primary signal. Behavioral availability
and location fit are small multiplicative adjustments whose cut points are real
pool percentiles, not absolute thresholds. Missing signals stay neutral.
Final order sorts by score descending, candidate_id ascending as tie-break.
"""

from __future__ import annotations

import re

import numpy as np

import config

_FACET_W = np.asarray(config.FACET_WEIGHTS, dtype=float)
_ML_RE = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in config.APPLIED_ML_HINTS) + r")\b")


def facet_relevance(scores_row) -> float:
    """Weighted mean over a candidate's strongest facets, in [0, 1]."""
    wv = _FACET_W * scores_row
    top = np.argsort(wv)[::-1][:config.RELEVANCE_TOP_FACETS]
    return float(wv[top].sum() / _FACET_W[top].sum())


def pool_stats(gaps, resps) -> dict:
    """Percentile cut points over the whole pool for the availability modifier."""
    gaps = np.asarray(gaps, dtype=float)
    resps = np.asarray(resps, dtype=float)
    return {
        "gap_stale": float(np.percentile(gaps, config.INACTIVE_PCTL * 100)),
        "gap_median": float(np.median(gaps)),
        "resp_floor": float(np.percentile(resps, config.RESPONSE_FLOOR_PCTL * 100)),
        "resp_median": float(np.median(resps)),
    }


def availability_modifier(cand, stats) -> float:
    """Behavioral availability in [-1, 1], averaged over the signals that are present."""
    s = cand.signals
    parts = []

    if s.last_active_date is not None:
        gap = (config.REFERENCE_DATE - s.last_active_date).days
        if gap <= stats["gap_median"]:
            parts.append(1.0)
        elif gap <= stats["gap_stale"]:
            parts.append(0.0)
        else:
            parts.append(-1.0)

    if s.recruiter_response_rate is not None:
        rr = s.recruiter_response_rate
        if rr < stats["resp_floor"]:
            parts.append(-1.0)
        elif rr >= stats["resp_median"]:
            parts.append(1.0)
        else:
            parts.append(0.0)

    parts.append(1.0 if s.open_to_work_flag else -1.0)
    return sum(parts) / len(parts)


def location_modifier(cand) -> float:
    """Location fit in [-1, 1] against the JD-preferred Indian metros."""
    text = f"{cand.profile.location} {cand.profile.country}".lower()
    if any(c in text for c in config.TARGET_LOCATIONS):
        return 1.0
    in_india = "india" in cand.profile.country.lower()
    if in_india:
        return 0.5 if cand.signals.willing_to_relocate else 0.0
    return -1.0


def _ic_score(title):
    t = title.lower()
    if any(h in t for h in config.MANAGEMENT_HINTS):
        return -1
    if any(h in t for h in config.IC_ENGINEERING_HINTS):
        return 1
    return 0


def _months(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def recency_modifier(cand) -> float:
    """Hands-on recency in [-1, 1]: a current IC engineer is positive, a current
    manager or architect negative, an engineering role that ended long ago neutral."""
    roles = cand.career_history
    if not roles:
        return 0.0
    current = [r for r in roles if r.is_current and r.start]
    role = max(current or roles, key=lambda r: r.end or r.start or config.REFERENCE_DATE)
    ic = _ic_score(role.title)
    if ic == 0:
        return 0.0
    end = config.REFERENCE_DATE if (role.is_current or role.end is None) else role.end
    recency = max(0.0, 1.0 - max(0, _months(end, config.REFERENCE_DATE)) / config.RECENT_CODE_MONTHS)
    return ic * recency


def applied_ml_months(cand) -> int:
    """Months in roles whose title or description shows ML/AI/NLP/search/ranking/
    recommendation/retrieval work, distinct from total experience."""
    return sum(r.duration_months for r in cand.career_history
               if _ML_RE.search(f"{r.title} {r.description}".lower()))


def applied_ml_signal(cand) -> float:
    """Applied-ML experience in [0, 1], saturating at the JD's target years."""
    return min(applied_ml_months(cand) / 12.0 / config.APPLIED_ML_TARGET_YEARS, 1.0)


def is_title_chaser(cand) -> bool:
    """Average tenure per company below the cut across a genuine multi-company history.

    Tenure is summed per company (so internal promotions do not read as hops),
    and the pattern must repeat across several companies before it counts.
    """
    by_company = {}
    for r in cand.career_history:
        by_company[r.company] = by_company.get(r.company, 0) + r.duration_months
    if len(by_company) < config.TITLE_CHASER_MIN_COMPANIES:
        return False
    avg_years = sum(by_company.values()) / len(by_company) / 12.0
    return avg_years < config.TITLE_CHASER_TENURE_YEARS


def combine(relevance, avail, loc, recency, applied_ml, title_chaser, impact) -> float:
    """Fold the small modifiers into the (already [0, 1]) relevance score.

    Availability, location, recency, and the title-chaser penalty scale with
    relevance inside the bracket; applied-ML experience and quantified-impact
    are small additive bonuses that nudge without dominating relevance.
    """
    score = relevance * (
        1.0
        + config.AVAILABILITY_WEIGHT * avail
        + config.LOCATION_WEIGHT * loc
        + config.RECENCY_WEIGHT * recency
        - (config.TITLE_CHASER_PENALTY if title_chaser else 0.0)
    )
    return score + config.APPLIED_ML_WEIGHT * applied_ml + config.IMPACT_WEIGHT * impact


def order_top(scored, n):
    """Sort by score descending, candidate_id ascending, take the first n."""
    return sorted(scored, key=lambda x: (-x[1], x[0]))[:n]
