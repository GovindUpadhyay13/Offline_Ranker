"""Impact score: does a candidate quantify outcomes or list generic duties.

Read only from career_history descriptions (never the skills list). Three
components, each normalized to [0, 1], combined by config weights:

  metric density   distinct numbers, percentages, and dollar amounts
  action strength  strong outcome verbs minus weak duty-listing phrases
  scale indicators distinct mentions of users, throughput, data volume, team size

Pure text analysis, deterministic, no network. Empty or missing descriptions
yield 0.0, so the signal only ever adds a small bonus, never penalizes.
"""

from __future__ import annotations

import re

import config

# A number with an optional currency prefix and an optional magnitude/unit
# suffix: matches "$2M", "40%", "500GB", "12", "3x", "2.5k".
_METRIC = re.compile(
    r"\$?\d[\d,]*\.?\d*\s?"
    r"(?:%|x|k|m|bn?|gb|tb|mb|pb|qps|rps|ms|million|billion|thousand)?",
    re.I,
)

_STRONG = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in config.IMPACT_STRONG_VERBS) + r")\b", re.I
)
_WEAK = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in config.IMPACT_WEAK_VERBS) + r")\b", re.I
)

# Scale: a magnitude bound to a unit of reach, throughput, volume, or team size.
_SCALE = [
    re.compile(p, re.I) for p in (
        r"\d[\d,.]*\s?[kmb]?\+?\s?(?:users?|customers?|requests?|queries|"
        r"records?|rows|events?|transactions?|downloads?|installs?|dau|mau)",
        r"team\s+(?:of|size\s+of)\s+\d+",
        r"\d[\d,.]*\s?(?:qps|rps|tps|req/s|requests?\s+per\s+second)",
        r"\d[\d,.]*\s?(?:gb|tb|pb|mb)\b",
        r"\d+(?:\.\d+)?\s?%\s?(?:faster|slower|improvement|increase|reduction|"
        r"growth|uplift|lift|latency)",
        r"\d[\d,.]*\s?x\s+(?:faster|throughput|improvement|speedup|speed-up)",
    )
]


def _norm(s):
    return re.sub(r"\s+", "", s.lower())


def _metric_density(text) -> float:
    found = {_norm(m) for m in _METRIC.findall(text) if any(ch.isdigit() for ch in m)}
    return min(len(found) / config.IMPACT_METRIC_FULL, 1.0)


def _action_strength(text) -> float:
    net = len(_STRONG.findall(text)) - len(_WEAK.findall(text))
    return min(max(net, 0) / config.IMPACT_VERB_FULL, 1.0)


def _scale(text) -> float:
    found = {_norm(m) for r in _SCALE for m in r.findall(text)}
    return min(len(found) / config.IMPACT_SCALE_FULL, 1.0)


def impact_score(cand) -> float:
    """Quantified-achievement signal in [0, 1] from the candidate's descriptions."""
    text = " ".join(r.description for r in cand.career_history if r.description).strip()
    if not text:
        return 0.0
    wm, wv, ws = config.IMPACT_COMPONENT_WEIGHTS
    return wm * _metric_density(text) + wv * _action_strength(text) + ws * _scale(text)
