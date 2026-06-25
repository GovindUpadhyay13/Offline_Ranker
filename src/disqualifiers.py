"""Hard disqualifier flags from the JD, used to floor a final score, but only
when the evidence is unambiguous.

Each flag carries an escape clause so a mixed or borderline history never fires:
consulting_only needs every company to be a consulting firm, pure_research needs
every affiliation to be academic, domain_mismatch needs repeated CV/speech/robotics
evidence with no NLP/IR/search/recsys evidence anywhere. Word-boundary matching
keeps "search" from firing on "research". No behavioral sentinels enter here.
"""

from __future__ import annotations

import re

import config

FLAGS = ("consulting_only", "pure_research", "domain_mismatch")


def _word_re(terms):
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b")


_CV_RE = _word_re(config.CV_SPEECH_ROBOTICS_HINTS)
_NLP_RE = _word_re(config.NLP_IR_HINTS)


def _is_consulting(company):
    c = company.lower()
    return any(f in c for f in config.CONSULTING_FIRMS)


def _is_academic(company):
    c = company.lower()
    return any(h in c for h in config.ACADEMIC_HINTS)


def _companies(cand):
    return [r.company for r in cand.career_history if r.company]


def consulting_only(cand) -> bool:
    """Every company is a consulting firm, so no product-company role exists."""
    companies = _companies(cand)
    return bool(companies) and all(_is_consulting(c) for c in companies)


def pure_research(cand) -> bool:
    """Every affiliation is academic or a lab, so no product-company role exists."""
    companies = _companies(cand)
    return bool(companies) and all(_is_academic(c) for c in companies)


def _evidence(cand) -> str:
    p = cand.profile
    parts = [p.headline, p.summary, p.current_industry]
    parts += [f"{r.title} {r.description}" for r in cand.career_history]
    return " ".join(parts).lower()


def domain_mismatch(cand) -> bool:
    """Primary domain is CV, speech, or robotics with no NLP/IR evidence anywhere."""
    text = _evidence(cand)
    if len(_CV_RE.findall(text)) < config.CV_DOMAIN_MIN_MENTIONS:
        return False
    return _NLP_RE.search(text) is None


def flags(cand) -> list:
    """Names of the disqualifiers this candidate trips, empty when clear."""
    out = []
    if consulting_only(cand):
        out.append("consulting_only")
    if pure_research(cand):
        out.append("pure_research")
    if domain_mismatch(cand):
        out.append("domain_mismatch")
    return out
