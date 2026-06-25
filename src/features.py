"""Derive text evidence and numeric features from a Candidate.

evidence_text feeds retrieval and reranking (semantic relevance, not the skills
list). numeric_features feeds the integrity gate and the modifiers.
"""

from __future__ import annotations


def evidence_text(cand) -> str:
    """Concatenate the semantic evidence: summary then each role title and description.

    Deliberately excludes the skills list so matching is driven by demonstrated work,
    not by keyword stuffing.
    """
    parts = [cand.profile.summary]
    parts += [f"{r.title}. {r.description}" for r in cand.career_history]
    return " ".join(p for p in parts if p).strip()


def numeric_features(cand) -> dict:
    """Parsed numerics and consistency residuals used by integrity and modifiers."""
    raise NotImplementedError
