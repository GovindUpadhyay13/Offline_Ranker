"""Aspect-based cross-encoder reranking of the shortlist.

The full JD overflows the cross-encoder's 512-token window, truncating the
candidate evidence away. Instead we score each candidate's evidence against a
few short facet queries derived from the submitted JD and combine the facet
scores. Runs only on the shortlist, never the full pool, so it stays inside
the time budget.
"""

from __future__ import annotations

import re

import numpy as np

import config

# Strong requirement keywords signal demonstrated or required domain work.
# Weighted 3× to outrank sentences that only contain generic terms.
_STRONG_REQ = frozenset({
    "production", "hands-on", "proficiency", "expertise", "proven",
    "built", "shipped", "deployed",
})
# Weak requirement keywords add signal but can't dominate on their own.
_WEAK_REQ = frozenset({
    "experience", "knowledge", "skills", "must", "required", "should",
    "ability", "familiar", "background", "strong", "developed", "designed",
    "implemented", "worked", "years", "degree",
})
# Sentences beginning with these words are conditional or exclusionary — skip them.
_NEGATIVE_START = re.compile(r"^(if|unless|when|we will|we won|we don|please don)", re.I)


def extract_facets(jd_text: str, n: int = 3) -> list[str]:
    """Derive n requirement-bearing facet queries from a JD at request time.

    Deterministic: strips bullets, segments into lines/sentences, scores by
    requirement-keyword density (strong keywords weighted 3×), returns the
    top-n phrases truncated to 30 words so they fit inside the cross-encoder's
    512-token window alongside candidate evidence. Falls back to config.FACETS
    when fewer than n candidates survive filtering.
    """
    # Strip bullet markers but preserve newline structure for line segmentation.
    text = re.sub(r"(?m)^[ \t]*[\-\*•][ \t]*", "", jd_text)
    text = re.sub(r"[ \t]+", " ", text)

    raw_chunks: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Further split on sentence boundaries within each line.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
        raw_chunks.extend(p.strip() for p in parts if p.strip())

    candidates: list[tuple[int, str]] = []
    for s in raw_chunks:
        words = s.split()
        if len(words) < 5:
            continue
        low = s.lower()
        if _NEGATIVE_START.match(low):
            continue
        strong = sum(1 for kw in _STRONG_REQ if kw in low)
        weak = sum(1 for kw in _WEAK_REQ if kw in low)
        score = 3 * strong + weak
        if score == 0:
            continue
        phrase = " ".join(words[:30])
        candidates.append((score, phrase))

    # Stable sort: descending score, document order preserved for ties.
    candidates.sort(key=lambda x: -x[0])
    selected = [phrase for _, phrase in candidates[:n]]
    return selected if len(selected) >= n else list(config.FACETS)


def facet_scores(model, facets, texts):
    """Return a (len(texts), len(facets)) array of per-facet relevance in [0, 1]."""
    pairs = [[f, t] for t in texts for f in facets]
    raw = model.predict(pairs, convert_to_numpy=True).reshape(len(texts), len(facets))
    return 1.0 / (1.0 + np.exp(-raw))
