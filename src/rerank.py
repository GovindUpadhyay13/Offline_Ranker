"""Aspect-based cross-encoder reranking of the shortlist.

The full JD overflows the cross-encoder's 512-token window, truncating the
candidate evidence away. Instead we score each candidate's evidence against a
few short facet queries (see config.FACETS) and combine the facet scores. Runs
only on the shortlist, never the full pool, so it stays inside the time budget.
"""

from __future__ import annotations

import numpy as np


def facet_scores(model, facets, texts):
    """Return a (len(texts), len(facets)) array of per-facet relevance in [0, 1]."""
    pairs = [[f, t] for t in texts for f in facets]
    raw = model.predict(pairs, convert_to_numpy=True).reshape(len(texts), len(facets))
    return 1.0 / (1.0 + np.exp(-raw))
