"""Hybrid retrieval: dense FAISS plus lexical BM25, fused by reciprocal rank.

precompute builds and persists both indexes. rank loads them, embeds the JD
once, queries each, and fuses the two ranked lists with RRF. No keyword
whitelist: lexical matching is plain BM25 over the same evidence text.
"""

from __future__ import annotations

import pickle
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9+#.]+")


def tokenize(text) -> list:
    return _TOKEN.findall(text.lower())


def build_dense_index(embeddings, path):
    """Build a cosine (inner-product on normalized vectors) FAISS index and persist it."""
    embeddings_f32 = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(embeddings_f32.shape[1])
    index.add(embeddings_f32)
    faiss.write_index(index, path)
    return index


def build_bm25(texts, path):
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    return bm25


def load_dense(path):
    return faiss.read_index(path)


def load_bm25(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dense_search(index, jd_vector, topk):
    """Return row indices of the topk nearest evidence vectors to the JD."""
    _, idx = index.search(jd_vector.reshape(1, -1), topk)
    return idx[0].tolist()


def bm25_search(bm25, jd_tokens, topk):
    scores = bm25.get_scores(jd_tokens)
    return np.argsort(scores)[::-1][:topk].tolist()


def rrf_fuse(ranked_lists, k, n):
    """Fuse ranked row-index lists by reciprocal rank fusion, return top n rows."""
    fused = {}
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(fused, key=lambda r: (-fused[r], r))
    return order[:n]
