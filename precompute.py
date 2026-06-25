#!/usr/bin/env python3
"""Offline precompute (network allowed): vendor the models and build caches.

Embeds every candidate's evidence, builds the FAISS and BM25 indexes, and
persists vectors, both indexes, and the candidate_id order so rank.py can run
with no network. Run once. Optional argv[1] overrides the data path for a smoke
test on a smaller file.
"""

import json
import os
import sys
import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from src import features, io, retrieval


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    t = time.perf_counter()
    ids, texts = [], []
    for c in io.read_candidates(data_path):
        ids.append(c.candidate_id)
        texts.append(features.evidence_text(c))
    print(f"load: {len(ids)} candidates, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    embedder = SentenceTransformer(config.EMBED_MODEL)
    embedder.save(config.EMBED_DIR)
    CrossEncoder(config.CROSS_ENCODER_MODEL).save(config.CROSS_DIR)
    print(f"models: vendored to {config.MODELS_DIR}, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    emb = embedder.encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(config.EMB_NPY, emb)
    print(f"embed: {emb.shape}, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    retrieval.build_dense_index(emb, config.FAISS_PATH)
    print(f"faiss: index built, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    retrieval.build_bm25(texts, config.BM25_PATH)
    print(f"bm25: index built, {time.perf_counter() - t:.1f}s")

    with open(config.IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"ids: {len(ids)} written to {config.IDS_PATH}")


if __name__ == "__main__":
    main()
