#!/usr/bin/env python3
"""Runtime ranking (no network, under 5 minutes).

Loads precomputed artifacts, embeds the JD once, retrieves by hybrid search,
reranks the survivors with a local cross-encoder, applies pool-percentile
modifiers, writes the submission CSV, and validates it.

  python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # set before transformers import to forbid any network
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import json
import subprocess
import sys
import time

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from src import disqualifiers, features, impact, integrity, io, reasoning, rerank, retrieval, scoring

faiss.omp_set_num_threads(1)  # avoid faiss/torch OpenMP clash that segfaults on macOS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=config.DATA_PATH)
    ap.add_argument("--out", default=config.OUTPUT_CSV)
    ap.add_argument("--prove-offline", action="store_true",
                    help="block outbound TCP so any network attempt during ranking fails loudly")
    return ap.parse_args()


def block_network():
    """Make every outbound internet socket raise, proving ranking needs no network."""
    import socket
    inet = (socket.AF_INET, socket.AF_INET6)
    real_connect = socket.socket.connect

    def guard(self, address):
        if self.family in inet:
            raise RuntimeError("network access attempted during ranking")
        return real_connect(self, address)

    socket.socket.connect = guard

    def no_conn(*_a, **_k):
        raise RuntimeError("network access attempted during ranking")

    socket.create_connection = no_conn


def main():
    args = parse_args()
    if args.prove_offline:
        block_network()
        print("prove-offline: outbound network disabled")
    start = time.perf_counter()

    embedder = SentenceTransformer(config.EMBED_DIR)
    cross = CrossEncoder(config.CROSS_DIR)
    index = retrieval.load_dense(config.FAISS_PATH)
    bm25 = retrieval.load_bm25(config.BM25_PATH)
    with open(config.IDS_PATH, encoding="utf-8") as f:
        id_order = json.load(f)
    with open(config.JD_PATH, encoding="utf-8") as f:
        jd_text = f.read()
    print(f"load: artifacts and models, {time.perf_counter() - start:.1f}s")

    t = time.perf_counter()
    jd_vec = embedder.encode(jd_text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    dense = retrieval.dense_search(index, jd_vec, config.DENSE_TOPK)
    lexical = retrieval.bm25_search(bm25, retrieval.tokenize(jd_text), config.BM25_TOPK)
    rows = retrieval.rrf_fuse([dense, lexical], config.RRF_K, config.SHORTLIST)
    shortlist = [id_order[r] for r in rows]
    short_set = set(shortlist)
    print(f"retrieve: {len(shortlist)} shortlisted, {time.perf_counter() - t:.1f}s")

    # Pass 1: company earliest-start bounds, needed before consistency checks.
    t = time.perf_counter()
    bounds = {}
    for c in io.read_candidates(args.candidates):
        integrity.update_company_bounds(bounds, c)
    print(f"bounds: {len(bounds)} companies, {time.perf_counter() - t:.1f}s")

    # Pass 2: modifier percentiles, integrity feature matrix, consistency
    # violations over the whole pool, and capture of the shortlisted candidates.
    t = time.perf_counter()
    pool, gaps, resps = {}, [], []
    feat, ids_all, violations = [], [], {}
    for c in io.read_candidates(args.candidates):
        s = c.signals
        if s.last_active_date is not None:
            gaps.append((config.REFERENCE_DATE - s.last_active_date).days)
        if s.recruiter_response_rate is not None:
            resps.append(s.recruiter_response_rate)
        feat.append(integrity.feature_vector(c))
        ids_all.append(c.candidate_id)
        hard = integrity.hard_contradictions(c, bounds)
        soft = integrity.soft_anomalies(c)
        if hard or soft:
            violations[c.candidate_id] = (c.profile.current_title, c.profile.current_company, hard, soft)
        if c.candidate_id in short_set:
            pool[c.candidate_id] = c
    stats = scoring.pool_stats(gaps, resps)
    print(f"pool: stats over {len(gaps)} active, {time.perf_counter() - t:.1f}s")

    excluded = run_gate(feat, ids_all, violations, args.out)

    t = time.perf_counter()
    cands = [pool[cid] for cid in shortlist if cid in pool and cid not in excluded]
    texts = [features.evidence_text(c) for c in cands]
    facets = rerank.facet_scores(cross, config.FACETS, texts)
    print(f"rerank: {len(cands)} scored on {len(config.FACETS)} facets, {time.perf_counter() - t:.1f}s")

    scored, mods, fac, raw, flagged, imp = [], {}, {}, {}, {}, {}
    for c, row in zip(cands, facets):
        cid = c.candidate_id
        avail = scoring.availability_modifier(c, stats)
        loc = scoring.location_modifier(c)
        recency = scoring.recency_modifier(c)
        applied = scoring.applied_ml_signal(c)
        chaser = scoring.is_title_chaser(c)
        imp[cid] = impact.impact_score(c)
        mods[cid] = (c, avail, loc)
        fac[cid] = row
        rel = scoring.facet_relevance(row)
        raw[cid] = scoring.combine(rel, avail, loc, recency, applied, chaser, imp[cid])
        fl = disqualifiers.flags(c)
        if fl:
            flagged[cid] = fl
        scored.append((cid, config.DISQUALIFIER_FLOOR if fl else raw[cid]))

    report_disqualifiers(flagged, raw, mods)

    top = scoring.order_top(scored, config.TOP_N)
    print_top(top, mods, fac, imp)

    t = time.perf_counter()
    reasons, fell_back = reasoning.generate(config.REASONING_MODE, top, mods)
    if config.REASONING_MODE == "llm":
        print(f"reasoning: llm path, {fell_back} of {len(top)} fell back to template, "
              f"{time.perf_counter() - t:.1f}s")
    else:
        print(f"reasoning: template path, {time.perf_counter() - t:.1f}s")

    write_csv(args.out, top, reasons)
    print(f"write: {len(top)} rows to {args.out}")

    ok = validate(args.out)
    total = time.perf_counter() - start
    print(f"{'PASS' if ok else 'FAIL'}: total {total:.1f}s")
    if total > config.RUNTIME_WARN_S:
        print(f"WARN: total runtime {total:.1f}s exceeds the {config.RUNTIME_WARN_S}s budget")
    sys.exit(0 if ok else 1)


def run_gate(feat, ids_all, violations, out_path):
    """Exclude hard contradictions outright; soft anomalies only if the forest corroborates."""
    t = time.perf_counter()
    matrix = integrity.impute(np.asarray(feat, dtype=np.float64))
    outlier = integrity.outlier_flags(matrix)
    outlier_ids = {ids_all[i] for i in np.flatnonzero(outlier)}
    excluded = {}
    for cid, (title, company, hard, soft) in violations.items():
        if hard:
            excluded[cid] = (title, company, hard, "hard")
        elif soft and cid in outlier_ids:
            excluded[cid] = (title, company, soft, "soft+outlier")
    frac = len(excluded) / len(ids_all)
    n_hard = sum(1 for x in excluded.values() if x[3] == "hard")
    print(f"integrity: {len(excluded)} excluded ({frac:.2%} of pool), "
          f"{n_hard} hard contradictions, {len(excluded) - n_hard} corroborated soft, "
          f"{outlier.sum()} outliers, {time.perf_counter() - t:.1f}s")
    for cid, (title, company, v, kind) in list(excluded.items())[:15]:
        print(f"  excluded {cid} [{kind}]: {title} at {company} [{', '.join(v)}]")
    if frac > config.MAX_EXCLUSION_FRACTION:
        print(f"STOP: exclusions exceed {config.MAX_EXCLUSION_FRACTION:.0%} of the pool, "
              f"not finalizing {out_path}. Review the gate before proceeding.")
        sys.exit(2)
    return set(excluded)


def report_disqualifiers(flagged, raw, mods):
    """Per-flag counts, plus any pre-floor top-100 candidate the floor removes."""
    counts = {f: sum(1 for fl in flagged.values() if f in fl) for f in disqualifiers.FLAGS}
    summary = ", ".join(f"{f}={counts[f]}" for f in disqualifiers.FLAGS)
    print(f"disqualifiers: {summary} ({len(flagged)} candidates floored of {len(raw)} scored)")

    prefloor = {cid for cid, _ in scoring.order_top(list(raw.items()), config.TOP_N)}
    removed = [cid for cid in prefloor if cid in flagged]
    if not removed:
        print("  borderline: none of the pre-floor top 100 are floored")
        return
    print(f"  borderline: {len(removed)} pre-floor top-100 candidate(s) floored, review:")
    for cid in sorted(removed, key=lambda c: -raw[c]):
        cand = mods[cid][0]
        title = f"{cand.profile.current_title} at {cand.profile.current_company}"
        companies = ", ".join(dict.fromkeys(r.company for r in cand.career_history if r.company))
        print(f"    {cid} raw={raw[cid]:.4f} [{', '.join(flagged[cid])}] {title} | {companies}")


def print_top(top, mods, fac, imp):
    print("top 20 (per-facet relevance: shipped, embeddings, product-ml):")
    for rank, (cid, score) in enumerate(top[:20], 1):
        cand = mods[cid][0]
        fs = " ".join(f"{x:.2f}" for x in fac[cid])
        title = f"{cand.profile.current_title} at {cand.profile.current_company}"
        print(f"  {rank:2d} {cid} score={score:.4f} impact={imp[cid]:.2f} facets=[{fs}] {title}")


def write_csv(path, top, reasons):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score) in enumerate(top, 1):
            w.writerow([cid, rank, repr(score), reasons[cid]])


def validate(path):
    r = subprocess.run([sys.executable, config.VALIDATOR, path], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
    return r.returncode == 0


if __name__ == "__main__":
    main()
