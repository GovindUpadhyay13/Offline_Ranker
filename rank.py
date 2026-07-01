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


# ---------------------------------------------------------------------------
# Core pipeline — callable from both CLI and API
# ---------------------------------------------------------------------------

def _integrity_status(cid, violations):
    """Return "clean" or the name(s) of the soft check(s) that flagged this candidate.

    Hard-violation candidates are excluded before scoring and never appear here;
    any candidate that reaches the top 100 with a violations entry has only
    uncorroborated soft anomalies.
    """
    if not violations or cid not in violations:
        return "clean"
    _, _, hard, soft = violations[cid]
    checks = soft or hard
    return checks[0] if len(checks) == 1 else ", ".join(checks)


def rank_candidates(
    jd_text: str,
    *,
    embedder,
    cross,
    index,
    bm25,
    id_order: list,
    candidate_pool: dict,
    excluded: set,
    stats: dict,
    violations: dict | None = None,
) -> list[dict]:
    """Embed a JD and run retrieval → rerank → score → reasoning.

    candidate_pool: dict mapping candidate_id -> Candidate. Must contain at
        least the candidates that appear in the shortlist (CLI passes only the
        shortlisted subset; API passes the full pool loaded at startup).
    excluded: candidate IDs dropped by the integrity gate.
    stats: pool-percentile cut points from scoring.pool_stats().
    violations: optional dict from integrity gate (cid -> (title, company, hard, soft));
        when provided, each result includes an integrity_status field.

    Returns the top-100 as a list of dicts with keys:
        candidate_id, rank, score, reasoning, facet_scores,
        availability_modifier, location_modifier, recency_modifier,
        integrity_status.
    """
    jd_vec = (
        embedder.encode(jd_text, normalize_embeddings=True, convert_to_numpy=True)
        .astype(np.float32)
    )
    dense = retrieval.dense_search(index, jd_vec, config.DENSE_TOPK)
    lexical = retrieval.bm25_search(bm25, retrieval.tokenize(jd_text), config.BM25_TOPK)
    rows = retrieval.rrf_fuse([dense, lexical], config.RRF_K, config.SHORTLIST)
    shortlist = [id_order[r] for r in rows]

    cands = [
        candidate_pool[cid]
        for cid in shortlist
        if cid in candidate_pool and cid not in excluded
    ]
    texts = [features.evidence_text(c) for c in cands]
    active_facets = rerank.extract_facets(jd_text, n=len(config.FACETS))
    print(f"facets: {active_facets}")
    facets = rerank.facet_scores(cross, active_facets, texts)

    scored, mods, fac, raw, flagged, imp, rec = [], {}, {}, {}, {}, {}, {}
    applied_scores: dict = {}
    chasers: dict = {}
    for c, row in zip(cands, facets):
        cid = c.candidate_id
        avail = scoring.availability_modifier(c, stats)
        loc = scoring.location_modifier(c)
        recency = scoring.recency_modifier(c)
        applied = scoring.applied_ml_signal(c)
        chaser = scoring.is_title_chaser(c)
        imp[cid] = impact.impact_score(c)
        applied_scores[cid] = applied
        chasers[cid] = chaser
        mods[cid] = (c, avail, loc)
        rec[cid] = recency
        fac[cid] = row
        rel = scoring.facet_relevance(row)
        raw[cid] = scoring.combine(rel, avail, loc, recency, applied, chaser, imp[cid], c)
        fl = disqualifiers.flags(c)
        if fl:
            flagged[cid] = fl
        scored.append((cid, config.DISQUALIFIER_FLOOR if fl else raw[cid]))

    top = scoring.order_top(scored, config.TOP_N)
    from src import reasoning_a
    reasons, _ = reasoning_a.generate_variant_a(top, mods)

    return [
        {
            "candidate_id": cid,
            "rank": rank,
            "score": float(score),
            "reasoning": reasons[cid],
            "facet_scores": [
                {"facet": f, "score": float(fac[cid][i])}
                for i, f in enumerate(active_facets)
            ],
            "availability_modifier": float(mods[cid][1]),
            "location_modifier": float(mods[cid][2]),
            "recency_modifier": float(rec[cid]),
            "integrity_status": _integrity_status(cid, violations),
            "applied_ml_score": float(applied_scores[cid]),
            "impact_score": float(imp[cid]),
            "title_chaser": chasers[cid],
            "disqualifier_status": flagged.get(cid) or None,
        }
        for rank, (cid, score) in enumerate(top, 1)
    ]


# ---------------------------------------------------------------------------
# CLI entry point (unchanged behaviour)
# ---------------------------------------------------------------------------

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

    # Variant A Run
    t = time.perf_counter()
    results_a = rank_candidates(
        jd_text,
        embedder=embedder,
        cross=cross,
        index=index,
        bm25=bm25,
        id_order=id_order,
        candidate_pool=pool,
        excluded=excluded,
        stats=stats,
    )
    print(f"rank Variant A: {len(results_a)} candidates scored, {time.perf_counter() - t:.1f}s")
    _report_cli(results_a, pool, excluded)
    # Write outputs to the path specified by --out
    write_csv(args.out, results_a)
    print(f"write: {len(results_a)} rows to {args.out}")
    ok_a = validate(args.out)

    total = time.perf_counter() - start
    print(f"Validation: {'PASS' if ok_a else 'FAIL'}")
    print(f"Total time elapsed: {total:.1f}s")
    if total > config.RUNTIME_WARN_S:
        print(f"WARN: total runtime {total:.1f}s exceeds the {config.RUNTIME_WARN_S}s budget")
    sys.exit(0 if ok_a else 1)


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
    n_founding = sum(1 for cid, (_, _, hard, _) in violations.items() if "predates_company_founding" in hard)
    print(f"integrity: {len(excluded)} excluded ({frac:.2%} of pool), "
          f"{n_hard} hard contradictions (incl. {n_founding} company founding violations), "
          f"{len(excluded) - n_hard} corroborated soft, "
          f"{outlier.sum()} outliers, {time.perf_counter() - t:.1f}s")
    for cid, (title, company, v, kind) in list(excluded.items())[:15]:
        print(f"  excluded {cid} [{kind}]: {title} at {company} [{', '.join(v)}]")
    if frac > config.MAX_EXCLUSION_FRACTION:
        print(f"STOP: exclusions exceed {config.MAX_EXCLUSION_FRACTION:.0%} of the pool, "
              f"not finalizing {out_path}. Review the gate before proceeding.")
        sys.exit(2)
    return set(excluded)


def _report_cli(results, pool, excluded):
    print("top 20:")
    for row in results[:20]:
        cand = pool.get(row["candidate_id"])
        title = f"{cand.profile.current_title} at {cand.profile.current_company}" if cand else "?"
        print(f"  {row['rank']:2d} {row['candidate_id']} score={row['score']:.4f} {title}")


def write_csv(path, results):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in results:
            w.writerow([row["candidate_id"], row["rank"], repr(row["score"]), row["reasoning"]])


def validate(path):
    r = subprocess.run([sys.executable, config.VALIDATOR, path], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
    return r.returncode == 0


if __name__ == "__main__":
    main()
