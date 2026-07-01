"""FastAPI service wrapping the candidate ranking pipeline.

On startup, loads all precomputed artifacts and the full candidate pool once.
POST /rank accepts a job description and returns the top-100 ranked candidates.
GET /health returns {"status": "ok"} once startup completes.

Run:
  .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from rank import rank_candidates
from src import integrity, io, retrieval, scoring

faiss.omp_set_num_threads(1)  # avoid faiss/torch OpenMP clash on macOS


def _fmt_duration(months: int) -> str:
    if months <= 0:
        return "< 1 mo"
    y, m = divmod(months, 12)
    if y and m:
        return f"{y}yr {m}mo"
    return f"{y}yr" if y else f"{m}mo"


# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------

@dataclass
class _State:
    embedder: Any = None
    cross: Any = None
    index: Any = None
    bm25: Any = None
    id_order: list = field(default_factory=list)
    all_candidates: dict = field(default_factory=dict)
    excluded: set = field(default_factory=set)
    violations: dict = field(default_factory=dict)
    n_hard_excluded: int = 0
    n_soft_excluded: int = 0
    stats: dict = field(default_factory=dict)
    ready: bool = False


_state = _State()


def _load_artifacts():
    """Load all models and precompute pool-level data. Called once at startup."""
    t_total = time.perf_counter()

    print("startup: loading models...")
    t = time.perf_counter()
    _state.embedder = SentenceTransformer(config.EMBED_DIR)
    _state.cross = CrossEncoder(config.CROSS_DIR)
    print(f"startup: models loaded in {time.perf_counter() - t:.1f}s")

    print("startup: loading retrieval artifacts...")
    t = time.perf_counter()
    _state.index = retrieval.load_dense(config.FAISS_PATH)
    _state.bm25 = retrieval.load_bm25(config.BM25_PATH)
    with open(config.IDS_PATH, encoding="utf-8") as f:
        _state.id_order = json.load(f)
    print(f"startup: retrieval artifacts loaded in {time.perf_counter() - t:.1f}s")

    # Single file pass: load all candidates and build company bounds simultaneously.
    print("startup: loading candidate pool (this takes ~60s for 100k candidates)...")
    t = time.perf_counter()
    bounds: dict = {}
    all_candidates: dict = {}
    for c in io.read_candidates(config.DATA_PATH):
        all_candidates[c.candidate_id] = c
        integrity.update_company_bounds(bounds, c)
    print(f"startup: {len(all_candidates)} candidates loaded in {time.perf_counter() - t:.1f}s")

    # In-memory pass: integrity features, violations, pool stats.
    print("startup: computing integrity gate and pool statistics...")
    t = time.perf_counter()
    feat, ids_all, violations = [], [], {}
    gaps, resps = [], []
    for c in all_candidates.values():
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
            violations[c.candidate_id] = (
                c.profile.current_title, c.profile.current_company, hard, soft
            )

    _state.stats = scoring.pool_stats(gaps, resps)

    matrix = integrity.impute(np.asarray(feat, dtype=np.float64))
    outlier = integrity.outlier_flags(matrix)
    outlier_ids = {ids_all[i] for i in np.flatnonzero(outlier)}

    excluded: set = set()
    for cid, (_, _, hard, soft) in violations.items():
        if hard:
            excluded.add(cid)
        elif soft and cid in outlier_ids:
            excluded.add(cid)

    frac = len(excluded) / max(len(ids_all), 1)
    n_hard = sum(1 for cid in excluded if violations[cid][2])
    n_soft = len(excluded) - n_hard
    print(
        f"startup: integrity gate done — {len(excluded)} excluded ({frac:.2%}), "
        f"{n_hard} hard, {n_soft} soft+outlier, "
        f"{outlier.sum()} outliers, {time.perf_counter() - t:.1f}s"
    )

    if frac > config.MAX_EXCLUSION_FRACTION:
        print(
            f"WARN: exclusions ({frac:.2%}) exceed {config.MAX_EXCLUSION_FRACTION:.0%} — "
            "review the integrity gate; service will still start"
        )

    _state.all_candidates = all_candidates
    _state.excluded = excluded
    _state.violations = violations
    _state.n_hard_excluded = n_hard
    _state.n_soft_excluded = n_soft
    _state.ready = True
    print(f"startup: complete in {time.perf_counter() - t_total:.1f}s")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(title="Candidate Ranking API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    """Serve the recruiter dashboard frontend."""
    return FileResponse("index.html")


class RankRequest(BaseModel):
    job_description: str


@app.get("/health")
def health():
    if not _state.ready:
        raise HTTPException(status_code=503, detail="loading")
    return {"status": "ok"}


@app.post("/rank")
def rank(req: RankRequest):
    if not _state.ready:
        raise HTTPException(status_code=503, detail="loading")
    if not req.job_description.strip():
        raise HTTPException(status_code=422, detail="job_description must not be empty")

    t_start = time.perf_counter()

    results = rank_candidates(
        req.job_description,
        embedder=_state.embedder,
        cross=_state.cross,
        index=_state.index,
        bm25=_state.bm25,
        id_order=_state.id_order,
        candidate_pool=_state.all_candidates,
        excluded=_state.excluded,
        stats=_state.stats,
        violations=_state.violations,
    )

    elapsed = time.perf_counter() - t_start
    print(f"/rank: {elapsed:.1f}s for {len(results)} results")
    if elapsed < 10 or elapsed > 30:
        print(f"TIMING: {elapsed:.1f}s is {'fast' if elapsed < 10 else 'slow'} "
              f"(target 10–30s); check cross-encoder batch size or CPU load")

    for r in results:
        cand = _state.all_candidates.get(r["candidate_id"])
        if cand:
            p = cand.profile
            history = sorted(
                cand.career_history,
                key=lambda role: (not role.is_current, -(role.start.toordinal() if role.start else 0)),
            )
            r["profile"] = {
                "current_title": p.current_title,
                "current_company": p.current_company,
                "years_of_experience": p.years_of_experience,
                "career_history": [
                    {
                        "company": role.company,
                        "title": role.title,
                        "duration": _fmt_duration(role.duration_months),
                    }
                    for role in history[:6]
                ],
            }

    scores = [r["score"] for r in results]
    return {
        "summary": {
            "total_pool_size": len(_state.all_candidates),
            "total_excluded_by_integrity": len(_state.excluded),
            "excluded_hard": _state.n_hard_excluded,
            "excluded_soft": _state.n_soft_excluded,
            "score_range": {
                "min": float(min(scores)) if scores else 0.0,
                "max": float(max(scores)) if scores else 0.0,
            },
        },
        "candidates": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
