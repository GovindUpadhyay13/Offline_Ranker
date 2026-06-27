#!/usr/bin/env python3
"""Offline cohort scorecard for a submission CSV against the JD.

There was no eval harness in the repo, so this is a deliberately independent
check: it scores the top-100 cohort on the JD's own disqualifiers and
preferences, not on the ranker's cross-encoder score, so it can corroborate the
ranking rather than restate it. Each component is in [0, 1], higher is better,
and the composite is their unweighted mean. Read it as a directional cohort
quality measure, not an absolute ground truth (the dataset ships no labels).

  python3 eval/eval_harness.py submission.csv [--candidates candidates.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for config/src

import config
from src import disqualifiers, impact, io, scoring


def load_ids(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [row["candidate_id"] for row in csv.DictReader(f)]


def score(csv_path, data_path):
    ids = set(load_ids(csv_path))
    gaps, resps, cohort = [], [], {}
    for c in io.read_candidates(data_path):
        s = c.signals
        if s.last_active_date is not None:
            gaps.append((config.REFERENCE_DATE - s.last_active_date).days)
        if s.recruiter_response_rate is not None:
            resps.append(s.recruiter_response_rate)
        if c.candidate_id in ids:
            cohort[c.candidate_id] = c
    stats = scoring.pool_stats(gaps, resps)

    n = len(cohort)
    disq = sum(1 for c in cohort.values() if disqualifiers.flags(c))
    chasers = sum(1 for c in cohort.values() if scoring.is_title_chaser(c))
    applied = sum(scoring.applied_ml_signal(c) for c in cohort.values()) / n
    handson = sum(1 for c in cohort.values() if scoring.recency_modifier(c) > 0) / n
    located = sum(1 for c in cohort.values() if scoring.location_modifier(c) > 0) / n
    # availability is in [-1, 1]; rescale to [0, 1] for a comparable component
    avail = sum((scoring.availability_modifier(c, stats) + 1) / 2 for c in cohort.values()) / n

    parts = {
        "disqualifier_clean": 1 - disq / n,
        "applied_ml": applied,
        "not_title_chaser": 1 - chasers / n,
        "handson_recent": handson,
        "location_fit": located,
        "availability": avail,
    }
    composite = sum(parts.values()) / len(parts)
    # Diagnostic only, kept out of the composite so it stays independent of the
    # signal the ranker now folds in: mean quantified-impact of the cohort.
    diag = {"impact": sum(impact.impact_score(c) for c in cohort.values()) / n}
    return parts, composite, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--candidates", default=config.DATA_PATH)
    args = ap.parse_args()

    parts, composite, diag = score(args.submission, args.candidates)
    for k, v in parts.items():
        print(f"  {k:20s} {v:.4f}")
    print(f"  {'composite':20s} {composite:.4f}")
    for k, v in diag.items():
        print(f"  {k:20s} {v:.4f} (diagnostic, not in composite)")


if __name__ == "__main__":
    main()
