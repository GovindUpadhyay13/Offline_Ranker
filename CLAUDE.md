# Candidate Ranking - build rules

## Goal
Rank the top 100 candidates from a 100k pool for the Senior AI Engineer JD, with a score and a one-line justification per candidate, written to a CSV that passes validate_submission.py.

## Hard constraints
- CPU only, no GPU.
- No network access during ranking. No API calls during ranking. No model downloads during ranking.
- Ranking of 100k must finish under 5 minutes on 16 GB RAM.
- A one-time offline precompute step is allowed to use the network (to fetch a vendored model and build caches). Ranking must not.

## Architecture (two binaries)
- precompute.py (offline, network allowed): download a small embedding model into models/ (local, gitignored), embed every candidate's text evidence, build a FAISS index and a BM25 index, persist all artifacts to disk.
- rank.py (runtime, no network, under 5 min): load artifacts, embed only the JD once, retrieve with BM25 plus dense via FAISS fused by Reciprocal Rank Fusion (k=60) to a shortlist, run the integrity gate, cross-encoder rerank the survivors, apply data-derived modifiers, sort, write the CSV, then run validate_submission.py.

## Relevance signal
- Matching is semantic relevance of candidate evidence to the JD, read from career_history descriptions and summary, not from the skills list. No keyword whitelist is used as the core matching logic.
- Title, consulting-firm, and research-only checks are soft signals living in config, never hard string gates.

## Integrity gate (not hardcoded)
- Fit an Isolation Forest on the suspicious-signal subspace of the real pool so outliers are defined by the pool's own distribution. Subsample to avoid masking.
- Combine with interpretable consistency checks: sum of career duration vs years_of_experience, role duration vs start and end dates, claimed tenure vs the company's earliest appearance in the pool.
- Exclude a candidate from the top 100 only when the outlier score and a consistency violation corroborate, to avoid false positives.
- Treat -1 in github_activity_score and offer_acceptance_rate as missing, never as a low score.

## Modifiers (derived from data)
- Behavioral availability and location fit are computed from real percentiles of the pool, not fixed cutoffs.

## Output rules (validator enforced)
- Header exactly: candidate_id,rank,score,reasoning. Exactly 100 rows.
- rank 1..100 once each. score is a float, non-increasing as rank increases.
- Ties break by candidate_id ascending. Deterministic: fixed seeds, stable sort.

## Code style (strict)
- No em-dashes anywhere, in code, comments, docstrings, strings, prints, or README. Use commas, colons, or parentheses.
- No emojis anywhere.
- Comment only what is non-obvious: a unit, a reason, a gotcha. Do not write comments that restate the code. No step-by-step narration comments. No ASCII banner blocks.
- Write terse, idiomatic Python as an experienced engineer would. Short clear names. No redundant type hints on trivial locals. No try/except wrapping everything, only real IO and parse boundaries.
- No utils.py junk drawer. Name modules by responsibility.
- No scattered magic numbers in logic. Tunable constants sit in one small config with a one-line reason each.
- Plain minimal stdout: stage name, a count, a timing. Nothing decorative.

## Proof obligations
- rank.py must run with networking disabled and still complete. Provide a way to demonstrate this.
