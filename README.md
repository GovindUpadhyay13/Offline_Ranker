# Redrob Candidate Ranking

Ranks the top 100 candidates from a 100k pool for the Senior AI Engineer JD, with
a score and a one-line justification per candidate, written to a CSV that passes
`validate_submission.py`.

## Two-binary design

The work splits into a stage that may use the network and a stage that may not.

**precompute.py** (run once, network allowed). It vendors a small CPU embedding
model and a cross-encoder into `models/` (gitignored), reads every candidate,
builds each candidate's evidence text from career-history descriptions plus the
summary, embeds all of them, and builds two retrieval indexes: a dense FAISS
index over the vectors and a BM25 lexical index over the same text. It persists
the vectors, both indexes, and a candidate_id order file to `artifacts/`. This is
the only stage allowed to touch the network, and it touches it only to fetch the
two models, never to send candidate data anywhere.

**rank.py** (run at submission time, no network, under 5 minutes). It loads the
vendored models and the precomputed artifacts, embeds the JD once, retrieves with
BM25 and dense FAISS, fuses the two ranked lists with Reciprocal Rank Fusion
(k=60) into a shortlist, reranks the survivors with the local cross-encoder,
applies pool-percentile modifiers, sorts, writes `submission.csv`, and runs the
validator. It sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` before importing
any model code, so loading is strictly local.

Splitting this way keeps the expensive, network-dependent work (model download,
embedding 100k texts) out of the timed, offline path. The timed path only embeds
one JD and reranks a bounded shortlist.

## How the scoring maps to the JD

The JD is explicit that the right answer is not keyword matching. Two of its
sentences drive the design: "if their career history shows they built a
recommendation system at a product company, they're a fit," and "a candidate who
has all the AI keywords listed as skills but whose title is Marketing Manager is
not a fit." So matching reads demonstrated work, not the skills list.

- **Relevance comes from evidence text, not skills.** Evidence is the summary
  plus every role's title and description. The skills array is deliberately
  excluded from matching so a stuffed skills list earns nothing on its own. This
  answers the JD's "reasoning about the gap between what the JD says and what the
  JD means."
- **Hybrid retrieval, then a cross-encoder.** Dense FAISS captures semantic
  similarity (a candidate who built retrieval or ranking systems without using
  the exact words). BM25 captures direct lexical overlap. RRF fuses them so
  neither dominates. The cross-encoder then reads each shortlisted candidate's
  evidence against the JD as the primary relevance signal, which is what
  separates a real systems builder from a keyword match.
- **Behavioral availability as a modifier.** The JD and the signals doc both say
  a perfect-on-paper candidate who has not logged in for months and rarely
  answers recruiters is, for hiring, not available. Availability is a small
  multiplicative modifier built from real pool percentiles: recency of last
  activity, recruiter response rate, and the open-to-work flag. The `-1`
  sentinels in `github_activity_score` and `offer_acceptance_rate` mean "no
  history" and are treated as missing, never as a low score.
- **Location fit as a modifier.** The JD prefers Noida, Pune, and other Tier-1
  Indian metros and does not sponsor visas. Location fit is a small modifier:
  positive for target metros, neutral-to-positive for India with willingness to
  relocate, negative for outside India.
- **Applied-ML experience as an additive bonus.** The JD's ideal has 4 to 5
  years specifically in applied ML/AI roles, distinct from total experience.
  `applied_ml_signal` sums months in roles whose title or description shows
  ML/AI/NLP/search/ranking/recommendation/retrieval work and saturates at the
  target, adding a small bonus (`APPLIED_ML_WEIGHT`).
- **Title-chasing as a soft penalty.** The JD rejects candidates who hop
  companies roughly every 1.5 years. `is_title_chaser` averages tenure per
  company (so internal promotions are not hops) and, across several companies,
  applies a small fractional penalty (`TITLE_CHASER_PENALTY`), never a floor.
- **Hard disqualifiers as a score floor, only when unambiguous.** Three JD
  disqualifiers in `src/disqualifiers.py` floor the final score: `consulting_only`
  (every company a consulting firm), `pure_research` (every affiliation academic),
  `domain_mismatch` (repeated CV/speech/robotics evidence with no NLP/IR anywhere).
  Each has an escape clause so a mixed or borderline history never triggers.
- **Modifiers are small and multiplicative.** Relevance stays primary; the
  modifiers nudge. Weights live in `config.py`, each with a one-line reason.
- **One-line justification.** Each row gets a justification built from structured
  facts plus a counterfactual clause naming the single weakest lever (for
  example, "would rank higher if based in or open to Pune or Noida"). No LLM is
  involved.

Soft signals the JD names (consulting-firm-only careers, pure-research-without-
production) live in `config.py` as tunable lists, never as hard string gates.

## Layout

```
config.py            every tunable constant, one reason each
precompute.py        offline cache builder (network allowed)
rank.py              runtime ranker (no network, under 5 min)
src/io.py            streaming JSONL reader into typed records
src/features.py      evidence text and numeric features
src/retrieval.py     FAISS, BM25, and RRF fusion
src/integrity.py     integrity gate (isolation forest plus consistency checks)
src/rerank.py        cross-encoder reranking
src/scoring.py       modifiers, applied-ML and title-chaser signals, ordering
src/disqualifiers.py hard JD disqualifier flags applied as a score floor
src/reasoning.py     one-line justification builder
eval_harness.py      offline cohort scorecard for a submission CSV (auxiliary)
```

## Running it

Install dependencies, then build the caches once with network, then rank offline.

```
pip install -r requirements.txt
python precompute.py            # network allowed, vendors models, builds indexes
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Precompute takes roughly ten minutes on CPU, dominated by embedding 100k texts.
Ranking runs in well under five minutes.

## Proving the no-network guarantee

`rank.py` accepts `--prove-offline`, which installs a guard that makes every
outbound internet socket raise before any work begins. If any step tried to reach
the network, the run would fail loudly with "network access attempted during
ranking." A clean PASS under this flag is the proof that ranking is fully offline.

```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --prove-offline
```
