# Changelog

## Upgrades - Parallel A/B Reasoning Pipeline

### Shared Upstream Gaps Fixed
- **Multiplicative Scoring Modifiers**: Changed the location and availability modifier from a late additive modifier to an early multiplicative gating factor, preventing location mismatches from ranking high.
- **Evaluation-Framework Facet**: Added a new facet query to `config.py` checking for experience with metrics (NDCG, MRR, MAP) and A/B testing. Extended `src/features.py` to index candidates' `headline` as well as summaries and full career histories.
- **Title-Chaser Detection**: Added detailed title-chaser logic matching rapid Senior -> Staff -> Principal title progressions during short-tenure company hops.
- **Honeypot Company Founding Gate**: Compiled a lookup list of company founding years in `precompute.py`. Added a hard contradiction rule disqualifying candidates claiming starts predating company founding years.

### Variant A
- Implemented structured slot-filling sentence builders with a strict grounding verification filter and deterministic skeleton selection over 8 distinct templates to guarantee maximum diversity and limit template reuse below 20%.

### Variant B
- Integrated a LightGBM regressor trained offline on proxy labels (heuristic combiner scores).
- Computed TreeSHAP values for all candidates in the top 100 and mapped the top 3 positive and top 1 negative SHAP features directly to phrasing pools explaining the model's actual internal weights.
