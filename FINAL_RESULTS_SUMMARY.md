# Final Results Summary

All results are from completed experiments. No retraining, new split or new statistical method was used for finalization.

## Audit

- Labels 0-4 map to ratings 1-5. Reviewer IDs and per-class counts agree with saved manifests.
- Each evaluation split contains 3,000 shared unique comments across seeds, not 9,000 independent observations.
- Reviewer-cluster resampling and finite-sample class quantiles were verified.
- Zero-event [0,0] bootstrap intervals do not establish zero population coverage.
- Temperature/conformal rows are distinct but share reviewers; OOD validity is not established.
- The original 16 tests pass; existing experiment artifacts and weights are unchanged.

## Primary OOD validation comparison

Values are three-seed means after temperature scaling; CI is conditional on fitted models and calibration rules.

| Model | Method | Overall coverage | One-star coverage (95% CI) | Set size | Worst-class | Macro class |
|---|---|---:|---|---:|---:|---:|
| LR | marginal | 0.8999 | 0.0000 [0.0000 to 0.0000] | 1.8807 | 0.0000 | 0.5212 |
| LR | class_conditional | 0.8939 | 0.8981 [0.8301 to 0.9584] | 2.2880 | 0.8420 | 0.8929 |
| DistilBERT | marginal | 0.8996 | 0.4167 [0.3043 to 0.5299] | 1.7839 | 0.4143 | 0.7169 |
| DistilBERT | class_conditional | 0.8914 | 0.9815 [0.9510 to 1.0000] | 2.3433 | 0.8731 | 0.9164 |

The LR zero-event interval is degenerate and is not a valid claim of population certainty. Class-conditional sets redistribute coverage and are not necessarily supersets of marginal sets.

## Evidence and limits

LR/BERT validation set-size increases: 0.407/0.559 labels. One-star calibration counts are 32/24/26 for seeds 42/123/2026; fixed ID/validation/OOD test counts are 26/36/48. Large observed improvements do not make these rare-class estimates precise.

Marginal OOD-minus-ID coverage differences: LR -1.24 percentage points [-2.98, 0.35]; DistilBERT -0.84 [-2.22, 0.62]. Both conditional intervals cross zero.

DistilBERT improves Macro-F1 in the paired comparison; an accuracy advantage is not established. Reviewer-level conditional coverage is not guaranteed. Previous test exposure makes the study exploratory. The contribution is a reproducible empirical characterization of an existing-method trade-off, with explicit statistical limitations.

## Files

- [English report (PDF)](FINAL_RESEARCH_REPORT.pdf) and [Markdown](FINAL_RESEARCH_REPORT.md).
- [Classwise counts, coverage and size](results/stage3/class_coverage.csv).
- [Seed-averaged comparison and intervals](results/stage3/class_conditional_comparison.csv).
- [Paired method changes](results/stage3/mean_method_difference.csv).
- [Classifier metrics](results/stage2/multiseed_metrics.csv).
- [Final audit](results/final_audit/FINAL_STATISTICAL_AUDIT.json) and [verified counts](results/final_audit/verified_sample_counts.csv).
