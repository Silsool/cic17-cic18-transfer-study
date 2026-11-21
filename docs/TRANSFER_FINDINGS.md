# Binary transfer and coverage: working findings

Recorded: 2026-11-20. Status: initial observations from saved results; notebook review is not complete. Follow [the roadmap](ANALYSIS_ROADMAP.md) before expanding the experiments.

## Experiment being described

Train on 1,679,976 cleaned 2017 rows; evaluate on 419,995 held-out 2017 rows and 500,000 cleaned 2018 rows. The source split is random and stratified by original label. Preprocessing is training-fitted; both classifiers use a fixed 0.5 attack-score threshold. The forest uses resource-bounded trees and bootstrap samples drawn from the full training pool.

## Initial observations

From full-evaluation metrics (`binary_baseline_results/baseline_overall.csv`, local artifact), using `scope = all`:

| Model | Held-out 2017 attack recall | 2018 attack recall | Held-out 2017 benign FPR | 2018 benign FPR |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 95.84% | 23.69% | 1.82% | 5.49% |
| Random forest | 99.85% | 23.12% | 0.031% | 0.161% |

There is a substantial transfer gap for these models and settings. Excluding training feature-hash matches does not remove it: attack recall is 92.58% versus 21.86% for logistic regression and 99.72% versus 21.27% for the forest. These restricted subsets have different compositions, so this is a sensitivity comparison, not an isolated estimate of a duplicate effect.

From coverage population counts (`binary_baseline_results/coverage/coverage_population_counts.csv`, local artifact), based on 15,000 uniform evaluation queries per split:

| Population | Supported | Intermediate | Low support |
| --- | ---: | ---: | ---: |
| Held-out 2017 | 89.33% | 9.48% | 1.19% |
| 2018 | 33.80% | 26.95% | 39.25% |

Under the selected source-fitted continuous-feature PCA geometry and training-calibrated thresholds, many more 2018 queries have weak training support. This is a representation-dependent observation, not proof of distributional disjointness.

## Review still needed

- [ ] Interpret per-attack recall and benign FPR, including counts and raw labels absent from training.
- [ ] Review the distance definition, reference deduplication, and threshold calibration chapter by chapter.
- [ ] Inspect performance by coverage band (`binary_baseline_results/coverage/coverage_performance.csv`, local artifact), including the hash-match-excluded sensitivity view.
- [ ] Compare per-label coverage results (`binary_baseline_results/coverage/coverage_per_label.csv`, local artifact) where sample counts permit; distinguish composition effects from within-label associations.
- [ ] Write a bounded conclusion and select one justified follow-up experiment.

## Limits to carry forward

Random-flow splitting does not establish session/capture independence. Feature hashes identify candidate identical prepared inputs, not proven raw duplicates. Their label agreement has **not** been audited. Matching attack names across years do not guarantee equivalent attacks. Coverage omits protocol and flags from its PCA coordinates even though the classifiers use them. Pointwise flow-level intervals do not account for correlated flows; small per-band attack counts can be misleading. Coverage–error association alone is not a causal explanation.

Further questions are parked in [the roadmap](ANALYSIS_ROADMAP.md), not added to the current experiment by default.
