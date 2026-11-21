# Dataset overlap and transfer: current roadmap

Updated: 2026-11-20.

## Main question

Does a binary detector trained on 2017 perform worse on 2018, and is that deterioration associated with weaker coverage by 2017 training examples?

The current phase is **understanding and interpreting existing results**. No new models, clustering experiments, or feature changes are needed yet.

## Current state

- Dataset compatibility, descriptive PCA, and exploratory clustering are available.
- [Binary baseline](../notebooks/binary_classification_baseline.ipynb): executed, with train-only preprocessing, frozen 2017 80/20 split, logistic regression, random forest, and full evaluation predictions.
- [Coverage analysis](../notebooks/coverage_and_transfer_analysis.ipynb): executed, with training-fitted PCA, a full training-location reference, sampled queries, and source-calibrated bands.
- Automated artifact checks passed. Human review and interpretation are still in progress.
- [Working findings](TRANSFER_FINDINGS.md) records initial observations and their limits; it is not a completed scientific conclusion.

## Main path

1. **Finish reviewing the binary baseline notebook.**
   - We have discussed the split, preprocessing, models, and the meaning of transformed feature-hash matches.
   - **Resume at “Frozen full-evaluation predictions.”** Read the global metrics and compare `all` with `no_training_hash_match`, then review per-attack recall and benign false-positive rate.
   - Check counts, uncertainty, raw-label differences, and labels absent from training before interpreting individual attacks.
2. **Review the coverage notebook.**
   - Understand source-fitted PCA, unique training locations, five-neighbor distances, training-only thresholds, and the separate sampling purposes.
   - Compare coverage distributions for held-out 2017 and 2018. Use all retained PCs for distances; PC1–PC2 is illustrative.
3. **Connect coverage to performance.**
   - Compare attack recall and benign FPR across bands, with denominators and uncertainty.
   - Inspect the same raw attack labels across bands where counts permit. This is per-category reporting, not an investigation of category mixing.
   - Separate changes in category prevalence from within-category performance changes. Matching names do not prove equivalent attack populations.
4. **Consolidate the findings note.**
   - Record supported observations, unresolved explanations, and one next experiment justified by those observations.
   - Do not claim that coverage causes errors from these descriptive associations.

Completion criterion: we can explain the overall transfer gap, identify where it concentrates, and distinguish supported observations from small-sample or representation-dependent findings.

## Parked sidequests

| Topic | Question | Revisit when |
| --- | --- | --- |
| Conflicting labels for identical inputs | Do verified equal prepared vectors carry different binary labels within 2017 training, across its split, or between training and 2018? | Investigating residual errors in well-supported regions. **Not checked yet.** Verify vector equality after hash matching; distinguish binary conflicts from attack-category conflicts. Do not automatically remove or relabel rows. |
| Grouped or temporal splits | Does the random row split benefit from related flows crossing the split? | Before strong generalization claims; first inspect available capture/session grouping information. |
| Coverage sensitivity | Do conclusions survive changes to PC count, neighbor count, calibration samples, and thresholds? | After understanding the default run; freeze classifier predictions while varying coverage settings. |
| Preprocessing choices | How much do negative-value handling, excluded columns, protocol, and flags affect results? | If conclusions appear dependent on the representation. |
| Label harmonization | Which differently named attacks are genuinely comparable across years? | Before attack-family equivalence claims or multiclass transfer. Preserve original labels and attempted variants for review. |
| Clustering and category mixing | Do error pockets form coherent groups, and do categories share those regions? | As targeted deeper analysis after coverage. Existing clustering remains available, but is not the first-pass framework. |

## Further work

- Multiclass classification, with an explicit policy for labels absent from training.
- Additional classifiers and training-only hyperparameter tuning.
- Reverse transfer: train on 2018, evaluate on 2017.
- Uncertainty estimates accounting for related flows or capture groups.
- Neural-model repairs before benchmarking the custom neural implementations: target encoding, transformer feature identity, and batched inference.

## Working rule

Complete the main-path review before opening another experiment. Record new questions here with a reason to revisit them. The parked questions do not invalidate the descriptive baseline, but they limit how strongly we can interpret it.
