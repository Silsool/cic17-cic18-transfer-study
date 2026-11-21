# CIC2017 → CIC2018 transfer and training coverage

A reproducible notebook study of how binary network-intrusion detectors trained on improved CIC2017 transfer to an improved CIC2018 sample, and whether transfer errors are associated with weak coverage by source-training examples.

**Current focus:** the binary baseline and training-coverage analysis. PCA and HDBSCAN are earlier exploratory work, retained for context and targeted follow-up. Category-mixing analysis, label-conflict auditing, and multiclass classification are deferred.

## Organization

```text
notebooks/       Five study notebooks, with published execution outputs cleared
src/nids_transfer/
                 Reusable preprocessing, model, clustering, coverage and path code
docs/            Current roadmap and preliminary findings with limitations
requirements-tested.txt
                 Recorded local experiment versions
```

`data/`, `artifacts/`, environments, and `.local/` are ignored by Git. This repository does not distribute datasets, per-flow predictions, trained models, feature matrices, or raw notebook outputs. The original executed notebook snapshots are retained locally under `.local/notebook_snapshots/` when migrating the existing workspace; they are not published. Existing source datasets and results remain in their original locations.

## Notebook order

| Order | Notebook | Purpose / status |
| --- | --- | --- |
| 1 | [dataset_compatibility.ipynb](notebooks/dataset_compatibility.ipynb) | Intersect columns, cast 2018 types to the 2017 schema, validate and save new compatible copies. Does not overwrite raw files. |
| 2 | [dataset_comparison.ipynb](notebooks/dataset_comparison.ipynb) | Feature audits, distributions, descriptive pooled PCA, per-year/category variance and plots. Exploratory; pooled preprocessing is not reused for classifier fitting. |
| 3 | [dataset_clustering.ipynb](notebooks/dataset_clustering.ipynb) | Pooled-year HDBSCAN, benign and selected-label experiments, protocol/flag variants and sensitivity checks. Optional exploratory branch. |
| 4 | [binary_classification_baseline.ipynb](notebooks/binary_classification_baseline.ipynb) | **Main path:** freeze a raw-label-stratified 2017 80/20 row split, train source-only preprocessing and two binary classifiers, evaluate every cleaned held-out 2017 and sampled 2018 row. |
| 5 | [coverage_and_transfer_analysis.ipynb](notebooks/coverage_and_transfer_analysis.ipynb) | **Main path:** reuse frozen predictions, fit source-training continuous-feature PCA, measure training-neighbor coverage, compare performance across bands. No classifier retraining. |

With compatible data available, start at notebook 4 and then 5. Notebook 3 is not a prerequisite. Each notebook explains its method and should be run top to bottom in a fresh kernel.

## Installation

From the repository root, using Python 3.10+ (the recorded experiment used Python 3.12.2):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name cic-transfer --display-name 'CIC transfer study'
jupyter lab
```

Select the `CIC transfer study` kernel. The notebooks can also use the existing `IMT_data_analyser` environment. `requirements-tested.txt` records the versions used locally; no exact cross-version numerical equivalence is claimed.

Notebook startup finds the checkout and adds its `src/` directory to the import path. Launch Jupyter from this repository (or a notebook subdirectory); set `NIDS_STUDY_ROOT` if the kernel's working directory is elsewhere.

## Data and output paths

Input data are not supplied. Acquire the appropriate improved datasets independently and record their provenance. Do not substitute original CIC datasets under these filenames without reviewing their schemas and label policies.

Default input layout:

```text
data/
  CIC2017_improved.parquet
  CIC2018_improved_sample.parquet
  compatible/
    CIC2017_shared.parquet
    CIC2018_shared.parquet
```

The first notebook generates `compatible/` from the two raw Parquet inputs. If the compatible files already exist, it stops before overwriting them; skip this preparation stage when reusing validated compatible data.

Paths can be configured before launching Jupyter:

```bash
export NIDS_DATA_DIR=/absolute/path/to/data
export NIDS_ARTIFACT_DIR=/absolute/path/to/study-artifacts
jupyter lab
```

`NIDS_DATA_DIR` is the directory containing the input files and `compatible/`. `NIDS_ARTIFACT_DIR` is the parent of `binary_baseline_results/` and `clustering_results/`, and also receives exploratory PDF figures. The default output parent is `artifacts/` inside this checkout. The original local workspace is detected automatically when the sibling data and baseline-results directories exist, so its existing datasets/results need not be moved or recomputed. Explicit environment variables take precedence.

## Reproducibility and compute

- Stable row IDs combine year and Parquet source-row number. Schema alignment alone does not establish semantic dataset equivalence.
- The classifier target is `BENIGN = 0`, all other raw labels (including attempted variants) `= 1`. Original labels are retained for per-category reporting.
- Model preprocessing learns log means/scales, variable columns and protocol vocabulary from 2017 training only. Negative numeric values are treated as missing and audited. Identifiers, addresses, timestamps, labels, ports and ICMP codes are excluded.
- Classifiers use prepared original features, not pooled exploratory PCA. Baseline constructors were extracted from the original `ml_models.py` into `src/nids_transfer/models.py`; only logistic regression and random forest are supported here. Resource and solver overrides are documented in the notebook.
- Float32 disk-backed matrices make full-training fitting practical; they require roughly 0.8 GB in the recorded run. Additional models, PCA locations and the neighbor index require more disk space and working RAM. The local run used a machine with about 6.7 GiB RAM.
- Logistic regression fits the complete source training matrix. The resource-bounded forest fits 100 trees, each drawing up to 200,000 bootstrap rows from the full training pool. Predictions cover all cleaned evaluation rows in batches.
- Coverage uses training-only, unwhitened PCA retaining 95% variance, exact five-neighbor search against all unique float32 training PCA locations, and training-only q90/q99 thresholds. Query sampling separates population summaries from rare-label supplementation.
- Complete stages are cached. Choose a new artifact directory for changed baseline settings; use a separate coverage subdirectory for coverage sensitivity experiments. The registry retains original fitted-run provenance when cached models are reused, so relocating the notebooks does not invalidate the existing coverage cache.
- Raw datasets are never rewritten. Derived artifacts and local snapshots can contain flow information and should remain local unless deliberately reviewed for sharing.

## Preliminary observations

The completed local run used 1,679,976 training rows, 419,995 held-out 2017 rows, and 500,000 rows from a previously prepared 2018 sample. These numbers and metrics describe that sample and configuration, not every possible dataset revision or model.

| Model | Held-out 2017 attack recall | 2018 attack recall | Held-out 2017 benign FPR | 2018 benign FPR |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 95.84% | 23.69% | 1.82% | 5.49% |
| Random forest | 99.85% | 23.12% | 0.031% | 0.161% |

Among 15,000 uniform coverage queries per evaluation set, 1.19% of held-out 2017 and 39.25% of 2018 queries were in the low-support band. Weaker source support is associated with poor target detection, but attack composition differs between bands; this is not a causal conclusion.

The random-flow split can place related traffic on both sides. Candidate identical prepared inputs occur in training for about 20.5% of held-out 2017 versus 0.58% of 2018. Excluding these hash matches does not remove the transfer gap; actual label agreement among equal vectors has not yet been audited. Pointwise Wilson intervals do not account for capture/session dependence. Matching attack names across years do not establish equivalent populations.

See [the roadmap](docs/ANALYSIS_ROADMAP.md) for the current review sequence and parked questions, and [working findings](docs/TRANSFER_FINDINGS.md) for the report outline. Do not tune models or thresholds against 2018 and then interpret it as an untouched final test.

## Quick validation

```bash
python -m unittest discover -s tests -v
```

Tests check notebook syntax/portability, source-only preprocessing, row alignment, both baselines, coverage calibration and frozen-prediction joins on small synthetic inputs. They do not rerun the expensive full-data experiments. GitHub Actions runs the same checks.

No license is selected yet; public visibility alone does not grant a software or dataset license.
