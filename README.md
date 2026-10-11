# forecastability-limits-proxy-bciaut-p300

Research code and saved results for **Forecastability Limits of a P300-Derived EEG Proxy in BCIAUT-P300**, by Ada Daniela Teodorescu and Alin Brîndușescu.

The original benchmark evaluates 17 model variants under leave-one-subject-out validation at horizons of 5, 10, 15, 20, and 25 online blocks. Models use a 10-block history within the same session. The context mean has the lowest mean raw-score MAE among these original variants; the specified paired comparisons establish no reliable residual-target improvement. The 11 October 2026 paper revision also includes five models at horizons 1-4, post-hoc calibration and expanding-online baselines, a training-only affine calibration correction, and residual association diagnostics. The expanding mean uses longer online history and improves on the context mean at eight of nine horizons after its separate correction.

Read the [supplementary methods and results](output/pdf/supplementary_results.pdf), including Tables S1-S19, and the [reproduction guide](docs/paper_revision/reproduction.md).

Publication DOI: pending. The published paper will be linked here once available.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/` | EEG loading, preprocessing, proxy construction, model fitting, and evaluation |
| `pipeline/` | Main workflow, historical paper-training command, robustness runs, additional analyses, and independent verifiers |
| `tests/` | Synthetic tests and data-dependent integration tests |
| `data/scores/` | Saved online and calibration proxy scores for all 105 subject-session sequences |
| `data/forecasting/` | Final forecasting metrics, model selections, ranking predictions, and robustness results |
| `results/figures/`, `outputs/forecasting/` | Selected analysis figures |
| `pyproject.toml`, `uv.lock` | Dependencies and resolved package versions |
| `CITATION.cff` | Software citation metadata |
| `docs/paper_revision/` | Supplement source, generated tables, protocols, reproduction guide, and release hashes |
| `output/pdf/supplementary_results.pdf` | Compiled supplementary methods and results |

## Setup

Use Python 3.10 or later and uv. Run commands from the repository root.

```powershell
uv sync --locked
```

The importable Python package is named `src`. The default development dependencies include pytest.

## Dataset

Obtain raw EEG from the [BCIAUT-P300 dataset release](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.568104/full) or its [Kaggle mirror](https://www.kaggle.com/datasets/disbeat/bciaut-p300).

The dataset contains 15 subjects with seven sessions each. Step 1 expects folders from `data/raw/SBJ01/S01/` through `data/raw/SBJ15/S07/`, each containing `Train/` and `Test/`.

- `Train/`: `trainData.mat`, `trainEvents.txt`, `trainTargets.txt`, `trainLabels.txt`.
- `Test/`: the corresponding `test*` files and `runs_per_block.txt`.

Raw EEG and full processed EEG features are obtained or generated separately.

## Reproduce the pipeline

Run these steps in order:

```powershell
uv run python pipeline/01_prepare_data.py
uv run python pipeline/02_step2_exploration.py
uv run python pipeline/03_step3_covariance.py
uv run python pipeline/04_step4_riemannian.py
uv run python pipeline/05_step5_scores.py
uv run python pipeline/06_step6_forecasting_task.py
uv run python pipeline/06b_step6_addendum.py
uv run python pipeline/06c_audit_upstream_target.py
uv run python pipeline/07_step7_benchmarks.py
uv run python pipeline/reproduce_paper_gru.py --experiment main
```

Step 1 writes preprocessed epochs to `data/processed/`; Steps 3 and 4 write covariance and tangent-space features to `data/features/`. Step 5 writes proxy scores to `data/scores/`. Later steps write forecasting targets, metrics, predictions, and model selections to `data/forecasting/`.

Step 1 accepts `--raw-root` and `--processed-root`. If the processed path changes, supply the corresponding `--inventory-path` to Steps 2 and 3. Keep downstream input and output roots aligned. Inspect each main step's options with `uv run python pipeline/<script>.py --help`.

The paper-training command uses the historical 200-epoch cap, patience 20, and no gradient clipping. It evaluates LOSO and writes to `data/forecasting/paper_reproduction/main/` by default. The general `pipeline/08_step8_temporal.py` command retains the later 100-epoch/patience-10/clipped defaults; those settings were not used for the saved paper fits. Training and refit parity with the preserved original implementation are tested on synthetic inputs.

Reruns can overwrite saved results. Use another checkout or the supported path flags to retain the provided results. Numerical results can vary with platform and library versions.

## Robustness experiments

```powershell
uv run python pipeline/reproduce_paper_gru.py --experiment exp30
uv run python pipeline/reproduce_paper_gru.py --experiment exp31
```

These run grouped subject-level inner validation and an expanded GRU hyperparameter search. Their saved results are in the corresponding folders under `data/forecasting/`.

## Additional analyses and verification

The added analyses can be rebuilt from the supplied scores and prediction tables without regenerating raw EEG or retraining the original forecasting models:

```powershell
uv run python pipeline/verify_paper_inference.py --output tmp/inference_check
uv run python pipeline/verify_short_horizons.py
uv run python pipeline/build_paper_supplement.py --keep-main-figure
uv run python pipeline/verify_paper_feedback_addendum.py
```

The supplement builder regenerates tables, figures, fixed-baseline diagnostics and the 15 training-only affine calibration fits. `--manuscript <external-paper.tex>` optionally checks all 85 main-table MAEs; the private manuscript is not required or packaged. Large prediction tables are stored as lossless `.csv.gz` files, read directly by these tools.

To refit the short-horizon models after generating the full upstream EEG artifacts, use a new output directory:

```powershell
uv run python pipeline/exp41_paper_short_horizons.py --stage all --output-root data/forecasting/short_horizon_rerun --workers 2
```

The shipped historical protocol records the original source/input snapshot. Use a new directory when running the adapted release sources. After deliberately regenerating outputs, refresh the active artifact and release hashes:

```powershell
uv run python pipeline/verify_paper_release.py --refresh-short-manifest --write-manifest
```

For an unchanged checkout, verify its release hashes with `uv run python pipeline/verify_paper_release.py`.

## Saved results

| Files under `data/forecasting/` | Contents |
| --- | --- |
| `step7_*` | Classical-model fold metrics, aggregate summaries, paired comparisons, and selected parameters |
| `step8_fold_metrics.csv`, `step8_summary.csv`, `step8_paired_stats.csv`, `step8_model_selection.csv` | GRU forecasting results and model selections |
| `step8_anomaly_predictions.csv`, `step8_anomaly_metrics.csv`, `step8_anomaly_paired_stats.csv` | Final low-proxy ranking predictions, subject metrics, and paired comparisons |
| `threshold_sensitivity.csv` | Ranking results at the evaluated engineering thresholds |
| `step6_detrended_exploration.csv`, `step6c_step5_score_decomposition.csv` | Score structure and descriptive diagnostics |
| `step7_predictions.csv.gz`, `step8_predictions.csv.gz` | Original held-out forecasting predictions, losslessly compressed |
| `review_revision/short_horizons/` | Five-model extension, predictions, selections, protocol, reference checks, and 32 paired tests |
| `review_revision/supplementary/` | Full model tables, corrected ranking summaries, additional baselines, calibration correction, residual diagnostics, and provenance |
| `review_revision/provenance/` | Preserved original thesis analysis manifests |

Point metrics give held-out subjects equal weight. The low-proxy endpoint is defined from calibration scores and has no clinical interpretation. Pooled-session ranking includes comparisons between sessions.

## Tests

```powershell
uv run python -m pytest -q
```

Synthetic tests run with the included files. Tests marked `requires_full_data` are skipped until the complete upstream EEG artifacts have been generated. Then run them with:

```powershell
uv run python -m pytest -m requires_full_data -q
```

## Citation

Use `CITATION.cff` for the code. The paper's DOI will be added after publication. A code license has not yet been assigned.
