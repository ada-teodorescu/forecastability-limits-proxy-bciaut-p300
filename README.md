# forecastability-limits-proxy-bciaut-p300

Research code and saved results for **Forecastability Limits of a P300-Derived EEG Proxy in BCIAUT-P300**, by Ada Daniela Teodorescu and Alin Brîndușescu.

The benchmark evaluates 17 model variants under leave-one-subject-out validation at horizons of 5, 10, 15, 20, and 25 online blocks. Models use a 10-block history within the same session. The context mean has the lowest mean raw-score MAE at every tested horizon; the specified paired comparisons establish no reliable residual-target improvement.

Publication DOI: pending. The published paper will be linked here once available.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/` | EEG loading, preprocessing, proxy construction, model fitting, and evaluation |
| `pipeline/` | Entry points for the main workflow and the two robustness experiments |
| `tests/` | Synthetic tests and data-dependent integration tests |
| `data/scores/` | Saved online proxy scores for the 105 subject-session sequences |
| `data/forecasting/` | Final forecasting metrics, model selections, ranking predictions, and robustness results |
| `results/figures/`, `outputs/forecasting/` | Selected analysis figures |
| `pyproject.toml`, `uv.lock` | Dependencies and resolved package versions |
| `CITATION.cff` | Software citation metadata |

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
uv run python pipeline/08_step8_temporal.py
```

Step 1 writes preprocessed epochs to `data/processed/`; Steps 3 and 4 write covariance and tangent-space features to `data/features/`. Step 5 writes proxy scores to `data/scores/`. Later steps write forecasting targets, metrics, predictions, and model selections to `data/forecasting/`.

Step 1 accepts `--raw-root` and `--processed-root`. If the processed path changes, supply the corresponding `--inventory-path` to Steps 2 and 3. Keep downstream input and output roots aligned. Inspect each main step's options with `uv run python pipeline/<script>.py --help`.

Reruns can overwrite saved results. Use another checkout or the supported path flags to retain the provided results. Numerical results can vary with platform and library versions.

## Robustness experiments

```powershell
uv run python pipeline/exp30_step8_grouped_inner_cv.py
uv run python pipeline/exp31_expanded_hp_search.py
```

These run grouped subject-level inner validation and an expanded GRU hyperparameter search. Their saved results are in the corresponding folders under `data/forecasting/`.

## Saved results

| Files under `data/forecasting/` | Contents |
| --- | --- |
| `step7_*` | Classical-model fold metrics, aggregate summaries, paired comparisons, and selected parameters |
| `step8_fold_metrics.csv`, `step8_summary.csv`, `step8_paired_stats.csv`, `step8_model_selection.csv` | GRU forecasting results and model selections |
| `step8_anomaly_predictions.csv`, `step8_anomaly_metrics.csv`, `step8_anomaly_paired_stats.csv` | Final low-proxy ranking predictions, subject metrics, and paired comparisons |
| `threshold_sensitivity.csv` | Ranking results at the evaluated engineering thresholds |
| `step6_detrended_exploration.csv`, `step6c_step5_score_decomposition.csv` | Score structure and descriptive diagnostics |

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
