# Reproducing the 11 October 2026 paper revision

The release includes the original 17-model benchmark at horizons 5-25, the five-model horizons 1-4 extension, corrected ranking, the additional fixed baselines and affine calibration correction, and residual association diagnostics. The [compiled supplement](../../output/pdf/supplementary_results.pdf) gives the methods and Tables S1-S19.

## Rebuild analyses from supplied artifacts

Run from the repository root after `uv sync --locked`:

```powershell
uv run python pipeline/verify_paper_release.py
uv run python pipeline/verify_paper_inference.py --output tmp/inference_check
uv run python pipeline/verify_short_horizons.py
uv run python pipeline/build_paper_supplement.py --keep-main-figure
uv run python pipeline/verify_paper_feedback_addendum.py
```

These commands verify the release, the original 340 contrasts and 40 Holm families, and all 161,700 short-horizon predictions/600 subject MAEs/32 paired tests. The supplement builder reproduces Tables S1-S19 and the additional diagnostics, including 29,820 examples for the fixed baselines and the 15 training-only affine calibration fits. It does not refit the original classical or GRU models.

Both calibration and online scores for all 105 sessions are supplied. Large original and short-horizon prediction tables are losslessly compressed as `.csv.gz`; analysis loaders prefer an uncompressed CSV if one has subsequently been regenerated. The release manifest records compressed and uncompressed SHA-256 hashes.

The manuscript is maintained separately. To check its 85 main-table entries, add `--manuscript <external-paper.tex>` to the supplement builder. This external file is neither needed for normal analysis rebuilding nor included in the release bundle.

## Refit the forecasting models

Obtain the dataset and run Steps 1-7 as described in the root README. Full upstream EEG epochs, covariance/tangent features and checkpoints are generated separately.

Use the dedicated historical training command for the paper's GRUs:

```powershell
uv run python pipeline/reproduce_paper_gru.py --experiment main
uv run python pipeline/reproduce_paper_gru.py --experiment exp30
uv run python pipeline/reproduce_paper_gru.py --experiment exp31
```

These use 200 epochs, patience 20 and no gradient clipping, with the paper's three ablations and original task/seed ordering. `main` uses the last two sorted training subjects for validation; `exp30` and `exp31` use grouped inner validation, with `exp31` adding dropout 0.0 to 0.2. They evaluate the paper's LOSO comparisons and write to separate `data/forecasting/paper_reproduction/<experiment>/` directories. Path flags and `--max-workers` are available. A regenerated Step 7 prediction CSV is required by the Step 8 pipeline for baseline reporting.

The preserved original implementation from thesis commit `48de5b0` is in `src/experiments/exp41_historical_step8.py`. The dedicated command uses the current pipeline with explicitly restored historical training settings and corrected session-aware ranking. Synthetic parity tests compare both validation training and final mean/scale predictions with the preserved original implementation. The general `08_step8_temporal.py` and old exp30/exp31 entrypoints retain their later defaults (100 epochs, patience 10, gradient clipping at norm 1.0); use the dedicated command for this paper.

For the five-model extension, generate upstream EEG artifacts and use a new output directory:

```powershell
uv run python pipeline/exp41_paper_short_horizons.py --stage all --output-root data/forecasting/short_horizon_rerun --workers 2
```

The existing historical protocol hashes the original environment, sources, upstream inputs and protected results. The portable release has adapted entrypoints and a different dependency lock file, so it must not resume that historical locked protocol. Original protocol and manifest records are retained as provenance. A new output directory locks the current rerun snapshot before fitting. Platform and library changes can affect numerical results; bitwise historical reproduction is not asserted.

## Compile and package the supplement

The supplement source, its generated tables, bibliography and bibliography style are included. Obtain IEEEtran through a TeX distribution, then compile from `docs/paper_revision/`:

```powershell
tectonic --outdir ../../output/pdf supplementary_results.tex
```

After deliberately regenerating analyses or compiling the supplement, refresh the short-horizon artifact hashes and the release manifest:

```powershell
uv run python pipeline/verify_paper_release.py --refresh-short-manifest --write-manifest
uv run python pipeline/build_paper_supplement.py --bundle-only
```

The ZIP is a generated local artifact under `output/supplementary/`; it is not committed. `docs/paper_revision/release_manifest.json` identifies the included files, while the analysis manifests identify the numerical inputs and bootstrap settings. The original thesis manifests under `data/forecasting/review_revision/provenance/` and `short_horizons/provenance/` retain historical hashes; they may name separately generated EEG or private manuscript files. They are historical records, not the current release inventory.
