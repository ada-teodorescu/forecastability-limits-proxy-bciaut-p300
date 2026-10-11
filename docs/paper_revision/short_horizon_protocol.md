# Short-horizon extension of the paper benchmark

This extension asks whether the negative result at 5–25 blocks misses useful forecasting at 1–4 blocks. The model set and comparisons were fixed before fitting these new horizons. It is an additional analysis on the existing participants, not an independent validation cohort.

The machine-readable specification, timestamps, source hashes and hashes of the frozen input artifacts are in data/forecasting/review_revision/short_horizons/protocol.json.

## What is held fixed

- The original session-calibrated proxy and tangent features are loaded from the saved Step 4/5 artifacts. Calibration references and proxy construction are not changed.
- A history contains ten observed online blocks, ending at anchor block b. The target is the score at b+h, where h is now 1, 2, 3 or 4. Every window stays in one subject/session.
- All 15 subjects and seven sessions per subject are used. Outer evaluation leaves out one complete subject at a time. Each subject contributes 280, 273, 266 or 259 test examples at the four horizons.
- Raw future scores and future scores minus the observed context mean are fitted separately, as in the original benchmark.
- Classical preprocessing and selection occur within five-fold GroupKFold on the 14 outer-training subjects. Selection minimizes mean inner-validation MAE, using the unchanged ridge and PCR grids.
- The GRU retains the original two-subject inner validation, Gaussian NLL objective, architecture, hyperparameter grid, optimizer, batch size, dropout, weight decay and refit rule. This experiment does not introduce MAE training or grouped-inner-CV GRU tuning.

## Five models

| Model | Information used |
| --- | --- |
| Context mean | Mean of the ten observed scores |
| Persistence | Last observed score |
| AR ridge | Five most recent scores |
| EEG PCR | Tangent features from the ten observed blocks |
| Score-only GRU | Ordered sequence of ten observed scores |

The reduced model set is a targeted extension, not a repeat of all 17 variants. Horizon-specific coefficients and GRU weights are fitted anew. No original results are overwritten.

## Comparisons and uncertainty

The primary metric is MAE. Average errors within a held-out subject, then average the 15 subject errors equally.

Compare each of the four alternatives directly with the context mean for every horizon and target view. This gives 32 comparisons. Use the existing two-sided paired Wilcoxon implementation with Pratt zero handling. The primary Holm correction covers all 32 additional MAE comparisons. Also expose four-test correction within each horizon/view, following the original paper's grouping by horizon and view. This is a different comparison family from the original seven classical and ten GRU contrasts.

Report signed differences as alternative minus context mean: negative values indicate lower MAE. Report every comparison, including favourable descriptive differences without established significance. Bootstrap intervals resample the 15 paired subject differences 10,000 times with seed 20261008 and remain unadjusted, descriptive intervals conditional on the fitted predictions. They omit refitting uncertainty. Shared LOSO training makes independence an approximation.

No ranking endpoint, new low-proxy threshold, alternative proxy, or new model architecture is introduced.

## Historical training settings

The original implementation is preserved in git commit `48de5b0`, at `src/models/step8_temporal.py`. It specifies a 200-epoch cap, patience 20 and no gradient clipping. The saved model-selection table contains best-epoch values up to 200. Current Step8Config defaults and the revised supplementary text instead state 100 epochs and patience 10; current code also adds gradient clipping.

The extension imports an immutable copy of the historical module, `src/experiments/exp41_historical_step8.py`, for point-forecast functions only. It uses 200/20 without gradient clipping. Preliminary fits from the current implementation are retained under `checkpoints/gru_current_code/` and excluded because of the configuration mismatch, before drawing conclusions from them. All production GRU tasks are restarted with the historical implementation. The supplementary training description should be reconciled with this source.

The score-only GRU retains the original three-ablation task enumeration for seed offsets, even though only score-only tasks are executed. Each worker process isolates its random state. Historical bitwise reproduction is not assumed.

As a reference check, the historical implementation was rerun at horizon 5 for held-out subject SBJ01 with the raw target. It selected hidden size 64, learning rate 0.001 and epoch 82, matching the saved paper selection. Validation NLL agrees to floating-point precision, and the maximum prediction difference is 4.77e-7. The evidence is in `gru_historical_reference_check.json`.

## Computation and verification

The same exact PCA principal subspace is computed through symmetric eigen-decomposition, with a full-SVD fallback for ill-conditioned inputs. Decompositions are reused across component candidates and target views. This leaves the estimator, grids and tuning unchanged. It is checked against the original full-SVD implementation in tests, the saved horizon-5 paper predictions for subject SBJ01, and completed full-SVD short-horizon fits for three subjects. The production agreement tolerance is an absolute prediction difference of 1e-8.

Completed fold checkpoints are retained. Execution stages can be resumed without discarding completed fits. Resource limits affect scheduling only.

Source snapshots and calculation changes are recorded under provenance/ and in protocol.json. A reporting precision fix validates the stored float32 residual targets directly. An allocation fix and the numerically checked PCA factorization reduce computation. The historical GRU implementation corrects the initially used current-code training procedure. The selected five models, targets, subject splits, tuning grids, objectives, seeds and statistical choices remain fixed; implementation changes and excluded preliminary fits are recorded separately.

The final verification reconstructs every subject MAE from its saved predictions, checks row counts, subject/session identity, anchor/target offsets, targets and exact fixed baselines, and checks that the frozen inputs and original result files are unchanged.

## Commands

~~~powershell
uv run python pipeline/exp41_paper_short_horizons.py --stage prepare
uv run python pipeline/exp41_paper_short_horizons.py --stage reference-check
uv run python pipeline/exp41_paper_short_horizons.py --stage classical --workers 2
uv run python pipeline/exp41_paper_short_horizons.py --stage gru --workers 2
uv run python pipeline/exp41_paper_short_horizons.py --stage report
~~~

The worker count can be increased if memory permits. The complete sequence is also available through --stage all.
