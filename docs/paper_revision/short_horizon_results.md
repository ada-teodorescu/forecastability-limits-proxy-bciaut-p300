# Paper benchmark extension: horizons 1–4

Completed with the paper's frozen proxy and ten-block history across all 15 subjects and 105 sessions.

The five models and all 32 baseline contrasts were specified before fitting. Both raw and context-residual targets were fitted separately.

## Mean absolute error

### raw_level

| Model | h=1 | h=2 | h=3 | h=4 |
| --- | ---: | ---: | ---: | ---: |
| Context mean | 0.172070 | 0.172444 | 0.173586 | 0.174084 |
| Persistence | 0.228350 | 0.229843 | 0.230899 | 0.230639 |
| AR ridge | 0.179276 | 0.181298 | 0.182993 | 0.182835 |
| EEG PCR | 0.342014 | 0.343274 | 0.342904 | 0.342537 |
| Score-only GRU | 0.173577 | 0.175451 | 0.175428 | 0.174803 |

### context_residual

| Model | h=1 | h=2 | h=3 | h=4 |
| --- | ---: | ---: | ---: | ---: |
| Context mean | 0.172070 | 0.172444 | 0.173586 | 0.174084 |
| Persistence | 0.228350 | 0.229843 | 0.230899 | 0.230639 |
| AR ridge | 0.172317 | 0.172598 | 0.173712 | 0.174110 |
| EEG PCR | 0.172071 | 0.172270 | 0.173040 | 0.173336 |
| Score-only GRU | 0.172950 | 0.173042 | 0.174020 | 0.174248 |

## Interpretation

No tested alternative achieves lower MAE with significance after Holm correction across all 32 extension comparisons. This is limited evidence about these models, rather than proof that short-term signal is absent.

3 of the 32 contrasts have a lower descriptive mean MAE. The CSV reports every effect, including improvements that are not statistically established.

## Protocol and assumptions

- Inputs are the original Step 4 tangent features and Step 5 raw proxy; no score construction, calibration reference, or normalization is refitted.
- Histories end at the observed anchor and stay within their subject/session. The 40, 39, 38, and 37 eligible anchors per session yield 4200, 4095, 3990, and 3885 examples at h=1–4.
- Outer validation leaves out each of the 15 subjects. Classical preprocessing and tuning use five-fold GroupKFold within the 14 training subjects, optimizing validation MAE.
- GRU validation uses the last two sorted training subjects; the four hidden-size/learning-rate candidates, NLL objective, batch size 64, dropout 0.2 and weight decay 0.0001 match the original specification. Refit uses all 14 training subjects for the selected best epoch.
- GRU point forecasts use the original implementation preserved in git 48de5b0, with a 200-epoch cap, patience 20 and no gradient clipping. Preliminary fits with the current implementation are archived separately and excluded because its training procedure differs from the original benchmark. The supplementary training description needs reconciliation with this historical source.
- Original full three-ablation task enumeration determines score-only GRU seeds. Worker processes isolate random states; historical bitwise identity is not claimed.
- EEG PCR uses the unchanged principal-subspace estimator and component grid. An exact symmetric eigen-decomposition computes the same PCA subspace, with full-SVD fallback for ill-conditioned inputs. Decompositions are shared across candidates and views. Tests, reference_check.json and eigen_factorization_check.json verify agreement with the original full-SVD predictions and selected settings.
- MAE averages examples within each subject, then averages the 15 subjects equally. Windows are not independent inference units.
- Two-sided Wilcoxon tests use the original Pratt zero handling. Primary Holm correction covers all 32 new MAE comparisons; four-test correction within each horizon/view is provided separately. These families differ from the original paper's seven-classical/ten-GRU contrast families because this is a five-model extension.
- Paired bootstrap intervals use 10000 subject resamples and are unadjusted descriptive intervals conditional on fitted predictions. Shared LOSO training limits independence; these intervals and tests do not prove equivalence.
- The extension follows reviewer feedback and uses already studied subjects; it is not a new independent validation cohort or externally preregistered experiment.

## Reproduction

~~~powershell
uv run python pipeline/exp41_paper_short_horizons.py --stage all --workers 2
~~~

Individual stages are prepare, reference-check, classical, gru, and report. Completed fold checkpoints are retained for resuming. The protocol hashes input files and source; changing the protocol requires a new output directory.

Results: data/forecasting/review_revision/short_horizons/summary.csv; full subject metrics, predictions, selections, paired statistics, protocol and verification are alongside it.

Original paper results and manuscript files were verified unchanged. No new ranking analysis was run.
