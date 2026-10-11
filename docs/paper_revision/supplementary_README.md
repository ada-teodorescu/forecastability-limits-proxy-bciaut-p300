# Forecastability limits of a P300-derived EEG proxy in BCIAUT-P300

This is the release package **forecastability-limits-proxy-bciaut-p300**, updated on 11 October 2026. Project repository: [forecastability-limits-proxy-bciaut-p300](https://github.com/ada-teodorescu/forecastability-limits-proxy-bciaut-p300).

The supplementary methods and results are provided as a separate companion document. They are not part of the six-page conference manuscript. Main-paper references to supplementary sections and Tables S1--S19 refer to this document. The local archive includes its source, generated inputs, and compiled PDF for repository upload; building the archive does not establish public repository availability.

The paper is accompanied by supplementary methods, complete saved results, the exact original comparison families, corrected ranking inputs, and code to regenerate the supplementary analyses. The original 17-variant, 5--25-block benchmark remains unchanged. An additional five-model experiment fits learned models at horizons 1--4 and contributes 32 new paired MAE comparisons, reported separately. Two further fixed baselines and a continuous-residual diagnostic use saved scores/predictions without model retraining. A final post-hoc affine calibration baseline is fitted only on training subjects and contributes a separate 18-test family.

## Documents

- [Supplement source](supplementary_results.tex) and [compiled supplement](../../output/pdf/supplementary_results.pdf).
- [Reproduction guide](reproduction.md), including historical GRU training and compressed prediction tables.
- The main manuscript and private reviewer correspondence are maintained separately.

## Reproduce the supplement

From the extracted package root, create the environment from pyproject.toml and uv.lock:

~~~powershell
uv sync --locked
uv run python pipeline/build_paper_supplement.py --keep-main-figure
~~~

Use `--keep-main-figure` to rebuild the supplement while preserving the existing main-paper Figure 5 files byte for byte.

The builder also regenerates main-paper Figure 5 from the saved lag
correlations and paired GRU MAE summaries. Its upper panel is a retrospective
diagnostic; its lower panel uses held-out forecasting results and the existing
descriptive bootstrap intervals. It fits no model and saves a vector PDF,
high-resolution PNG, and input/plot provenance JSON.
To regenerate only that figure, run:

~~~powershell
uv run python pipeline/make_conference_figures.py --figure structure
~~~

The script matches original paired means/differences to subject metrics, reproduces pooled AUC from corrected probabilities, and derives descriptive intervals, session-restricted discrimination, Gaussian width, and score variance. It also verifies all 40 short-horizon table entries, reconstructs their 600 subject MAEs from 161,700 predictions, and reproduces all 32 new paired tests, both Holm corrections, and bootstrap intervals. With `--manuscript <external-paper.tex>`, it additionally checks all 80 original main-table MAEs and the five expanding-mean entries. It fits only the two-parameter affine calibration baseline; it does not refit original classical or GRU models or alter historical result CSVs.

Compile with Tectonic or a compatible LaTeX installation:

~~~powershell
Set-Location docs/paper_revision
tectonic --keep-logs --outdir ../../output/pdf supplementary_results.tex
Set-Location ../..
~~~

Run the original inference audit separately:

~~~powershell
uv run python pipeline/verify_paper_inference.py --output tmp/inference_verification
~~~

After compiling the supplement, refresh release hashes and rebuild the local archive if needed:

~~~powershell
uv run python pipeline/verify_paper_release.py --refresh-short-manifest --write-manifest
uv run python pipeline/build_paper_supplement.py --bundle-only
~~~

The archive contains derived results and relevant code. Reproducing the complete upstream pipeline requires raw EEG from the [BCIAUT-P300 dataset release](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.568104/full). Raw EEG and third-party research-paper PDFs are not included.

## Result files

Canonical outputs are under data/forecasting/review_revision/supplementary/.

| File | Contents |
| --- | --- |
| full_model_summary.csv | 170 summaries: all 17 variants, two target views, five horizons; subject SDs, paired differences, descriptive intervals, original direct-test status |
| subject_metrics.csv | 2,550 original subject-fold metric rows |
| paired_mae_differences.csv | 2,400 subject differences: 16 alternatives minus the context mean |
| original_tested_contrasts.csv | Original 340 comparisons with p-values, Holm families, original intervals and method metadata |
| contrast_templates.csv | Seven classical and ten GRU comparison templates |
| contrast_definitions.csv | 23 distinct instantiated ordered model pairs |
| selected_classical_comparators.csv | 20 metric/view/horizon-specific selected classical comparators |
| ranking_session_metrics.csv | Session AUC, average precision, counts and eligibility |
| ranking_subject_metrics.csv | Pooled-session and session-restricted subject aggregates |
| ranking_summary.csv | Equal-subject rankings and descriptive intervals for both schemes |
| robustness_summary.csv | All three GRUs, views and horizons under main, grouped CV and expanded dropout search |
| probabilistic_summary.csv | Gaussian NLL, interval coverage and width for all variants |
| variance_components.csv | Empirical population-variance partition of the constructed score |
| score_diagnostics.json | Recomputed context associations and variance share |
| snapshot_manifest.json | SHA-256 hashes, environment versions, bootstrap settings and ranking provenance |
| short_horizon_summary.csv | 40 summaries: five models, two target views, horizons 1--4 |
| short_horizon_subject_metrics.csv | 600 additional held-out subject metrics |
| short_horizon_paired_mae_stats.csv | All 32 new baseline contrasts, signed effects, intervals, raw p-values and both Holm corrections |
| additional_baseline_predictions.csv | 29,820 matched examples, original recent mean, avatar-calibration mean and expanding online mean |
| additional_baseline_subject_metrics.csv | 405 equal-subject baseline MAEs: three fixed rules, nine horizons, 15 subjects |
| additional_baseline_summary.csv | 27 additional fixed-baseline summaries |
| additional_baseline_paired_stats.csv | 18 post-hoc tests, paired differences, descriptive intervals and separate all-18 Holm correction |
| calibration_mean_session_diagnostics.csv / calibration_mean_diagnostics.json | All 105 avatar-calibration means, signs, distribution, calibration/online bias and both mean correlations |
| residual_association_session_metrics.csv | Within-session continuous-residual Pearson correlations, eligibility and predicted/observed SD ratios |
| residual_association_subject_metrics.csv / residual_association_summary.csv | Equal-session then equal-subject aggregates and descriptive subject uncertainty for 13 learned variants, both fitted target views and five horizons |
| residual_diagnostic_h5_points.csv / residual_diagnostic_h5_bins.csv | All points and five descriptive bins underlying supplementary Figure S1 |
| additional_analysis_provenance.json | New analysis definitions, source/input hashes, verification and bootstrap settings |
| calibration_correction_fold_parameters.csv | 15 training-only intercepts/slopes, fold identities and training-subject lists |
| calibration_correction_predictions.csv | 29,820 held-out affine-calibration forecasts on the same valid examples |
| calibration_correction_subject_metrics.csv / calibration_correction_summary.csv | 135 subject MAEs and nine equal-subject summaries with descriptive intervals |
| calibration_correction_paired_stats.csv | 18 correction-versus-context/expanding tests with their own all-18 Holm correction |
| additional_analysis_verification.json | Independent numerical audit of fixed baselines, the affine correction and displayed residual diagnostics |

Point metrics give held-out subjects equal weight. New paired MAE differences are **model minus context mean**; negative values indicate lower descriptive error. Original contrast differences are **model B minus model A**. Original Holm families are separate by horizon, view, metric and classical/GRU family. The original tested metrics are MAE and Gaussian NLL, not RMSE. Not every variant has an original direct baseline comparison.

Three GRU templates compare against the best of nine trained classical candidates, selected from aggregate outer-subject results separately for each metric, target view and horizon. These selected-comparator contrasts are exploratory. Their 20 selections are explicit in the comparator CSV; the 17 templates instantiate 23 ordered pairs. The context-mean comparisons do not use that selection rule.

New 95% intervals for the original benchmark use 10,000 subject bootstrap resamples, NumPy seed 20261007. They condition on fitted predictions, omit refitting/model-selection uncertainty, and inherit the shared-LOSO-fold independence limitation. They are unadjusted descriptive intervals; the original 340 tests are unchanged. The short-horizon experiment and the two additional fixed baselines have separate inference and bootstrap settings, described below.

Pooled AUC includes pairs from different sessions. The session-restricted diagnostic averages eligible sessions within subjects, then averages subjects. Horizon 5 has 103 eligible sessions and all 15 subjects; two sessions have no low-proxy examples. Average precision uses sklearn.metrics.average_precision_score. Session-restricted AP uses the same both-class eligibility set as AUC.

## Corrected ranking provenance

Only data/forecasting/review_revision/step8_anomaly_predictions.csv is consumed by the supplement for ranking. The released ranking files directly under data/forecasting/ are corrected copies and agree with this canonical input. The exp30 and exp31 point-forecasting results remain usable for the documented robustness checks; their historical ranking outputs are not used.

The proxy uses task-designated target annotations. Online computation would require synchronized cue and flash metadata, which may be accessible to an avatar-task controller. It does not establish actual participant attention or validate engagement.

Full-session means include future observations and are retrospective diagnostics, not causal forecasting inputs.

## Additional fixed baselines and residual diagnostics

Tables S14--S15 evaluate the avatar-calibration mean and expanding online mean on the original valid examples at horizons 1--5, 10, 15, 20 and 25. Calibration uses blocks 10--19. The expanding mean uses online blocks 0 through the anchor, including earlier observed blocks outside the original ten-block model inputs. Neither rule fits a model or selects a parameter. Fixed rules are evaluated once on the raw-score scale; their residual-coordinate errors would duplicate the same comparisons.

The expanding mean has lower descriptive MAE at all nine horizons and lower MAE after correction at eight; the horizon-25 difference remains inconclusive. At horizon 5, MAE is 0.170122 versus 0.174550 (all-18 Holm p=0.005798). The calibration mean has MAE 0.4517--0.4623: high session-mean correlation does not remove its level bias. Online means average 0.444 below avatar-calibration means. All 105 avatar-calibration means are positive, ranging from 1.279 to 3.280. Correlations are 0.958 for all calibration blocks and 0.946 for avatar calibration alone. Calibration blocks contribute to both prototypes and the threshold, making the endpoint an in-sample reference despite its causal availability.

There are 18 new post-hoc two-sided paired Wilcoxon tests with Pratt zero handling and one all-18 Holm family, separate from all historical families. Descriptive paired intervals use 10,000 subject bootstrap resamples with seed 20261011. These results reuse the original participants.

Tables S16--S17 and Figure S1 use saved held-out predictions to compare predicted and observed continuous context residuals within each session, average correlations within subjects, then average subjects equally. Both raw- and residual-target fits are included in the CSVs; the printed tables emphasize residual fits. Constant-output sessions have undefined correlation and are excluded explicitly. The context-mean correction is zero and also has undefined correlation. Residual elastic-net is constant at horizons 10 and 15; at horizon 20, 13 subjects/91 sessions are eligible. Other residual fits use all 15 subjects/105 sessions. The unadjusted descriptive intervals use 10,000 subject draws and seed 20261011, conditional on saved fits; no new association tests are performed. Small positive associations do not establish calibrated forecast magnitudes, behavioural validity or an MAE advantage.

Figure S1 centers predicted and observed residuals within sessions for display only. Five pooled predicted-deviation quantile bins give subjects equal weight within each bin; intervals resample subject means. Binning and full-session centering are retrospective diagnostics and do not tune a predictor. Original target rounding is retained for MAE verification, while the residual association is computed in float64 from saved float32 scores.

The supplement builder generates all additions automatically. To regenerate only these analyses:

~~~powershell
uv run python pipeline/build_paper_feedback_addendum.py
uv run python pipeline/verify_paper_feedback_addendum.py
~~~

The release includes the saved online/calibration scores, original point predictions, subject metrics, new analysis code, generated tables and display inputs. All original context MAEs and all 1,950 learned-model subject MAEs are reproduced at absolute tolerance 1e-12. The independent verifier reconstructs all 29,820 fixed-baseline examples and 405 subject MAEs, exhaustively enumerates signed-rank distributions for their 18 tests, checks Holm correction and all paired intervals, and reconstructs the displayed residual GRU's 105 within-session correlations from saved predictions. It also verifies the affine correction described below. No original GRU or classical model is retrained.

## Training-only affine calibration correction

Tables S18--S19 evaluate one post-hoc correction, `prediction = intercept + slope * avatar_calibration_mean`. The specification in `docs/paper_revision/calibration_correction_protocol.md` was written before evaluating this baseline. In each LOSO fold, equal-session OLS maps calibration means to full 50-block online-session means for the other 14 subjects (98 sessions). Full online outcomes are used only for training subjects. Held-out forecasting uses only the session's calibration mean and the training-fitted coefficients, shared across all anchors/horizons. No blend, clipping, tuning or selection among corrections is performed.

Corrected MAE is 0.1919--0.1946, compared with uncorrected calibration MAE 0.4517--0.4623. At horizon 5, correction reduces MAE from 0.4523 to 0.1933 (57.3%). Both online means nevertheless outperform the correction at all nine horizons after one separate Holm correction covering its 18 two-sided paired tests. This preserves the earlier fixed-baseline 18-test family and all historical families. The largest adjusted p-value is 0.0215 versus the context mean and 0.0137 versus the expanding mean. The analysis reuses the original participants and is exploratory.

The independent verifier computes slopes from centered covariance, confirms that each fit excludes the held-out subject, reconstructs all 29,820 forecasts and 135 subject MAEs, checks nine summary intervals, and independently audits all 18 correction tests, paired bootstrap intervals and Holm correction. The leakage test changes all held-out online outcomes and confirms that coefficients remain unchanged, with a training-outcome mutation as a positive control. Unadjusted descriptive intervals use 10,000 subject resamples and seed 20261011, conditional on fitted predictions; they omit refitting uncertainty and inherit shared-LOSO-fold dependence.

## Additional short-horizon experiment

Supplementary Tables S10--S13 show the context mean, persistence, five-score AR ridge, EEG-context PCR, and score-only GRU at horizons 1--4. Both target views use the original frozen proxy and ten-block histories under 15-subject LOSO. Classical tuning and the historical main GRU procedure are retained. The GRU uses a 200-epoch cap, patience 20 and no gradient clipping, through the original implementation preserved in git 48de5b0.

The four alternatives are compared directly with the context mean at each horizon/view. Primary Holm correction covers all 32 additional MAE tests; local four-test corrections are exposed separately. Descriptive intervals use 10,000 subject resamples and seed 20261008. This experiment reuses the original participants, covers five core models, and adds no ranking analysis.

No tested alternative establishes a gain. Residual EEG PCR's largest descriptive gain is 0.000748 MAE (0.43%) at horizon 4, with primary adjusted p=0.886. The context mean has the lowest raw-score MAE at every short horizon.

Completed predictions, selected settings, the locked protocol, reference checks, and inference audit are under `data/forecasting/review_revision/short_horizons/`. Large prediction tables are supplied as lossless `.csv.gz` files. The original experiment snapshot predates manuscript integration; its original-file hashes describe the state during fitting. Those records are preserved as historical provenance. The active release manifest inventories the shipped files and does not include the private manuscript.

To refit the experiment, use the complete repository with its frozen Step 4/5 artifacts. These large upstream feature artifacts are identified by hashes and are not duplicated in the release package. After manuscript or source revisions, select a new output directory:

~~~powershell
uv run python pipeline/exp41_paper_short_horizons.py --stage all --output-root data/forecasting/review_revision/short_horizons_reproduction --workers 2
~~~

The supplement generator consumes completed outputs; it never invokes this training command.

## Snapshot and release

The analysis manifest identifies a content-based source/input snapshot and records the project repository URL. `docs/paper_revision/release_manifest.json` inventories the portable release and records compressed/uncompressed prediction hashes. Original thesis manifests are retained separately as historical provenance. Building the archive is a local packaging operation; remote publication and access are not checked by these scripts.
