# Post-hoc calibration correction: fixed analysis specification

Analysis date: 11 October 2026. This specification was written before evaluating the corrected baseline. The analysis was motivated by the review and earlier calibration diagnostics; it is exploratory, not preregistered or confirmatory.

Use one affine correction, with no blend, clipping, horizon-specific adjustment, or model selection. In each leave-one-subject-out fold, fit ordinary least squares to the 98 sessions of the other 14 subjects:

`online_session_mean = intercept + slope * avatar_calibration_mean`.

Each training session has equal weight (seven sessions per subject). The training outcome is the mean of all 50 online scores. Full online-session outcomes are used only for training subjects. Neither the held-out subject's online scores nor its online session means enter fitting. The held-out session forecast is the fitted intercept plus slope times its avatar-calibration mean, available before online forecasting. The same forecast and coefficients apply at all anchors and horizons.

Evaluate the original valid examples at horizons 1, 2, 3, 4, 5, 10, 15, 20 and 25. Report equal-subject MAE. Compare the correction with both the ten-block context mean and the expanding online mean, using 15 paired subject errors per comparison. Apply one separate Holm correction to all 18 new two-sided Wilcoxon tests with Pratt zero handling. Use 10,000 descriptive subject-bootstrap resamples with seed 20261011. Preserve the previous 18 fixed-baseline tests, original 340 tests and short-horizon 32-test families.

Save coefficients, training-subject identities, held-out predictions, subject metrics, summaries and paired statistics. Independently reconstruct coefficients using centered covariance, verify the absence of held-out fitting outcomes, enumerate signed-rank distributions and verify bootstrap intervals and Holm correction. These intervals condition on fitted predictions; they do not include refitting uncertainty and share the existing LOSO dependence limitation.

Do not refit original classical or GRU models. Keep main-paper Figure 5 unchanged. The corrected baseline is evaluated on the raw-score scale; residual-coordinate conversion would duplicate its errors.
