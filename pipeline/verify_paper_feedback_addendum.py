"""Independently audit fixed baselines, the LOSO calibration correction and residuals."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.paper_artifacts import resolve_artifact
OUT = ROOT / "data/forecasting/review_revision/supplementary"


def exact_signed_rank(differences):
    ranks = rankdata(np.abs(differences))
    weights = np.rint(2 * ranks[differences != 0]).astype(int)
    observed = int(round(2 * ranks[differences > 0].sum()))
    totals = np.zeros(1 << len(weights), dtype=int)
    signs = np.arange(len(totals))
    for i, weight in enumerate(weights):
        totals += ((signs >> i) & 1) * weight
    return min(1.0, 2 * min(np.mean(totals <= observed), np.mean(totals >= observed)))


def audit_pairs(tests, metrics, draws):
    assert len(tests) == 18
    raw_p = []
    for row in tests.itertuples():
        wide = metrics[metrics.horizon == row.horizon].pivot(index="subject_id", columns="model_key", values="mae")
        delta = (wide[row.model_b] - wide[row.model_a]).to_numpy()
        p = exact_signed_rank(delta)
        low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
        np.testing.assert_allclose([p, delta.mean(), low, high],
            [row.p_value, row.mean_delta_b_minus_a, row.bootstrap_ci_lower, row.bootstrap_ci_upper],
            rtol=0, atol=1e-12)
        raw_p.append(p)
    raw_p = np.asarray(raw_p)
    order = np.argsort(raw_p)
    corrected = np.empty(18)
    corrected[order] = np.minimum(1, np.maximum.accumulate(raw_p[order] * np.arange(18, 0, -1)))
    np.testing.assert_allclose(corrected, tests.p_value_holm_all_18, rtol=0, atol=1e-12)


def main():
    predictions = pd.read_csv(OUT / "additional_baseline_predictions.csv")
    keys = ["subject_id", "session_id", "horizon", "anchor_idx", "target_idx"]
    assert len(predictions) == 29820 and not predictions.duplicated(keys).any()
    session_values = []
    for (subject, session), group in predictions.groupby(["subject_id", "session_id"]):
        prefix = ROOT / f"data/scores/sub-{int(subject[-2:]):02d}_ses-{int(session[-2:]):02d}_split-online_scores.npz"
        with np.load(prefix) as saved:
            scores = saved["score_raw"].astype(float)
            calibration = float(saved["session_cal_mean_raw"])
        session_values.append((subject, session, calibration, scores.mean()))
        for row in group.itertuples():
            b, t = row.anchor_idx, row.target_idx
            assert b >= 9 and t < 50 and t - b == row.horizon
            expected = [scores[t], float(np.float32(np.sum(scores[b-9:b+1]) / 10)),
                        calibration, np.sum(scores[:b+1]) / (b+1)]
            actual = [row.y_true_raw, row.context_score_mean, row.calibration_mean, row.expanding_online_mean]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    rows = []
    for model in ["context_score_mean", "calibration_mean", "expanding_online_mean"]:
        frame = predictions.assign(error=np.abs(predictions.y_true_raw - predictions[model]))
        recomputed = frame.groupby(["subject_id", "horizon"]).error.mean().reset_index(name="mae")
        rows.append(recomputed.assign(model_key=model))
    metrics = pd.concat(rows)
    names = ["subject_id", "horizon", "model_key"]
    stored = pd.read_csv(OUT / "additional_baseline_subject_metrics.csv")
    np.testing.assert_allclose(metrics.set_index(names).mae.sort_index(),
        stored.set_index(names).mae.sort_index(), rtol=0, atol=1e-12)
    tests = pd.read_csv(OUT / "additional_baseline_paired_stats.csv")
    draws = np.random.default_rng(20261011).integers(0, 15, (10000, 15))
    audit_pairs(tests, metrics, draws)

    # Independent centered-covariance OLS reconstruction, using only training
    # subjects' outcomes. The generator uses a design-matrix least-squares fit.
    sessions = pd.DataFrame(session_values, columns=["subject_id", "session_id", "cal", "online"])
    fits = pd.read_csv(OUT / "calibration_correction_fold_parameters.csv")
    corrected_predictions = pd.read_csv(OUT / "calibration_correction_predictions.csv")
    assert len(fits) == 15 and len(corrected_predictions) == len(predictions)
    matched_correction = predictions.merge(corrected_predictions, on=keys,
                                            validate="one_to_one", suffixes=("", "_corrected"))
    np.testing.assert_allclose(matched_correction.y_true_raw, matched_correction.y_true_raw_corrected,
                               rtol=0, atol=2e-15)
    np.testing.assert_allclose(matched_correction.calibration_mean, matched_correction.calibration_mean_corrected,
                               rtol=0, atol=2e-15)
    assert (matched_correction.fold_subject_id == matched_correction.subject_id).all()
    model = "corrected_calibration_mean"
    for row in fits.itertuples():
        training = sessions[sessions.subject_id != row.fold_subject_id]
        assert len(training) == row.n_training_sessions == 98
        subjects = set(training.subject_id)
        assert len(subjects) == row.n_training_subjects == 14
        assert row.fold_subject_id not in subjects
        assert subjects == set(row.training_subject_ids.split("|"))
        x, y = training.cal.to_numpy(), training.online.to_numpy()
        dx = x - x.mean()
        slope = np.dot(dx, y - y.mean()) / np.dot(dx, dx)
        intercept = y.mean() - slope * x.mean()
        np.testing.assert_allclose([row.intercept, row.slope], [intercept, slope], rtol=0, atol=1e-12)
        held_out = matched_correction[matched_correction.subject_id == row.fold_subject_id]
        np.testing.assert_allclose(held_out[model], intercept + slope * held_out.calibration_mean,
                                   rtol=0, atol=1e-12)
    assert corrected_predictions.groupby(["subject_id", "session_id"])[model].nunique().eq(1).all()
    frame = matched_correction.assign(error=np.abs(matched_correction.y_true_raw - matched_correction[model]))
    correction_metrics = frame.groupby(["subject_id", "horizon"]).error.mean().reset_index(name="mae")
    correction_metrics["model_key"] = model
    saved_correction = pd.read_csv(OUT / "calibration_correction_subject_metrics.csv")
    assert len(saved_correction) == 135
    np.testing.assert_allclose(correction_metrics.set_index(names).mae.sort_index(),
        saved_correction.set_index(names).mae.sort_index(), rtol=0, atol=1e-12)
    correction_summary = pd.read_csv(OUT / "calibration_correction_summary.csv")
    for row in correction_summary.itertuples():
        values = correction_metrics[correction_metrics.horizon == row.horizon].sort_values("subject_id").mae.to_numpy()
        low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
        np.testing.assert_allclose([values.mean(), values.std(ddof=1), low, high],
            [row.mae_mean, row.mae_sd_subjects, row.mae_ci_lower, row.mae_ci_upper], rtol=0, atol=1e-12)
    correction_tests = pd.read_csv(OUT / "calibration_correction_paired_stats.csv")
    assert correction_tests.model_b.eq(model).all()
    assert set(correction_tests.model_a) == {"context_score_mean", "expanding_online_mean"}
    audit_pairs(correction_tests, pd.concat([metrics, correction_metrics]), draws)

    # Independently reconstruct the displayed GRU's session correlations from
    # archived predictions, using explicit centered dot products.
    points = pd.read_csv(OUT / "residual_diagnostic_h5_points.csv")
    cols = keys + ["ablation", "target_view", "y_true", "y_pred_mu"]
    archive = pd.read_csv(resolve_artifact(ROOT / "data/forecasting/step8_predictions.csv"), usecols=cols)
    archive = archive[(archive.ablation == "score_only") &
                      (archive.target_view == "context_residual") & (archive.horizon == 5)]
    matched = points.merge(archive, on=keys, validate="one_to_one")
    assert len(matched) == 3780
    np.testing.assert_allclose(matched.predicted_residual, matched.y_pred_mu, rtol=0, atol=1e-12)
    correlations = []
    for (subject, session), group in matched.groupby(["subject_id", "session_id"]):
        x, y = group.predicted_residual.to_numpy(), group.observed_residual.to_numpy()
        dx, dy = x - x.mean(), y - y.mean()
        np.testing.assert_allclose(dx, group.predicted_residual_centered, rtol=0, atol=1e-12)
        np.testing.assert_allclose(dy, group.observed_residual_centered, rtol=0, atol=1e-12)
        correlations.append((subject, np.dot(dx, dy) / np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)),
                             np.sqrt(np.dot(dx, dx) / np.dot(dy, dy))))
    subject = pd.DataFrame(correlations, columns=["subject_id", "r", "ratio"]).groupby("subject_id").mean()
    summary = pd.read_csv(OUT / "residual_association_summary.csv")
    target = summary[(summary.model_key == "gru_score_only") &
                     (summary.target_view == "context_residual") & (summary.horizon == 5)].iloc[0]
    low, high = np.quantile(subject.r.to_numpy()[draws].mean(axis=1), [.025, .975])
    np.testing.assert_allclose([subject.r.mean(), subject.ratio.mean(), low, high],
        [target.pearson_r_mean, target.sd_ratio_mean, target.pearson_r_ci_lower, target.pearson_r_ci_upper],
        rtol=0, atol=1e-12)
    provenance = json.loads((OUT / "additional_analysis_provenance.json").read_text(encoding="utf-8"))
    for item in provenance["input_files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    report = {"passed": True, "baseline_examples_verified": 29820, "subject_maes_verified": 405,
        "exact_signed_rank_tests_verified": 18, "holm_family_size": 18,
        "paired_bootstrap_intervals_verified": 18, "h5_residual_gru_pairs_verified": 3780,
        "h5_residual_gru_session_correlations_verified": 105,
        "training_only_affine_fits_verified": 15, "corrected_calibration_predictions_verified": 29820,
        "corrected_calibration_subject_maes_verified": 135, "calibration_correction_tests_verified": 18,
        "calibration_correction_holm_family_size": 18, "calibration_correction_summary_intervals_verified": 9,
        "provenance_input_hashes_verified": len(provenance["input_files"])}
    (OUT / "additional_analysis_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
