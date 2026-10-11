"""Reproduce additional baselines and residual diagnostics from saved artifacts.

The two fixed baselines use calibration or online blocks available at the
anchor. One affine calibration correction is fitted on training subjects only;
original forecasting models are not retrained. The fixed baselines and the
correction have separate 18-test post-hoc Holm families. Residual associations
and intervals are descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.paper_artifacts import resolve_artifact
from src.evaluation.forecasting_metrics import compute_paired_statistic, holm_correction

OUT = ROOT / "data/forecasting/review_revision/supplementary"
TEX = ROOT / "docs/paper_revision/supplementary_generated"
FIG = ROOT / "outputs/forecasting"
HORIZONS = (1, 2, 3, 4, 5, 10, 15, 20, 25)
MAIN_HORIZONS = (5, 10, 15, 20, 25)
SUBJECTS = [f"SBJ{i:02d}" for i in range(1, 16)]
SEED = 20261011
REPLICATES = 10000
CORRECTED = "corrected_calibration_mean"
LABELS = {
    "context_score_mean": "Context mean", "calibration_mean": "Calibration mean",
    "expanding_online_mean": "Expanding mean", "autoreg_ridge_5": "AR ridge (5)",
    "ridge_anchor_ts": "EEG ridge", "elasticnet_anchor_ts": "EEG elastic-net",
    "pcr_context_ts": "EEG PCR", "pls_context_ts": "EEG PLS",
    "riemannian_knn_anchor": "Riemannian KNN",
    "pcr_context_ts_permuted_target": "Permuted EEG PCR",
    "ridge_score5_anchor_ts": "Hybrid ridge", "pcr_score5_context_ts": "Hybrid PCR",
    "pls_score5_context_ts": "Hybrid PLS", "gru_score_only": "GRU score",
    "gru_eeg_reduced": "GRU EEG", "gru_hybrid": "GRU hybrid",
    CORRECTED: "Corrected calibration",
}
LEARNED = [k for k in LABELS if k not in
           {"context_score_mean", "calibration_mean", "expanding_online_mean", CORRECTED}]
INPUTS: set[Path] = set()


def ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    if not len(values):
        return (np.nan, np.nan)
    draws = np.random.default_rng(SEED).integers(0, len(values), (REPLICATES, len(values)))
    return tuple(np.quantile(values[draws].mean(axis=1), [.025, .975]))


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.17g")


def read(relative: str, **kwargs) -> pd.DataFrame:
    path = resolve_artifact(ROOT / relative)
    INPUTS.add(path)
    return pd.read_csv(path, **kwargs)


def causal_predictions(scores: np.ndarray, calibration_mean: float, horizon: int) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=np.float64)
    anchors = np.arange(9, len(scores) - horizon)
    context64 = np.array([scores[a - 9:a + 1].mean() for a in anchors])
    return pd.DataFrame({
        "anchor_idx": anchors, "target_idx": anchors + horizon,
        "y_true_raw": scores[anchors + horizon],
        # Match the original forecasting table's float32 context field.
        "context_score_mean": context64.astype(np.float32).astype(float),
        "context_mean_float64": context64,
        "calibration_mean": calibration_mean,
        "expanding_online_mean": np.cumsum(scores)[anchors] / (anchors + 1),
    })


def load_examples() -> tuple[pd.DataFrame, pd.DataFrame]:
    examples, sessions = [], []
    for u in range(1, 16):
        for j in range(1, 8):
            prefix = f"data/scores/sub-{u:02d}_ses-{j:02d}_split-"
            files = [ROOT / (prefix + kind + "_scores.npz") for kind in ["online", "calibration"]]
            INPUTS.update(files)
            with np.load(files[0]) as payload:
                online = payload["score_raw"].astype(float)
                cal_mean = float(payload["session_cal_mean_raw"])
            with np.load(files[1]) as payload:
                calibration = payload["score_raw"].astype(float)
                assert float(payload["session_cal_mean_raw"]) == cal_mean
            assert online.shape == (50,) and calibration.shape == (20,)
            np.testing.assert_allclose(calibration[10:20].mean(), cal_mean, rtol=0, atol=2e-7)
            subject, session = f"SBJ{u:02d}", f"S{j:02d}"
            sessions.append({"subject_id": subject, "session_id": session,
                "avatar_calibration_mean": cal_mean,
                "all_calibration_mean": calibration.mean(), "online_mean": online.mean(),
                "online_minus_avatar_calibration_mean": online.mean() - cal_mean})
            for horizon in HORIZONS:
                frame = causal_predictions(online, cal_mean, horizon)
                frame = frame.assign(subject_id=subject, session_id=session, horizon=horizon)
                examples.append(frame)
    return pd.concat(examples, ignore_index=True), pd.DataFrame(sessions)


def baseline_results(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    for model in ["context_score_mean", "calibration_mean", "expanding_online_mean"]:
        error = examples.assign(error=np.abs(examples.y_true_raw - examples[model]))
        group = error.groupby(["subject_id", "horizon"]).agg(mae=("error", "mean"),
                                                               n_examples=("error", "size")).reset_index()
        metrics.append(group.assign(model_key=model))
    metrics = pd.concat(metrics, ignore_index=True)
    assert len(metrics) == 405 and metrics.n_examples.eq(7 * (41 - metrics.horizon)).all()
    original = read("data/forecasting/step7_fold_metrics.csv")
    short = read("data/forecasting/review_revision/short_horizons/subject_metrics.csv")
    for stored in [original, short]:
        stored = stored[(stored.model_name == "context_score_mean") & (stored.target_view == "raw_level")]
        for row in stored.itertuples():
            actual = metrics[(metrics.subject_id == row.fold_subject_id) &
                (metrics.horizon == row.horizon) & (metrics.model_key == "context_score_mean")].mae.item()
            np.testing.assert_allclose(actual, row.mae, rtol=0, atol=1e-12)
    summaries, paired = [], []
    for horizon in HORIZONS:
        wide = metrics[metrics.horizon == horizon].pivot(index="subject_id", columns="model_key", values="mae").loc[SUBJECTS]
        for model in ["context_score_mean", "calibration_mean", "expanding_online_mean"]:
            values = wide[model].to_numpy()
            low, high = ci(values)
            summaries.append({"horizon": horizon, "model_key": model, "n_subjects": 15,
                "mae_mean": values.mean(), "mae_sd_subjects": values.std(ddof=1),
                "mae_ci_lower": low, "mae_ci_upper": high})
            if model != "context_score_mean":
                result = compute_paired_statistic(comparison_id=model + "_vs_context",
                    horizon=horizon, target_view="raw_level", metric="mae",
                    model_a="context_score_mean", model_b=model,
                    metric_a=wide.context_score_mean.to_numpy(), metric_b=values,
                    bootstrap_seed=SEED, bootstrap_replicates=REPLICATES)
                paired.append(vars(result))
    summary, tests = pd.DataFrame(summaries), pd.DataFrame(paired)
    tests["p_value_holm_all_18"] = holm_correction(tests.p_value.to_numpy())
    tests["p_value_holm"] = tests.p_value_holm_all_18
    save(examples, "additional_baseline_predictions.csv")
    save(metrics, "additional_baseline_subject_metrics.csv")
    save(summary, "additional_baseline_summary.csv")
    save(tests, "additional_baseline_paired_stats.csv")
    return summary, tests


def fit_calibration_correction(sessions: pd.DataFrame, held_out: str) -> dict[str, object]:
    """Fit a fixed two-parameter level map without held-out online outcomes."""
    training = sessions.loc[sessions.subject_id != held_out,
                            ["subject_id", "avatar_calibration_mean", "online_mean"]]
    assert len(training) == 98 and set(training.subject_id) == set(SUBJECTS) - {held_out}
    x = training.avatar_calibration_mean.to_numpy(dtype=float)
    y = training.online_mean.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    assert rank == 2 and np.isfinite(coefficients).all()
    return {"fold_subject_id": held_out, "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]), "n_training_subjects": 14,
            "n_training_sessions": 98,
            "training_subject_ids": "|".join(sorted(set(training.subject_id)))}


def corrected_calibration_results(examples: pd.DataFrame, sessions: pd.DataFrame
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    INPUTS.add(ROOT / "docs/paper_revision/calibration_correction_protocol.md")
    fits = pd.DataFrame([fit_calibration_correction(sessions, subject) for subject in SUBJECTS])
    predictions = examples.merge(fits[["fold_subject_id", "intercept", "slope"]],
        left_on="subject_id", right_on="fold_subject_id", validate="many_to_one")
    predictions[CORRECTED] = predictions.intercept + predictions.slope * predictions.calibration_mean
    assert len(predictions) == len(examples)
    error = predictions.assign(error=np.abs(predictions.y_true_raw - predictions[CORRECTED]))
    metrics = error.groupby(["subject_id", "horizon"]).agg(
        mae=("error", "mean"), n_examples=("error", "size")).reset_index().assign(model_key=CORRECTED)
    assert len(metrics) == 135 and metrics.n_examples.eq(7 * (41 - metrics.horizon)).all()
    references = pd.read_csv(OUT / "additional_baseline_subject_metrics.csv")
    combined = pd.concat([references, metrics], ignore_index=True)
    summaries, paired = [], []
    for horizon in HORIZONS:
        wide = combined[combined.horizon == horizon].pivot(
            index="subject_id", columns="model_key", values="mae").loc[SUBJECTS]
        values = wide[CORRECTED].to_numpy()
        low, high = ci(values)
        summaries.append({"horizon": horizon, "model_key": CORRECTED, "n_subjects": 15,
            "mae_mean": values.mean(), "mae_sd_subjects": values.std(ddof=1),
            "mae_ci_lower": low, "mae_ci_upper": high})
        for comparator in ["context_score_mean", "expanding_online_mean"]:
            result = compute_paired_statistic(comparison_id=CORRECTED + "_vs_" + comparator,
                horizon=horizon, target_view="raw_level", metric="mae", model_a=comparator,
                model_b=CORRECTED, metric_a=wide[comparator].to_numpy(), metric_b=values,
                bootstrap_seed=SEED, bootstrap_replicates=REPLICATES)
            paired.append(vars(result))
    summary, tests = pd.DataFrame(summaries), pd.DataFrame(paired)
    tests["p_value_holm_all_18"] = holm_correction(tests.p_value.to_numpy())
    tests["p_value_holm"] = tests.p_value_holm_all_18
    keys = ["subject_id", "session_id", "horizon", "anchor_idx", "target_idx"]
    save(fits, "calibration_correction_fold_parameters.csv")
    save(predictions[keys + ["fold_subject_id", "y_true_raw", "calibration_mean", CORRECTED]],
         "calibration_correction_predictions.csv")
    save(metrics, "calibration_correction_subject_metrics.csv")
    save(summary, "calibration_correction_summary.csv")
    save(tests, "calibration_correction_paired_stats.csv")
    return summary, tests


def residual_results(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = ["fold_subject_id", "subject_id", "session_id", "horizon", "target_view",
              "anchor_idx", "target_idx", "y_true"]
    classical = read("data/forecasting/step7_predictions.csv", usecols=common + ["model_name", "y_pred_mean"])
    neural = read("data/forecasting/step8_predictions.csv", usecols=common + ["ablation", "y_pred_mu"])
    classical = classical.rename(columns={"model_name": "model_key", "y_pred_mean": "prediction"})
    neural = neural.assign(model_key="gru_" + neural.ablation).rename(columns={"y_pred_mu": "prediction"})
    predictions = pd.concat([classical, neural.drop(columns="ablation")], ignore_index=True)
    predictions = predictions[predictions.model_key.isin(LEARNED)].copy()
    keys = ["subject_id", "session_id", "horizon", "anchor_idx", "target_idx"]
    assert not predictions.duplicated(keys + ["model_key", "target_view"]).any()
    assert (predictions.fold_subject_id == predictions.subject_id).all()
    matched = predictions.merge(examples[keys + ["y_true_raw", "context_mean_float64"]],
                                on=keys, how="left", validate="many_to_one")
    assert matched.y_true_raw.notna().all()
    residual_fit = matched.target_view.eq("context_residual")
    residual = matched.y_true_raw - matched.context_mean_float64
    expected_target = np.where(residual_fit, residual.astype(np.float32).astype(float), matched.y_true_raw)
    np.testing.assert_allclose(matched.y_true, expected_target, rtol=0, atol=1e-12)
    matched["observed_residual"] = residual
    matched["predicted_residual"] = np.where(residual_fit, matched.prediction,
                                            matched.prediction - matched.context_mean_float64)
    # Reproduce original MAE from archived targets and predictions, independently
    # of the canonical float64 residual used for association diagnostics.
    matched["original_error"] = np.abs(matched.y_true - matched.prediction)
    recomputed = matched.groupby(["fold_subject_id", "horizon", "target_view", "model_key"]).original_error.mean()
    originals = []
    for stage in [7, 8]:
        original = read(f"data/forecasting/step{stage}_fold_metrics.csv")
        original["model_key"] = "gru_" + original.ablation if stage == 8 else original.model_name
        originals.append(original)
    stored = pd.concat(originals)
    stored = stored[stored.model_key.isin(LEARNED)].set_index(recomputed.index.names).mae.sort_index()
    np.testing.assert_allclose(recomputed.sort_index(), stored, rtol=0, atol=1e-12)
    rows = []
    for key, group in matched.groupby(["model_key", "target_view", "horizon", "subject_id", "session_id"]):
        model, view, horizon, subject, session = key
        x, y = group.predicted_residual.to_numpy(), group.observed_residual.to_numpy()
        assert len(x) == 41 - horizon
        sx, sy = x.std(ddof=0), y.std(ddof=0)
        eligible = sx > 1e-12 and sy > 1e-12
        correlation = float(np.corrcoef(x, y)[0, 1]) if eligible else np.nan
        rows.append(dict(model_key=model, target_view=view, horizon=horizon,
            subject_id=subject, session_id=session, n_examples=len(x), eligible=eligible,
            pearson_r=correlation, predicted_sd=sx, observed_sd=sy,
            sd_ratio=sx / sy if sy > 1e-12 else np.nan))
    sessions = pd.DataFrame(rows)
    subjects = sessions.groupby(["model_key", "target_view", "horizon", "subject_id"]).agg(
        pearson_r=("pearson_r", "mean"), sd_ratio=("sd_ratio", "mean"),
        eligible_sessions=("eligible", "sum")).reset_index()
    summaries = []
    for (model, view, horizon), group in subjects.groupby(["model_key", "target_view", "horizon"]):
        assert set(group.subject_id) == set(SUBJECTS)
        eligible = group.pearson_r.notna()
        values = group.loc[eligible, "pearson_r"].to_numpy()
        low, high = ci(values)
        summaries.append(dict(model_key=model, target_view=view, horizon=horizon,
            n_subjects=len(values), eligible_sessions=int(group.eligible_sessions.sum()),
            pearson_r_mean=values.mean() if len(values) else np.nan,
            pearson_r_ci_lower=low, pearson_r_ci_upper=high,
            sd_ratio_mean=group.sd_ratio.mean()))
    summary = pd.DataFrame(summaries)
    assert len(summary) == 130 and len(subjects) == 1950 and len(sessions) == 13650
    save(sessions, "residual_association_session_metrics.csv")
    save(subjects, "residual_association_subject_metrics.csv")
    save(summary, "residual_association_summary.csv")
    # Display the score-only residual GRU at h=5, chosen for the paper's main
    # diagnostic rather than selected for an extreme observed association.
    display = matched[(matched.model_key == "gru_score_only") &
                      (matched.target_view == "context_residual") & (matched.horizon == 5)].copy()
    group = display.groupby(["subject_id", "session_id"])
    for name in ["observed_residual", "predicted_residual"]:
        display[name + "_centered"] = display[name] - group[name].transform("mean")
    save(display[keys + ["observed_residual", "predicted_residual",
                         "observed_residual_centered", "predicted_residual_centered"]],
         "residual_diagnostic_h5_points.csv")
    return summary, display


def table(filename: str, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [r"\begin{table}[!ht]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    (TEX / filename).write_text("\n".join(lines), encoding="utf-8")


def tables(baselines: pd.DataFrame, tests: pd.DataFrame, residuals: pd.DataFrame) -> None:
    wide = baselines.pivot(index="horizon", columns="model_key", values="mae_mean")
    table("additional_baselines.tex", "Additional fixed baselines on the original valid examples. "
        "MAE gives 15 subjects equal weight; the expanding mean uses all online blocks observed through the anchor.",
        "tab:supp_additional_baselines", ["Horizon", "Context mean", "Calibration mean", "Expanding mean"],
        [[str(h), *[f"{wide.loc[h, k]:.4f}" for k in
          ["context_score_mean", "calibration_mean", "expanding_online_mean"]]] for h in HORIZONS])
    table("additional_baseline_pairs.tex", r"Additional post-hoc paired MAE contrasts (alternative minus context mean). "
        r"Intervals are unadjusted descriptive 95\% subject bootstraps. $p_{18}$ corrects all 18 new two-sided tests.",
        "tab:supp_additional_baseline_pairs", ["Alternative", "$h$", r"$\Delta$ MAE", r"95\% interval", "$p_{18}$"],
        [[LABELS[r.model_b], str(r.horizon), f"{r.mean_delta_b_minus_a:+.5f}",
          f"[{r.bootstrap_ci_lower:+.5f}, {r.bootstrap_ci_upper:+.5f}]", f"{r.p_value_holm_all_18:.4f}"]
         for r in tests.itertuples()])
    residual = residuals[residuals.target_view == "context_residual"].set_index(["model_key", "horizon"])
    table("residual_associations.tex", "Within-session Pearson associations between predicted and observed "
        "context residuals for separately fitted residual-target models. Session correlations are averaged "
        "within subjects, then subjects equally. Subject uncertainty is in the companion CSV.",
        "tab:supp_residual_associations", ["Residual-target model", *[str(h) for h in MAIN_HORIZONS]],
        [[LABELS[k], *[f"{residual.loc[(k, h), 'pearson_r_mean']:+.3f}"
                       if pd.notna(residual.loc[(k, h), 'pearson_r_mean']) else "---"
                       for h in MAIN_HORIZONS]] for k in LEARNED])
    gru = residuals[(residuals.target_view == "context_residual") & residuals.model_key.str.startswith("gru_")]
    table("residual_gru_intervals.tex", r"Residual-target GRU associations with unadjusted descriptive 95\% "
        r"subject-bootstrap intervals. SD ratio averages within-session predicted/observed residual SDs "
        "within subjects and then across subjects; it describes variation, not forecast calibration.",
        "tab:supp_residual_gru_intervals", ["Model", "$h$", "Pearson $r$", r"95\% interval", "SD ratio"],
        [[LABELS[r.model_key], str(r.horizon), f"{r.pearson_r_mean:+.3f}",
          f"[{r.pearson_r_ci_lower:+.3f}, {r.pearson_r_ci_upper:+.3f}]", f"{r.sd_ratio_mean:.3f}"]
         for r in gru.itertuples()])


def correction_tables(baselines: pd.DataFrame, correction: pd.DataFrame, tests: pd.DataFrame) -> None:
    wide = pd.concat([baselines, correction]).pivot(index="horizon", columns="model_key", values="mae_mean")
    table("calibration_correction.tex", "Post-hoc training-subject-fitted affine calibration correction. "
        "MAE gives subjects equal weight. Context and expanding means use observed online history; "
        "the correction uses only the held-out session's calibration mean.",
        "tab:supp_calibration_correction", ["Horizon", "Context mean", "Expanding mean", "Corrected calibration"],
        [[str(h), *[f"{wide.loc[h, k]:.4f}" for k in
          ["context_score_mean", "expanding_online_mean", CORRECTED]]] for h in HORIZONS])
    table("calibration_correction_pairs.tex", r"Post-hoc corrected-calibration contrasts (correction minus comparator). "
        r"Intervals are unadjusted descriptive 95\% subject bootstraps. $p_{18}$ corrects all 18 "
        "correction tests in a family separate from Table S15.",
        "tab:supp_calibration_correction_pairs", ["Comparator", "$h$", r"$\Delta$ MAE", r"95\% interval", "$p_{18}$"],
        [[LABELS[r.model_a], str(r.horizon), f"{r.mean_delta_b_minus_a:+.5f}",
          f"[{r.bootstrap_ci_lower:+.5f}, {r.bootstrap_ci_upper:+.5f}]", f"{r.p_value_holm_all_18:.4f}"]
         for r in tests.itertuples()])


def residual_figure(display: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    x = display.predicted_residual_centered.to_numpy()
    y = display.observed_residual_centered.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), constrained_layout=True)
    axes[0].scatter(x, y, s=5, alpha=.16, color="#32688e", rasterized=True, edgecolors="none")
    axes[0].set_title("(a) Within-session deviations", loc="left", fontsize=9)
    axes[0].set_ylabel("Observed residual deviation")
    for ax in axes:
        ax.axhline(0, color=".75", lw=.6)
        ax.axvline(0, color=".75", lw=.6)
        ax.set_xlabel("Predicted residual deviation")
        ax.spines[["top", "right"]].set_visible(False)
    # Bins are a retrospective display, not predictor tuning. Give each subject
    # equal weight within a bin; intervals resample the 15 subject means.
    edges = np.quantile(x, np.linspace(0, 1, 6))
    display = display.copy()
    display["bin"] = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 4)
    bins = []
    for b, group in display.groupby("bin"):
        per_subject = group.groupby("subject_id").agg(
            predicted=("predicted_residual_centered", "mean"),
            observed=("observed_residual_centered", "mean"))
        assert set(per_subject.index) == set(SUBJECTS)
        low, high = ci(per_subject.observed.to_numpy())
        bins.append(dict(bin=b, n_examples=len(group), n_subjects=15,
            predicted_mean=per_subject.predicted.mean(), observed_mean=per_subject.observed.mean(),
            observed_ci_lower=low, observed_ci_upper=high))
    bins = pd.DataFrame(bins)
    save(bins, "residual_diagnostic_h5_bins.csv")
    axes[1].errorbar(bins.predicted_mean, bins.observed_mean,
        yerr=[bins.observed_mean - bins.observed_ci_lower, bins.observed_ci_upper - bins.observed_mean],
        fmt="o-", color="#32688e", capsize=3, lw=1)
    limits = [min(x.min(), bins.observed_ci_lower.min()), max(x.max(), bins.observed_ci_upper.max())]
    axes[1].plot(limits, limits, "--", color=".5", lw=.8, label="Identity")
    axes[1].set_ylabel("Mean observed residual deviation")
    axes[1].set_title("(b) Five descriptive bins", loc="left", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ["pdf", "png"]:
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {"dpi": 300}
        fig.savefig(FIG / f"residual_prediction_diagnostic_h5.{suffix}", **kwargs)
    plt.close(fig)


def build_feedback_additions() -> dict[str, object]:
    INPUTS.clear()
    for path in [OUT, TEX, FIG]:
        path.mkdir(parents=True, exist_ok=True)
    examples, sessions = load_examples()
    baselines, tests = baseline_results(examples)
    correction, correction_tests = corrected_calibration_results(examples, sessions)
    residuals, display = residual_results(examples)
    tables(baselines, tests, residuals)
    correction_tables(baselines, correction, correction_tests)
    residual_figure(display)
    save(sessions, "calibration_mean_session_diagnostics.csv")
    cal = sessions.avatar_calibration_mean
    diagnostics = {
        "n_sessions": len(sessions), "nonpositive_avatar_means": int((cal <= 0).sum()),
        "avatar_calibration_mean_quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                                                     cal.quantile([0, .25, .5, .75, 1]).tolist())),
        "avatar_calibration_online_correlation": float(cal.corr(sessions.online_mean)),
        "all_calibration_online_correlation": float(sessions.all_calibration_mean.corr(sessions.online_mean)),
        "online_minus_avatar_calibration_mean_bias": float(sessions.online_minus_avatar_calibration_mean.mean()),
    }
    (OUT / "calibration_mean_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    report = {
        "analysis_date": "2026-10-11", "models_retrained": False,
        "new_baseline_fitted": "One affine OLS calibration correction in each of 15 LOSO folds; no tuning or blending.",
        "baseline_forecasting_examples": len(examples), "baseline_subject_metrics": 405,
        "new_posthoc_baseline_tests": 18, "holm_family": "All 18 additional fixed-baseline contrasts",
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
        "calibration_correction": {
            "protocol": "docs/paper_revision/calibration_correction_protocol.md",
            "fit": "Equal-session OLS from avatar-calibration mean to full online-session mean, using 98 sessions of 14 training subjects only.",
            "held_out_information": "Avatar-calibration mean only; coefficients and forecasts shared across anchors and horizons.",
            "n_fold_fits": 15, "n_parameters_per_fold": 2, "n_predictions": len(examples),
            "n_subject_metrics": len(correction) * 15, "posthoc_tests": len(correction_tests),
            "holm_family": "All 18 correction-versus-context/expanding tests; separate from the earlier fixed-baseline family.",
        },
        "baseline_target_views": "Fixed predictions evaluated once on the raw-score scale; residual conversion does not add a comparison.",
        "expanding_mean_information": "Online blocks 0 through anchor, including blocks preceding the original ten-block context.",
        "residual_diagnostic": "Pearson correlation within sessions, arithmetic session means within subjects, equal-subject mean; descriptive, no significance test.",
        "residual_models": LEARNED, "residual_summary_rows": len(residuals),
        "interval_scope": "Unadjusted descriptive subject percentiles; conditional on saved fits; shared LOSO training and overlapping windows remain limitations.",
        "calibration_threshold": "In-sample reference: avatar blocks also contribute to calibration prototypes; no future online blocks are used.",
        "calibration_diagnostics": diagnostics,
        "verification": "Original context MAEs and all 1950 learned-model subject MAEs reproduced at absolute tolerance 1e-12; all archived targets and keys matched.",
        "input_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(INPUTS | {Path(__file__).resolve()})],
    }
    (OUT / "additional_analysis_provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = build_feedback_additions()
    print(json.dumps({k: v for k, v in report.items() if k != "input_files"}, indent=2))
