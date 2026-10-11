"""Build the paper supplement from saved results and one new affine baseline.

New intervals are descriptive, unadjusted subject-fold percentile bootstraps.
Original p-values retain their explicitly identified test families.
The completed five-model short-horizon experiment is verified and reported
separately, with its prespecified all-32 Holm family.
Only the session-corrected review_revision ranking predictions are consumed.
The sole new fit is a training-subject-only affine calibration correction;
original classical and GRU models are not refitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_conference_figures import make_proxy_structure_and_forecasting
from build_paper_feedback_addendum import build_feedback_additions
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.paper_artifacts import resolve_artifact
DATA = ROOT / "data/forecasting"
OUT = DATA / "review_revision/supplementary"
TEX = ROOT / "docs/paper_revision/supplementary_generated"
FIG = ROOT / "outputs/forecasting"
HORIZONS = (5, 10, 15, 20, 25)
VIEWS = ("raw_level", "context_residual")
SUBJECTS = [f"SBJ{i:02d}" for i in range(1, 16)]
MODELS = {
    "context_score_mean": ("Context mean", "score_only"),
    "persistence": ("Persistence", "score_only"),
    "moving_average_5": ("Moving average (5)", "score_only"),
    "train_mean": ("Training mean", "score_only"),
    "autoreg_ridge_5": ("AR ridge (5)", "score_only"),
    "gru_score_only": ("GRU score", "score_only"),
    "ridge_anchor_ts": ("EEG ridge", "eeg_only"),
    "elasticnet_anchor_ts": ("EEG elastic-net", "eeg_only"),
    "pcr_context_ts": ("EEG PCR", "eeg_only"),
    "pls_context_ts": ("EEG PLS", "eeg_only"),
    "riemannian_knn_anchor": ("Riemannian KNN", "eeg_only"),
    "gru_eeg_reduced": ("GRU EEG", "eeg_only"),
    "pcr_context_ts_permuted_target": ("Permuted EEG PCR", "eeg_only"),
    "ridge_score5_anchor_ts": ("Hybrid ridge", "hybrid"),
    "pcr_score5_context_ts": ("Hybrid PCR", "hybrid"),
    "pls_score5_context_ts": ("Hybrid PLS", "hybrid"),
    "gru_hybrid": ("GRU hybrid", "hybrid"),
}
INPUTS: set[Path] = set()


def read_csv(relative: str) -> pd.DataFrame:
    path = resolve_artifact(DATA / relative)
    INPUTS.add(path)
    return pd.read_csv(path)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.17g")


def normalize_folds(frame: pd.DataFrame, *, neural: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame["model_key"] = (
        "gru_" + frame["ablation"] if neural else frame["model_name"]
    )
    frame["sigma"] = frame[
        "mean_learned_sigma" if neural else "mean_oof_sigma"
    ]
    frame["gaussian_width_80"] = 2 * norm.ppf(0.9) * frame["sigma"]
    frame["model_label"] = frame["model_key"].map(lambda k: MODELS[k][0])
    frame["input_family"] = frame["model_key"].map(lambda k: MODELS[k][1])
    return frame[
        [
            "fold_subject_id", "horizon", "target_view", "model_key",
            "model_label", "input_family", "mae", "rmse", "gaussian_nll",
            "gaussian_picp_80", "gaussian_picp_90", "gaussian_width_80",
            "sigma", "n_test_rows",
        ]
    ]


def bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    indices = np.random.default_rng(seed).integers(
        0, len(values), size=(replicates, len(values))
    )
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def tex_table(
    filename: str, caption: str, label: str, headers: list[str],
    rows: list[list[str]], *, alignment: str | None = None,
) -> None:
    alignment = alignment or ("l" + "r" * (len(headers) - 1))
    lines = [
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{" + alignment + "}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" if row else r"\midrule" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TEX / filename).write_text("\n".join(lines), encoding="utf-8")


def mean_ci(value: float, lower: float, upper: float, *, signed: bool = False) -> str:
    fmt = "+.4f" if signed else ".4f"
    return (
        format(value, fmt) + " [" + format(lower, fmt)
        + ", " + format(upper, fmt) + "]"
    )


def build_point_results(seed: int, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    classical = read_csv("step7_fold_metrics.csv")
    neural = read_csv("step8_fold_metrics.csv")
    folds = pd.concat(
        [normalize_folds(classical, neural=False), normalize_folds(neural, neural=True)],
        ignore_index=True,
    )
    key = ["fold_subject_id", "horizon", "target_view", "model_key"]
    assert not folds.duplicated(key).any()
    assert len(folds) == 15 * 5 * 2 * 17
    save_csv(folds, "subject_metrics.csv")

    contrasts = pd.concat(
        [
            read_csv("step7_paired_stats.csv").assign(test_family="classical_7"),
            read_csv("step8_paired_stats.csv").assign(test_family="gru_10"),
        ], ignore_index=True,
    )
    assert len(contrasts) == 340
    for row in contrasts.itertuples():
        means = []
        for model in [row.model_a, row.model_b]:
            selected = folds[
                (folds.horizon == row.horizon) & (folds.target_view == row.target_view)
                & (folds.model_key == model)
            ]
            assert len(selected) == 15
            means.append(selected[row.metric].mean())
        assert np.allclose(
            means, [row.mean_metric_a, row.mean_metric_b], atol=1e-12, rtol=0
        )
        assert np.isclose(
            means[1] - means[0], row.mean_delta_b_minus_a, atol=1e-12, rtol=0
        )
    save_csv(contrasts, "original_tested_contrasts.csv")
    definitions = contrasts[
        ["test_family", "comparison_id", "model_a", "model_b"]
    ].drop_duplicates().reset_index(drop=True)
    assert len(definitions) == 23
    save_csv(definitions, "contrast_definitions.csv")
    templates = contrasts.assign(
        model_b_template=np.where(
            contrasts.comparison_id.str.endswith("step7_best"),
            "__selected_classical__", contrasts.model_b,
        )
    )[["test_family", "comparison_id", "model_a", "model_b_template"]].drop_duplicates()
    assert len(templates) == 17
    save_csv(templates, "contrast_templates.csv")
    selected_comparators = contrasts[
        contrasts.comparison_id.str.endswith("step7_best")
    ][["horizon", "target_view", "metric", "model_b"]].drop_duplicates()
    assert len(selected_comparators) == 20
    save_csv(selected_comparators, "selected_classical_comparators.csv")

    saved_summaries = {}
    for filename, column, prefix in [
        ("step7_summary.csv", "model_name", ""),
        ("step8_summary.csv", "ablation", "gru_"),
    ]:
        for row in read_csv(filename).to_dict("records"):
            saved_summaries[(row["horizon"], row["target_view"], prefix + row[column])] = row

    summaries, differences = [], []
    for (horizon, view), group in folds.groupby(["horizon", "target_view"]):
        baseline = group[group.model_key == "context_score_mean"].set_index(
            "fold_subject_id"
        ).loc[SUBJECTS]
        for model in MODELS:
            selected = group[group.model_key == model].set_index(
                "fold_subject_id"
            ).loc[SUBJECTS]
            delta = selected.mae.to_numpy() - baseline.mae.to_numpy()
            lo, hi = bootstrap_ci(delta, seed=seed, replicates=replicates)
            direct = contrasts[
                (contrasts.horizon == horizon) & (contrasts.target_view == view)
                & (contrasts.metric == "mae")
                & (
                    ((contrasts.model_a == model) & (contrasts.model_b == "context_score_mean"))
                    | ((contrasts.model_b == model) & (contrasts.model_a == "context_score_mean"))
                )
            ]
            assert len(direct) <= 1
            nll_direct = contrasts[
                (contrasts.horizon == horizon) & (contrasts.target_view == view)
                & (contrasts.metric == "gaussian_nll")
                & (
                    ((contrasts.model_a == model) & (contrasts.model_b == "context_score_mean"))
                    | ((contrasts.model_b == model) & (contrasts.model_a == "context_score_mean"))
                )
            ]
            record = {
                "horizon": horizon, "target_view": view, "model_key": model,
                "model_label": MODELS[model][0], "input_family": MODELS[model][1],
                "n_subjects": len(selected), "mae_mean": selected.mae.mean(),
                "mae_sd_subjects": selected.mae.std(ddof=1),
                "rmse_mean": selected.rmse.mean(),
                "delta_mae_model_minus_context": delta.mean(),
                "delta_ci_95_lower": lo, "delta_ci_95_upper": hi,
                "original_direct_baseline_mae_test": not direct.empty,
                "original_direct_baseline_mae_p_holm": (
                    direct.iloc[0].p_value_holm if not direct.empty else np.nan
                ),
                "original_test_family": (
                    direct.iloc[0].test_family if not direct.empty else ""
                ),
                "gaussian_nll_mean": selected.gaussian_nll.mean(),
                "original_direct_baseline_nll_p_holm": (
                    nll_direct.iloc[0].p_value_holm if not nll_direct.empty else np.nan
                ),
                "gaussian_coverage_80": selected.gaussian_picp_80.mean(),
                "gaussian_coverage_90": selected.gaussian_picp_90.mean(),
                "gaussian_width_80": selected.gaussian_width_80.mean(),
            }
            source = saved_summaries[(horizon, view, model)]
            for metric in ["mae", "rmse", "gaussian_nll"]:
                assert np.isclose(
                    record[metric + "_mean"], source[metric + "_mean"],
                    atol=1e-12, rtol=0,
                )
            summaries.append(record)
            if model != "context_score_mean":
                differences += [
                    {
                        "fold_subject_id": subject, "horizon": horizon,
                        "target_view": view, "model_key": model,
                        "model_mae": selected.loc[subject, "mae"],
                        "context_mean_mae": baseline.loc[subject, "mae"],
                        "delta_mae_model_minus_context": value,
                    }
                    for subject, value in zip(SUBJECTS, delta, strict=True)
                ]
    summary = pd.DataFrame(summaries)
    assert len(summary) == 170
    save_csv(summary, "full_model_summary.csv")
    save_csv(pd.DataFrame(differences), "paired_mae_differences.csv")
    save_csv(
        summary[
            ["horizon", "target_view", "model_key", "model_label", "gaussian_nll_mean",
             "original_direct_baseline_nll_p_holm", "gaussian_coverage_80",
             "gaussian_coverage_90", "gaussian_width_80"]
        ], "probabilistic_summary.csv",
    )
    return folds, summary


def build_ranking(seed: int, replicates: int) -> pd.DataFrame:
    predictions = read_csv("review_revision/step8_anomaly_predictions.csv")
    assert len(predictions) == 81900
    assert np.allclose(
        predictions.baseline_mu_raw, predictions.context_score_mean_value,
        atol=1e-12, rtol=0,
    )
    for prefix in ["", "baseline_"]:
        expected = ndtr(
            (predictions.drop20_threshold_raw - predictions[prefix + "mu_raw"])
            / predictions[prefix + "sigma_raw"]
        )
        assert np.allclose(
            predictions[prefix + "p_drop20_state"], expected, atol=1e-12, rtol=0
        )
    groups = ["horizon", "target_view", "ablation"]
    session_rows, subject_rows = [], []

    def metrics(group: pd.DataFrame) -> dict[str, float]:
        if group.target_drop20_state.nunique() != 2:
            return {name: np.nan for name in ["gru_auc", "baseline_auc", "gru_ap", "baseline_ap"]}
        result = {}
        for name, prefix in [("gru", ""), ("baseline", "baseline_")]:
            y, p = group.target_drop20_state, group[prefix + "p_drop20_state"]
            result[name + "_auc"] = roc_auc_score(y, p)
            result[name + "_ap"] = average_precision_score(y, p)
        return result

    for keys, group in predictions.groupby(groups + ["subject_id", "session_id"]):
        horizon, view, ablation, subject, session = keys
        session_rows.append(
            {
                "horizon": horizon, "target_view": view, "model_key": "gru_" + ablation,
                "subject_id": subject, "session_id": session,
                "n_rows": len(group), "n_positive": int(group.target_drop20_state.sum()),
                "eligible_both_classes": group.target_drop20_state.nunique() == 2,
                **metrics(group),
            }
        )
    sessions = pd.DataFrame(session_rows)
    for keys, group in predictions.groupby(groups + ["subject_id"]):
        horizon, view, ablation, subject = keys
        subset = sessions[
            (sessions.horizon == horizon) & (sessions.target_view == view)
            & (sessions.model_key == "gru_" + ablation) & (sessions.subject_id == subject)
        ]
        common = {
            "horizon": horizon, "target_view": view, "model_key": "gru_" + ablation,
            "subject_id": subject, "n_eligible_sessions": int(subset.eligible_both_classes.sum()),
            "n_rows": len(group), "n_positive": int(group.target_drop20_state.sum()),
        }
        subject_rows.append({**common, "aggregation": "pooled_sessions", **metrics(group)})
        restricted = subset[subset.eligible_both_classes]
        subject_rows.append(
            {
                **common, "aggregation": "within_session_then_subject",
                **restricted[["gru_auc", "baseline_auc", "gru_ap", "baseline_ap"]].mean().to_dict(),
            }
        )
    subjects = pd.DataFrame(subject_rows)
    save_csv(sessions, "ranking_session_metrics.csv")
    save_csv(subjects, "ranking_subject_metrics.csv")
    records = []
    for keys, group in subjects.groupby(["horizon", "target_view", "model_key", "aggregation"]):
        horizon, view, model, aggregation = keys
        valid = group.dropna(subset=["gru_auc", "baseline_auc"])
        record = {
            "horizon": horizon, "target_view": view, "model_key": model,
            "aggregation": aggregation, "n_subjects": len(valid),
            "eligible_sessions": int(group.n_eligible_sessions.sum()),
            "total_sessions": len(group) * 7,
        }
        for metric in ["auc", "ap"]:
            for predictor in ["gru", "baseline"]:
                values = valid[predictor + "_" + metric].to_numpy()
                lo, hi = bootstrap_ci(values, seed=seed, replicates=replicates)
                record[predictor + "_" + metric + "_mean"] = values.mean()
                record[predictor + "_" + metric + "_ci_lower"] = lo
                record[predictor + "_" + metric + "_ci_upper"] = hi
            delta = valid["gru_" + metric].to_numpy() - valid["baseline_" + metric].to_numpy()
            lo, hi = bootstrap_ci(delta, seed=seed, replicates=replicates)
            record["delta_" + metric + "_gru_minus_baseline"] = delta.mean()
            record["delta_" + metric + "_ci_lower"] = lo
            record["delta_" + metric + "_ci_upper"] = hi
        records.append(record)
    summary = pd.DataFrame(records)
    save_csv(summary, "ranking_summary.csv")
    original = read_csv("review_revision/step8_anomaly_metrics.csv")
    for row in summary[summary.aggregation == "pooled_sessions"].itertuples():
        source = original[
            (original.horizon == row.horizon) & (original.target_view == row.target_view)
            & (original.ablation == row.model_key.removeprefix("gru_"))
        ]
        assert np.isclose(
            row.gru_auc_mean, source.prospective_roc_auc_p_drop20_state_for_state.mean(),
            atol=1e-12, rtol=0,
        )
        assert np.isclose(
            row.baseline_auc_mean, source.baseline_roc_auc_p_drop20_state_for_state.mean(),
            atol=1e-12, rtol=0,
        )
    diagnostic = summary[
        (summary.horizon == 5) & (summary.target_view == "raw_level")
        & (summary.model_key == "gru_score_only")
        & (summary.aggregation == "within_session_then_subject")
    ].iloc[0]
    assert diagnostic.eligible_sessions == 103
    assert np.isclose(diagnostic.gru_auc_mean, 0.46919835218145617, atol=1e-12)
    assert np.isclose(diagnostic.baseline_auc_mean, 0.47419977779245903, atol=1e-12)
    return summary


def build_robustness(folds: pd.DataFrame, seed: int, replicates: int) -> pd.DataFrame:
    rows = []
    runs = [
        ("last2_subject_holdout", "Main GRU", ""),
        ("group_kfold", "Grouped inner CV", "exp30_step8_grouped_inner_cv/"),
        ("group_kfold_expanded", "Grouped CV + dropout search", "exp31_expanded_hp_search/"),
    ]
    for protocol, label, prefix in runs:
        run = normalize_folds(read_csv(prefix + "step8_fold_metrics.csv"), neural=True)
        read_csv(prefix + "step8_model_selection.csv")
        saved = read_csv(prefix + "step8_summary.csv")
        for (horizon, view, model), group in run.groupby(["horizon", "target_view", "model_key"]):
            selected = group.set_index("fold_subject_id").loc[SUBJECTS]
            baseline = folds[
                (folds.horizon == horizon) & (folds.target_view == view)
                & (folds.model_key == "context_score_mean")
            ].set_index("fold_subject_id").loc[SUBJECTS]
            delta = selected.mae.to_numpy() - baseline.mae.to_numpy()
            lo, hi = bootstrap_ci(delta, seed=seed, replicates=replicates)
            original = saved[
                (saved.horizon == horizon) & (saved.target_view == view)
                & (saved.ablation == model.removeprefix("gru_"))
            ].iloc[0]
            assert np.isclose(selected.mae.mean(), original.mae_mean, atol=1e-12, rtol=0)
            rows.append(
                {
                    "protocol": protocol, "protocol_label": label, "horizon": horizon,
                    "target_view": view, "model_key": model, "mae_mean": selected.mae.mean(),
                    "mae_sd_subjects": selected.mae.std(ddof=1),
                    "baseline_mae_mean": baseline.mae.mean(),
                    "delta_mae_model_minus_context": delta.mean(),
                    "delta_ci_95_lower": lo, "delta_ci_95_upper": hi,
                }
            )
    result = pd.DataFrame(rows)
    assert len(result) == 90
    check = result[(result.target_view == "raw_level") & (result.model_key == "gru_score_only")]
    assert (check.delta_mae_model_minus_context > 0).all()
    save_csv(result, "robustness_summary.csv")
    return result


def build_variance() -> dict[str, object]:
    arrays = []
    for subject in range(1, 16):
        for session in range(1, 8):
            path = ROOT / f"data/scores/sub-{subject:02d}_ses-{session:02d}_split-online_scores.npz"
            INPUTS.add(path)
            with np.load(path) as payload:
                arrays.append(payload["score_raw"].astype(float))
    values = np.asarray(arrays).reshape(15, 7, 50)
    components = np.array(
        [
            values.mean(axis=(1, 2)).var(ddof=0),
            values.mean(axis=2).var(axis=1, ddof=0).mean(),
            values.var(axis=2, ddof=0).mean(),
        ]
    )
    assert np.isclose(components.sum(), values.var(ddof=0), atol=1e-12, rtol=0)
    partition = pd.DataFrame(
        {
            "component": ["between_subjects", "between_sessions_within_subjects", "within_sessions"],
            "population_variance": components, "share_percent": 100 * components / components.sum(),
        }
    )
    save_csv(partition, "variance_components.csv")
    contexts, futures, anchors, full_means = [], [], [], []
    for sequence in values.reshape(105, 50):
        for anchor in range(9, 45):
            contexts.append(sequence[anchor - 9:anchor + 1].mean())
            futures.append(sequence[anchor + 5])
            anchors.append(sequence[anchor])
            full_means.append(sequence.mean())
    diagnostics = {
        "between_subject_and_session_share_percent": float(100 * components[:2].sum() / components.sum()),
        "context_vs_full_session_mean_correlation": float(np.corrcoef(contexts, full_means)[0, 1]),
        "context_vs_full_session_mean_mae": float(np.mean(np.abs(np.asarray(contexts) - full_means))),
        "context_vs_future_h5_correlation": float(np.corrcoef(contexts, futures)[0, 1]),
        "anchor_vs_future_h5_correlation": float(np.corrcoef(anchors, futures)[0, 1]),
    }
    (OUT / "score_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return diagnostics


def build_tables(point: pd.DataFrame, ranking: pd.DataFrame, robustness: pd.DataFrame) -> None:
    for name in ["references.bib", "IEEEtran.bst"]:
        assert (TEX / name).is_file(), f"Missing included supplement resource: {name}"
    for view, short in [("raw_level", "raw"), ("context_residual", "residual")]:
        subset = point[point.target_view == view].set_index(["model_key", "horizon"])
        best = point[point.target_view == view].groupby("horizon").mae_mean.min()
        rows, family = [], None
        for model, (label, current_family) in MODELS.items():
            if family is not None and current_family != family:
                rows.append([])
            family = current_family
            cells = []
            for horizon in HORIZONS:
                value = subset.loc[(model, horizon), "mae_mean"]
                cell = f"{value:.4f}"
                if np.isclose(value, best[horizon], atol=1e-12, rtol=0):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            rows.append([label, *cells])
        tex_table(
            f"mae_{short}.tex",
            ("All 17 raw-score variants." if short == "raw" else "All 17 separately fitted residual-target variants.")
            + " Equal-subject mean MAE; bold denotes the lowest descriptive mean, not a significance claim.",
            f"tab:supp_mae_{short}", ["Model", *[str(h) for h in HORIZONS]], rows,
        )
        paired = []
        for model, (label, _) in MODELS.items():
            if model == "context_score_mean":
                continue
            cells = []
            for horizon in [5, 25]:
                record = subset.loc[(model, horizon)]
                cells.append(
                    mean_ci(record.delta_mae_model_minus_context,
                            record.delta_ci_95_lower, record.delta_ci_95_upper, signed=True)
                )
            paired.append(
                [label, *cells, "Yes" if subset.loc[(model, 5), "original_direct_baseline_mae_test"] else "--"]
            )
        tex_table(
            f"deltas_{short}.tex",
            ("Raw-target" if short == "raw" else "Residual-target")
            + r" MAE differences (model minus context mean), with unadjusted descriptive 95\% subject-fold bootstrap intervals."
            + " The final column identifies an original direct baseline test.",
            f"tab:supp_deltas_{short}", ["Model", r"$h=5$: difference [CI]", r"$h=25$: difference [CI]", "Direct test"],
            paired,
        )
    definitions = pd.read_csv(OUT / "contrast_templates.csv")
    for family, filename, label, prefix in [
        ("classical_7", "contrasts_classical.tex", "tab:supp_contrasts_classical", "C"),
        ("gru_10", "contrasts_gru.tex", "tab:supp_contrasts_gru", "G"),
    ]:
        rows = [
            [
                prefix + str(i + 1), MODELS[row.model_a][0],
                ("Selected trained classical" if row.model_b_template == "__selected_classical__"
                 else MODELS[row.model_b_template][0]),
            ]
            for i, row in enumerate(definitions[definitions.test_family == family].itertuples())
        ]
        tex_table(
            filename, "Original " + ("seven classical" if prefix == "C" else "ten GRU")
            + " contrast templates. Archived differences use model B minus model A.",
            label, ["ID", "Model A", "Model B"], rows, alignment="lll",
        )
    rank_rows = []
    for view, fit in [("raw_level", "Raw"), ("context_residual", "Residual")]:
        for model in ["context_score_mean", "gru_score_only", "gru_eeg_reduced", "gru_hybrid"]:
            cells = []
            for aggregation in ["pooled_sessions", "within_session_then_subject"]:
                record = ranking[
                    (ranking.horizon == 5) & (ranking.target_view == view)
                    & (ranking.model_key == ("gru_score_only" if model == "context_score_mean" else model))
                    & (ranking.aggregation == aggregation)
                ].iloc[0]
                prefix = "baseline" if model == "context_score_mean" else "gru"
                cells.append(mean_ci(record[prefix + "_auc_mean"], record[prefix + "_auc_ci_lower"], record[prefix + "_auc_ci_upper"]))
            rank_rows.append([MODELS[model][0], fit, *cells])
    tex_table(
        "ranking_h5.tex",
        r"Horizon-5 ROC-AUC, with unadjusted descriptive 95\% subject-bootstrap intervals. Session-restricted results average 103 eligible sessions within subjects, then give the 15 subjects equal weight.",
        "tab:supp_ranking", ["Predictor", "Fit", "Sessions pooled", "Within-session"], rank_rows,
        alignment="llrr",
    )
    robust_rows = [
        ["Context mean", *[
            f"{point[(point.model_key == 'context_score_mean') & (point.target_view == 'raw_level') & (point.horizon == h)].iloc[0].mae_mean:.4f}"
            for h in HORIZONS
        ]]
    ]
    for protocol, label in [
        ("last2_subject_holdout", "Main GRU"),
        ("group_kfold", "Grouped inner CV"),
        ("group_kfold_expanded", "Grouped CV + dropout search"),
    ]:
        selected = robustness[
            (robustness.protocol == protocol) & (robustness.target_view == "raw_level")
            & (robustness.model_key == "gru_score_only")
        ].set_index("horizon")
        robust_rows.append([label, *[f"{selected.loc[h, 'mae_mean']:.4f}" for h in HORIZONS]])
    tex_table(
        "robustness.tex", "Saved robustness runs: raw-target score-only GRU mean MAE.",
        "tab:supp_robustness", ["Protocol", *[str(h) for h in HORIZONS]], robust_rows,
    )
    probability_rows = []
    for view, fit in [("raw_level", "Raw"), ("context_residual", "Residual")]:
        for model in ["context_score_mean", "gru_score_only", "gru_eeg_reduced", "gru_hybrid"]:
            record = point[(point.horizon == 5) & (point.target_view == view) & (point.model_key == model)].iloc[0]
            probability_rows.append(
                [MODELS[model][0], fit, f"{record.gaussian_nll_mean:.4f}",
                 f"{100 * record.gaussian_coverage_80:.1f}", f"{record.gaussian_width_80:.4f}"]
            )
    tex_table(
        "uncertainty.tex",
        r"Horizon-5 Gaussian NLL (lower is better), nominal 80\% interval coverage, and mean interval width in proxy-score units.",
        "tab:supp_uncertainty", ["Predictor", "Fit", "NLL", r"Coverage (\%)", "Width"],
        probability_rows, alignment="llrrr",
    )


def build_figure(point: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharex=True)
    panels = [
        (axes[0], ["gru_score_only", "gru_hybrid"], ["GRU score", "GRU hybrid"], "GRU comparisons"),
        (axes[1], ["pcr_context_ts"], ["EEG PCR"], "EEG PCR comparison"),
    ]
    for ax, keys, labels, title in panels:
        for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
            values = point[(point.target_view == "raw_level") & (point.model_key == key)].set_index("horizon").loc[list(HORIZONS)]
            means = values.delta_mae_model_minus_context.to_numpy()
            lower = values.delta_ci_95_lower.to_numpy()
            upper = values.delta_ci_95_upper.to_numpy()
            ax.errorbar(
                HORIZONS, means, yerr=np.vstack([means - lower, upper - means]),
                label=label, marker="o" if index == 0 else "s", markersize=3,
                linewidth=1, elinewidth=.7, capsize=2, color=["#1764a0", "#9658a8"][index],
            )
        ax.axhline(0, color="#444444", linestyle="--", linewidth=.7)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon (blocks)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MAE minus context mean", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[1].set_ylabel("MAE difference", fontsize=8)
    fig.tight_layout(pad=.4, w_pad=.6)
    fig.savefig(FIG / "main_result_delta_mae_paired.pdf")
    fig.savefig(FIG / "main_result_delta_mae_paired.png", dpi=350)
    plt.close(fig)


def verify_printed_tables(point: pd.DataFrame, manuscript: Path) -> int:
    paper = manuscript.read_text(encoding="utf-8")
    labels = {
        "Context mean": "context_score_mean", "Persistence": "persistence",
        "Moving avg": "moving_average_5", "AR ridge-5": "autoreg_ridge_5",
        "PCR EEG": "pcr_context_ts", "Permuted EEG": "pcr_context_ts_permuted_target",
        "GRU score": "gru_score_only", "GRU EEG": "gru_eeg_reduced", "GRU hybrid": "gru_hybrid",
    }
    checked = 0
    for view, label in zip(VIEWS, ["tab:raw_mae_compact", "tab:residual_mae_compact"], strict=True):
        segment = paper.split(r"\label{" + label + "}")[1].split(r"\end{table}")[0]
        for display, model in labels.items():
            match = re.search(re.escape(display) + r" & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+)", segment)
            if match:
                values = point[(point.target_view == view) & (point.model_key == model)].sort_values("horizon").mae_mean
                assert np.all(np.abs(values.to_numpy() - np.asarray(match.groups(), dtype=float)) <= .00005)
                checked += 5
    assert checked == 80
    return checked


def verify_expanding_mean_table(manuscript: Path) -> int:
    paper = manuscript.read_text(encoding="utf-8")
    segment = paper.split(r"\label{tab:raw_mae_compact}")[1].split(r"\end{table}")[0]
    match = re.search(r"Expanding mean & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+)", segment)
    assert match, "The post-hoc expanding baseline must be visible in main Table I."
    summary = pd.read_csv(OUT / "additional_baseline_summary.csv")
    values = summary[(summary.model_key == "expanding_online_mean") &
                     summary.horizon.isin(HORIZONS)].sort_values("horizon").mae_mean
    assert len(values) == 5
    assert np.all(np.abs(values.to_numpy() - np.asarray(match.groups(), dtype=float)) <= .00005)
    return 5


def build_short_horizon_results() -> dict[str, object]:
    """Verify completed exp41 outputs and generate four supplementary tables."""
    sys.path.insert(0, str(ROOT))
    from src.evaluation.forecasting_metrics import compute_paired_statistic, holm_correction

    directory = DATA / "review_revision/short_horizons"
    delivered = json.loads((directory / "deliverable_sha256.json").read_text(encoding="utf-8"))
    INPUTS.add(directory / "deliverable_sha256.json")
    artifacts = ["protocol.json", "verification.json", "independent_statistical_audit.json",
                 "reference_check.json", "eigen_factorization_check.json",
                 "gru_historical_reference_check.json", "execution_manifest.json",
                 "summary.csv", "subject_metrics.csv", "paired_mae_stats.csv",
                 "classical/predictions.csv", "gru/predictions.csv",
                 "classical/model_selection.csv", "gru/model_selection.csv"]
    for name in artifacts:
        path = resolve_artifact(directory / name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == delivered[path.relative_to(ROOT).as_posix()]
        INPUTS.add(path)
    verification = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "independent_statistical_audit.json").read_text(encoding="utf-8"))
    reference = json.loads((directory / "gru_historical_reference_check.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    assert verification["passed"] and audit["passed"] and reference["predictions_agree_at_1e_6"]
    assert verification["total_subject_metric_rows"] == 600 and audit["paired_contrasts"] == 32

    summary = read_csv("review_revision/short_horizons/summary.csv")
    metrics = read_csv("review_revision/short_horizons/subject_metrics.csv")
    paired = read_csv("review_revision/short_horizons/paired_mae_stats.csv")
    assert (len(summary), len(metrics), len(paired)) == (40, 600, 32)
    models = ("context_score_mean", "persistence", "autoreg_ridge_5", "pcr_context_ts", "gru_score_only")
    assert set(summary.model_name) == set(metrics.model_name) == set(models)
    assert set(summary.horizon) == set(metrics.horizon) == {1, 2, 3, 4}
    assert set(summary.target_view) == set(metrics.target_view) == set(VIEWS)
    assert summary.n_subjects.eq(15).all()
    assert not metrics.duplicated(["horizon", "target_view", "model_name", "fold_subject_id"]).any()
    for (horizon, view, model), subset in metrics.groupby(["horizon", "target_view", "model_name"]):
        assert set(subset.fold_subject_id) == set(SUBJECTS)
        assert subset.n_test_rows.eq(7 * (41 - horizon)).all()
        saved = summary[(summary.horizon == horizon) & (summary.target_view == view) &
                        (summary.model_name == model)].iloc[0]
        np.testing.assert_allclose(subset.mae.mean(), saved.mae_mean, rtol=0, atol=1e-12)
    for kind, prediction_field in [("classical", "y_pred_mean"), ("gru", "y_pred_mu")]:
        predictions = read_csv(f"review_revision/short_horizons/{kind}/predictions.csv")
        if kind == "gru":
            predictions["model_name"] = "gru_score_only"
        assert (predictions.target_idx - predictions.anchor_idx == predictions.horizon).all()
        assert (predictions.subject_id == predictions.fold_subject_id).all()
        assert (predictions.anchor_idx >= 9).all() and (predictions.target_idx <= 49).all()
        keys = ["horizon", "target_view", "model_name", "fold_subject_id"]
        predictions["error"] = np.abs(predictions.y_true - predictions[prediction_field])
        recalculated = predictions.groupby(keys).error.mean().sort_index()
        stored = metrics[metrics.model_name.isin(predictions.model_name.unique())].set_index(keys).mae.sort_index()
        np.testing.assert_allclose(recalculated, stored, rtol=0, atol=1e-12)
    for row in paired.itertuples():
        subset = metrics[(metrics.horizon == row.horizon) & (metrics.target_view == row.target_view)]
        wide = subset.pivot(index="fold_subject_id", columns="model_name", values="mae").sort_index()
        computed = compute_paired_statistic(comparison_id=row.comparison_id,
            horizon=row.horizon, target_view=row.target_view, metric="mae",
            model_a=row.model_a, model_b=row.model_b,
            metric_a=wide[row.model_a].to_numpy(), metric_b=wide[row.model_b].to_numpy(),
            bootstrap_seed=20261008, bootstrap_replicates=10000)
        for field in ["mean_delta_b_minus_a", "bootstrap_ci_lower", "bootstrap_ci_upper", "p_value"]:
            np.testing.assert_allclose(getattr(computed, field), getattr(row, field), rtol=0, atol=1e-12)
    np.testing.assert_allclose(holm_correction(paired.p_value.to_numpy()), paired.p_value_holm_all_32,
                               rtol=0, atol=1e-12)
    for _, subset in paired.groupby(["horizon", "target_view"]):
        np.testing.assert_allclose(holm_correction(subset.p_value.to_numpy()),
            subset.p_value_holm_within_horizon_view, rtol=0, atol=1e-12)
    save_csv(summary, "short_horizon_summary.csv")
    save_csv(metrics, "short_horizon_subject_metrics.csv")
    save_csv(paired, "short_horizon_paired_mae_stats.csv")
    for view, short in [("raw_level", "raw"), ("context_residual", "residual")]:
        rows = []
        for model in models:
            values = summary[(summary.target_view == view) & (summary.model_name == model)].sort_values("horizon")
            rows.append([MODELS[model][0], *[f"{value:.4f}" for value in values.mae_mean]])
        tex_table(f"short_mae_{short}.tex",
            ("Raw-score" if short == "raw" else "Separately fitted residual-target") +
            r" short-horizon MAE for the five-model extension. Values average 15 held-out subjects equally.",
            f"tab:supp_short_mae_{short}", ["Model", "1", "2", "3", "4"], rows)
        effects = []
        for model in models[1:]:
            for row in paired[(paired.target_view == view) & (paired.model_b == model)].sort_values("horizon").itertuples():
                effects.append([MODELS[model][0], str(row.horizon), f"{row.mean_delta_b_minus_a:+.5f}",
                    "[" + f"{row.bootstrap_ci_lower:+.5f}, {row.bootstrap_ci_upper:+.5f}" + "]",
                    f"{row.p_value_holm_all_32:.4f}", f"{row.p_value_holm_within_horizon_view:.4f}"])
        tex_table(f"short_pairs_{short}.tex",
            ("Raw-target" if short == "raw" else "Residual-target") +
            r" short-horizon MAE differences (alternative minus context mean). Intervals are unadjusted descriptive 95\% paired-subject bootstraps. Primary $p_{32}$ corrects all 32 additional comparisons; $p_4$ corrects within each horizon/view.",
            f"tab:supp_short_pairs_{short}", ["Alternative", "$h$", r"$\Delta$ MAE", r"95\% interval", "$p_{32}$", "$p_4$"],
            effects, alignment="lrrlrr")
    # These printed entries are generated directly from, and checked against, the saved summaries.
    for view, short in [("raw_level", "raw"), ("context_residual", "residual")]:
        text = (TEX / f"short_mae_{short}.tex").read_text(encoding="utf-8")
        for model in models:
            values = summary[(summary.target_view == view) & (summary.model_name == model)].sort_values("horizon").mae_mean
            expected = MODELS[model][0] + " & " + " & ".join(f"{value:.4f}" for value in values)
            assert expected in text
    return {"models": list(models), "horizons": [1, 2, 3, 4], "summary_rows": 40,
            "subject_metric_rows": 600, "paired_tests": 32, "printed_mae_entries_verified": 40,
            "holm_primary_family_size": 32, "holm_local_family_size": 4,
            "bootstrap_seed": 20261008, "bootstrap_replicates": 10000,
            "original_benchmark_retrained": False, "short_horizon_models_fitted_separately": True,
            "protocol_locked_at_utc": protocol["locked_at_utc"],
            "gru_historical_reference_prediction_difference": reference["max_prediction_difference"]}


def write_manifest(seed: int, replicates: int, diagnostics: dict[str, object], checked: int | None,
                   short_horizon: dict[str, object], feedback: dict[str, object], expanding_checked: int | None) -> None:
    for name in [
        "README.md", "CITATION.cff", "pyproject.toml", "uv.lock", ".gitignore", ".gitattributes",
        "docs/paper_revision/reproduction.md", "docs/paper_revision/supplementary_results.tex",
        "docs/paper_revision/supplementary_README.md", "docs/paper_revision/calibration_correction_protocol.md",
        "docs/paper_revision/short_horizon_protocol.md", "docs/paper_revision/short_horizon_results.md",
        "data/forecasting/review_revision/step8_anomaly_paired_stats.csv",
        "data/forecasting/review_revision/threshold_sensitivity.csv",
        "data/forecasting/review_revision/audit.json",
        "data/forecasting/step6_detrended_exploration.csv",
        "data/forecasting/step6c_step5_score_decomposition.csv",
        "results/figures/step2/grand_average_pz_target_vs_nontarget.png",
        "outputs/forecasting/proxy_structure_and_forecasting.pdf",
        "outputs/forecasting/proxy_structure_and_forecasting.png",
        "outputs/forecasting/proxy_structure_and_forecasting.json",
        "outputs/forecasting/residual_prediction_diagnostic_h5.pdf",
        "outputs/forecasting/residual_prediction_diagnostic_h5.png",
    ]:
        path = ROOT / name
        if path.is_file():
            INPUTS.add(path)
    INPUTS.update((ROOT / "src").rglob("*.py"))
    INPUTS.update((ROOT / "pipeline").glob("*.py"))
    INPUTS.update((ROOT / "tests").glob("*.py"))
    INPUTS.update(ROOT / item["path"] for item in feedback["input_files"])
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyriemann", "torch"]:
        packages[name] = importlib.metadata.version(name)
    manifest = {
        "snapshot_date": "2026-10-11",
        "analysis_snapshot_date": "2026-10-08",
        "package_name": "forecastability-limits-proxy-bciaut-p300",
        "public_repository_url": "https://github.com/ada-teodorescu/forecastability-limits-proxy-bciaut-p300",
        "public_repository_status": "Local release prepared; remote availability is not verified by this script.",
        "python_version": platform.python_version(), "analysis_package_versions": packages,
        "bootstrap_seed": seed, "bootstrap_replicates": replicates,
        "new_intervals": "Descriptive unadjusted percentile bootstraps over held-out subjects, conditional on fitted predictions.",
        "original_benchmark_retrained": False, "supplement_generator_fits_models": True,
        "generator_fit_scope": "Only the two-parameter affine calibration baseline, fitted on training subjects in each LOSO fold; no original classical/GRU refits.",
        "additional_short_horizon_analysis": short_horizon,
        "additional_fixed_baselines_and_residual_diagnostics": {
            key: value for key, value in feedback.items() if key != "input_files"
        },
        "printed_mae_entries_verified": checked, "original_paired_comparisons": 340,
        "printed_expanding_mean_entries_verified": expanding_checked,
        "manuscript_table_check": "External manuscript checked" if checked is not None else "Not requested; manuscript is maintained separately",
        "score_diagnostics": diagnostics,
        "canonical_ranking_predictions": "data/forecasting/review_revision/step8_anomaly_predictions.csv",
        "excluded_ranking_inputs": [
            "data/forecasting/step8_anomaly_predictions.csv",
            "data/forecasting/exp30_step8_grouped_inner_cv/step8_anomaly_predictions.csv",
            "data/forecasting/exp31_expanded_hp_search/step8_anomaly_predictions.csv",
        ],
        "input_files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(INPUTS)
        ],
    }
    (OUT / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_bundle() -> Path:
    manifest = json.loads((OUT / "snapshot_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["input_files"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"Missing snapshot input: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], (
            f"Snapshot input changed; regenerate the supplement first: {path}"
        )
    files = {ROOT / item["path"] for item in manifest["input_files"]}
    release_path = ROOT / "docs/paper_revision/release_manifest.json"
    if release_path.is_file():
        release = json.loads(release_path.read_text(encoding="utf-8"))
        for item in release["files"]:
            path = ROOT / item["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
            files.add(path)
        files.add(release_path)
        files.add(ROOT / "docs/paper_revision/release_verification.json")
    files.update(OUT.iterdir())
    files.update(path for path in TEX.iterdir() if path.is_file())
    files.update(
        ROOT / path for path in [
            "docs/paper_revision/supplementary_README.md",
            "output/pdf/supplementary_results.pdf",
            "outputs/forecasting/main_result_delta_mae_paired.pdf",
            "outputs/forecasting/main_result_delta_mae_paired.png",
        ]
    )
    destination = ROOT / "output/supplementary/forecastability-limits-proxy-bciaut-p300.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-seed", type=int, default=20261007)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument("--keep-main-figure", action="store_true",
                        help="Keep the existing Figure 5 artifacts unchanged.")
    parser.add_argument("--manuscript", type=Path, help="Optional external manuscript to check its 85 printed main-table MAEs; not packaged")
    args = parser.parse_args()
    if args.bundle_only:
        print(build_bundle())
        return
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    folds, point = build_point_results(args.bootstrap_seed, args.bootstrap_replicates)
    ranking = build_ranking(args.bootstrap_seed, args.bootstrap_replicates)
    robustness = build_robustness(folds, args.bootstrap_seed, args.bootstrap_replicates)
    diagnostics = build_variance()
    build_tables(point, ranking, robustness)
    build_figure(point)
    if args.keep_main_figure:
        assert all((FIG / ("proxy_structure_and_forecasting." + suffix)).is_file()
                   for suffix in ("pdf", "png", "json"))
    else:
        make_proxy_structure_and_forecasting()
    checked = verify_printed_tables(point, args.manuscript) if args.manuscript is not None else None
    short_horizon = build_short_horizon_results()
    feedback = build_feedback_additions()
    expanding_checked = verify_expanding_mean_table(args.manuscript) if args.manuscript is not None else None
    write_manifest(args.bootstrap_seed, args.bootstrap_replicates, diagnostics, checked, short_horizon, feedback, expanding_checked)
    print(json.dumps(
        {"model_summary_rows": len(point), "subject_metric_rows": len(folds),
         "ranking_summary_rows": len(ranking), "robustness_rows": len(robustness),
         "printed_mae_values_verified": checked, "short_horizon": short_horizon,
         "additional_baseline_tests": feedback["new_posthoc_baseline_tests"],
         "calibration_correction_tests": feedback["calibration_correction"]["posthoc_tests"],
         "printed_expanding_mean_values_verified": expanding_checked,
         "score_diagnostics": diagnostics},
        indent=2,
    ))


if __name__ == "__main__":
    main()
