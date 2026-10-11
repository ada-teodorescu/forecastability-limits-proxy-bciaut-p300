"""Recompute the paper's ranking analysis after fixing session matching.

Uses archived predictions only: no model is fitted and no endpoint is changed.
Original result files are retained; corrected artifacts go to review_revision/.
Run from the repository root: python pipeline/revise_paper_analysis.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.anomaly_detection import binary_auc_metrics
from src.models import step8_temporal as step8


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source = REPO_ROOT / "data/forecasting"
    output = source / "review_revision"
    output.mkdir(parents=True, exist_ok=True)
    archived = read_rows(source / "step8_anomaly_predictions.csv")
    # The original archive predates this metadata column. Its inner protocol
    # was the two-subject holdout, as recorded in the original manuscript.
    for row in archived:
        row.setdefault(step8.PRIMARY_INNER_PROTOCOL_COLUMN, "last2_subject_holdout")
    config = step8.Step8Config(step7_root=source)
    predictions, metrics = step8._build_anomaly_rows(archived, config=config)
    summary, _ = step8._summarize_anomaly(metrics)
    paired = step8._anomaly_paired_stats_rows(anomaly_metric_rows=metrics, config=config)
    for name, rows in [("predictions", predictions), ("metrics", metrics),
                       ("summary", summary), ("paired_stats", paired)]:
        write_rows(output / f"step8_anomaly_{name}.csv", rows)

    def key(row: dict) -> tuple:
        return tuple(str(row[k]) for k in ["fold_subject_id", "subject_id", "session_id",
                                           "horizon", "ablation", "anchor_idx", "target_idx"])

    raw = {key(row): row for row in predictions if row["target_view"] == "raw_level"}
    residual = {key(row): row for row in predictions if row["target_view"] == "context_residual"}
    assert raw.keys() == residual.keys()
    assert all(raw[k]["target_drop20_state"] == residual[k]["target_drop20_state"] for k in raw)
    max_mu_error = max(abs(r["baseline_mu_raw"] - r["context_score_mean_value"]) for r in predictions)
    max_view_probability_error = max(abs(raw[k]["baseline_p_drop20_state"] - residual[k]["baseline_p_drop20_state"]) for k in raw)
    assert max_mu_error < 1e-6
    assert max_view_probability_error < 1e-6
    unchanged_gru = max(abs(float(old["p_drop20_state"]) - new["p_drop20_state"]) for old, new in zip(archived, predictions))
    assert unchanged_gru < 1e-12
    changed_baselines = sum(abs(float(old["baseline_mu_raw"]) - new["baseline_mu_raw"]) > 1e-6 for old, new in zip(archived, predictions))
    audit = {"prediction_rows": len(predictions), "changed_baseline_mean_rows": changed_baselines,
             "max_baseline_mean_error": max_mu_error,
             "max_baseline_probability_difference_between_views": max_view_probability_error,
             "max_gru_probability_change": unchanged_gru,
             "endpoint_labels_identical_between_views": True}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    sensitivity = []
    for fraction in [0.7, 0.8, 0.9]:
        rows = [r for r in predictions if r["horizon"] == 5 and r["target_view"] == "raw_level" and r["ablation"] == "score_only"]
        fold_values = []
        for subject in sorted({r["fold_subject_id"] for r in rows}):
            fold = [r for r in rows if r["fold_subject_id"] == subject]
            threshold = fraction * np.asarray([r["session_cal_mean_raw"] for r in fold])
            labels = np.asarray([r["y_true"] for r in fold]) < threshold
            values = []
            for prefix in ["", "baseline_"]:
                risk = step8.prospective_drop20_probability(
                    mu_raw=np.asarray([r[prefix + "mu_raw"] for r in fold]),
                    sigma_raw=np.asarray([r[prefix + "sigma_raw"] for r in fold]),
                    drop20_threshold_raw=threshold)
                values.append(binary_auc_metrics(labels, risk)[0])
            fold_values.append(values)
        means = np.nanmean(fold_values, axis=0)
        sensitivity.append({"calibration_fraction": fraction, "gru_roc_auc": means[0], "baseline_roc_auc": means[1]})
    write_rows(output / "threshold_sensitivity.csv", sensitivity)
    print(json.dumps(audit, indent=2))
    for row in paired:
        if row["horizon"] == 5 and row["endpoint_label"] == "target_drop20_state":
            print({k: row[k] for k in ["target_view", "model_name", "summary_metric", "mean_metric_model", "mean_metric_baseline", "bootstrap_ci_lower", "bootstrap_ci_upper", "p_value_holm"]})
    print(sensitivity)


if __name__ == "__main__":
    main()
