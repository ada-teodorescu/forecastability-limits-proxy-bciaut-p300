"""The ranking comparator must use the prediction's own session."""

import csv

import numpy as np

from src.models import step8_temporal as step8


def test_anomaly_baseline_matches_session_and_agrees_between_views(tmp_path):
    baselines = []
    predictions = []
    selection = []
    for view in ("raw_level", "context_residual"):
        selection.append({"fold_subject_id": "SBJ01", "horizon": 5,
                          "target_view": view, "model_name": "context_score_mean",
                          "oof_residual_sigma": 0.2})
        for session, context in [("S01", 1.0), ("S02", 2.0)]:
            for anchor, label in [(9, 0), (10, 1)]:
                shared = {"fold_subject_id": "SBJ01", "subject_id": "SBJ01",
                          "session_id": session, "horizon": 5, "target_view": view,
                          "anchor_idx": anchor, "target_idx": anchor + 5}
                baselines.append({**shared, "model_name": "context_score_mean",
                                  "y_pred_mean": context if view == "raw_level" else 0.0})
                predictions.append({**shared, "ablation": "score_only",
                                    step8.PRIMARY_INNER_PROTOCOL_COLUMN: "last2_subject_holdout",
                                    "y_pred_mu": context if view == "raw_level" else 0.0,
                                    "y_pred_sigma": 0.2, "y_true": context if view == "raw_level" else 0.0,
                                    "context_score_mean_value": context,
                                    "session_cal_mean_raw": 1.5, "drop20_threshold_raw": 1.2,
                                    "z_score": 0.0, "target_drop20_state": label,
                                    "future_drop20_onset": label})
    for name, rows in [("step7_predictions.csv", baselines), ("step7_model_selection.csv", selection)]:
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    rows, _ = step8._build_anomaly_rows(predictions, config=step8.Step8Config(step7_root=tmp_path))
    for row in rows:
        assert row["baseline_mu_raw"] == row["context_score_mean_value"]
    raw = [r["baseline_p_drop20_state"] for r in rows if r["target_view"] == "raw_level"]
    residual = [r["baseline_p_drop20_state"] for r in rows if r["target_view"] == "context_residual"]
    np.testing.assert_array_equal(raw, residual)
    assert raw[0] != raw[2]
