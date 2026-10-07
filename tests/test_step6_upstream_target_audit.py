from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from src.features.step6_forecasting_task import detrended_exploration_path, load_csv_rows
from src.features.step6_upstream_target_audit import (
    AuditSessionInputs,
    Step6AuditConfig,
    build_step5_session_diagnostics,
    build_step6_target_lagged_records,
    build_step6_target_variance_records,
    compute_nested_dispersion,
    load_audit_sessions,
    load_saved_step6_variance_decomposition,
    run_step6_upstream_target_audit,
    target_view_series,
)


def _dummy_session(subject_id: str, session_id: str, score_raw: np.ndarray) -> AuditSessionInputs:
    score_raw = np.asarray(score_raw, dtype=np.float32)
    n_blocks = score_raw.shape[0]
    covariance = np.eye(24, dtype=np.float64)
    cov_stack = np.repeat(covariance[None, :, :], n_blocks, axis=0)
    aligned_cov_stack = np.repeat(covariance[None, :, :], max(n_blocks, 2), axis=0)
    features = np.zeros((n_blocks, 300), dtype=np.float32)
    return AuditSessionInputs(
        subject_id=subject_id,
        session_id=session_id,
        session_k=5,
        step3_online_covs=cov_stack,
        step4_calibration_aligned_covs=aligned_cov_stack,
        step4_online_aligned_covs=cov_stack,
        step4_online_features=features,
        session_ref_mean=covariance,
        score_raw_cal=np.asarray([1.0, 1.2, 1.1, 1.3], dtype=np.float32),
        score_raw_online=score_raw,
        d_target_cal=np.asarray([0.6, 0.5, 0.55, 0.45], dtype=np.float32),
        d_nontarget_cal=np.asarray([1.6, 1.7, 1.65, 1.75], dtype=np.float32),
        d_target_online=np.asarray(score_raw.shape[0] * [0.5], dtype=np.float32),
        d_nontarget_online=(np.asarray(score_raw, dtype=np.float32) + np.float32(0.5)).astype(np.float32),
        online_mdm_accuracy=1.0,
        session_cal_mean_raw=1.15,
        ref_separation_riemann=10.0,
    )


def test_compute_nested_dispersion_shares_sum_to_one() -> None:
    grouped = {
        "SBJ01": {
            "S01": np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
            "S02": np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float64),
        },
        "SBJ02": {
            "S01": np.asarray([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]], dtype=np.float64),
            "S02": np.asarray([[5.0, 4.0], [6.0, 5.0], [7.0, 6.0]], dtype=np.float64),
        },
    }
    result = compute_nested_dispersion(grouped)
    total_share = (
        result["share_between_subject"]
        + result["share_between_session_within_subject"]
        + result["share_within_session"]
    )
    assert total_share == pytest.approx(1.0)
    assert result["session_identity_share"] == pytest.approx(
        result["share_between_subject"] + result["share_between_session_within_subject"]
    )


def test_step5_session_diagnostics_roundtrip_score_exactly() -> None:
    session = _dummy_session("SBJ01", "S01", np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32))
    rows = build_step5_session_diagnostics([session], config=Step6AuditConfig(online_blocks=4, horizons=(1,)))
    assert rows[0]["score_roundtrip_error_max"] == pytest.approx(0.0)


def test_target_view_series_and_lagged_metrics_are_deterministic() -> None:
    session_a = _dummy_session("SBJ01", "S01", np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    session_b = _dummy_session("SBJ02", "S01", np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    config = Step6AuditConfig(online_blocks=4, horizons=(1,))

    centered = target_view_series(session_a.score_raw_online, target_view="session_mean_centered")
    diffs = target_view_series(session_a.score_raw_online, target_view="first_difference")
    assert np.array_equal(centered, np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64))
    assert np.array_equal(diffs, np.asarray([1.0, 1.0, 1.0], dtype=np.float64))

    lagged = build_step6_target_lagged_records([session_a, session_b], config=config)
    raw_row = next(row for row in lagged if row["target_view"] == "raw_level")
    centered_row = next(row for row in lagged if row["target_view"] == "session_mean_centered")
    diff_row = next(row for row in lagged if row["target_view"] == "first_difference")
    assert raw_row["lagged_corr"] == pytest.approx(1.0)
    assert centered_row["lagged_corr"] == pytest.approx(1.0)
    assert diff_row["persistence_mae"] == pytest.approx(0.0)


def test_step6c_raw_level_variance_matches_saved_step6_addendum() -> None:
    config = Step6AuditConfig()
    sessions = load_audit_sessions(config)
    raw_row = next(
        row for row in build_step6_target_variance_records(sessions, config=config) if row["target_view"] == "raw_level"
    )
    saved = load_saved_step6_variance_decomposition(config)

    assert raw_row["var_total"] == pytest.approx(saved["var_total"])
    assert raw_row["var_between_subject"] == pytest.approx(saved["var_between_subject"])
    assert raw_row["var_between_session_within_subject"] == pytest.approx(saved["var_between_session_within_subject"])
    assert raw_row["var_within_session"] == pytest.approx(saved["var_within_session"])
    assert raw_row["share_between_subject"] == pytest.approx(saved["share_between_subject"])
    assert raw_row["share_between_session_within_subject"] == pytest.approx(saved["share_between_session_within_subject"])
    assert raw_row["share_within_session"] == pytest.approx(saved["share_within_session"])


def test_step6c_centered_raw_lagged_metrics_match_saved_detrended_addendum() -> None:
    config = Step6AuditConfig()
    sessions = load_audit_sessions(config)
    centered_rows = {
        int(row["horizon"]): row
        for row in build_step6_target_lagged_records(sessions, config=config)
        if row["target_view"] == "session_mean_centered"
    }
    _, saved_rows = load_csv_rows(detrended_exploration_path(output_root=config.step6_root))
    for saved_row in saved_rows:
        horizon = int(saved_row["horizon"])
        row = centered_rows[horizon]
        assert row["lagged_corr"] == pytest.approx(float(saved_row["detrended_lagged_corr"]))
        assert row["persistence_mae"] == pytest.approx(float(saved_row["detrended_persistence_mae"]))


def test_run_step6_upstream_target_audit_writes_new_outputs_only() -> None:
    preserved_path = Path("data/forecasting/step6_exploration_summary.csv")
    before_mtime = preserved_path.stat().st_mtime_ns
    temp_root = Path("test_artifacts_step6c")
    shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        summary = run_step6_upstream_target_audit(
            Step6AuditConfig(
                output_root=temp_root / "forecasting",
                figure_root=temp_root / "figures",
            )
        )
        assert summary.session_count == 105
        assert (temp_root / "forecasting" / "step6c_step4_dispersion_summary.csv").is_file()
        assert (temp_root / "forecasting" / "step6c_step5_score_decomposition.csv").is_file()
        assert (temp_root / "forecasting" / "step6c_step6_target_variance.csv").is_file()
        assert (temp_root / "forecasting" / "step6c_hypothesis_judgments.csv").is_file()
        assert summary.conclusion_path.is_file()
        assert preserved_path.stat().st_mtime_ns == before_mtime
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
