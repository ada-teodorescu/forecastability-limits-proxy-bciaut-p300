"""Unit tests for Step 7 benchmarking helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.forecasting_loader import (
    Step7Config,
    build_primary_folds,
    build_rolling_folds,
    load_forecasting_table,
    validate_step7_config,
)
from src.evaluation.forecasting_metrics import compute_paired_statistic
from src.models.step7_benchmarks import (
    PRIMARY_MODELS,
    _PrimaryTaskSpec,
    _build_primary_task_specs,
    _checkpoint_dir,
    _execute_primary_task_specs,
    _load_checkpoint,
    _primary_checkpoint_key,
    _primary_result_from_checkpoint,
    _primary_result_to_checkpoint,
    _resolved_max_workers,
    _rolling_checkpoint_key,
    _RollingTaskSpec,
    _save_checkpoint,
    fit_and_predict_model,
    model_family,
)


@pytest.fixture(scope="module")
def step7_config() -> Step7Config:
    return Step7Config()


@pytest.fixture(scope="module")
def table_h10(step7_config: Step7Config):
    return load_forecasting_table(horizon=10, config=step7_config)


@pytest.fixture(scope="module")
def table_h05(step7_config: Step7Config):
    return load_forecasting_table(horizon=5, config=step7_config)


@pytest.fixture(scope="module")
def table_h25(step7_config: Step7Config):
    return load_forecasting_table(horizon=25, config=step7_config)


def test_load_forecasting_table_h10_matches_contract(table_h10) -> None:
    assert table_h10.n_rows == 3255
    assert table_h10.score_history_5.shape == (3255, 5)
    assert table_h10.score_history_l.shape == (3255, 10)
    assert table_h10.anchor_ts.shape == (3255, 300)
    assert table_h10.context_ts_flat.shape == (3255, 3000)
    assert table_h10.anchor_aligned_cov.shape == (3255, 24, 24)

    assert str(table_h10.subject_id[0]) == "SBJ01"
    assert str(table_h10.session_id[0]) == "S01"
    assert int(table_h10.anchor_idx[0]) == 9
    assert int(table_h10.target_idx[0]) == 19
    assert float(table_h10.anchor_score_raw[0]) == pytest.approx(float(table_h10.score_history_l[0, -1]))
    assert float(table_h10.target_score_context_residual[0]) == pytest.approx(
        float(table_h10.target_score_raw[0] - table_h10.context_score_mean_value[0]),
    )


def test_primary_and_rolling_folds_are_leakage_free(table_h10, table_h25) -> None:
    primary_folds = build_primary_folds(table_h10)
    assert len(primary_folds) == 15
    for fold in primary_folds:
        train_subjects = set(table_h10.subject_id[fold.train_idx].tolist())
        test_subjects = set(table_h10.subject_id[fold.test_idx].tolist())
        assert test_subjects == {fold.fold_subject_id}
        assert fold.fold_subject_id not in train_subjects
        assert fold.train_idx.size == 3038
        assert fold.test_idx.size == 217

    rolling_folds = build_rolling_folds(table_h25)
    assert len(rolling_folds) == 90
    for fold in rolling_folds:
        train_sessions = set(table_h25.session_id[fold.train_idx].tolist())
        test_sessions = set(table_h25.session_id[fold.test_idx].tolist())
        assert test_sessions == {fold.test_session}
        assert all(session <= fold.train_through_session for session in train_sessions)
        assert fold.test_session not in train_sessions


def test_exact_score_baselines_match_raw_and_residual_contracts(table_h10, step7_config: Step7Config) -> None:
    fold = build_primary_folds(table_h10)[0]
    sample_idx = fold.test_idx[:20]

    persistence_raw = fit_and_predict_model(
        table=table_h10,
        train_idx=fold.train_idx,
        test_idx=sample_idx,
        model_name="persistence",
        target_view="raw_level",
        config=step7_config,
        protocol="loso",
    )
    assert np.allclose(persistence_raw.test_pred, table_h10.anchor_score_raw[sample_idx], rtol=0.0, atol=1e-10)

    persistence_residual = fit_and_predict_model(
        table=table_h10,
        train_idx=fold.train_idx,
        test_idx=sample_idx,
        model_name="persistence",
        target_view="context_residual",
        config=step7_config,
        protocol="loso",
    )
    expected_persistence_residual = (
        np.asarray(table_h10.anchor_score_raw[sample_idx], dtype=np.float64)
        - np.asarray(table_h10.context_score_mean_value[sample_idx], dtype=np.float64)
    )
    assert np.allclose(persistence_residual.test_pred, expected_persistence_residual, rtol=0.0, atol=1e-10)

    moving_average_residual = fit_and_predict_model(
        table=table_h10,
        train_idx=fold.train_idx,
        test_idx=sample_idx,
        model_name="moving_average_5",
        target_view="context_residual",
        config=step7_config,
        protocol="loso",
    )
    expected_moving_average_residual = np.mean(
        np.asarray(table_h10.score_history_5[sample_idx], dtype=np.float64),
        axis=1,
    ) - np.asarray(table_h10.context_score_mean_value[sample_idx], dtype=np.float64)
    assert np.allclose(moving_average_residual.test_pred, expected_moving_average_residual, rtol=0.0, atol=1e-10)

    context_mean_residual = fit_and_predict_model(
        table=table_h10,
        train_idx=fold.train_idx,
        test_idx=sample_idx,
        model_name="context_score_mean",
        target_view="context_residual",
        config=step7_config,
        protocol="loso",
    )
    assert np.allclose(context_mean_residual.test_pred, 0.0, rtol=0.0, atol=1e-10)


def test_model_family_covers_all_primary_models() -> None:
    assert len(PRIMARY_MODELS) == 14
    families = {model_name: model_family(model_name) for model_name in PRIMARY_MODELS}
    assert families["train_mean"] == "score_only"
    assert families["ridge_anchor_ts"] == "eeg_only"
    assert families["elasticnet_anchor_ts"] == "eeg_only"
    assert families["pcr_score5_context_ts"] == "hybrid"
    assert families["riemannian_knn_anchor"] == "geometry_control"
    assert families["pcr_context_ts_permuted_target"] == "falsification_control"


def test_fit_and_predict_rejects_empty_splits(table_h10, step7_config: Step7Config) -> None:
    with pytest.raises(ValueError, match="zero train rows"):
        fit_and_predict_model(
            table=table_h10,
            train_idx=np.asarray([], dtype=np.int64),
            test_idx=np.asarray([0], dtype=np.int64),
            model_name="pcr_context_ts",
            target_view="raw_level",
            config=step7_config,
            protocol="loso",
        )
    with pytest.raises(ValueError, match="zero test rows"):
        fit_and_predict_model(
            table=table_h10,
            train_idx=np.asarray([0], dtype=np.int64),
            test_idx=np.asarray([], dtype=np.int64),
            model_name="pcr_context_ts",
            target_view="raw_level",
            config=step7_config,
            protocol="loso",
        )


def test_geometry_control_returns_finite_predictions(table_h05, step7_config: Step7Config) -> None:
    train_idx = np.arange(0, 40, dtype=np.int64)
    test_idx = np.arange(40, 46, dtype=np.int64)
    result = fit_and_predict_model(
        table=table_h05,
        train_idx=train_idx,
        test_idx=test_idx,
        model_name="riemannian_knn_anchor",
        target_view="raw_level",
        config=Step7Config(geometry_batch_size=3),
        protocol="loso",
    )
    assert result.train_pred.shape == (40,)
    assert result.test_pred.shape == (6,)
    assert result.calibration_residuals.shape == (40,)
    assert np.all(np.isfinite(result.train_pred))
    assert np.all(np.isfinite(result.test_pred))
    assert result.used_geometry_chunking is True


def test_paired_statistic_sign_permutation_and_bootstrap_are_deterministic() -> None:
    metric_a = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float64)
    metric_b = np.asarray([1.0, 0.5, 2.0, 2.5, 3.0], dtype=np.float64)
    first = compute_paired_statistic(
        comparison_id="demo",
        horizon=5,
        target_view="raw_level",
        metric="mae",
        model_a="a",
        model_b="b",
        metric_a=metric_a,
        metric_b=metric_b,
        bootstrap_seed=20260324,
        bootstrap_replicates=1000,
    )
    second = compute_paired_statistic(
        comparison_id="demo",
        horizon=5,
        target_view="raw_level",
        metric="mae",
        model_a="a",
        model_b="b",
        metric_a=metric_a,
        metric_b=metric_b,
        bootstrap_seed=20260324,
        bootstrap_replicates=1000,
    )
    assert first.p_value_method == "sign_permutation_pratt"
    assert first.p_value == pytest.approx(second.p_value)
    assert first.rank_biserial == pytest.approx(second.rank_biserial)
    assert first.bootstrap_ci_lower == pytest.approx(second.bootstrap_ci_lower)
    assert first.bootstrap_ci_upper == pytest.approx(second.bootstrap_ci_upper)
    assert np.isfinite(first.std_delta) and first.std_delta >= 0.0
    assert np.isfinite(first.min_detectable_delta_80) and first.min_detectable_delta_80 >= 0.0
    assert first.std_delta == pytest.approx(second.std_delta)
    assert first.min_detectable_delta_80 == pytest.approx(second.min_detectable_delta_80)


def test_step7_config_rejects_negative_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        validate_step7_config(Step7Config(max_workers=-1))


def test_primary_task_parallel_executor_matches_serial(table_h10) -> None:
    primary_tables = {10: table_h10}
    serial_config = Step7Config(horizons=(10,), rolling_horizons=(10,), max_workers=1)
    parallel_config = Step7Config(horizons=(10,), rolling_horizons=(10,), max_workers=2)
    task_specs = _build_primary_task_specs(primary_tables=primary_tables, config=serial_config)[:4]

    serial_rows = _execute_primary_task_specs(
        task_specs,
        primary_tables=primary_tables,
        config=serial_config,
        max_workers=1,
    )
    parallel_rows = _execute_primary_task_specs(
        task_specs,
        primary_tables=primary_tables,
        config=parallel_config,
        max_workers=2,
    )

    assert parallel_rows == serial_rows


# ---------------------------------------------------------------------------
# reserve_cores tests
# ---------------------------------------------------------------------------


def test_reserve_cores_default_is_two() -> None:
    config = Step7Config()
    assert config.reserve_cores == 2


def test_reserve_cores_validation_rejects_negative() -> None:
    with pytest.raises(ValueError, match="reserve_cores"):
        validate_step7_config(Step7Config(reserve_cores=-1))


def test_resolved_max_workers_subtracts_reserve_cores() -> None:
    from unittest.mock import patch

    with patch("src.models.step7_benchmarks.os.cpu_count", return_value=28):
        assert _resolved_max_workers(Step7Config(max_workers=0, reserve_cores=2)) == 26

    with patch("src.models.step7_benchmarks.os.cpu_count", return_value=4):
        assert _resolved_max_workers(Step7Config(max_workers=0, reserve_cores=2)) == 2

    # Explicit max_workers ignores reserve_cores.
    assert _resolved_max_workers(Step7Config(max_workers=10, reserve_cores=2)) == 10

    # reserve_cores >= cpu_count clamps to 1.
    with patch("src.models.step7_benchmarks.os.cpu_count", return_value=2):
        assert _resolved_max_workers(Step7Config(max_workers=0, reserve_cores=3)) == 1


# ---------------------------------------------------------------------------
# Checkpoint helpers tests
# ---------------------------------------------------------------------------


def test_checkpoint_dir_location() -> None:
    from pathlib import Path

    assert _checkpoint_dir(Path("data/forecasting")) == Path("data/forecasting/.step7_checkpoint")


def test_primary_checkpoint_key_scheme() -> None:
    task = _PrimaryTaskSpec(
        task_index=42,
        horizon=10,
        fold_subject_id="SBJ01",
        model_name="pcr_context_ts",
        train_idx=np.array([0]),
        test_idx=np.array([1]),
    )
    key = _primary_checkpoint_key(task)
    assert key == "primary/10/SBJ01/pcr_context_ts.json"
    # Key must NOT contain task_index (positional, not semantic).
    assert "42" not in key


def test_rolling_checkpoint_key_scheme() -> None:
    task = _RollingTaskSpec(
        task_index=99,
        horizon=5,
        subject_id="SBJ03",
        train_through_session="S04",
        test_session="S05",
        model_name="persistence",
        train_idx=np.array([0]),
        test_idx=np.array([1]),
    )
    key = _rolling_checkpoint_key(task)
    assert key == "rolling/5/SBJ03/S04/S05/persistence.json"
    assert "99" not in key


def test_checkpoint_roundtrip_primary(tmp_path) -> None:
    key = "primary/10/SBJ01/pcr_context_ts.json"
    payload = {
        "fold_metric_rows": [{"mae": 0.5, "rmse": 0.6}],
        "model_selection_rows": [{"selected_alpha": 1.0}],
        "prediction_rows": [{"y_true": 1.0, "y_pred_mean": 1.1}],
        "used_geometry_chunking": False,
    }
    _save_checkpoint(tmp_path, key, payload)
    loaded = _load_checkpoint(tmp_path, key)
    assert loaded == payload


def test_corrupted_checkpoint_returns_none(tmp_path) -> None:
    key = "primary/10/raw_level/SBJ01/bad.json"
    bad_path = tmp_path / key
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{truncated", encoding="utf-8")
    loaded = _load_checkpoint(tmp_path, key)
    assert loaded is None


def test_missing_checkpoint_returns_none(tmp_path) -> None:
    loaded = _load_checkpoint(tmp_path, "nonexistent/key.json")
    assert loaded is None


def test_primary_result_checkpoint_roundtrip() -> None:
    from src.models.step7_benchmarks import _PrimaryTaskResult

    result = _PrimaryTaskResult(
        task_index=7,
        fold_metric_rows=[{"mae": 0.123, "model_name": "persistence"}],
        model_selection_rows=[{"selected_alpha": "NA"}],
        prediction_rows=[{"y_true": 1.0, "y_pred_mean": 1.05}],
        used_geometry_chunking=False,
    )
    data = _primary_result_to_checkpoint(result)
    restored = _primary_result_from_checkpoint(task_index=7, data=data)
    assert restored.fold_metric_rows == result.fold_metric_rows
    assert restored.model_selection_rows == result.model_selection_rows
    assert restored.prediction_rows == result.prediction_rows
    assert restored.used_geometry_chunking == result.used_geometry_chunking


def test_execute_primary_with_checkpoint_resume(table_h10, tmp_path) -> None:
    """Run a small set of primary tasks, then re-run and verify cache is used."""
    primary_tables = {10: table_h10}
    config = Step7Config(horizons=(10,), rolling_horizons=(10,), max_workers=1)
    task_specs = _build_primary_task_specs(primary_tables=primary_tables, config=config)[:2]

    checkpoint_dir = tmp_path / ".step7_checkpoint"
    checkpoint_dir.mkdir()

    # First run: compute and cache.
    result1 = _execute_primary_task_specs(
        task_specs,
        primary_tables=primary_tables,
        config=config,
        max_workers=1,
        checkpoint_dir=checkpoint_dir,
    )

    # Verify checkpoint files were created.
    json_files = list(checkpoint_dir.rglob("*.json"))
    assert len(json_files) == 2

    # Second run: should load from cache, produce identical results.
    result2 = _execute_primary_task_specs(
        task_specs,
        primary_tables=primary_tables,
        config=config,
        max_workers=1,
        checkpoint_dir=checkpoint_dir,
    )
    assert result1 == result2
