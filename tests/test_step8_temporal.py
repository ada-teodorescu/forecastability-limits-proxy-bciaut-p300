"""Unit tests for Step 8 temporal forecasting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.forecasting_loader import (
    PrimaryFoldDefinition,
    RollingFoldDefinition,
    Step7Config,
    build_primary_folds,
    build_rolling_folds,
    load_forecasting_table,
)
from src.evaluation.anomaly_detection import binary_auc_metrics, prospective_drop20_probability, raw_space_prediction
from src.evaluation.forecasting_metrics import compute_paired_statistic, compute_regression_metrics_heteroscedastic
from src.features.step6_forecasting_task import load_csv_rows, write_csv
from src.models import step8_temporal
from src.models.step8_temporal import (
    ANOMALY_METRICS_COLUMNS,
    ANOMALY_PAIRED_STATS_COLUMNS,
    ANOMALY_PREDICTIONS_COLUMNS,
    FOLD_METRICS_COLUMNS,
    MODEL_SELECTION_COLUMNS,
    PAIRED_STATS_COLUMNS,
    PREDICTIONS_COLUMNS,
    PRIMARY_ABLATIONS,
    EXPECTED_FIGURE_FILENAMES,
    ROLLING_FOLD_COLUMNS,
    ROLLING_SUMMARY_COLUMNS,
    SUMMARY_COLUMNS,
    ArrayStandardizer,
    PAIRED_COMPARISON_TEMPLATES,
    ProbabilisticGRU,
    SequencePreprocessor,
    Step8Config,
    Step8ForecastingTable,
    _build_primary_task_specs,
    _build_rolling_task_specs,
    _checkpoint_dir,
    _checkpoint_config_fingerprint,
    _inner_validation_split_loso,
    _inner_validation_split_rolling,
    _primary_inner_validation_splits,
    _load_checkpoint,
    _primary_checkpoint_key,
    _primary_result_from_checkpoint,
    _primary_result_to_checkpoint,
    _rolling_checkpoint_key,
    _save_checkpoint,
    _strongest_step7_baseline_lookup,
    _raw_drop_threshold,
    _weighted_inner_cv_score,
    gaussian_nll_loss,
    load_step8_forecasting_table,
    run_step8,
)


@pytest.fixture(scope="module")
def step8_config() -> Step8Config:
    return Step8Config()


@pytest.fixture(scope="module")
def table_h10(step8_config: Step8Config) -> Step8ForecastingTable:
    return load_step8_forecasting_table(horizon=10, config=step8_config)


@pytest.fixture(scope="module")
def table_h25(step8_config: Step8Config) -> Step8ForecastingTable:
    return load_step8_forecasting_table(horizon=25, config=step8_config)


def _synthetic_table(
    *,
    horizon: int,
    subjects: tuple[str, ...],
    sessions: tuple[str, ...],
    context_length: int,
    tangent_dim: int,
) -> Step8ForecastingTable:
    rows: list[dict[str, object]] = []
    for subject_index, subject_id in enumerate(subjects):
        for session_ord, session_id in enumerate(sessions, start=1):
            session_cal_mean_raw = 2.0 + 0.1 * subject_index + 0.05 * session_ord
            drop20_threshold_raw = 0.8 * session_cal_mean_raw
            if session_ord == 1:
                target_score_raw = drop20_threshold_raw + 0.25 + 0.01 * subject_index
                future_drop20_onset = 1
                target_drop20_state = 0
            elif session_ord == 2:
                target_score_raw = drop20_threshold_raw - 0.18 - 0.01 * subject_index
                future_drop20_onset = 1
                target_drop20_state = 1
            else:
                target_score_raw = drop20_threshold_raw + 0.15 + 0.01 * subject_index
                future_drop20_onset = 0
                target_drop20_state = 0

            score_start = 1.0 + 0.2 * subject_index + 0.05 * session_ord
            score_sequence = np.linspace(score_start, score_start + 0.2, context_length, dtype=np.float32)
            context_score_mean_value = float(np.mean(score_sequence))
            anchor_idx = context_length - 1 + session_ord - 1
            target_idx = anchor_idx + horizon
            ts_sequence = (
                0.5 * subject_index
                + 0.1 * session_ord
                + 0.01 * np.arange(context_length, dtype=np.float32)[:, None]
                + 0.001 * np.arange(tangent_dim, dtype=np.float32)[None, :]
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "session_ord": session_ord,
                    "anchor_idx": anchor_idx,
                    "target_idx": target_idx,
                    "score_sequence": score_sequence,
                    "ts_sequence": ts_sequence,
                    "target_score_raw": target_score_raw,
                    "context_score_mean_value": context_score_mean_value,
                    "session_cal_mean_raw": session_cal_mean_raw,
                    "drop20_threshold_raw": drop20_threshold_raw,
                    "target_score_context_residual": target_score_raw - context_score_mean_value,
                    "target_drop20_state": target_drop20_state,
                    "future_drop20_onset": future_drop20_onset,
                },
            )

    rows.sort(key=lambda row: (str(row["subject_id"]), str(row["session_id"]), int(row["target_idx"])))
    return Step8ForecastingTable(
        horizon=horizon,
        subject_id=np.asarray([row["subject_id"] for row in rows], dtype=object),
        session_id=np.asarray([row["session_id"] for row in rows], dtype=object),
        session_ord=np.asarray([row["session_ord"] for row in rows], dtype=np.int16),
        session_k=np.asarray([len(sessions)] * len(rows), dtype=np.int16),
        target_block_id=np.asarray([row["session_ord"] for row in rows], dtype=np.int16),
        elapsed_runs=np.asarray([1] * len(rows), dtype=np.int16),
        anchor_idx=np.asarray([row["anchor_idx"] for row in rows], dtype=np.int16),
        target_idx=np.asarray([row["target_idx"] for row in rows], dtype=np.int16),
        score_sequence=np.asarray([row["score_sequence"] for row in rows], dtype=np.float32),
        ts_sequence=np.asarray([row["ts_sequence"] for row in rows], dtype=np.float32),
        target_score_raw=np.asarray([row["target_score_raw"] for row in rows], dtype=np.float32),
        context_score_mean_value=np.asarray([row["context_score_mean_value"] for row in rows], dtype=np.float32),
        session_cal_mean_raw=np.asarray([row["session_cal_mean_raw"] for row in rows], dtype=np.float64),
        drop20_threshold_raw=np.asarray([row["drop20_threshold_raw"] for row in rows], dtype=np.float64),
        target_score_context_residual=np.asarray(
            [row["target_score_context_residual"] for row in rows],
            dtype=np.float32,
        ),
        target_drop20_state=np.asarray([row["target_drop20_state"] for row in rows], dtype=np.int8),
        future_drop20_onset=np.asarray([row["future_drop20_onset"] for row in rows], dtype=np.int8),
    )


def _synthetic_primary_folds(table: Step8ForecastingTable) -> list[PrimaryFoldDefinition]:
    folds: list[PrimaryFoldDefinition] = []
    for subject_id in sorted({str(value) for value in table.subject_id.tolist()}):
        test_idx = np.flatnonzero(table.subject_id == subject_id).astype(np.int64)
        train_idx = np.flatnonzero(table.subject_id != subject_id).astype(np.int64)
        folds.append(
            PrimaryFoldDefinition(
                fold_subject_id=subject_id,
                horizon=table.horizon,
                train_idx=train_idx,
                test_idx=test_idx,
            ),
        )
    return folds


def _synthetic_rolling_folds(table: Step8ForecastingTable) -> list[RollingFoldDefinition]:
    folds: list[RollingFoldDefinition] = []
    for subject_id in sorted({str(value) for value in table.subject_id.tolist()}):
        subject_mask = table.subject_id == subject_id
        for train_through_session, test_session, max_session_ord in (("S01", "S02", 1), ("S02", "S03", 2)):
            train_idx = np.flatnonzero(subject_mask & (table.session_ord <= max_session_ord)).astype(np.int64)
            test_idx = np.flatnonzero(subject_mask & (table.session_id == test_session)).astype(np.int64)
            folds.append(
                RollingFoldDefinition(
                    subject_id=subject_id,
                    train_through_session=train_through_session,
                    test_session=test_session,
                    horizon=table.horizon,
                    train_idx=train_idx,
                    test_idx=test_idx,
                ),
            )
    return folds


def _write_step7_fixture_csvs(*, root: Path, table: Step8ForecastingTable, subjects: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary_columns = ("horizon", "target_view", "model_name", "mae_mean", "gaussian_nll_mean")
    fold_metric_columns = ("horizon", "target_view", "model_name", "fold_subject_id", "mae", "gaussian_nll")
    prediction_columns = ("fold_subject_id", "subject_id", "session_id", "horizon", "target_view", "anchor_idx", "target_idx", "model_name", "y_pred_mean")
    model_selection_columns = ("fold_subject_id", "horizon", "target_view", "model_name", "oof_residual_sigma")

    base_metrics = {
        "raw_level": {
            "context_score_mean": (0.30, 0.66),
            "persistence": (0.36, 0.74),
            "autoreg_ridge_5": (0.24, 0.52),
        },
        "context_residual": {
            "context_score_mean": (0.26, 0.58),
            "persistence": (0.33, 0.69),
            "autoreg_ridge_5": (0.21, 0.47),
        },
    }

    summary_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_selection_rows: list[dict[str, object]] = []

    for target_view, model_metrics in base_metrics.items():
        for model_name, (mae_mean, gaussian_nll_mean) in model_metrics.items():
            summary_rows.append(
                {
                    "horizon": table.horizon,
                    "target_view": target_view,
                    "model_name": model_name,
                    "mae_mean": mae_mean,
                    "gaussian_nll_mean": gaussian_nll_mean,
                },
            )
            for subject_index, subject_id in enumerate(subjects):
                fold_metric_rows.append(
                    {
                        "horizon": table.horizon,
                        "target_view": target_view,
                        "model_name": model_name,
                        "fold_subject_id": subject_id,
                        "mae": mae_mean + 0.01 * subject_index,
                        "gaussian_nll": gaussian_nll_mean + 0.02 * subject_index,
                    },
                )
        for subject_index, subject_id in enumerate(subjects):
            model_selection_rows.append(
                {
                    "fold_subject_id": subject_id,
                    "horizon": table.horizon,
                    "target_view": target_view,
                    "model_name": "context_score_mean",
                    "oof_residual_sigma": 0.18 + 0.01 * subject_index,
                },
            )
            row_indices = np.flatnonzero(table.subject_id == subject_id)
            for row_index in row_indices:
                prediction_rows.append(
                    {
                        "fold_subject_id": subject_id,
                        "subject_id": subject_id,
                        "session_id": str(table.session_id[row_index]),
                        "horizon": table.horizon,
                        "target_view": target_view,
                        "anchor_idx": int(table.anchor_idx[row_index]),
                        "target_idx": int(table.target_idx[row_index]),
                        "model_name": "context_score_mean",
                        "y_pred_mean": float(table.context_score_mean_value[row_index] if target_view == "raw_level" else 0.0),
                    },
                )

    write_csv(root / "step7_summary.csv", columns=summary_columns, records=summary_rows)
    write_csv(root / "step7_fold_metrics.csv", columns=fold_metric_columns, records=fold_metric_rows)
    write_csv(root / "step7_predictions.csv", columns=prediction_columns, records=prediction_rows)
    write_csv(root / "step7_model_selection.csv", columns=model_selection_columns, records=model_selection_rows)


def _expected_anomaly_summary_columns() -> tuple[str, ...]:
    base_keys = {"fold_subject_id", "horizon", "target_view", "ablation", "primary_inner_protocol"}
    columns = ["horizon", "target_view", "ablation", "primary_inner_protocol", "n_folds"]
    for column in ANOMALY_METRICS_COLUMNS:
        if column in base_keys:
            continue
        columns.append(f"{column}_mean")
        columns.append(f"{column}_std")
    return tuple(columns)


def _synthetic_anomaly_metric_rows(*, horizons: tuple[int, ...], target_views: tuple[str, ...], ablations: tuple[str, ...], folds: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        for target_view_index, target_view in enumerate(target_views):
            for ablation_index, ablation in enumerate(ablations):
                for fold_index, fold_subject_id in enumerate(folds):
                    base = 0.55 + 0.01 * horizon + 0.01 * target_view_index + 0.015 * ablation_index + 0.005 * fold_index
                    rows.append(
                        {
                            "fold_subject_id": fold_subject_id,
                            "horizon": horizon,
                            "target_view": target_view,
                            "ablation": ablation,
                            "primary_inner_protocol": "last2_subject_holdout",
                            "n_test_rows": 20,
                            "n_positive_drop20_state": 5,
                            "n_positive_drop20_onset": 4,
                            "prospective_roc_auc_p_drop20_state_for_state": base + 0.06,
                            "prospective_pr_auc_p_drop20_state_for_state": base + 0.04,
                            "prospective_roc_auc_p_drop20_state_for_onset": base + 0.05,
                            "prospective_pr_auc_p_drop20_state_for_onset": base + 0.03,
                            "prospective_roc_auc_neg_mu_raw_for_state": base - 0.01,
                            "prospective_pr_auc_neg_mu_raw_for_state": base - 0.02,
                            "prospective_roc_auc_neg_mu_raw_for_onset": base - 0.015,
                            "prospective_pr_auc_neg_mu_raw_for_onset": base - 0.025,
                            "prospective_roc_auc_sigma_raw_for_state": base - 0.005,
                            "prospective_pr_auc_sigma_raw_for_state": base - 0.015,
                            "prospective_roc_auc_sigma_raw_for_onset": base - 0.01,
                            "prospective_pr_auc_sigma_raw_for_onset": base - 0.02,
                            "baseline_roc_auc_p_drop20_state_for_state": base - 0.03,
                            "baseline_pr_auc_p_drop20_state_for_state": base - 0.04,
                            "baseline_roc_auc_p_drop20_state_for_onset": base - 0.025,
                            "baseline_pr_auc_p_drop20_state_for_onset": base - 0.035,
                            "baseline_roc_auc_neg_mu_raw_for_state": base - 0.02,
                            "baseline_pr_auc_neg_mu_raw_for_state": base - 0.03,
                            "baseline_roc_auc_neg_mu_raw_for_onset": base - 0.02,
                            "baseline_pr_auc_neg_mu_raw_for_onset": base - 0.03,
                            "baseline_roc_auc_sigma_raw_for_state": base - 0.02,
                            "baseline_pr_auc_sigma_raw_for_state": base - 0.03,
                            "baseline_roc_auc_sigma_raw_for_onset": base - 0.02,
                            "baseline_pr_auc_sigma_raw_for_onset": base - 0.03,
                            "z_score_mean": 0.0,
                            "z_score_std": 1.0,
                            "z_score_ks_stat": 0.1,
                            "z_score_ks_pvalue": 0.9,
                        },
                    )
    return rows


def _write_figures_only_fixture_csvs(*, step7_root: Path, output_root: Path) -> None:
    def _anomaly_metric_row(
        *,
        fold_subject_id: str,
        target_view: str,
        ablation: str,
        state_roc: float,
        state_pr: float,
        onset_roc: float,
        onset_pr: float,
        baseline_state_roc: float,
        baseline_state_pr: float,
        baseline_onset_roc: float,
        baseline_onset_pr: float,
    ) -> dict[str, object]:
        return {
            "fold_subject_id": fold_subject_id,
            "horizon": 5,
            "target_view": target_view,
            "ablation": ablation,
            "primary_inner_protocol": "last2_subject_holdout",
            "n_test_rows": 20,
            "n_positive_drop20_state": 5,
            "n_positive_drop20_onset": 4,
            "prospective_roc_auc_p_drop20_state_for_state": state_roc,
            "prospective_pr_auc_p_drop20_state_for_state": state_pr,
            "prospective_roc_auc_p_drop20_state_for_onset": onset_roc,
            "prospective_pr_auc_p_drop20_state_for_onset": onset_pr,
            "prospective_roc_auc_neg_mu_raw_for_state": 0.50,
            "prospective_pr_auc_neg_mu_raw_for_state": 0.48,
            "prospective_roc_auc_neg_mu_raw_for_onset": 0.49,
            "prospective_pr_auc_neg_mu_raw_for_onset": 0.47,
            "prospective_roc_auc_sigma_raw_for_state": 0.52,
            "prospective_pr_auc_sigma_raw_for_state": 0.50,
            "prospective_roc_auc_sigma_raw_for_onset": 0.51,
            "prospective_pr_auc_sigma_raw_for_onset": 0.49,
            "baseline_roc_auc_p_drop20_state_for_state": baseline_state_roc,
            "baseline_pr_auc_p_drop20_state_for_state": baseline_state_pr,
            "baseline_roc_auc_p_drop20_state_for_onset": baseline_onset_roc,
            "baseline_pr_auc_p_drop20_state_for_onset": baseline_onset_pr,
            "baseline_roc_auc_neg_mu_raw_for_state": 0.49,
            "baseline_pr_auc_neg_mu_raw_for_state": 0.47,
            "baseline_roc_auc_neg_mu_raw_for_onset": 0.48,
            "baseline_pr_auc_neg_mu_raw_for_onset": 0.46,
            "baseline_roc_auc_sigma_raw_for_state": 0.50,
            "baseline_pr_auc_sigma_raw_for_state": 0.48,
            "baseline_roc_auc_sigma_raw_for_onset": 0.49,
            "baseline_pr_auc_sigma_raw_for_onset": 0.47,
            "z_score_mean": 0.0,
            "z_score_std": 1.0,
            "z_score_ks_stat": 0.1,
            "z_score_ks_pvalue": 0.9,
        }

    write_csv(
        output_root / "step8_summary.csv",
        columns=step8_temporal.FIGURES_ONLY_SUMMARY_COLUMNS,
        records=[
            {
                "horizon": 5,
                "target_view": "raw_level",
                "ablation": "score_only",
                "mae_mean": 0.31,
                "gaussian_nll_mean": 0.42,
                "gaussian_picp_80_mean": 0.81,
                "conformal_picp_80_mean": 0.79,
            },
            {
                "horizon": 5,
                "target_view": "context_residual",
                "ablation": "score_only",
                "mae_mean": 0.21,
                "gaussian_nll_mean": 0.33,
                "gaussian_picp_80_mean": 0.82,
                "conformal_picp_80_mean": 0.8,
            },
        ],
    )
    write_csv(
        output_root / "step8_rolling_summary.csv",
        columns=step8_temporal.FIGURES_ONLY_ROLLING_SUMMARY_COLUMNS,
        records=[
            {"horizon": 5, "target_view": "raw_level", "first_difference_pearson_r_mean": 0.12},
            {"horizon": 5, "target_view": "context_residual", "first_difference_pearson_r_mean": 0.04},
        ],
    )
    write_csv(
        output_root / "step8_anomaly_predictions.csv",
        columns=step8_temporal.FIGURES_ONLY_ANOMALY_PREDICTIONS_COLUMNS,
        records=[
            {
                "y_pred_sigma": 0.2,
                "z_score": -0.9,
                "target_drop20_state": 0,
                "future_drop20_onset": 0,
                "sigma_raw": 0.2,
                "p_drop20_state": 0.2,
                "baseline_p_drop20_state": 0.15,
            },
            {
                "y_pred_sigma": 0.25,
                "z_score": -0.1,
                "target_drop20_state": 1,
                "future_drop20_onset": 0,
                "sigma_raw": 0.25,
                "p_drop20_state": 0.6,
                "baseline_p_drop20_state": 0.45,
            },
            {
                "y_pred_sigma": 0.3,
                "z_score": 0.2,
                "target_drop20_state": 0,
                "future_drop20_onset": 1,
                "sigma_raw": 0.3,
                "p_drop20_state": 0.4,
                "baseline_p_drop20_state": 0.35,
            },
            {
                "y_pred_sigma": 0.35,
                "z_score": 1.1,
                "target_drop20_state": 1,
                "future_drop20_onset": 1,
                "sigma_raw": 0.35,
                "p_drop20_state": 0.8,
                "baseline_p_drop20_state": 0.7,
            },
        ],
    )
    write_csv(
        output_root / "step8_anomaly_metrics.csv",
        columns=ANOMALY_METRICS_COLUMNS,
        records=[
            _anomaly_metric_row(
                fold_subject_id="SBJ01",
                target_view="raw_level",
                ablation="score_only",
                state_roc=0.68,
                state_pr=0.55,
                onset_roc=0.63,
                onset_pr=0.49,
                baseline_state_roc=0.56,
                baseline_state_pr=0.43,
                baseline_onset_roc=0.52,
                baseline_onset_pr=0.39,
            ),
            _anomaly_metric_row(
                fold_subject_id="SBJ02",
                target_view="raw_level",
                ablation="score_only",
                state_roc=0.66,
                state_pr=0.53,
                onset_roc=0.61,
                onset_pr=0.47,
                baseline_state_roc=0.55,
                baseline_state_pr=0.41,
                baseline_onset_roc=0.51,
                baseline_onset_pr=0.38,
            ),
            _anomaly_metric_row(
                fold_subject_id="SBJ01",
                target_view="context_residual",
                ablation="score_only",
                state_roc=0.60,
                state_pr=0.47,
                onset_roc=0.58,
                onset_pr=0.44,
                baseline_state_roc=0.57,
                baseline_state_pr=0.45,
                baseline_onset_roc=0.53,
                baseline_onset_pr=0.41,
            ),
            _anomaly_metric_row(
                fold_subject_id="SBJ02",
                target_view="context_residual",
                ablation="score_only",
                state_roc=0.59,
                state_pr=0.46,
                onset_roc=0.57,
                onset_pr=0.43,
                baseline_state_roc=0.56,
                baseline_state_pr=0.44,
                baseline_onset_roc=0.52,
                baseline_onset_pr=0.40,
            ),
        ],
    )
    write_csv(
        step7_root / "step7_summary.csv",
        columns=step8_temporal.FIGURES_ONLY_STEP7_SUMMARY_COLUMNS,
        records=[
            {
                "horizon": 5,
                "target_view": "raw_level",
                "model_name": "context_score_mean",
                "mae_mean": 0.29,
                "gaussian_nll_mean": 0.4,
            },
            {
                "horizon": 5,
                "target_view": "context_residual",
                "model_name": "context_score_mean",
                "mae_mean": 0.22,
                "gaussian_nll_mean": 0.34,
            },
        ],
    )


def test_gru_forward_shapes_and_sigma_floor() -> None:
    for input_dim in (1, 4, 5):
        model = ProbabilisticGRU(input_dim=input_dim, hidden_dim=8, dropout=0.2, sigma_floor=1e-6)
        x = torch.zeros((6, 10, input_dim), dtype=torch.float32)
        mu, sigma = model(x)
        assert mu.shape == (6,)
        assert sigma.shape == (6,)
        assert torch.all(sigma >= 1e-6)


def test_gaussian_nll_loss_matches_manual_formula() -> None:
    y_true = torch.tensor([1.0, 2.0], dtype=torch.float32)
    mu = torch.tensor([1.5, 1.0], dtype=torch.float32)
    sigma = torch.tensor([0.5, 2.0], dtype=torch.float32)
    actual = float(gaussian_nll_loss(y_true, mu, sigma).item())

    y_true_np = np.asarray([1.0, 2.0], dtype=np.float64)
    mu_np = np.asarray([1.5, 1.0], dtype=np.float64)
    sigma_np = np.asarray([0.5, 2.0], dtype=np.float64)
    variance = sigma_np * sigma_np
    expected = float(
        np.mean(0.5 * np.log(2.0 * np.pi * variance) + 0.5 * ((y_true_np - mu_np) ** 2) / variance),
    )
    assert actual == pytest.approx(expected, rel=1e-6)


def test_train_only_preprocessor_uses_train_statistics() -> None:
    train_scores = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    test_scores = np.asarray([[100.0, 101.0, 102.0]], dtype=np.float32)
    train_ts = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
        ],
        dtype=np.float32,
    )
    test_ts = np.asarray([[[1000.0, 5.0], [1001.0, 5.0], [1002.0, 5.0]]], dtype=np.float32)
    table = Step8ForecastingTable(
        horizon=5,
        subject_id=np.asarray(["A", "A", "B"], dtype=object),
        session_id=np.asarray(["S01", "S02", "S03"], dtype=object),
        session_ord=np.asarray([1, 2, 3], dtype=np.int16),
        session_k=np.asarray([3, 3, 3], dtype=np.int16),
        target_block_id=np.asarray([1, 1, 1], dtype=np.int16),
        elapsed_runs=np.asarray([1, 1, 1], dtype=np.int16),
        anchor_idx=np.asarray([2, 2, 2], dtype=np.int16),
        target_idx=np.asarray([7, 7, 7], dtype=np.int16),
        score_sequence=np.concatenate([train_scores, test_scores], axis=0),
        ts_sequence=np.concatenate([train_ts, test_ts], axis=0),
        target_score_raw=np.asarray([3.5, 4.5, 5.5], dtype=np.float32),
        context_score_mean_value=np.asarray([2.0, 3.0, 101.0], dtype=np.float32),
        session_cal_mean_raw=np.asarray([1.4, 1.4, 1.4], dtype=np.float64),
        drop20_threshold_raw=np.asarray([1.12, 1.12, 1.12], dtype=np.float64),
        target_score_context_residual=np.asarray([1.5, 1.5, -95.5], dtype=np.float32),
        target_drop20_state=np.asarray([0, 0, 1], dtype=np.int8),
        future_drop20_onset=np.asarray([0, 1, 1], dtype=np.int8),
    )
    train_idx = np.asarray([0, 1], dtype=np.int64)
    preprocessor = step8_temporal._fit_preprocessor(
        table=table,
        fit_idx=train_idx,
        ablation="hybrid",
        context_length=3,
        eeg_pca_components=2,
    )

    expected_score_mean = float(np.mean(train_scores))
    assert float(preprocessor.score_standardizer.mean[0]) == pytest.approx(expected_score_mean)
    assert preprocessor.score_standardizer.mean.dtype == np.float32
    assert preprocessor.score_standardizer.scale.dtype == np.float32
    assert np.allclose(
        np.asarray(preprocessor.eeg_pca.mean_, dtype=np.float64),
        np.mean(train_ts.reshape(-1, 2), axis=0),
    )
    assembled = step8_temporal._assemble_inputs(
        table=table,
        indices=np.asarray([2], dtype=np.int64),
        preprocessor=preprocessor,
        context_length=3,
    )
    assert assembled.dtype == np.float32
    assert assembled.shape == (1, 3, 3)


def test_load_step8_table_reconstructs_sequences_labels_and_threshold(table_h10: Step8ForecastingTable) -> None:
    assert table_h10.n_rows == 3255
    assert table_h10.score_sequence.shape == (3255, 10)
    assert table_h10.ts_sequence.shape == (3255, 10, 300)
    assert float(table_h10.target_score_context_residual[0]) == pytest.approx(
        float(table_h10.target_score_raw[0] - table_h10.context_score_mean_value[0]),
    )
    assert float(table_h10.drop20_threshold_raw[0]) == pytest.approx(0.8 * float(table_h10.session_cal_mean_raw[0]))

    _, manifest_rows = load_csv_rows(Path("data/forecasting/step6_h10_manifest.csv"))
    filtered_rows = [row for row in manifest_rows if int(row["anchor_idx"]) >= 9]
    filtered_rows.sort(key=lambda row: (row["subject_id"], row["session_id"], int(row["target_idx"])))
    first = filtered_rows[0]
    assert int(table_h10.target_drop20_state[0]) == int(first["target_drop20_state"])
    assert int(table_h10.future_drop20_onset[0]) == int(first["future_drop20_onset"])


def test_step8_folds_match_step7_exactly(table_h10: Step8ForecastingTable, table_h25: Step8ForecastingTable, step8_config: Step8Config) -> None:
    step7_h10 = load_forecasting_table(horizon=10, config=Step7Config())
    step7_h25 = load_forecasting_table(horizon=25, config=Step7Config())

    step8_primary = build_primary_folds(table_h10)
    step7_primary = build_primary_folds(step7_h10)
    assert len(step8_primary) == len(step7_primary) == 15
    for fold8, fold7 in zip(step8_primary, step7_primary, strict=True):
        assert fold8.fold_subject_id == fold7.fold_subject_id
        assert np.array_equal(fold8.test_idx, fold7.test_idx)
        assert np.array_equal(fold8.train_idx, fold7.train_idx)

    step8_rolling = build_rolling_folds(table_h25)
    step7_rolling = build_rolling_folds(step7_h25)
    assert len(step8_rolling) == len(step7_rolling) == 90
    for fold8, fold7 in zip(step8_rolling, step7_rolling, strict=True):
        assert fold8.subject_id == fold7.subject_id
        assert fold8.train_through_session == fold7.train_through_session
        assert fold8.test_session == fold7.test_session
        assert np.array_equal(fold8.test_idx, fold7.test_idx)


def test_inner_validation_splits_are_deterministic(table_h10: Step8ForecastingTable, table_h25: Step8ForecastingTable) -> None:
    fold = build_primary_folds(table_h10)[0]
    inner_train_idx, val_idx = _inner_validation_split_loso(table_h10, fold.train_idx)
    train_subjects = sorted({str(value) for value in table_h10.subject_id[fold.train_idx]})
    assert set(table_h10.subject_id[val_idx].tolist()) == set(train_subjects[-2:])
    assert set(table_h10.subject_id[inner_train_idx].tolist()).isdisjoint(set(table_h10.subject_id[val_idx].tolist()))

    first_rolling_fold = build_rolling_folds(table_h25)[0]
    _, val_none = _inner_validation_split_rolling(table_h25, first_rolling_fold.train_idx)
    assert val_none is None

    later_fold = next(fold for fold in build_rolling_folds(table_h25) if fold.subject_id == "SBJ01" and fold.train_through_session == "S03")
    _, rolling_val_idx = _inner_validation_split_rolling(table_h25, later_fold.train_idx)
    assert rolling_val_idx is not None
    assert set(table_h25.session_id[rolling_val_idx].tolist()) == {"S03"}


def test_group_kfold_primary_inner_validation_splits_cover_training_subjects(table_h10: Step8ForecastingTable) -> None:
    fold = build_primary_folds(table_h10)[0]
    train_subjects = set(table_h10.subject_id[fold.train_idx].tolist())
    grouped_splits = _primary_inner_validation_splits(
        table_h10,
        fold.train_idx,
        protocol="group_kfold",
    )
    assert len(grouped_splits) == 5
    covered_validation_subjects: set[object] = set()
    for inner_train_idx, val_idx in grouped_splits:
        train_subject_set = set(table_h10.subject_id[inner_train_idx].tolist())
        val_subject_set = set(table_h10.subject_id[val_idx].tolist())
        assert train_subject_set.isdisjoint(val_subject_set)
        assert val_subject_set
        covered_validation_subjects.update(val_subject_set)
    assert covered_validation_subjects == train_subjects


def test_weighted_inner_cv_score_uses_validation_row_counts() -> None:
    score = _weighted_inner_cv_score([1.0, 3.0], [3, 1])
    assert score == pytest.approx(1.5)


def test_checkpoint_dir_changes_when_primary_inner_protocol_changes(tmp_path: Path) -> None:
    output_root = tmp_path / "step8"
    default_dir = _checkpoint_dir(output_root, Step8Config())
    grouped_dir = _checkpoint_dir(output_root, Step8Config(primary_inner_protocol="group_kfold"))
    assert default_dir != grouped_dir
    assert _checkpoint_config_fingerprint(Step8Config()) != _checkpoint_config_fingerprint(
        Step8Config(primary_inner_protocol="group_kfold"),
    )


def test_raw_drop_threshold_respects_configured_drop_threshold() -> None:
    assert _raw_drop_threshold(session_cal_mean=2.0, drop_threshold=-0.20) == pytest.approx(1.6)
    assert _raw_drop_threshold(session_cal_mean=2.0, drop_threshold=-0.10) == pytest.approx(1.8)


def test_validate_step8_config_rejects_unknown_primary_inner_protocol() -> None:
    with pytest.raises(ValueError, match="primary_inner_protocol"):
        step8_temporal.validate_step8_config(Step8Config(primary_inner_protocol="unknown"))


def test_checkpoint_roundtrip_and_corruption_handling(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / ".step8_checkpoint"
    checkpoint_dir.mkdir()
    task = step8_temporal._PrimaryTaskSpec(
        task_index=7,
        horizon=5,
        target_view="raw_level",
        ablation="score_only",
        fold_subject_id="SBJ03",
        train_idx=np.asarray([0, 1], dtype=np.int64),
        test_idx=np.asarray([2], dtype=np.int64),
    )
    result = step8_temporal._PrimaryTaskResult(
        task_index=7,
        fold_metric_rows=[{"fold_subject_id": "SBJ03", "horizon": 5, "target_view": "raw_level", "ablation": "score_only", "mae": 0.1, "rmse": 0.2, "r2": 0.3, "pearson_r": 0.4, "gaussian_nll": 0.5, "gaussian_picp_80": 0.8, "gaussian_picp_90": 0.9, "conformal_picp_80": 0.8, "conformal_picp_90": 0.9, "mean_learned_sigma": 0.2, "mean_oof_sigma": 0.3, "n_test_rows": 1}],
        model_selection_rows=[{"fold_subject_id": "SBJ03", "horizon": 5, "target_view": "raw_level", "ablation": "score_only", "selected_hidden_dim": 32, "selected_learning_rate": 1e-3, "selected_dropout": 0.2, "selected_weight_decay": 1e-4, "selected_eeg_pca_components": 0, "selected_context_length": 5, "best_val_nll": 0.1, "n_epochs_trained": 2, "oof_residual_sigma": 0.2, "conformal_q80": 0.3, "conformal_q90": 0.4, "n_train_rows": 2, "n_test_rows": 1}],
        prediction_rows=[{"fold_subject_id": "SBJ03", "subject_id": "SBJ03", "session_id": "S01", "horizon": 5, "target_view": "raw_level", "ablation": "score_only", "anchor_idx": 2, "target_idx": 7, "y_true": 1.0, "y_pred_mu": 1.1, "y_pred_sigma": 0.2, "oof_sigma": 0.25, "gaussian_lower_80": 0.8, "gaussian_upper_80": 1.4, "gaussian_lower_90": 0.7, "gaussian_upper_90": 1.5, "conformal_lower_80": 0.9, "conformal_upper_80": 1.3, "conformal_lower_90": 0.8, "conformal_upper_90": 1.4, "z_score": -0.5, "context_score_mean_value": 1.0, "session_cal_mean_raw": 1.5, "drop20_threshold_raw": 1.2, "target_drop20_state": 0, "future_drop20_onset": 1}],
    )
    key = _primary_checkpoint_key(task)
    _save_checkpoint(checkpoint_dir, key, _primary_result_to_checkpoint(result))
    loaded = _load_checkpoint(checkpoint_dir, key)
    restored = _primary_result_from_checkpoint(task_index=7, data=loaded)
    assert restored.fold_metric_rows == result.fold_metric_rows

    bad_key = _rolling_checkpoint_key(
        step8_temporal._RollingTaskSpec(
            task_index=1,
            horizon=5,
            target_view="raw_level",
            ablation="score_only",
            subject_id="SBJ03",
            train_through_session="S01",
            test_session="S02",
            train_idx=np.asarray([0], dtype=np.int64),
            test_idx=np.asarray([1], dtype=np.int64),
        ),
    )
    bad_path = checkpoint_dir / bad_key
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{broken", encoding="utf-8")
    assert _load_checkpoint(checkpoint_dir, bad_key) is None


def test_execute_primary_task_specs_retries_missing_tasks_serially_after_memory_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / ".step8_checkpoint"
    checkpoint_dir.mkdir()
    task_specs = [
        step8_temporal._PrimaryTaskSpec(
            task_index=0,
            horizon=5,
            target_view="raw_level",
            ablation="score_only",
            fold_subject_id="SBJ01",
            train_idx=np.asarray([0, 1], dtype=np.int64),
            test_idx=np.asarray([2], dtype=np.int64),
        ),
        step8_temporal._PrimaryTaskSpec(
            task_index=1,
            horizon=5,
            target_view="raw_level",
            ablation="score_only",
            fold_subject_id="SBJ02",
            train_idx=np.asarray([0, 1], dtype=np.int64),
            test_idx=np.asarray([2], dtype=np.int64),
        ),
    ]
    run_calls: list[int] = []

    def _fake_run_primary_task(task, *, primary_tables, config):
        run_calls.append(int(task.task_index))
        return step8_temporal._PrimaryTaskResult(
            task_index=int(task.task_index),
            fold_metric_rows=[{"task_index": int(task.task_index)}],
            model_selection_rows=[{"task_index": int(task.task_index)}],
            prediction_rows=[{"task_index": int(task.task_index)}],
        )

    def _fake_execute_ordered_tasks(task_specs, *, max_workers, run_task):
        run_task(task_specs[0])
        raise MemoryError("simulated array allocation failure")

    monkeypatch.setattr(step8_temporal, "_run_primary_task", _fake_run_primary_task)
    monkeypatch.setattr(step8_temporal, "_execute_ordered_tasks", _fake_execute_ordered_tasks)

    fold_metric_rows, prediction_rows, model_selection_rows = step8_temporal._execute_primary_task_specs(
        task_specs,
        primary_tables={},
        config=Step8Config(),
        max_workers=2,
        checkpoint_dir=checkpoint_dir,
    )

    assert run_calls == [0, 1]
    assert [row["task_index"] for row in fold_metric_rows] == [0, 1]
    assert [row["task_index"] for row in prediction_rows] == [0, 1]
    assert [row["task_index"] for row in model_selection_rows] == [0, 1]
    assert _load_checkpoint(checkpoint_dir, _primary_checkpoint_key(task_specs[0])) is not None
    assert _load_checkpoint(checkpoint_dir, _primary_checkpoint_key(task_specs[1])) is not None


def test_heteroscedastic_metrics_are_finite_and_intervals_ordered() -> None:
    metrics = compute_regression_metrics_heteroscedastic(
        y_true=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
        y_pred=np.asarray([1.1, 1.4, 2.1], dtype=np.float64),
        gaussian_sigma=np.asarray([0.2, 0.3, 0.4], dtype=np.float64),
        conformal_q80=0.25,
        conformal_q90=0.35,
        corr_epsilon=1e-12,
        mean_oof_sigma=0.3,
        include_first_difference=True,
    )
    for value in metrics.values():
        assert np.isfinite(float(value))
    assert 0.0 <= float(metrics["gaussian_picp_80"]) <= 1.0
    assert 0.0 <= float(metrics["conformal_picp_90"]) <= 1.0


def test_strongest_step7_baseline_lookup_matches_manual_candidate_scan(step8_config: Step8Config) -> None:
    lookup = _strongest_step7_baseline_lookup(step8_config)
    _, rows = load_csv_rows(Path("data/forecasting/step7_summary.csv"))
    candidates = [
        row
        for row in rows
        if int(row["horizon"]) == 5
        and str(row["target_view"]) == "raw_level"
        and str(row["model_name"]) in step8_temporal.STEP7_TRAINED_BASELINE_CANDIDATES
    ]
    manual = min(
        candidates,
        key=lambda row: (
            float(row["mae_mean"]),
            float(row["gaussian_nll_mean"]),
            str(row["model_name"]),
        ),
    )
    assert lookup[(5, "raw_level", "mae")] == str(manual["model_name"])


def test_anomaly_transform_and_nan_safe_auc() -> None:
    mu_raw, sigma_raw = raw_space_prediction(
        target_view="context_residual",
        y_pred_mu=np.asarray([0.2], dtype=np.float64),
        y_pred_sigma=np.asarray([0.5], dtype=np.float64),
        context_score_mean_value=np.asarray([1.1], dtype=np.float64),
    )
    assert float(mu_raw[0]) == pytest.approx(1.3)
    assert float(sigma_raw[0]) == pytest.approx(0.5)
    prob = prospective_drop20_probability(
        mu_raw=mu_raw,
        sigma_raw=sigma_raw,
        drop20_threshold_raw=np.asarray([1.0], dtype=np.float64),
    )
    assert 0.0 <= float(prob[0]) <= 1.0
    roc_auc, pr_auc = binary_auc_metrics(np.asarray([1, 1, 1], dtype=np.int8), np.asarray([0.1, 0.2, 0.3], dtype=np.float64))
    assert np.isnan(roc_auc)
    assert np.isnan(pr_auc)


def test_compute_paired_statistic_supports_higher_is_better_direction() -> None:
    statistic = compute_paired_statistic(
        comparison_id="gru_score_only_vs_context_score_mean",
        horizon=5,
        target_view="raw_level",
        metric="roc_auc",
        model_a="context_score_mean",
        model_b="gru_score_only",
        metric_a=np.asarray([0.52, 0.55, 0.54, 0.53], dtype=np.float64),
        metric_b=np.asarray([0.66, 0.69, 0.67, 0.68], dtype=np.float64),
        bootstrap_seed=123,
        bootstrap_replicates=128,
        higher_is_better=True,
    )
    assert statistic.mean_delta_b_minus_a > 0.0
    assert statistic.better_model == "gru_score_only"


def test_anomaly_paired_stats_generation_returns_expected_schema_and_family_counts() -> None:
    anomaly_metric_rows = _synthetic_anomaly_metric_rows(
        horizons=(5, 10, 15, 20, 25),
        target_views=("raw_level", "context_residual"),
        ablations=PRIMARY_ABLATIONS,
        folds=("SBJ01", "SBJ02", "SBJ03"),
    )
    rows = step8_temporal._anomaly_paired_stats_rows(
        anomaly_metric_rows=anomaly_metric_rows,
        config=Step8Config(bootstrap_replicates=64, bootstrap_seed=77),
    )

    assert len(rows) == 120
    assert tuple(rows[0].keys()) == ANOMALY_PAIRED_STATS_COLUMNS
    family_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        family_key = (str(row["endpoint_label"]), str(row["summary_metric"]))
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        assert np.isfinite(float(row["p_value_holm"]))
    assert family_counts == {
        ("target_drop20_state", "roc_auc"): 30,
        ("target_drop20_state", "pr_auc"): 30,
        ("future_drop20_onset", "roc_auc"): 30,
        ("future_drop20_onset", "pr_auc"): 30,
    }


def test_anomaly_paired_stats_drop_only_nonfinite_folds() -> None:
    anomaly_metric_rows = _synthetic_anomaly_metric_rows(
        horizons=(5,),
        target_views=("raw_level",),
        ablations=("score_only",),
        folds=("SBJ01", "SBJ02", "SBJ03"),
    )
    anomaly_metric_rows[1]["prospective_roc_auc_p_drop20_state_for_state"] = float("nan")
    rows = step8_temporal._anomaly_paired_stats_rows(
        anomaly_metric_rows=anomaly_metric_rows,
        config=Step8Config(horizons=(5,), bootstrap_replicates=32, bootstrap_seed=11),
    )

    target_row = next(
        row
        for row in rows
        if str(row["endpoint_label"]) == "target_drop20_state" and str(row["summary_metric"]) == "roc_auc"
    )
    assert int(target_row["n_folds_total"]) == 3
    assert int(target_row["n_folds_valid"]) == 2
    assert int(target_row["n_nonzero_deltas"]) <= 2


def test_run_step8_figures_only_writes_expected_figures_from_saved_csvs(tmp_path: Path) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)

    figure_count = step8_temporal.run_step8_figures_only(
        Step8Config(step7_root=step7_root, output_root=output_root, figure_root=figure_root),
    )

    assert figure_count == len(EXPECTED_FIGURE_FILENAMES)
    for filename in EXPECTED_FIGURE_FILENAMES:
        assert (figure_root / filename).is_file()


def test_run_step8_reports_only_writes_anomaly_paired_stats_from_saved_csvs(tmp_path: Path) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)

    summary = step8_temporal.run_step8_reports_only(
        Step8Config(step7_root=step7_root, output_root=output_root, figure_root=figure_root),
    )

    assert summary.anomaly_paired_stats_rows == 8
    assert summary.figure_count == len(EXPECTED_FIGURE_FILENAMES)
    header, rows = load_csv_rows(output_root / "step8_anomaly_paired_stats.csv")
    assert tuple(header) == ANOMALY_PAIRED_STATS_COLUMNS
    assert len(rows) == 8
    for filename in EXPECTED_FIGURE_FILENAMES:
        assert (figure_root / filename).is_file()


def test_main_figures_only_skips_training_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)

    def _fail_run_step8(*args, **kwargs):
        raise AssertionError("run_step8 should not be called in --figures-only mode")

    monkeypatch.setattr(step8_temporal, "run_step8", _fail_run_step8)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "step8_temporal.py",
            "--step7-root",
            str(step7_root),
            "--output-root",
            str(output_root),
            "--figure-root",
            str(figure_root),
            "--figures-only",
        ],
    )

    step8_temporal.main()

    for filename in EXPECTED_FIGURE_FILENAMES:
        assert (figure_root / filename).is_file()


def test_main_reports_only_skips_training_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)

    def _fail_run_step8(*args, **kwargs):
        raise AssertionError("run_step8 should not be called in --reports-only mode")

    monkeypatch.setattr(step8_temporal, "run_step8", _fail_run_step8)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "step8_temporal.py",
            "--step7-root",
            str(step7_root),
            "--output-root",
            str(output_root),
            "--figure-root",
            str(figure_root),
            "--reports-only",
        ],
    )

    step8_temporal.main()

    assert (output_root / "step8_anomaly_paired_stats.csv").is_file()
    for filename in EXPECTED_FIGURE_FILENAMES:
        assert (figure_root / filename).is_file()


def test_main_figures_only_rejects_no_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["step8_temporal.py", "--figures-only", "--no-resume"])

    with pytest.raises(SystemExit) as exc_info:
        step8_temporal.main()

    assert exc_info.value.code == 2


def test_main_reports_only_rejects_no_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["step8_temporal.py", "--reports-only", "--no-resume"])

    with pytest.raises(SystemExit) as exc_info:
        step8_temporal.main()

    assert exc_info.value.code == 2


def test_run_step8_figures_only_fails_on_missing_required_csv(tmp_path: Path) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)
    (output_root / "step8_anomaly_predictions.csv").unlink()

    with pytest.raises(FileNotFoundError, match="step8_anomaly_predictions.csv"):
        step8_temporal.run_step8_figures_only(
            Step8Config(step7_root=step7_root, output_root=output_root, figure_root=figure_root),
        )


def test_run_step8_reports_only_fails_on_missing_required_csv(tmp_path: Path) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)
    (output_root / "step8_anomaly_metrics.csv").unlink()

    with pytest.raises(FileNotFoundError, match="step8_anomaly_metrics.csv"):
        step8_temporal.run_step8_reports_only(
            Step8Config(step7_root=step7_root, output_root=output_root, figure_root=figure_root),
        )


def test_run_step8_figures_only_fails_on_missing_required_column(tmp_path: Path) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)
    write_csv(
        output_root / "step8_summary.csv",
        columns=("horizon", "target_view", "ablation", "mae_mean", "gaussian_picp_80_mean", "conformal_picp_80_mean"),
        records=[
            {
                "horizon": 5,
                "target_view": "raw_level",
                "ablation": "score_only",
                "mae_mean": 0.31,
                "gaussian_picp_80_mean": 0.81,
                "conformal_picp_80_mean": 0.79,
            },
        ],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        step8_temporal.run_step8_figures_only(
            Step8Config(step7_root=step7_root, output_root=output_root, figure_root=figure_root),
        )


def test_run_step8_figures_only_fails_on_empty_required_csv(tmp_path: Path) -> None:
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_figures_only_fixture_csvs(step7_root=step7_root, output_root=output_root)
    write_csv(
        output_root / "step8_rolling_summary.csv",
        columns=step8_temporal.FIGURES_ONLY_ROLLING_SUMMARY_COLUMNS,
        records=[],
    )

    with pytest.raises(ValueError, match="contains no data rows"):
        step8_temporal.run_step8_figures_only(
            Step8Config(step7_root=step7_root, output_root=output_root, figure_root=figure_root),
        )


@pytest.mark.parametrize("primary_inner_protocol", ["last2_subject_holdout", "group_kfold"])
def test_run_step8_writes_expected_csv_schemas_and_counts_on_synthetic_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_inner_protocol: str,
) -> None:
    subjects = ("SBJ01", "SBJ02", "SBJ03", "SBJ04")
    sessions = ("S01", "S02", "S03")
    table = _synthetic_table(
        horizon=5,
        subjects=subjects,
        sessions=sessions,
        context_length=3,
        tangent_dim=4,
    )
    step7_root = tmp_path / "step7"
    output_root = tmp_path / "step8"
    figure_root = tmp_path / "figures"
    _write_step7_fixture_csvs(root=step7_root, table=table, subjects=subjects)

    monkeypatch.setattr(step8_temporal, "SUBJECT_IDS", subjects)
    monkeypatch.setattr(
        step8_temporal,
        "expected_session_ids",
        lambda: [(subject_id, session_id) for subject_id in subjects for session_id in sessions],
    )
    monkeypatch.setattr(step8_temporal, "load_step8_tables", lambda config: {5: table})
    monkeypatch.setattr(step8_temporal, "build_primary_folds", lambda table_arg: _synthetic_primary_folds(table_arg))
    monkeypatch.setattr(step8_temporal, "build_rolling_folds", lambda table_arg: _synthetic_rolling_folds(table_arg))

    config = Step8Config(
        step7_root=step7_root,
        output_root=output_root,
        figure_root=figure_root,
        horizons=(5,),
        rolling_horizons=(5,),
        online_blocks=8,
        context_length_grid=(3,),
        tangent_dim=4,
        eeg_pca_components_grid=(2,),
        primary_inner_protocol=primary_inner_protocol,
        hidden_dim_grid=(4,),
        learning_rate_grid=(1e-3,),
        batch_size=4,
        max_epochs=2,
        patience=1,
        rolling_small_fold_max_epochs=2,
        rolling_small_fold_row_threshold=99,
        bootstrap_replicates=20,
        bootstrap_seed=123,
        seed_base=321,
        max_workers=1,
    )
    summary = run_step8(config=config)

    assert summary.fold_metric_rows == 24
    assert summary.summary_rows == 6
    assert summary.prediction_rows == 72
    assert summary.model_selection_rows == 24
    assert summary.paired_stats_rows == 40
    assert summary.rolling_fold_rows == 16
    assert summary.rolling_summary_rows == 2
    assert summary.anomaly_metrics_rows == 24
    assert summary.anomaly_summary_rows == 6
    assert summary.anomaly_prediction_rows == 72
    assert summary.anomaly_paired_stats_rows == 24
    assert summary.figure_count == len(EXPECTED_FIGURE_FILENAMES)

    csv_expectations = [
        ("step8_fold_metrics.csv", FOLD_METRICS_COLUMNS, 24),
        ("step8_summary.csv", SUMMARY_COLUMNS, 6),
        ("step8_predictions.csv", PREDICTIONS_COLUMNS, 72),
        ("step8_model_selection.csv", MODEL_SELECTION_COLUMNS, 24),
        ("step8_paired_stats.csv", PAIRED_STATS_COLUMNS, 40),
        ("step8_rolling_fold_metrics.csv", ROLLING_FOLD_COLUMNS, 16),
        ("step8_rolling_summary.csv", ROLLING_SUMMARY_COLUMNS, 2),
        ("step8_anomaly_metrics.csv", ANOMALY_METRICS_COLUMNS, 24),
        ("step8_anomaly_summary.csv", _expected_anomaly_summary_columns(), 6),
        ("step8_anomaly_predictions.csv", ANOMALY_PREDICTIONS_COLUMNS, 72),
        ("step8_anomaly_paired_stats.csv", ANOMALY_PAIRED_STATS_COLUMNS, 24),
    ]
    for filename, expected_columns, expected_rows in csv_expectations:
        header, rows = load_csv_rows(output_root / filename)
        assert tuple(header) == expected_columns
        assert len(rows) == expected_rows
        assert rows
        if "primary_inner_protocol" in header:
            assert {str(row["primary_inner_protocol"]) for row in rows} == {primary_inner_protocol}

    _, model_selection_rows = load_csv_rows(output_root / "step8_model_selection.csv")
    assert {int(row["selected_context_length"]) for row in model_selection_rows} == {3}
    score_only_rows = [row for row in model_selection_rows if str(row["ablation"]) == "score_only"]
    assert {int(row["selected_eeg_pca_components"]) for row in score_only_rows} == {0}

    assert not (output_root / ".step8_checkpoint").exists()
    for filename in EXPECTED_FIGURE_FILENAMES:
        assert (figure_root / filename).is_file()
