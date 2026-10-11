"""Step 8 generic GRU forecasting and anomaly-detection layer."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_f
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.forecasting_loader import build_primary_folds, build_rolling_folds
from src.data.loader import SUBJECT_IDS, expected_session_ids
from src.evaluation.anomaly_detection import (
    binary_auc_metrics,
    prospective_drop20_probability,
    raw_space_prediction,
    retrospective_z_scores,
    standard_normal_ks,
)
from src.evaluation.forecasting_metrics import (
    PairedStatistic,
    compute_paired_statistic,
    compute_regression_metrics_heteroscedastic,
    holm_correction,
)
from src.features.step5_scores import Step5Config, _validate_saved_score_payload, score_split_path
from src.features.step6_forecasting_task import (
    Step6Config,
    compute_drop20_state,
    compute_future_drop20_onset,
    csv_float,
    load_csv_rows,
    load_online_session_inputs,
    manifest_path,
    write_csv,
)

logger = logging.getLogger(__name__)

PRIMARY_ABLATIONS: tuple[str, ...] = ("score_only", "eeg_reduced", "hybrid")
ROLLING_ABLATIONS: tuple[str, ...] = ("score_only",)
TARGET_VIEWS: tuple[str, ...] = ("raw_level", "context_residual")
PAIRED_METRICS: tuple[str, ...] = ("mae", "gaussian_nll")
STEP7_TRAINED_BASELINE_CANDIDATES: tuple[str, ...] = (
    "autoreg_ridge_5",
    "ridge_anchor_ts",
    "elasticnet_anchor_ts",
    "pcr_context_ts",
    "pls_context_ts",
    "ridge_score5_anchor_ts",
    "pcr_score5_context_ts",
    "pls_score5_context_ts",
    "riemannian_knn_anchor",
)
PAIRED_COMPARISON_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("gru_score_only_vs_persistence", "gru_score_only", "persistence"),
    ("gru_score_only_vs_context_score_mean", "gru_score_only", "context_score_mean"),
    ("gru_score_only_vs_step7_best", "gru_score_only", "__step7_best__"),
    ("gru_eeg_reduced_vs_persistence", "gru_eeg_reduced", "persistence"),
    ("gru_eeg_reduced_vs_context_score_mean", "gru_eeg_reduced", "context_score_mean"),
    ("gru_eeg_reduced_vs_step7_best", "gru_eeg_reduced", "__step7_best__"),
    ("gru_hybrid_vs_persistence", "gru_hybrid", "persistence"),
    ("gru_hybrid_vs_context_score_mean", "gru_hybrid", "context_score_mean"),
    ("gru_hybrid_vs_step7_best", "gru_hybrid", "__step7_best__"),
    ("gru_hybrid_vs_gru_score_only", "gru_hybrid", "gru_score_only"),
)

FOLD_METRICS_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "horizon",
    "target_view",
    "ablation",
    "mae",
    "rmse",
    "r2",
    "pearson_r",
    "gaussian_nll",
    "gaussian_picp_80",
    "gaussian_picp_90",
    "conformal_picp_80",
    "conformal_picp_90",
    "mean_learned_sigma",
    "mean_oof_sigma",
    "n_test_rows",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "target_view",
    "ablation",
    "n_folds",
    "mae_mean",
    "mae_std",
    "gaussian_nll_mean",
    "gaussian_nll_std",
    "rmse_mean",
    "rmse_std",
    "r2_mean",
    "r2_std",
    "pearson_r_mean",
    "pearson_r_std",
    "gaussian_picp_80_mean",
    "gaussian_picp_80_std",
    "gaussian_picp_90_mean",
    "gaussian_picp_90_std",
    "conformal_picp_80_mean",
    "conformal_picp_80_std",
    "conformal_picp_90_mean",
    "conformal_picp_90_std",
    "mean_learned_sigma_mean",
    "mean_learned_sigma_std",
)

PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "subject_id",
    "session_id",
    "horizon",
    "target_view",
    "ablation",
    "anchor_idx",
    "target_idx",
    "y_true",
    "y_pred_mu",
    "y_pred_sigma",
    "oof_sigma",
    "gaussian_lower_80",
    "gaussian_upper_80",
    "gaussian_lower_90",
    "gaussian_upper_90",
    "conformal_lower_80",
    "conformal_upper_80",
    "conformal_lower_90",
    "conformal_upper_90",
    "z_score",
)

MODEL_SELECTION_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "horizon",
    "target_view",
    "ablation",
    "selected_hidden_dim",
    "selected_learning_rate",
    "selected_dropout",
    "selected_weight_decay",
    "selected_eeg_pca_components",
    "best_val_nll",
    "n_epochs_trained",
    "oof_residual_sigma",
    "conformal_q80",
    "conformal_q90",
    "n_train_rows",
    "n_test_rows",
)

PAIRED_STATS_COLUMNS: tuple[str, ...] = (
    "comparison_id",
    "horizon",
    "target_view",
    "metric",
    "model_a",
    "model_b",
    "n_folds",
    "n_nonzero_deltas",
    "test_method",
    "zero_method",
    "p_value_method",
    "effect_size_method",
    "mean_metric_a",
    "mean_metric_b",
    "mean_delta_b_minus_a",
    "median_delta_b_minus_a",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "wilcoxon_stat",
    "p_value",
    "p_value_holm",
    "rank_biserial",
    "better_model",
)

ROLLING_FOLD_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "train_through_session",
    "test_session",
    "horizon",
    "target_view",
    "ablation",
    "mae",
    "gaussian_nll",
    "rmse",
    "pearson_r",
    "first_difference_pearson_r",
    "n_train_rows",
    "n_test_rows",
)

ROLLING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "target_view",
    "ablation",
    "n_folds",
    "mae_mean",
    "mae_std",
    "gaussian_nll_mean",
    "gaussian_nll_std",
    "rmse_mean",
    "rmse_std",
    "pearson_r_mean",
    "pearson_r_std",
    "first_difference_pearson_r_mean",
    "first_difference_pearson_r_std",
)

ANOMALY_METRICS_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "horizon",
    "target_view",
    "ablation",
    "n_test_rows",
    "n_positive_drop20_state",
    "n_positive_drop20_onset",
    "prospective_roc_auc_p_drop20_state_for_state",
    "prospective_pr_auc_p_drop20_state_for_state",
    "prospective_roc_auc_p_drop20_state_for_onset",
    "prospective_pr_auc_p_drop20_state_for_onset",
    "prospective_roc_auc_neg_mu_raw_for_state",
    "prospective_pr_auc_neg_mu_raw_for_state",
    "prospective_roc_auc_neg_mu_raw_for_onset",
    "prospective_pr_auc_neg_mu_raw_for_onset",
    "prospective_roc_auc_sigma_raw_for_state",
    "prospective_pr_auc_sigma_raw_for_state",
    "prospective_roc_auc_sigma_raw_for_onset",
    "prospective_pr_auc_sigma_raw_for_onset",
    "baseline_roc_auc_p_drop20_state_for_state",
    "baseline_pr_auc_p_drop20_state_for_state",
    "baseline_roc_auc_p_drop20_state_for_onset",
    "baseline_pr_auc_p_drop20_state_for_onset",
    "baseline_roc_auc_neg_mu_raw_for_state",
    "baseline_pr_auc_neg_mu_raw_for_state",
    "baseline_roc_auc_neg_mu_raw_for_onset",
    "baseline_pr_auc_neg_mu_raw_for_onset",
    "baseline_roc_auc_sigma_raw_for_state",
    "baseline_pr_auc_sigma_raw_for_state",
    "baseline_roc_auc_sigma_raw_for_onset",
    "baseline_pr_auc_sigma_raw_for_onset",
    "z_score_mean",
    "z_score_std",
    "z_score_ks_stat",
    "z_score_ks_pvalue",
)

ANOMALY_PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "subject_id",
    "session_id",
    "horizon",
    "target_view",
    "ablation",
    "anchor_idx",
    "target_idx",
    "y_true",
    "y_pred_mu",
    "y_pred_sigma",
    "context_score_mean_value",
    "session_cal_mean_raw",
    "drop20_threshold_raw",
    "mu_raw",
    "sigma_raw",
    "p_drop20_state",
    "baseline_mu_raw",
    "baseline_sigma_raw",
    "baseline_p_drop20_state",
    "z_score",
    "target_drop20_state",
    "future_drop20_onset",
)

EXPECTED_FIGURE_FILENAMES: tuple[str, ...] = (
    "step8_ablation_mae_by_horizon.png",
    "step8_ablation_nll_by_horizon.png",
    "step8_raw_vs_step7_best.png",
    "step8_residual_vs_step7_best.png",
    "step8_calibration_comparison.png",
    "step8_learned_sigma_distribution.png",
    "step8_zscore_qq_plot.png",
    "step8_zscore_histogram.png",
    "step8_anomaly_roc_curves.png",
    "step8_anomaly_pr_curves.png",
    "step8_anomaly_sigma_vs_decline.png",
    "step8_rolling_first_difference.png",
)

FIGURES_ONLY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "target_view",
    "ablation",
    "mae_mean",
    "gaussian_nll_mean",
    "gaussian_picp_80_mean",
    "conformal_picp_80_mean",
)

FIGURES_ONLY_ROLLING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "target_view",
    "first_difference_pearson_r_mean",
)

FIGURES_ONLY_ANOMALY_PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "y_pred_sigma",
    "z_score",
    "target_drop20_state",
    "future_drop20_onset",
    "sigma_raw",
    "p_drop20_state",
    "baseline_p_drop20_state",
)

FIGURES_ONLY_STEP7_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "target_view",
    "model_name",
    "mae_mean",
    "gaussian_nll_mean",
)


@dataclass(frozen=True)
class Step8Config:
    """Configuration for the Step 8 neural forecasting layer."""

    step4_root: Path = Path("data/features")
    step5_root: Path = Path("data/scores")
    step6_root: Path = Path("data/forecasting")
    step7_root: Path = Path("data/forecasting")
    output_root: Path = Path("data/forecasting")
    figure_root: Path = Path("outputs/forecasting/step8")
    horizons: tuple[int, ...] = (5, 10, 15, 20, 25)
    rolling_horizons: tuple[int, ...] = (5, 25)
    target_views: tuple[str, ...] = TARGET_VIEWS
    ablations: tuple[str, ...] = PRIMARY_ABLATIONS
    rolling_ablations: tuple[str, ...] = ROLLING_ABLATIONS
    online_blocks: int = 50
    calibration_blocks: int = 20
    calibration_social_start: int = 10
    tangent_dim: int = 300
    context_length: int = 10
    online_k_min: int = 3
    online_k_max: int = 10
    drop_threshold: float = -0.20
    sigma_floor: float = 1e-6
    corr_epsilon: float = 1e-12
    hidden_dim_grid: tuple[int, ...] = (32, 64)
    learning_rate_grid: tuple[float, ...] = (1e-3, 5e-4)
    dropout: float = 0.2
    weight_decay: float = 1e-4
    eeg_pca_components: int = 20
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    rolling_small_fold_max_epochs: int = 50
    rolling_small_fold_row_threshold: int = 80
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 20260329
    seed_base: int = 20260330
    max_workers: int = 1
    reserve_cores: int = 2


@dataclass(frozen=True)
class Step8ForecastingTable:
    """One horizon-specific Step 8 row table with ordered sequences and labels."""

    horizon: int
    subject_id: np.ndarray
    session_id: np.ndarray
    session_ord: np.ndarray
    session_k: np.ndarray
    target_block_id: np.ndarray
    elapsed_runs: np.ndarray
    anchor_idx: np.ndarray
    target_idx: np.ndarray
    score_sequence: np.ndarray
    ts_sequence: np.ndarray
    target_score_raw: np.ndarray
    context_score_mean_value: np.ndarray
    session_cal_mean_raw: np.ndarray
    drop20_threshold_raw: np.ndarray
    target_score_context_residual: np.ndarray
    target_drop20_state: np.ndarray
    future_drop20_onset: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.target_score_raw.shape[0])


@dataclass(frozen=True)
class Step8Summary:
    """High-level summary returned after a successful Step 8 run."""

    output_root: Path
    figure_root: Path
    fold_metric_rows: int
    summary_rows: int
    prediction_rows: int
    model_selection_rows: int
    paired_stats_rows: int
    rolling_fold_rows: int
    rolling_summary_rows: int
    anomaly_metrics_rows: int
    anomaly_summary_rows: int
    anomaly_prediction_rows: int
    figure_count: int


@dataclass(frozen=True)
class ArrayStandardizer:
    """Minimal train-only feature standardizer."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_train(cls, x: np.ndarray) -> "ArrayStandardizer":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        mean = np.mean(x, axis=0, dtype=np.float64)
        scale = np.std(x, axis=0, ddof=0, dtype=np.float64)
        scale[scale <= 1e-12] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / self.scale


@dataclass(frozen=True)
class SequencePreprocessor:
    """Train-fold-only preprocessing for one Step 8 ablation."""

    ablation: str
    score_standardizer: ArrayStandardizer
    eeg_pca: PCA | None
    eeg_standardizer: ArrayStandardizer | None

    def transform(self, *, score_sequence: np.ndarray, ts_sequence: np.ndarray) -> np.ndarray:
        score_sequence = np.asarray(score_sequence, dtype=np.float64)
        ts_sequence = np.asarray(ts_sequence, dtype=np.float64)
        score_feature = self.score_standardizer.transform(score_sequence.reshape(-1, 1)).reshape(
            score_sequence.shape[0],
            score_sequence.shape[1],
            1,
        )
        if self.ablation == "score_only":
            return np.asarray(score_feature, dtype=np.float32)
        if self.eeg_pca is None or self.eeg_standardizer is None:
            raise ValueError(f"Ablation {self.ablation} requires PCA and EEG standardization")
        flat_ts = ts_sequence.reshape(-1, ts_sequence.shape[-1])
        pca_scores = self.eeg_pca.transform(flat_ts)
        pca_scores = self.eeg_standardizer.transform(pca_scores).reshape(
            ts_sequence.shape[0],
            ts_sequence.shape[1],
            -1,
        )
        if self.ablation == "eeg_reduced":
            return np.asarray(pca_scores, dtype=np.float32)
        if self.ablation == "hybrid":
            return np.asarray(np.concatenate([score_feature, pca_scores], axis=2), dtype=np.float32)
        raise KeyError(f"Unknown Step 8 ablation: {self.ablation}")

    @property
    def selected_eeg_pca_components(self) -> int | str:
        if self.eeg_pca is None:
            return "NA"
        return int(self.eeg_pca.n_components_)


@dataclass(frozen=True)
class HyperparameterSelection:
    """Selected hyperparameters and calibration stats for one primary task."""

    hidden_dim: int
    learning_rate: float
    best_epoch: int
    best_val_nll: float
    dropout: float
    weight_decay: float
    calibration_residuals: np.ndarray
    preprocessor: SequencePreprocessor


@dataclass(frozen=True)
class _PrimaryTaskSpec:
    task_index: int
    horizon: int
    target_view: str
    ablation: str
    fold_subject_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class _PrimaryTaskResult:
    task_index: int
    fold_metric_rows: list[dict[str, Any]]
    model_selection_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class _RollingTaskSpec:
    task_index: int
    horizon: int
    target_view: str
    ablation: str
    subject_id: str
    train_through_session: str
    test_session: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class _RollingTaskResult:
    task_index: int
    rolling_fold_rows: list[dict[str, Any]]


def validate_step8_config(config: Step8Config) -> None:
    """Validate the Step 8 configuration."""

    if not config.horizons:
        raise ValueError("At least one primary horizon is required")
    if len(set(config.horizons)) != len(config.horizons):
        raise ValueError(f"Primary horizons must be unique, got {config.horizons}")
    if len(set(config.rolling_horizons)) != len(config.rolling_horizons):
        raise ValueError(f"Rolling horizons must be unique, got {config.rolling_horizons}")
    if not set(config.rolling_horizons).issubset(set(config.horizons)):
        raise ValueError("rolling_horizons must be a subset of horizons")
    if tuple(config.target_views) != TARGET_VIEWS:
        raise ValueError(f"target_views must be exactly {TARGET_VIEWS}, got {config.target_views}")
    if not set(config.rolling_ablations).issubset(set(config.ablations)):
        raise ValueError("rolling_ablations must be a subset of ablations")
    if config.sigma_floor <= 0.0:
        raise ValueError("sigma_floor must be positive")
    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.tangent_dim <= 0:
        raise ValueError("tangent_dim must be positive")
    if config.eeg_pca_components <= 0:
        raise ValueError("eeg_pca_components must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_epochs <= 0 or config.rolling_small_fold_max_epochs <= 0:
        raise ValueError("epoch counts must be positive")
    if config.patience <= 0:
        raise ValueError("patience must be positive")
    if config.max_workers < 0:
        raise ValueError("max_workers must be >= 0")
    if config.reserve_cores < 0:
        raise ValueError("reserve_cores must be >= 0")
    for horizon in (*config.horizons, *config.rolling_horizons):
        if horizon <= 0 or horizon >= config.online_blocks:
            raise ValueError(f"Invalid horizon {horizon} for online_blocks={config.online_blocks}")


def _step6_loader_config(config: Step8Config) -> Step6Config:
    return Step6Config(
        step4_root=config.step4_root,
        step5_root=config.step5_root,
        output_root=config.step6_root,
        horizons=config.horizons,
        online_blocks=config.online_blocks,
        calibration_blocks=config.calibration_blocks,
        calibration_social_start=config.calibration_social_start,
        tangent_dim=config.tangent_dim,
        default_context_length=config.context_length,
        drop_threshold=config.drop_threshold,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
        corr_epsilon=config.corr_epsilon,
    )


def _step5_validation_config(config: Step8Config) -> Step5Config:
    return Step5Config(
        output_root=config.step5_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
    )


def _expected_usable_rows_per_session(*, horizon: int, config: Step8Config) -> int:
    usable = config.online_blocks - int(horizon) - config.context_length + 1
    return max(0, usable)


def target_view_values(table: Step8ForecastingTable, *, target_view: str) -> np.ndarray:
    """Return the requested target-view vector from a Step 8 table."""

    if target_view == "raw_level":
        return np.asarray(table.target_score_raw, dtype=np.float64)
    if target_view == "context_residual":
        return np.asarray(table.target_score_context_residual, dtype=np.float64)
    raise KeyError(f"Unknown Step 8 target view: {target_view}")


def _parse_manifest_rows_for_step8(*, horizon: int, config: Step8Config) -> list[dict[str, str]]:
    manifest_file = manifest_path(output_root=config.step6_root, horizon=horizon)
    fieldnames, rows = load_csv_rows(manifest_file)
    required = {
        "subject_id",
        "session_id",
        "session_k",
        "horizon",
        "anchor_idx",
        "target_idx",
        "target_block_id",
        "elapsed_runs",
        "anchor_score_raw",
        "target_score_raw",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"{manifest_file.name} is missing required Step 8 columns: {missing}")
    filtered = [row for row in rows if int(row["anchor_idx"]) >= config.context_length - 1]
    filtered.sort(key=lambda row: (row["subject_id"], row["session_id"], int(row["target_idx"])))
    return filtered


def _session_ord_from_session_id(session_id: str) -> int:
    if not session_id.startswith("S") or len(session_id) != 3:
        raise ValueError(f"Unexpected session_id format: {session_id}")
    return int(session_id[1:])


def load_step8_forecasting_table(*, horizon: int, config: Step8Config) -> Step8ForecastingTable:
    """Load one horizon-specific Step 8 forecasting table."""

    if horizon not in config.horizons:
        raise ValueError(f"Horizon {horizon} is not part of the Step 8 horizon set {config.horizons}")
    step6_config = _step6_loader_config(config)
    step5_config = _step5_validation_config(config)
    rows = _parse_manifest_rows_for_step8(horizon=horizon, config=config)
    session_cache: dict[tuple[str, str], tuple[Any, float]] = {}

    subject_id: list[str] = []
    session_id: list[str] = []
    session_ord: list[int] = []
    session_k: list[int] = []
    target_block_id: list[int] = []
    elapsed_runs: list[int] = []
    anchor_idx: list[int] = []
    target_idx: list[int] = []
    score_sequence: list[np.ndarray] = []
    ts_sequence: list[np.ndarray] = []
    target_score_raw: list[float] = []
    context_score_mean_value: list[float] = []
    session_cal_mean_raw: list[float] = []
    drop20_threshold_raw: list[float] = []
    target_score_context_residual: list[float] = []
    target_drop20_state: list[int] = []
    future_drop20_onset: list[int] = []

    for row in rows:
        row_subject_id = row["subject_id"]
        row_session_id = row["session_id"]
        cache_key = (row_subject_id, row_session_id)
        cached = session_cache.get(cache_key)
        if cached is None:
            session_inputs = load_online_session_inputs(
                subject_id=row_subject_id,
                session_id=row_session_id,
                config=step6_config,
            )
            score_payload = _validate_saved_score_payload(
                path=score_split_path(
                    output_root=config.step5_root,
                    subject_id=row_subject_id,
                    session_id=row_session_id,
                    split="online",
                ),
                split="online",
                config=step5_config,
            )
            cached = (session_inputs, float(np.asarray(score_payload["session_cal_mean_raw"], dtype=np.float64)))
            session_cache[cache_key] = cached
        session_inputs, session_cal_mean = cached

        anchor = int(row["anchor_idx"])
        target = int(row["target_idx"])
        if int(row["horizon"]) != horizon or target - anchor != horizon:
            raise ValueError(f"Manifest target/anchor mismatch for horizon {horizon}: {row}")

        sequence = np.asarray(
            session_inputs.score_raw[anchor - config.context_length + 1 : anchor + 1],
            dtype=np.float32,
        )
        ts_context = np.asarray(
            session_inputs.features[anchor - config.context_length + 1 : anchor + 1],
            dtype=np.float32,
        )
        target_score = float(np.asarray(session_inputs.score_raw[target], dtype=np.float32))
        context_mean = float(np.mean(sequence, dtype=np.float64))
        score_drop = np.asarray(session_inputs.score_drop_rel_session, dtype=np.float64)
        target_drop_label = (
            int(row["target_drop20_state"])
            if "target_drop20_state" in row
            else compute_drop20_state(float(score_drop[target]), threshold=config.drop_threshold)
        )
        future_onset_label = (
            int(row["future_drop20_onset"])
            if "future_drop20_onset" in row
            else compute_future_drop20_onset(
                score_drop,
                anchor_idx=anchor,
                target_idx=target,
                threshold=config.drop_threshold,
            )
        )

        subject_id.append(row_subject_id)
        session_id.append(row_session_id)
        session_ord.append(_session_ord_from_session_id(row_session_id))
        session_k.append(int(row["session_k"]))
        target_block_id.append(int(row["target_block_id"]))
        elapsed_runs.append(int(row["elapsed_runs"]))
        anchor_idx.append(anchor)
        target_idx.append(target)
        score_sequence.append(sequence)
        ts_sequence.append(ts_context)
        target_score_raw.append(target_score)
        context_score_mean_value.append(context_mean)
        session_cal_mean_raw.append(session_cal_mean)
        drop20_threshold_raw.append(0.8 * session_cal_mean)
        target_score_context_residual.append(target_score - context_mean)
        target_drop20_state.append(target_drop_label)
        future_drop20_onset.append(future_onset_label)

    expected_rows = len(expected_session_ids()) * _expected_usable_rows_per_session(horizon=horizon, config=config)
    if len(subject_id) != expected_rows:
        raise ValueError(f"Horizon {horizon} loaded {len(subject_id)} rows, expected {expected_rows}")

    return Step8ForecastingTable(
        horizon=horizon,
        subject_id=np.asarray(subject_id, dtype=object),
        session_id=np.asarray(session_id, dtype=object),
        session_ord=np.asarray(session_ord, dtype=np.int16),
        session_k=np.asarray(session_k, dtype=np.int16),
        target_block_id=np.asarray(target_block_id, dtype=np.int16),
        elapsed_runs=np.asarray(elapsed_runs, dtype=np.int16),
        anchor_idx=np.asarray(anchor_idx, dtype=np.int16),
        target_idx=np.asarray(target_idx, dtype=np.int16),
        score_sequence=np.stack(score_sequence, axis=0).astype(np.float32),
        ts_sequence=np.stack(ts_sequence, axis=0).astype(np.float32),
        target_score_raw=np.asarray(target_score_raw, dtype=np.float32),
        context_score_mean_value=np.asarray(context_score_mean_value, dtype=np.float32),
        session_cal_mean_raw=np.asarray(session_cal_mean_raw, dtype=np.float64),
        drop20_threshold_raw=np.asarray(drop20_threshold_raw, dtype=np.float64),
        target_score_context_residual=np.asarray(target_score_context_residual, dtype=np.float32),
        target_drop20_state=np.asarray(target_drop20_state, dtype=np.int8),
        future_drop20_onset=np.asarray(future_drop20_onset, dtype=np.int8),
    )


def load_step8_tables(config: Step8Config) -> dict[int, Step8ForecastingTable]:
    """Load all Step 8 forecasting tables."""

    validate_step8_config(config)
    return {horizon: load_step8_forecasting_table(horizon=horizon, config=config) for horizon in config.horizons}


def _fit_preprocessor(
    *,
    table: Step8ForecastingTable,
    fit_idx: np.ndarray,
    ablation: str,
    config: Step8Config,
) -> SequencePreprocessor:
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    if fit_idx.size == 0:
        raise ValueError("Cannot fit a Step 8 preprocessor with zero rows")
    score_flat = np.asarray(table.score_sequence[fit_idx], dtype=np.float64).reshape(-1, 1)
    score_standardizer = ArrayStandardizer.from_train(score_flat)
    if ablation == "score_only":
        return SequencePreprocessor(ablation=ablation, score_standardizer=score_standardizer, eeg_pca=None, eeg_standardizer=None)
    ts_flat = np.asarray(table.ts_sequence[fit_idx], dtype=np.float64).reshape(-1, table.ts_sequence.shape[-1])
    pca = PCA(n_components=int(config.eeg_pca_components), svd_solver="full")
    pca.fit(ts_flat)
    reduced = pca.transform(ts_flat)
    eeg_standardizer = ArrayStandardizer.from_train(reduced)
    return SequencePreprocessor(
        ablation=ablation,
        score_standardizer=score_standardizer,
        eeg_pca=pca,
        eeg_standardizer=eeg_standardizer,
    )


def _assemble_inputs(
    *,
    table: Step8ForecastingTable,
    indices: np.ndarray,
    preprocessor: SequencePreprocessor,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    return preprocessor.transform(
        score_sequence=np.asarray(table.score_sequence[indices], dtype=np.float64),
        ts_sequence=np.asarray(table.ts_sequence[indices], dtype=np.float64),
    )


def _torch_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32),
    )


def _set_torch_determinism(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class ProbabilisticGRU(nn.Module):
    """Small GRU with a Gaussian output head."""

    def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float, sigma_floor: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden_dim), 2)
        self.sigma_floor = float(sigma_floor)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(x)
        final_hidden = self.dropout(hidden[-1])
        output = self.head(final_hidden)
        mu = output[:, 0]
        log_sigma_sq = output[:, 1]
        sigma = torch.sqrt(torch_f.softplus(log_sigma_sq) + self.sigma_floor)
        return mu, sigma


def gaussian_nll_loss(y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return the mean Gaussian NLL for one batch."""

    variance = sigma * sigma
    return torch.mean(0.5 * torch.log(2.0 * math.pi * variance) + 0.5 * ((y_true - mu) ** 2) / variance)


def _predict_with_model(model: ProbabilisticGRU, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        mu, sigma = model(x_tensor)
    return np.asarray(mu.cpu().numpy(), dtype=np.float64), np.asarray(sigma.cpu().numpy(), dtype=np.float64)


def _train_with_validation(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    sigma_floor: float,
    batch_size: int,
    seed: int,
) -> tuple[int, float, np.ndarray]:
    _set_torch_determinism(seed)
    model = ProbabilisticGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        sigma_floor=sigma_floor,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_loader = DataLoader(
        _torch_dataset(x_train, y_train),
        batch_size=min(int(batch_size), int(x_train.shape[0])),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    x_val_tensor = torch.tensor(np.asarray(x_val, dtype=np.float32), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.asarray(y_val, dtype=np.float32), dtype=torch.float32)

    best_val_nll = float("inf")
    best_epoch = 1
    best_val_pred = np.zeros_like(np.asarray(y_val, dtype=np.float64))
    stale_epochs = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            mu, sigma = model(x_batch)
            loss = gaussian_nll_loss(y_batch, mu, sigma)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mu, val_sigma = model(x_val_tensor)
            val_loss = gaussian_nll_loss(y_val_tensor, val_mu, val_sigma)
        val_nll = float(val_loss.item())
        if not math.isfinite(val_nll):
            raise ValueError(f"Validation loss became non-finite at epoch {epoch}")
        if val_nll < best_val_nll - 1e-12:
            best_val_nll = val_nll
            best_epoch = epoch
            best_val_pred = np.asarray(val_mu.cpu().numpy(), dtype=np.float64)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    return best_epoch, best_val_nll, best_val_pred


def _refit_and_predict(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    epochs: int,
    sigma_floor: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    _set_torch_determinism(seed)
    model = ProbabilisticGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        sigma_floor=sigma_floor,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_loader = DataLoader(
        _torch_dataset(x_train, y_train),
        batch_size=min(int(batch_size), int(x_train.shape[0])),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            mu, sigma = model(x_batch)
            loss = gaussian_nll_loss(y_batch, mu, sigma)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite refit loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
    return _predict_with_model(model, x_test)


def _task_seed(*, config: Step8Config, offset: int) -> int:
    return int(config.seed_base + offset)


def _inner_validation_split_loso(table: Step8ForecastingTable, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_subjects = sorted({str(value) for value in table.subject_id[train_idx]})
    validation_subjects = set(train_subjects[-2:])
    val_mask = np.asarray([str(subject) in validation_subjects for subject in table.subject_id[train_idx]], dtype=bool)
    val_idx = train_idx[val_mask]
    inner_train_idx = train_idx[~val_mask]
    if inner_train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("LOSO inner split produced an empty train or validation split")
    return inner_train_idx, val_idx


def _inner_validation_split_rolling(table: Step8ForecastingTable, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    unique_sessions = sorted({int(value) for value in table.session_ord[train_idx]})
    if len(unique_sessions) < 2:
        return train_idx, None
    validation_session_ord = unique_sessions[-1]
    val_mask = np.asarray(table.session_ord[train_idx] == validation_session_ord, dtype=bool)
    val_idx = train_idx[val_mask]
    inner_train_idx = train_idx[~val_mask]
    if inner_train_idx.size == 0 or val_idx.size == 0:
        return train_idx, None
    return inner_train_idx, val_idx


def _tune_primary_hyperparameters(
    *,
    table: Step8ForecastingTable,
    train_idx: np.ndarray,
    target_view: str,
    ablation: str,
    task_seed: int,
    config: Step8Config,
) -> HyperparameterSelection:
    inner_train_idx, val_idx = _inner_validation_split_loso(table, train_idx)
    y = target_view_values(table, target_view=target_view)
    best_selection: HyperparameterSelection | None = None
    grid_index = 0
    for hidden_dim in config.hidden_dim_grid:
        for learning_rate in config.learning_rate_grid:
            preprocessor = _fit_preprocessor(table=table, fit_idx=inner_train_idx, ablation=ablation, config=config)
            x_inner_train = _assemble_inputs(table=table, indices=inner_train_idx, preprocessor=preprocessor)
            x_val = _assemble_inputs(table=table, indices=val_idx, preprocessor=preprocessor)
            best_epoch, best_val_nll, val_pred = _train_with_validation(
                x_train=x_inner_train,
                y_train=y[inner_train_idx],
                x_val=x_val,
                y_val=y[val_idx],
                input_dim=int(x_inner_train.shape[2]),
                hidden_dim=int(hidden_dim),
                learning_rate=float(learning_rate),
                dropout=float(config.dropout),
                weight_decay=float(config.weight_decay),
                max_epochs=int(config.max_epochs),
                patience=int(config.patience),
                sigma_floor=float(config.sigma_floor),
                batch_size=int(config.batch_size),
                seed=_task_seed(config=config, offset=task_seed + grid_index),
            )
            calibration_residuals = np.asarray(y[val_idx], dtype=np.float64) - np.asarray(val_pred, dtype=np.float64)
            candidate = HyperparameterSelection(
                hidden_dim=int(hidden_dim),
                learning_rate=float(learning_rate),
                best_epoch=int(best_epoch),
                best_val_nll=float(best_val_nll),
                dropout=float(config.dropout),
                weight_decay=float(config.weight_decay),
                calibration_residuals=calibration_residuals,
                preprocessor=preprocessor,
            )
            if best_selection is None or candidate.best_val_nll < best_selection.best_val_nll - 1e-12:
                best_selection = candidate
            grid_index += 1
    if best_selection is None:
        raise RuntimeError("Primary Step 8 hyperparameter search produced no candidate")
    return best_selection


def _default_rolling_hyperparameters(
    *,
    table: Step8ForecastingTable,
    train_idx: np.ndarray,
    config: Step8Config,
) -> tuple[int, float, int]:
    unique_sessions = sorted({int(value) for value in table.session_ord[np.asarray(train_idx, dtype=np.int64)]})
    max_epochs = int(config.max_epochs)
    if len(unique_sessions) < 2 or int(train_idx.size) < int(config.rolling_small_fold_row_threshold):
        max_epochs = int(config.rolling_small_fold_max_epochs)
    return 32, 1e-3, max_epochs


def _primary_checkpoint_key(task: _PrimaryTaskSpec) -> str:
    return f"primary/{task.ablation}/{task.horizon}/{task.target_view}/{task.fold_subject_id}.json"


def _rolling_checkpoint_key(task: _RollingTaskSpec) -> str:
    return f"rolling/{task.ablation}/{task.horizon}/{task.target_view}/{task.subject_id}/{task.train_through_session}.json"


def _checkpoint_dir(output_root: Path) -> Path:
    return output_root / ".step8_checkpoint"


def _save_checkpoint(checkpoint_dir: Path, key: str, payload: dict[str, Any]) -> None:
    path = checkpoint_dir / key
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".ckpt_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        os.replace(tmp_path_str, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _load_checkpoint(checkpoint_dir: Path, key: str) -> dict[str, Any] | None:
    path = checkpoint_dir / key
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload is not a dict")
        return payload
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("Corrupted Step 8 checkpoint %s, will recompute: %s", key, exc)
        return None


def _primary_result_to_checkpoint(result: _PrimaryTaskResult) -> dict[str, Any]:
    return {
        "fold_metric_rows": result.fold_metric_rows,
        "model_selection_rows": result.model_selection_rows,
        "prediction_rows": result.prediction_rows,
    }


def _primary_result_from_checkpoint(task_index: int, data: dict[str, Any]) -> _PrimaryTaskResult:
    return _PrimaryTaskResult(
        task_index=task_index,
        fold_metric_rows=data["fold_metric_rows"],
        model_selection_rows=data["model_selection_rows"],
        prediction_rows=data["prediction_rows"],
    )


def _rolling_result_to_checkpoint(result: _RollingTaskResult) -> dict[str, Any]:
    return {"rolling_fold_rows": result.rolling_fold_rows}


def _rolling_result_from_checkpoint(task_index: int, data: dict[str, Any]) -> _RollingTaskResult:
    return _RollingTaskResult(task_index=task_index, rolling_fold_rows=data["rolling_fold_rows"])


def fold_metrics_path(*, output_root: Path) -> Path:
    return output_root / "step8_fold_metrics.csv"


def summary_path(*, output_root: Path) -> Path:
    return output_root / "step8_summary.csv"


def predictions_path(*, output_root: Path) -> Path:
    return output_root / "step8_predictions.csv"


def model_selection_path(*, output_root: Path) -> Path:
    return output_root / "step8_model_selection.csv"


def paired_stats_path(*, output_root: Path) -> Path:
    return output_root / "step8_paired_stats.csv"


def rolling_fold_metrics_path(*, output_root: Path) -> Path:
    return output_root / "step8_rolling_fold_metrics.csv"


def rolling_summary_path(*, output_root: Path) -> Path:
    return output_root / "step8_rolling_summary.csv"


def anomaly_metrics_path(*, output_root: Path) -> Path:
    return output_root / "step8_anomaly_metrics.csv"


def anomaly_summary_path(*, output_root: Path) -> Path:
    return output_root / "step8_anomaly_summary.csv"


def anomaly_predictions_path(*, output_root: Path) -> Path:
    return output_root / "step8_anomaly_predictions.csv"


def figure_path(*, figure_root: Path, filename: str) -> Path:
    return figure_root / filename


def _records_to_csv(path: Path, *, columns: tuple[str, ...], records: list[dict[str, Any]]) -> None:
    write_csv(path, columns=columns, records=records)


def _load_required_csv_rows(
    *,
    path: Path,
    required_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    header, rows = load_csv_rows(path)
    missing_columns = [column for column in required_columns if column not in header]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def _resolved_max_workers(config: Step8Config) -> int:
    if config.max_workers == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - config.reserve_cores)
    return int(config.max_workers)


def _execute_ordered_tasks(task_specs: list[Any], *, max_workers: int, run_task: Callable[[Any], Any]) -> list[Any]:
    if not task_specs:
        return []
    if max_workers <= 1:
        return [run_task(task) for task in task_specs]
    ordered_results: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="step8") as executor:
        futures = [executor.submit(run_task, task) for task in task_specs]
        for future in as_completed(futures):
            result = future.result()
            ordered_results[int(result.task_index)] = result
    return [ordered_results[int(task.task_index)] for task in task_specs]


def _gru_model_name(ablation: str) -> str:
    return f"gru_{ablation}"


def _describe_primary_task(task: _PrimaryTaskSpec) -> str:
    return (
        f"h={task.horizon} "
        f"view={task.target_view} "
        f"ablation={task.ablation} "
        f"fold={task.fold_subject_id}"
    )


def _describe_rolling_task(task: _RollingTaskSpec) -> str:
    return (
        f"h={task.horizon} "
        f"view={task.target_view} "
        f"ablation={task.ablation} "
        f"subject={task.subject_id} "
        f"{task.train_through_session}->{task.test_session}"
    )


def _build_primary_task_specs(
    *,
    primary_tables: dict[int, Step8ForecastingTable],
    config: Step8Config,
) -> list[_PrimaryTaskSpec]:
    task_specs: list[_PrimaryTaskSpec] = []
    task_index = 0
    for horizon in config.horizons:
        table = primary_tables[horizon]
        for target_view in config.target_views:
            for ablation in config.ablations:
                for fold in build_primary_folds(table):
                    task_specs.append(
                        _PrimaryTaskSpec(
                            task_index=task_index,
                            horizon=horizon,
                            target_view=target_view,
                            ablation=ablation,
                            fold_subject_id=str(fold.fold_subject_id),
                            train_idx=np.asarray(fold.train_idx, dtype=np.int64),
                            test_idx=np.asarray(fold.test_idx, dtype=np.int64),
                        ),
                    )
                    task_index += 1
    return task_specs


def _build_rolling_task_specs(
    *,
    primary_tables: dict[int, Step8ForecastingTable],
    config: Step8Config,
) -> list[_RollingTaskSpec]:
    task_specs: list[_RollingTaskSpec] = []
    task_index = 0
    for horizon in config.rolling_horizons:
        table = primary_tables[horizon]
        for target_view in config.target_views:
            for ablation in config.rolling_ablations:
                for fold in build_rolling_folds(table):
                    task_specs.append(
                        _RollingTaskSpec(
                            task_index=task_index,
                            horizon=horizon,
                            target_view=target_view,
                            ablation=ablation,
                            subject_id=str(fold.subject_id),
                            train_through_session=str(fold.train_through_session),
                            test_session=str(fold.test_session),
                            train_idx=np.asarray(fold.train_idx, dtype=np.int64),
                            test_idx=np.asarray(fold.test_idx, dtype=np.int64),
                        ),
                    )
                    task_index += 1
    return task_specs


def _run_primary_task(task: _PrimaryTaskSpec, *, primary_tables: dict[int, Step8ForecastingTable], config: Step8Config) -> _PrimaryTaskResult:
    table = primary_tables[task.horizon]
    y = target_view_values(table, target_view=task.target_view)
    selection = _tune_primary_hyperparameters(
        table=table,
        train_idx=task.train_idx,
        target_view=task.target_view,
        ablation=task.ablation,
        task_seed=task.task_index * 101,
        config=config,
    )
    oof_sigma = max(float(np.std(selection.calibration_residuals, ddof=0)), float(config.sigma_floor))
    abs_residuals = np.abs(np.asarray(selection.calibration_residuals, dtype=np.float64))
    conformal_q80 = float(np.quantile(abs_residuals, 0.80, method="higher"))
    conformal_q90 = float(np.quantile(abs_residuals, 0.90, method="higher"))

    final_preprocessor = _fit_preprocessor(table=table, fit_idx=task.train_idx, ablation=task.ablation, config=config)
    x_train = _assemble_inputs(table=table, indices=task.train_idx, preprocessor=final_preprocessor)
    x_test = _assemble_inputs(table=table, indices=task.test_idx, preprocessor=final_preprocessor)
    test_mu, test_sigma = _refit_and_predict(
        x_train=x_train,
        y_train=y[task.train_idx],
        x_test=x_test,
        input_dim=int(x_train.shape[2]),
        hidden_dim=int(selection.hidden_dim),
        learning_rate=float(selection.learning_rate),
        dropout=float(selection.dropout),
        weight_decay=float(selection.weight_decay),
        epochs=int(selection.best_epoch),
        sigma_floor=float(config.sigma_floor),
        batch_size=int(config.batch_size),
        seed=_task_seed(config=config, offset=task.task_index * 101 + 97),
    )

    gaussian_lower_80 = test_mu - 1.2815515655446004 * test_sigma
    gaussian_upper_80 = test_mu + 1.2815515655446004 * test_sigma
    gaussian_lower_90 = test_mu - 1.6448536269514722 * test_sigma
    gaussian_upper_90 = test_mu + 1.6448536269514722 * test_sigma
    conformal_lower_80 = test_mu - conformal_q80
    conformal_upper_80 = test_mu + conformal_q80
    conformal_lower_90 = test_mu - conformal_q90
    conformal_upper_90 = test_mu + conformal_q90
    z_score = retrospective_z_scores(y_true=y[task.test_idx], y_pred_mu=test_mu, y_pred_sigma=test_sigma)

    metrics = compute_regression_metrics_heteroscedastic(
        y_true=y[task.test_idx],
        y_pred=test_mu,
        gaussian_sigma=test_sigma,
        conformal_q80=conformal_q80,
        conformal_q90=conformal_q90,
        corr_epsilon=config.corr_epsilon,
        mean_oof_sigma=oof_sigma,
    )
    fold_metric_row = {
        "fold_subject_id": task.fold_subject_id,
        "horizon": task.horizon,
        "target_view": task.target_view,
        "ablation": task.ablation,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "pearson_r": float(metrics["pearson_r"]),
        "gaussian_nll": float(metrics["gaussian_nll"]),
        "gaussian_picp_80": float(metrics["gaussian_picp_80"]),
        "gaussian_picp_90": float(metrics["gaussian_picp_90"]),
        "conformal_picp_80": float(metrics["conformal_picp_80"]),
        "conformal_picp_90": float(metrics["conformal_picp_90"]),
        "mean_learned_sigma": float(metrics["mean_learned_sigma"]),
        "mean_oof_sigma": float(metrics["mean_oof_sigma"]),
        "n_test_rows": int(task.test_idx.size),
    }
    model_selection_row = {
        "fold_subject_id": task.fold_subject_id,
        "horizon": task.horizon,
        "target_view": task.target_view,
        "ablation": task.ablation,
        "selected_hidden_dim": int(selection.hidden_dim),
        "selected_learning_rate": float(selection.learning_rate),
        "selected_dropout": float(selection.dropout),
        "selected_weight_decay": float(selection.weight_decay),
        "selected_eeg_pca_components": final_preprocessor.selected_eeg_pca_components,
        "best_val_nll": float(selection.best_val_nll),
        "n_epochs_trained": int(selection.best_epoch),
        "oof_residual_sigma": float(oof_sigma),
        "conformal_q80": float(conformal_q80),
        "conformal_q90": float(conformal_q90),
        "n_train_rows": int(task.train_idx.size),
        "n_test_rows": int(task.test_idx.size),
    }

    prediction_rows: list[dict[str, Any]] = []
    for offset, row_index in enumerate(np.asarray(task.test_idx, dtype=np.int64)):
        prediction_rows.append(
            {
                "fold_subject_id": task.fold_subject_id,
                "subject_id": str(table.subject_id[row_index]),
                "session_id": str(table.session_id[row_index]),
                "horizon": task.horizon,
                "target_view": task.target_view,
                "ablation": task.ablation,
                "anchor_idx": int(table.anchor_idx[row_index]),
                "target_idx": int(table.target_idx[row_index]),
                "y_true": float(y[row_index]),
                "y_pred_mu": float(test_mu[offset]),
                "y_pred_sigma": float(test_sigma[offset]),
                "oof_sigma": float(oof_sigma),
                "gaussian_lower_80": float(gaussian_lower_80[offset]),
                "gaussian_upper_80": float(gaussian_upper_80[offset]),
                "gaussian_lower_90": float(gaussian_lower_90[offset]),
                "gaussian_upper_90": float(gaussian_upper_90[offset]),
                "conformal_lower_80": float(conformal_lower_80[offset]),
                "conformal_upper_80": float(conformal_upper_80[offset]),
                "conformal_lower_90": float(conformal_lower_90[offset]),
                "conformal_upper_90": float(conformal_upper_90[offset]),
                "z_score": float(z_score[offset]),
                "context_score_mean_value": float(table.context_score_mean_value[row_index]),
                "session_cal_mean_raw": float(table.session_cal_mean_raw[row_index]),
                "drop20_threshold_raw": float(table.drop20_threshold_raw[row_index]),
                "target_drop20_state": int(table.target_drop20_state[row_index]),
                "future_drop20_onset": int(table.future_drop20_onset[row_index]),
            },
        )
    return _PrimaryTaskResult(
        task_index=task.task_index,
        fold_metric_rows=[fold_metric_row],
        model_selection_rows=[model_selection_row],
        prediction_rows=prediction_rows,
    )


def _run_rolling_task(task: _RollingTaskSpec, *, primary_tables: dict[int, Step8ForecastingTable], config: Step8Config) -> _RollingTaskResult:
    table = primary_tables[task.horizon]
    y = target_view_values(table, target_view=task.target_view)
    inner_train_idx, val_idx = _inner_validation_split_rolling(table, task.train_idx)
    hidden_dim, learning_rate, max_epochs = _default_rolling_hyperparameters(table=table, train_idx=task.train_idx, config=config)
    best_epoch = max_epochs
    if val_idx is not None and int(task.train_idx.size) >= int(config.rolling_small_fold_row_threshold):
        preprocessor = _fit_preprocessor(table=table, fit_idx=inner_train_idx, ablation=task.ablation, config=config)
        x_inner_train = _assemble_inputs(table=table, indices=inner_train_idx, preprocessor=preprocessor)
        x_val = _assemble_inputs(table=table, indices=val_idx, preprocessor=preprocessor)
        best_epoch, _, _ = _train_with_validation(
            x_train=x_inner_train,
            y_train=y[inner_train_idx],
            x_val=x_val,
            y_val=y[val_idx],
            input_dim=int(x_inner_train.shape[2]),
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout=float(config.dropout),
            weight_decay=float(config.weight_decay),
            max_epochs=int(config.max_epochs),
            patience=int(config.patience),
            sigma_floor=float(config.sigma_floor),
            batch_size=int(config.batch_size),
            seed=_task_seed(config=config, offset=task.task_index * 73),
        )
    final_preprocessor = _fit_preprocessor(table=table, fit_idx=task.train_idx, ablation=task.ablation, config=config)
    x_train = _assemble_inputs(table=table, indices=task.train_idx, preprocessor=final_preprocessor)
    x_test = _assemble_inputs(table=table, indices=task.test_idx, preprocessor=final_preprocessor)
    test_mu, test_sigma = _refit_and_predict(
        x_train=x_train,
        y_train=y[task.train_idx],
        x_test=x_test,
        input_dim=int(x_train.shape[2]),
        hidden_dim=int(hidden_dim),
        learning_rate=float(learning_rate),
        dropout=float(config.dropout),
        weight_decay=float(config.weight_decay),
        epochs=int(best_epoch),
        sigma_floor=float(config.sigma_floor),
        batch_size=int(config.batch_size),
        seed=_task_seed(config=config, offset=task.task_index * 73 + 37),
    )
    rolling_metrics = compute_regression_metrics_heteroscedastic(
        y_true=y[task.test_idx],
        y_pred=test_mu,
        gaussian_sigma=test_sigma,
        conformal_q80=0.0,
        conformal_q90=0.0,
        corr_epsilon=config.corr_epsilon,
        mean_oof_sigma=float(np.mean(test_sigma)),
        include_first_difference=True,
    )
    return _RollingTaskResult(
        task_index=task.task_index,
        rolling_fold_rows=[
            {
                "subject_id": task.subject_id,
                "train_through_session": task.train_through_session,
                "test_session": task.test_session,
                "horizon": task.horizon,
                "target_view": task.target_view,
                "ablation": task.ablation,
                "mae": float(rolling_metrics["mae"]),
                "gaussian_nll": float(rolling_metrics["gaussian_nll"]),
                "rmse": float(rolling_metrics["rmse"]),
                "pearson_r": float(rolling_metrics["pearson_r"]),
                "first_difference_pearson_r": float(rolling_metrics["first_difference_pearson_r"]),
                "n_train_rows": int(task.train_idx.size),
                "n_test_rows": int(task.test_idx.size),
            },
        ],
    )


def _execute_primary_task_specs(
    task_specs: list[_PrimaryTaskSpec],
    *,
    primary_tables: dict[int, Step8ForecastingTable],
    config: Step8Config,
    max_workers: int,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cached_results: dict[int, _PrimaryTaskResult] = {}
    remaining_specs: list[_PrimaryTaskSpec] = []
    if checkpoint_dir is not None:
        for task in task_specs:
            key = _primary_checkpoint_key(task)
            data = _load_checkpoint(checkpoint_dir, key)
            if data is not None:
                cached_results[task.task_index] = _primary_result_from_checkpoint(task.task_index, data)
            else:
                remaining_specs.append(task)
        logger.info("Step 8 primary: %d/%d cached, %d to compute", len(cached_results), len(task_specs), len(remaining_specs))
    else:
        remaining_specs = list(task_specs)

    def _run_and_save(task: _PrimaryTaskSpec) -> _PrimaryTaskResult:
        result = _run_primary_task(task, primary_tables=primary_tables, config=config)
        if checkpoint_dir is not None:
            _save_checkpoint(checkpoint_dir, _primary_checkpoint_key(task), _primary_result_to_checkpoint(result))
        return result

    started_at = time.perf_counter()
    if max_workers <= 1:
        new_results = []
        for offset, task in enumerate(remaining_specs, start=1):
            logger.info("Step 8 primary: starting %d/%d (%s)", offset, len(remaining_specs), _describe_primary_task(task))
            result = _run_and_save(task)
            new_results.append(result)
            elapsed_minutes = (time.perf_counter() - started_at) / 60.0
            logger.info(
                "Step 8 primary: completed %d/%d (%s) elapsed=%.1f min",
                offset,
                len(remaining_specs),
                _describe_primary_task(task),
                elapsed_minutes,
            )
    else:
        task_positions = {int(task.task_index): position for position, task in enumerate(remaining_specs, start=1)}
        task_descriptions = {int(task.task_index): _describe_primary_task(task) for task in remaining_specs}

        def _run_with_task_log(task: _PrimaryTaskSpec) -> _PrimaryTaskResult:
            logger.info(
                "Step 8 primary: submitted %d/%d (%s)",
                task_positions[int(task.task_index)],
                len(remaining_specs),
                task_descriptions[int(task.task_index)],
            )
            return _run_and_save(task)

        new_results = _execute_ordered_tasks(remaining_specs, max_workers=max_workers, run_task=_run_with_task_log)
        for result in new_results:
            elapsed_minutes = (time.perf_counter() - started_at) / 60.0
            logger.info(
                "Step 8 primary: completed %d/%d (%s) elapsed=%.1f min",
                task_positions[int(result.task_index)],
                len(remaining_specs),
                task_descriptions[int(result.task_index)],
                elapsed_minutes,
            )
    for result in new_results:
        cached_results[result.task_index] = result
    all_results = [cached_results[task.task_index] for task in task_specs]
    fold_metric_rows: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for task_result in all_results:
        fold_metric_rows.extend(task_result.fold_metric_rows)
        model_selection_rows.extend(task_result.model_selection_rows)
        prediction_rows.extend(task_result.prediction_rows)
    return fold_metric_rows, prediction_rows, model_selection_rows


def _execute_rolling_task_specs(
    task_specs: list[_RollingTaskSpec],
    *,
    primary_tables: dict[int, Step8ForecastingTable],
    config: Step8Config,
    max_workers: int,
    checkpoint_dir: Path | None = None,
) -> list[dict[str, Any]]:
    cached_results: dict[int, _RollingTaskResult] = {}
    remaining_specs: list[_RollingTaskSpec] = []
    if checkpoint_dir is not None:
        for task in task_specs:
            key = _rolling_checkpoint_key(task)
            data = _load_checkpoint(checkpoint_dir, key)
            if data is not None:
                cached_results[task.task_index] = _rolling_result_from_checkpoint(task.task_index, data)
            else:
                remaining_specs.append(task)
        logger.info("Step 8 rolling: %d/%d cached, %d to compute", len(cached_results), len(task_specs), len(remaining_specs))
    else:
        remaining_specs = list(task_specs)

    def _run_and_save(task: _RollingTaskSpec) -> _RollingTaskResult:
        result = _run_rolling_task(task, primary_tables=primary_tables, config=config)
        if checkpoint_dir is not None:
            _save_checkpoint(checkpoint_dir, _rolling_checkpoint_key(task), _rolling_result_to_checkpoint(result))
        return result

    started_at = time.perf_counter()
    if max_workers <= 1:
        new_results = []
        for offset, task in enumerate(remaining_specs, start=1):
            logger.info("Step 8 rolling: starting %d/%d (%s)", offset, len(remaining_specs), _describe_rolling_task(task))
            result = _run_and_save(task)
            new_results.append(result)
            elapsed_minutes = (time.perf_counter() - started_at) / 60.0
            logger.info(
                "Step 8 rolling: completed %d/%d (%s) elapsed=%.1f min",
                offset,
                len(remaining_specs),
                _describe_rolling_task(task),
                elapsed_minutes,
            )
    else:
        task_positions = {int(task.task_index): position for position, task in enumerate(remaining_specs, start=1)}
        task_descriptions = {int(task.task_index): _describe_rolling_task(task) for task in remaining_specs}

        def _run_with_task_log(task: _RollingTaskSpec) -> _RollingTaskResult:
            logger.info(
                "Step 8 rolling: submitted %d/%d (%s)",
                task_positions[int(task.task_index)],
                len(remaining_specs),
                task_descriptions[int(task.task_index)],
            )
            return _run_and_save(task)

        new_results = _execute_ordered_tasks(remaining_specs, max_workers=max_workers, run_task=_run_with_task_log)
        for result in new_results:
            elapsed_minutes = (time.perf_counter() - started_at) / 60.0
            logger.info(
                "Step 8 rolling: completed %d/%d (%s) elapsed=%.1f min",
                task_positions[int(result.task_index)],
                len(remaining_specs),
                task_descriptions[int(result.task_index)],
                elapsed_minutes,
            )
    for result in new_results:
        cached_results[result.task_index] = result
    all_results = [cached_results[task.task_index] for task in task_specs]
    rolling_rows: list[dict[str, Any]] = []
    for task_result in all_results:
        rolling_rows.extend(task_result.rolling_fold_rows)
    return rolling_rows


def _summarize_primary(fold_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in fold_metric_rows}):
        for target_view in sorted({str(row["target_view"]) for row in fold_metric_rows}):
            for ablation in sorted({str(row["ablation"]) for row in fold_metric_rows}):
                matching = [
                    row
                    for row in fold_metric_rows
                    if int(row["horizon"]) == horizon
                    and str(row["target_view"]) == target_view
                    and str(row["ablation"]) == ablation
                ]
                if not matching:
                    continue
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "target_view": target_view,
                        "ablation": ablation,
                        "n_folds": len(matching),
                        "mae_mean": float(np.mean([float(row["mae"]) for row in matching])),
                        "mae_std": float(np.std([float(row["mae"]) for row in matching], ddof=0)),
                        "gaussian_nll_mean": float(np.mean([float(row["gaussian_nll"]) for row in matching])),
                        "gaussian_nll_std": float(np.std([float(row["gaussian_nll"]) for row in matching], ddof=0)),
                        "rmse_mean": float(np.mean([float(row["rmse"]) for row in matching])),
                        "rmse_std": float(np.std([float(row["rmse"]) for row in matching], ddof=0)),
                        "r2_mean": float(np.mean([float(row["r2"]) for row in matching])),
                        "r2_std": float(np.std([float(row["r2"]) for row in matching], ddof=0)),
                        "pearson_r_mean": float(np.mean([float(row["pearson_r"]) for row in matching])),
                        "pearson_r_std": float(np.std([float(row["pearson_r"]) for row in matching], ddof=0)),
                        "gaussian_picp_80_mean": float(np.mean([float(row["gaussian_picp_80"]) for row in matching])),
                        "gaussian_picp_80_std": float(np.std([float(row["gaussian_picp_80"]) for row in matching], ddof=0)),
                        "gaussian_picp_90_mean": float(np.mean([float(row["gaussian_picp_90"]) for row in matching])),
                        "gaussian_picp_90_std": float(np.std([float(row["gaussian_picp_90"]) for row in matching], ddof=0)),
                        "conformal_picp_80_mean": float(np.mean([float(row["conformal_picp_80"]) for row in matching])),
                        "conformal_picp_80_std": float(np.std([float(row["conformal_picp_80"]) for row in matching], ddof=0)),
                        "conformal_picp_90_mean": float(np.mean([float(row["conformal_picp_90"]) for row in matching])),
                        "conformal_picp_90_std": float(np.std([float(row["conformal_picp_90"]) for row in matching], ddof=0)),
                        "mean_learned_sigma_mean": float(np.mean([float(row["mean_learned_sigma"]) for row in matching])),
                        "mean_learned_sigma_std": float(np.std([float(row["mean_learned_sigma"]) for row in matching], ddof=0)),
                    },
                )
    return summary_rows


def _summarize_rolling(rolling_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in rolling_rows}):
        for target_view in sorted({str(row["target_view"]) for row in rolling_rows}):
            for ablation in sorted({str(row["ablation"]) for row in rolling_rows}):
                matching = [
                    row
                    for row in rolling_rows
                    if int(row["horizon"]) == horizon
                    and str(row["target_view"]) == target_view
                    and str(row["ablation"]) == ablation
                ]
                if not matching:
                    continue
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "target_view": target_view,
                        "ablation": ablation,
                        "n_folds": len(matching),
                        "mae_mean": float(np.mean([float(row["mae"]) for row in matching])),
                        "mae_std": float(np.std([float(row["mae"]) for row in matching], ddof=0)),
                        "gaussian_nll_mean": float(np.mean([float(row["gaussian_nll"]) for row in matching])),
                        "gaussian_nll_std": float(np.std([float(row["gaussian_nll"]) for row in matching], ddof=0)),
                        "rmse_mean": float(np.mean([float(row["rmse"]) for row in matching])),
                        "rmse_std": float(np.std([float(row["rmse"]) for row in matching], ddof=0)),
                        "pearson_r_mean": float(np.mean([float(row["pearson_r"]) for row in matching])),
                        "pearson_r_std": float(np.std([float(row["pearson_r"]) for row in matching], ddof=0)),
                        "first_difference_pearson_r_mean": float(np.mean([float(row["first_difference_pearson_r"]) for row in matching])),
                        "first_difference_pearson_r_std": float(np.std([float(row["first_difference_pearson_r"]) for row in matching], ddof=0)),
                    },
                )
    return summary_rows


def _load_step7_summary_rows(config: Step8Config) -> list[dict[str, str]]:
    return _load_required_csv_rows(
        path=config.step7_root / "step7_summary.csv",
        required_columns=FIGURES_ONLY_STEP7_SUMMARY_COLUMNS,
    )


def _load_step7_fold_metric_lookup(config: Step8Config) -> dict[tuple[int, str, str, str], dict[str, str]]:
    _, rows = load_csv_rows(config.step7_root / "step7_fold_metrics.csv")
    lookup: dict[tuple[int, str, str, str], dict[str, str]] = {}
    for row in rows:
        lookup[(int(row["horizon"]), str(row["target_view"]), str(row["model_name"]), str(row["fold_subject_id"]))] = row
    return lookup


def _load_step7_context_baseline_lookup(config: Step8Config) -> dict[tuple[str, str, int, str, int, int], dict[str, str]]:
    _, rows = load_csv_rows(config.step7_root / "step7_predictions.csv")
    lookup: dict[tuple[str, str, int, str, int, int], dict[str, str]] = {}
    for row in rows:
        if str(row["model_name"]) != "context_score_mean":
            continue
        lookup[
            (
                str(row["fold_subject_id"]),
                str(row["subject_id"]),
                int(row["horizon"]),
                str(row["target_view"]),
                int(row["anchor_idx"]),
                int(row["target_idx"]),
            )
        ] = row
    return lookup


def _load_step7_context_sigma_lookup(config: Step8Config) -> dict[tuple[str, int, str], float]:
    _, rows = load_csv_rows(config.step7_root / "step7_model_selection.csv")
    lookup: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if str(row["model_name"]) != "context_score_mean":
            continue
        lookup[(str(row["fold_subject_id"]), int(row["horizon"]), str(row["target_view"]))] = float(row["oof_residual_sigma"])
    return lookup


def _strongest_step7_baseline_lookup(config: Step8Config) -> dict[tuple[int, str, str], str]:
    summary_rows = _load_step7_summary_rows(config)
    lookup: dict[tuple[int, str, str], str] = {}
    for horizon in config.horizons:
        for target_view in config.target_views:
            matching = [
                row
                for row in summary_rows
                if int(row["horizon"]) == horizon
                and str(row["target_view"]) == target_view
                and str(row["model_name"]) in STEP7_TRAINED_BASELINE_CANDIDATES
            ]
            if not matching:
                raise ValueError(f"Missing Step 7 baseline candidates for horizon={horizon}, target_view={target_view}")
            for metric_name in PAIRED_METRICS:
                other_metric = "gaussian_nll_mean" if metric_name == "mae" else "mae_mean"
                best_row = min(
                    matching,
                    key=lambda row: (
                        float(row[f"{metric_name}_mean"]),
                        float(row[other_metric]),
                        str(row["model_name"]),
                    ),
                )
                lookup[(horizon, target_view, metric_name)] = str(best_row["model_name"])
    return lookup


def _paired_stats_rows(*, fold_metric_rows: list[dict[str, Any]], config: Step8Config) -> list[dict[str, Any]]:
    step7_lookup = _load_step7_fold_metric_lookup(config)
    step7_best_lookup = _strongest_step7_baseline_lookup(config)
    step8_lookup: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in fold_metric_rows:
        step8_lookup[(int(row["horizon"]), str(row["target_view"]), _gru_model_name(str(row["ablation"])), str(row["fold_subject_id"]))] = row

    paired_results: list[PairedStatistic] = []
    for horizon in sorted({int(row["horizon"]) for row in fold_metric_rows}):
        for target_view in sorted({str(row["target_view"]) for row in fold_metric_rows}):
            for metric_name in PAIRED_METRICS:
                for comparison_id, model_a, model_b_template in PAIRED_COMPARISON_TEMPLATES:
                    model_b = step7_best_lookup[(horizon, target_view, metric_name)] if model_b_template == "__step7_best__" else model_b_template
                    values_a: list[float] = []
                    values_b: list[float] = []
                    for subject_id in SUBJECT_IDS:
                        key_a = (horizon, target_view, model_a, subject_id)
                        values_a.append(float(step8_lookup[key_a][metric_name]))
                        if model_b.startswith("gru_"):
                            values_b.append(float(step8_lookup[(horizon, target_view, model_b, subject_id)][metric_name]))
                        else:
                            values_b.append(float(step7_lookup[(horizon, target_view, model_b, subject_id)][metric_name]))
                    paired_results.append(
                        compute_paired_statistic(
                            comparison_id=comparison_id,
                            horizon=horizon,
                            target_view=target_view,
                            metric=metric_name,
                            model_a=model_a,
                            model_b=model_b,
                            metric_a=np.asarray(values_a, dtype=np.float64),
                            metric_b=np.asarray(values_b, dtype=np.float64),
                            bootstrap_seed=config.bootstrap_seed,
                            bootstrap_replicates=config.bootstrap_replicates,
                        ),
                    )

    corrected_by_family: dict[tuple[int, str, str], np.ndarray] = {}
    for horizon in sorted({result.horizon for result in paired_results}):
        for target_view in sorted({result.target_view for result in paired_results}):
            for metric_name in PAIRED_METRICS:
                family_rows = [
                    result
                    for result in paired_results
                    if result.horizon == horizon and result.target_view == target_view and result.metric == metric_name
                ]
                corrected_by_family[(horizon, target_view, metric_name)] = holm_correction(
                    np.asarray([result.p_value for result in family_rows], dtype=np.float64),
                )

    finalized_rows: list[dict[str, Any]] = []
    family_position: dict[tuple[int, str, str], int] = {}
    for result in paired_results:
        key = (result.horizon, result.target_view, result.metric)
        offset = family_position.get(key, 0)
        finalized_rows.append(
            {
                "comparison_id": result.comparison_id,
                "horizon": result.horizon,
                "target_view": result.target_view,
                "metric": result.metric,
                "model_a": result.model_a,
                "model_b": result.model_b,
                "n_folds": result.n_folds,
                "n_nonzero_deltas": result.n_nonzero_deltas,
                "test_method": result.test_method,
                "zero_method": result.zero_method,
                "p_value_method": result.p_value_method,
                "effect_size_method": result.effect_size_method,
                "mean_metric_a": result.mean_metric_a,
                "mean_metric_b": result.mean_metric_b,
                "mean_delta_b_minus_a": result.mean_delta_b_minus_a,
                "median_delta_b_minus_a": result.median_delta_b_minus_a,
                "bootstrap_ci_lower": result.bootstrap_ci_lower,
                "bootstrap_ci_upper": result.bootstrap_ci_upper,
                "wilcoxon_stat": result.wilcoxon_stat,
                "p_value": result.p_value,
                "p_value_holm": float(corrected_by_family[key][offset]),
                "rank_biserial": result.rank_biserial,
                "better_model": result.better_model,
            },
        )
        family_position[key] = offset + 1
    return finalized_rows


def _project_prediction_rows_for_csv(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row[column] for column in PREDICTIONS_COLUMNS} for row in prediction_rows]


def _build_anomaly_rows(prediction_rows: list[dict[str, Any]], *, config: Step8Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step7_prediction_lookup = _load_step7_context_baseline_lookup(config)
    step7_sigma_lookup = _load_step7_context_sigma_lookup(config)
    anomaly_prediction_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        key = (
            str(row["fold_subject_id"]),
            str(row["subject_id"]),
            int(row["horizon"]),
            str(row["target_view"]),
            int(row["anchor_idx"]),
            int(row["target_idx"]),
        )
        step7_row = step7_prediction_lookup[key]
        sigma_key = (str(row["fold_subject_id"]), int(row["horizon"]), str(row["target_view"]))
        baseline_sigma_raw = float(step7_sigma_lookup[sigma_key])
        mu_raw, sigma_raw = raw_space_prediction(
            target_view=str(row["target_view"]),
            y_pred_mu=np.asarray([float(row["y_pred_mu"])], dtype=np.float64),
            y_pred_sigma=np.asarray([float(row["y_pred_sigma"])], dtype=np.float64),
            context_score_mean_value=np.asarray([float(row["context_score_mean_value"])], dtype=np.float64),
        )
        baseline_residual_mu = float(step7_row["y_pred_mean"])
        baseline_mu_raw = baseline_residual_mu if str(row["target_view"]) == "raw_level" else float(row["context_score_mean_value"]) + baseline_residual_mu
        threshold = float(row["drop20_threshold_raw"])
        anomaly_prediction_rows.append(
            {
                "fold_subject_id": str(row["fold_subject_id"]),
                "subject_id": str(row["subject_id"]),
                "session_id": str(row["session_id"]),
                "horizon": int(row["horizon"]),
                "target_view": str(row["target_view"]),
                "ablation": str(row["ablation"]),
                "anchor_idx": int(row["anchor_idx"]),
                "target_idx": int(row["target_idx"]),
                "y_true": float(row["y_true"]),
                "y_pred_mu": float(row["y_pred_mu"]),
                "y_pred_sigma": float(row["y_pred_sigma"]),
                "context_score_mean_value": float(row["context_score_mean_value"]),
                "session_cal_mean_raw": float(row["session_cal_mean_raw"]),
                "drop20_threshold_raw": threshold,
                "mu_raw": float(mu_raw[0]),
                "sigma_raw": float(sigma_raw[0]),
                "p_drop20_state": float(
                    prospective_drop20_probability(
                        mu_raw=mu_raw,
                        sigma_raw=sigma_raw,
                        drop20_threshold_raw=np.asarray([threshold], dtype=np.float64),
                    )[0],
                ),
                "baseline_mu_raw": baseline_mu_raw,
                "baseline_sigma_raw": baseline_sigma_raw,
                "baseline_p_drop20_state": float(
                    prospective_drop20_probability(
                        mu_raw=np.asarray([baseline_mu_raw], dtype=np.float64),
                        sigma_raw=np.asarray([baseline_sigma_raw], dtype=np.float64),
                        drop20_threshold_raw=np.asarray([threshold], dtype=np.float64),
                    )[0],
                ),
                "z_score": float(row["z_score"]),
                "target_drop20_state": int(row["target_drop20_state"]),
                "future_drop20_onset": int(row["future_drop20_onset"]),
            },
        )

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in anomaly_prediction_rows:
        grouped.setdefault((str(row["fold_subject_id"]), int(row["horizon"]), str(row["target_view"]), str(row["ablation"])), []).append(row)

    anomaly_metric_rows: list[dict[str, Any]] = []
    for (fold_subject_id, horizon, target_view, ablation), rows in grouped.items():
        state_labels = np.asarray([int(row["target_drop20_state"]) for row in rows], dtype=np.int8)
        onset_labels = np.asarray([int(row["future_drop20_onset"]) for row in rows], dtype=np.int8)
        p_drop = np.asarray([float(row["p_drop20_state"]) for row in rows], dtype=np.float64)
        neg_mu = -np.asarray([float(row["mu_raw"]) for row in rows], dtype=np.float64)
        sigma = np.asarray([float(row["sigma_raw"]) for row in rows], dtype=np.float64)
        baseline_p = np.asarray([float(row["baseline_p_drop20_state"]) for row in rows], dtype=np.float64)
        baseline_neg_mu = -np.asarray([float(row["baseline_mu_raw"]) for row in rows], dtype=np.float64)
        baseline_sigma = np.asarray([float(row["baseline_sigma_raw"]) for row in rows], dtype=np.float64)
        z_score = np.asarray([float(row["z_score"]) for row in rows], dtype=np.float64)
        state_p_roc, state_p_pr = binary_auc_metrics(state_labels, p_drop)
        onset_p_roc, onset_p_pr = binary_auc_metrics(onset_labels, p_drop)
        state_mu_roc, state_mu_pr = binary_auc_metrics(state_labels, neg_mu)
        onset_mu_roc, onset_mu_pr = binary_auc_metrics(onset_labels, neg_mu)
        state_sigma_roc, state_sigma_pr = binary_auc_metrics(state_labels, sigma)
        onset_sigma_roc, onset_sigma_pr = binary_auc_metrics(onset_labels, sigma)
        baseline_state_p_roc, baseline_state_p_pr = binary_auc_metrics(state_labels, baseline_p)
        baseline_onset_p_roc, baseline_onset_p_pr = binary_auc_metrics(onset_labels, baseline_p)
        baseline_state_mu_roc, baseline_state_mu_pr = binary_auc_metrics(state_labels, baseline_neg_mu)
        baseline_onset_mu_roc, baseline_onset_mu_pr = binary_auc_metrics(onset_labels, baseline_neg_mu)
        baseline_state_sigma_roc, baseline_state_sigma_pr = binary_auc_metrics(state_labels, baseline_sigma)
        baseline_onset_sigma_roc, baseline_onset_sigma_pr = binary_auc_metrics(onset_labels, baseline_sigma)
        z_ks_stat, z_ks_pvalue = standard_normal_ks(z_score)
        anomaly_metric_rows.append(
            {
                "fold_subject_id": fold_subject_id,
                "horizon": horizon,
                "target_view": target_view,
                "ablation": ablation,
                "n_test_rows": len(rows),
                "n_positive_drop20_state": int(np.sum(state_labels)),
                "n_positive_drop20_onset": int(np.sum(onset_labels)),
                "prospective_roc_auc_p_drop20_state_for_state": state_p_roc,
                "prospective_pr_auc_p_drop20_state_for_state": state_p_pr,
                "prospective_roc_auc_p_drop20_state_for_onset": onset_p_roc,
                "prospective_pr_auc_p_drop20_state_for_onset": onset_p_pr,
                "prospective_roc_auc_neg_mu_raw_for_state": state_mu_roc,
                "prospective_pr_auc_neg_mu_raw_for_state": state_mu_pr,
                "prospective_roc_auc_neg_mu_raw_for_onset": onset_mu_roc,
                "prospective_pr_auc_neg_mu_raw_for_onset": onset_mu_pr,
                "prospective_roc_auc_sigma_raw_for_state": state_sigma_roc,
                "prospective_pr_auc_sigma_raw_for_state": state_sigma_pr,
                "prospective_roc_auc_sigma_raw_for_onset": onset_sigma_roc,
                "prospective_pr_auc_sigma_raw_for_onset": onset_sigma_pr,
                "baseline_roc_auc_p_drop20_state_for_state": baseline_state_p_roc,
                "baseline_pr_auc_p_drop20_state_for_state": baseline_state_p_pr,
                "baseline_roc_auc_p_drop20_state_for_onset": baseline_onset_p_roc,
                "baseline_pr_auc_p_drop20_state_for_onset": baseline_onset_p_pr,
                "baseline_roc_auc_neg_mu_raw_for_state": baseline_state_mu_roc,
                "baseline_pr_auc_neg_mu_raw_for_state": baseline_state_mu_pr,
                "baseline_roc_auc_neg_mu_raw_for_onset": baseline_onset_mu_roc,
                "baseline_pr_auc_neg_mu_raw_for_onset": baseline_onset_mu_pr,
                "baseline_roc_auc_sigma_raw_for_state": baseline_state_sigma_roc,
                "baseline_pr_auc_sigma_raw_for_state": baseline_state_sigma_pr,
                "baseline_roc_auc_sigma_raw_for_onset": baseline_onset_sigma_roc,
                "baseline_pr_auc_sigma_raw_for_onset": baseline_onset_sigma_pr,
                "z_score_mean": float(np.mean(z_score)),
                "z_score_std": float(np.std(z_score, ddof=0)),
                "z_score_ks_stat": float(z_ks_stat),
                "z_score_ks_pvalue": float(z_ks_pvalue),
            },
        )
    return anomaly_prediction_rows, anomaly_metric_rows


def _summarize_anomaly(anomaly_metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    base_keys = ("horizon", "target_view", "ablation")
    numeric_columns = [column for column in ANOMALY_METRICS_COLUMNS if column not in {"fold_subject_id", *base_keys}]
    summary_columns: list[str] = ["horizon", "target_view", "ablation", "n_folds"]
    for column in numeric_columns:
        summary_columns.append(f"{column}_mean")
        summary_columns.append(f"{column}_std")

    summary_rows: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in anomaly_metric_rows}):
        for target_view in sorted({str(row["target_view"]) for row in anomaly_metric_rows}):
            for ablation in sorted({str(row["ablation"]) for row in anomaly_metric_rows}):
                matching = [
                    row
                    for row in anomaly_metric_rows
                    if int(row["horizon"]) == horizon
                    and str(row["target_view"]) == target_view
                    and str(row["ablation"]) == ablation
                ]
                if not matching:
                    continue
                summary_row: dict[str, Any] = {
                    "horizon": horizon,
                    "target_view": target_view,
                    "ablation": ablation,
                    "n_folds": len(matching),
                }
                for column in numeric_columns:
                    values = np.asarray([float(row[column]) for row in matching], dtype=np.float64)
                    summary_row[f"{column}_mean"] = float(np.nanmean(values))
                    summary_row[f"{column}_std"] = float(np.nanstd(values, ddof=0))
                summary_rows.append(summary_row)
    return summary_rows, tuple(summary_columns)


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_figures(
    *,
    summary_rows: list[dict[str, Any]],
    rolling_summary_rows: list[dict[str, Any]],
    anomaly_prediction_rows: list[dict[str, Any]],
    step7_summary_rows: list[dict[str, str]],
    figure_root: Path,
) -> None:
    step7_lookup = {(int(row["horizon"]), str(row["target_view"]), str(row["model_name"])): row for row in step7_summary_rows}
    horizons = sorted({int(row["horizon"]) for row in summary_rows})
    ablations = sorted({str(row["ablation"]) for row in summary_rows})
    target_views = sorted({str(row["target_view"]) for row in summary_rows})

    for metric_name, filename, y_label in (
        ("mae_mean", "step8_ablation_mae_by_horizon.png", "MAE"),
        ("gaussian_nll_mean", "step8_ablation_nll_by_horizon.png", "Gaussian NLL"),
    ):
        fig, axes = plt.subplots(1, len(target_views), figsize=(5 * len(target_views), 4), squeeze=False)
        for col_index, target_view in enumerate(target_views):
            axis = axes[0, col_index]
            for ablation in ablations:
                values = [
                    float(
                        next(
                            row[metric_name]
                            for row in summary_rows
                            if int(row["horizon"]) == horizon and str(row["target_view"]) == target_view and str(row["ablation"]) == ablation
                        ),
                    )
                    for horizon in horizons
                ]
                axis.plot(horizons, values, marker="o", label=ablation)
            baseline_values = [float(step7_lookup[(horizon, target_view, "context_score_mean")][metric_name]) for horizon in horizons]
            axis.plot(horizons, baseline_values, marker="s", linestyle="--", label="context_score_mean")
            axis.set_title(target_view)
            axis.set_xlabel("Horizon")
            axis.set_ylabel(y_label)
            axis.legend()
        _save_figure(figure_path(figure_root=figure_root, filename=filename))

    def _best_step8_value(target_view: str, metric_name: str, horizon: int) -> float:
        matching = [row for row in summary_rows if str(row["target_view"]) == target_view and int(row["horizon"]) == horizon]
        return float(min(matching, key=lambda row: float(row[metric_name]))[metric_name])

    for target_view, filename in (
        ("raw_level", "step8_raw_vs_step7_best.png"),
        ("context_residual", "step8_residual_vs_step7_best.png"),
    ):
        fig, axis = plt.subplots(figsize=(6, 4))
        step8_values = [_best_step8_value(target_view, "mae_mean", horizon) for horizon in horizons]
        step7_values = [
            min(float(row["mae_mean"]) for row in step7_summary_rows if int(row["horizon"]) == horizon and str(row["target_view"]) == target_view)
            for horizon in horizons
        ]
        axis.plot(horizons, step8_values, marker="o", label="Step 8 best")
        axis.plot(horizons, step7_values, marker="s", linestyle="--", label="Step 7 best")
        axis.set_xlabel("Horizon")
        axis.set_ylabel("MAE")
        axis.set_title(target_view)
        axis.legend()
        _save_figure(figure_path(figure_root=figure_root, filename=filename))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        np.arange(len(ablations)),
        [float(np.mean([float(row["gaussian_picp_80_mean"]) for row in summary_rows if str(row["ablation"]) == ablation])) for ablation in ablations],
    )
    axes[0].set_xticks(np.arange(len(ablations)), ablations)
    axes[0].set_title("Gaussian 80% coverage")
    axes[1].bar(
        np.arange(len(ablations)),
        [float(np.mean([float(row["conformal_picp_80_mean"]) for row in summary_rows if str(row["ablation"]) == ablation])) for ablation in ablations],
    )
    axes[1].set_xticks(np.arange(len(ablations)), ablations)
    axes[1].set_title("Conformal 80% coverage")
    _save_figure(figure_path(figure_root=figure_root, filename="step8_calibration_comparison.png"))

    learned_sigma = np.asarray([float(row["y_pred_sigma"]) for row in anomaly_prediction_rows], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(learned_sigma, bins=30)
    axis.set_xlabel("Learned sigma")
    axis.set_ylabel("Count")
    axis.set_title("Learned Sigma Distribution")
    _save_figure(figure_path(figure_root=figure_root, filename="step8_learned_sigma_distribution.png"))

    z_score = np.asarray([float(row["z_score"]) for row in anomaly_prediction_rows], dtype=np.float64)
    quantiles = np.linspace(0.01, 0.99, 99)
    rng = np.random.default_rng(0)
    theoretical = np.percentile(rng.normal(size=100000), quantiles * 100.0)
    empirical = np.percentile(z_score, quantiles * 100.0)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(theoretical, empirical, s=10)
    min_value = float(min(np.min(theoretical), np.min(empirical)))
    max_value = float(max(np.max(theoretical), np.max(empirical)))
    axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    axis.set_xlabel("Theoretical quantiles")
    axis.set_ylabel("Empirical quantiles")
    axis.set_title("Z-score QQ Plot")
    _save_figure(figure_path(figure_root=figure_root, filename="step8_zscore_qq_plot.png"))

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(z_score, bins=30, density=True)
    xs = np.linspace(float(np.min(z_score)), float(np.max(z_score)), 200)
    axis.plot(xs, (1.0 / np.sqrt(2.0 * math.pi)) * np.exp(-0.5 * xs * xs), color="black")
    axis.set_title("Z-score Histogram")
    _save_figure(figure_path(figure_root=figure_root, filename="step8_zscore_histogram.png"))

    for rows, filename, positive_key in (
        (anomaly_prediction_rows, "step8_anomaly_roc_curves.png", "target_drop20_state"),
        (anomaly_prediction_rows, "step8_anomaly_pr_curves.png", "future_drop20_onset"),
    ):
        labels = np.asarray([int(row[positive_key]) for row in rows], dtype=np.int8)
        fig, axis = plt.subplots(figsize=(6, 4))
        for label_name, score in (
            ("GRU p_drop20_state", np.asarray([float(row["p_drop20_state"]) for row in rows], dtype=np.float64)),
            ("Baseline p_drop20_state", np.asarray([float(row["baseline_p_drop20_state"]) for row in rows], dtype=np.float64)),
        ):
            if np.unique(labels).size < 2:
                continue
            order = np.argsort(score)
            sorted_labels = labels[order]
            if filename.endswith("roc_curves.png"):
                tpr = np.cumsum(sorted_labels[::-1]) / max(1, int(np.sum(sorted_labels)))
                fpr = np.cumsum((1 - sorted_labels)[::-1]) / max(1, int(np.sum(1 - sorted_labels)))
                axis.plot(np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr]), label=label_name)
                axis.set_xlabel("False Positive Rate")
                axis.set_ylabel("True Positive Rate")
            else:
                tp = np.cumsum(sorted_labels[::-1])
                fp = np.cumsum((1 - sorted_labels)[::-1])
                precision = tp / np.maximum(tp + fp, 1)
                recall = tp / max(1, int(np.sum(sorted_labels)))
                axis.plot(np.concatenate([[0.0], recall]), np.concatenate([[1.0], precision]), label=label_name)
                axis.set_xlabel("Recall")
                axis.set_ylabel("Precision")
        axis.legend()
        _save_figure(figure_path(figure_root=figure_root, filename=filename))

    fig, axis = plt.subplots(figsize=(6, 4))
    decline_sigma = np.asarray([float(row["sigma_raw"]) for row in anomaly_prediction_rows if int(row["target_drop20_state"]) == 1], dtype=np.float64)
    steady_sigma = np.asarray([float(row["sigma_raw"]) for row in anomaly_prediction_rows if int(row["target_drop20_state"]) == 0], dtype=np.float64)
    axis.boxplot([steady_sigma, decline_sigma], tick_labels=["non-decline", "decline"])
    axis.set_ylabel("Sigma raw")
    axis.set_title("Sigma vs Decline")
    _save_figure(figure_path(figure_root=figure_root, filename="step8_anomaly_sigma_vs_decline.png"))

    fig, axis = plt.subplots(figsize=(6, 4))
    if rolling_summary_rows:
        bar_positions: list[int] = []
        bar_values: list[float] = []
        bar_labels: list[str] = []
        for horizon in sorted({int(row["horizon"]) for row in rolling_summary_rows}):
            values = [
                float(row["first_difference_pearson_r_mean"])
                for row in rolling_summary_rows
                if int(row["horizon"]) == horizon and str(row["target_view"]) == "raw_level"
            ]
            if values:
                bar_positions.append(len(bar_positions))
                bar_values.append(values[0])
                bar_labels.append(str(horizon))
        if bar_values:
            axis.bar(bar_positions, bar_values)
            axis.set_xticks(bar_positions, bar_labels)
            axis.set_xlabel("Horizon")
    axis.set_ylabel("First-difference Pearson r")
    axis.set_title("Rolling First Difference")
    _save_figure(figure_path(figure_root=figure_root, filename="step8_rolling_first_difference.png"))


def _validate_expected_figure_files(figure_root: Path) -> None:
    required_figures = [figure_path(figure_root=figure_root, filename=name) for name in EXPECTED_FIGURE_FILENAMES]
    missing_figures = [path.name for path in required_figures if not path.is_file()]
    if missing_figures:
        raise FileNotFoundError(f"Missing Step 8 figures: {missing_figures}")


def run_step8_figures_only(config: Step8Config) -> int:
    """Regenerate Step 8 figures from saved CSV outputs without retraining."""

    summary_rows = _load_required_csv_rows(
        path=summary_path(output_root=config.output_root),
        required_columns=FIGURES_ONLY_SUMMARY_COLUMNS,
    )
    rolling_summary_rows = _load_required_csv_rows(
        path=rolling_summary_path(output_root=config.output_root),
        required_columns=FIGURES_ONLY_ROLLING_SUMMARY_COLUMNS,
    )
    anomaly_prediction_rows = _load_required_csv_rows(
        path=anomaly_predictions_path(output_root=config.output_root),
        required_columns=FIGURES_ONLY_ANOMALY_PREDICTIONS_COLUMNS,
    )
    step7_summary_rows = _load_step7_summary_rows(config)
    _save_figures(
        summary_rows=summary_rows,
        rolling_summary_rows=rolling_summary_rows,
        anomaly_prediction_rows=anomaly_prediction_rows,
        step7_summary_rows=step7_summary_rows,
        figure_root=config.figure_root,
    )
    _validate_expected_figure_files(config.figure_root)
    return len(EXPECTED_FIGURE_FILENAMES)


def validate_outputs(
    *,
    config: Step8Config,
    primary_tables: dict[int, Step8ForecastingTable],
    primary_task_specs: list[_PrimaryTaskSpec],
    rolling_task_specs: list[_RollingTaskSpec],
    fold_metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    model_selection_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    rolling_fold_rows: list[dict[str, Any]],
    rolling_summary_rows: list[dict[str, Any]],
    anomaly_metric_rows: list[dict[str, Any]],
    anomaly_summary_rows: list[dict[str, Any]],
    anomaly_prediction_rows: list[dict[str, Any]],
    anomaly_summary_columns: tuple[str, ...],
) -> None:
    expected_primary_rows = len(primary_task_specs)
    expected_prediction_rows = sum(table.n_rows for table in primary_tables.values()) * len(config.ablations) * len(config.target_views)
    expected_summary_rows = len(config.horizons) * len(config.ablations) * len(config.target_views)
    expected_paired_rows = len(config.horizons) * len(PAIRED_COMPARISON_TEMPLATES) * len(config.target_views) * len(PAIRED_METRICS)
    expected_rolling_rows = len(rolling_task_specs)
    expected_rolling_summary_rows = len(config.rolling_horizons) * len(config.rolling_ablations) * len(config.target_views)
    if len(fold_metric_rows) != expected_primary_rows or len(summary_rows) != expected_summary_rows:
        raise ValueError("Primary Step 8 row counts do not match expected values")
    if len(prediction_rows) != expected_prediction_rows or len(model_selection_rows) != expected_primary_rows:
        raise ValueError("Primary Step 8 prediction/model-selection counts do not match expected values")
    if len(paired_rows) != expected_paired_rows:
        raise ValueError(f"Paired rows {len(paired_rows)} != expected {expected_paired_rows}")
    if len(rolling_fold_rows) != expected_rolling_rows or len(rolling_summary_rows) != expected_rolling_summary_rows:
        raise ValueError("Rolling Step 8 row counts do not match expected values")
    if len(anomaly_metric_rows) != expected_primary_rows or len(anomaly_summary_rows) != expected_summary_rows:
        raise ValueError("Anomaly Step 8 row counts do not match expected values")
    if len(anomaly_prediction_rows) != expected_prediction_rows:
        raise ValueError("Anomaly prediction row count does not match expected values")

    for horizon, table in primary_tables.items():
        expected_table_rows = len(expected_session_ids()) * _expected_usable_rows_per_session(horizon=horizon, config=config)
        if table.n_rows != expected_table_rows:
            raise ValueError(f"Horizon {horizon} loaded {table.n_rows} rows, expected {expected_table_rows}")
    for row in prediction_rows:
        if float(row["y_pred_sigma"]) < config.sigma_floor:
            raise ValueError(f"Prediction sigma below sigma_floor: {row}")
        if float(row["gaussian_lower_80"]) > float(row["gaussian_upper_80"]):
            raise ValueError(f"Inverted Gaussian interval: {row}")
        if float(row["conformal_lower_80"]) > float(row["conformal_upper_80"]):
            raise ValueError(f"Inverted conformal interval: {row}")
    if not anomaly_summary_columns:
        raise ValueError("Anomaly summary columns must not be empty")
    _validate_expected_figure_files(config.figure_root)


def run_step8(config: Step8Config | None = None, *, no_resume: bool = False) -> Step8Summary:
    """Run the full Step 8 neural forecasting layer."""

    config = config or Step8Config()
    validate_step8_config(config)
    primary_tables = load_step8_tables(config)
    max_workers = _resolved_max_workers(config)
    checkpoint_dir: Path | None = None
    if not no_resume:
        checkpoint_dir = _checkpoint_dir(config.output_root)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoint directory: %s", checkpoint_dir)
    else:
        stale_dir = _checkpoint_dir(config.output_root)
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)

    primary_task_specs = _build_primary_task_specs(primary_tables=primary_tables, config=config)
    fold_metric_rows, prediction_rows, model_selection_rows = _execute_primary_task_specs(
        primary_task_specs,
        primary_tables=primary_tables,
        config=config,
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
    )
    summary_rows = _summarize_primary(fold_metric_rows)
    paired_rows = _paired_stats_rows(fold_metric_rows=fold_metric_rows, config=config)
    rolling_task_specs = _build_rolling_task_specs(primary_tables=primary_tables, config=config)
    rolling_fold_rows = _execute_rolling_task_specs(
        rolling_task_specs,
        primary_tables=primary_tables,
        config=config,
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
    )
    rolling_summary_rows = _summarize_rolling(rolling_fold_rows)
    anomaly_prediction_rows, anomaly_metric_rows = _build_anomaly_rows(prediction_rows, config=config)
    anomaly_summary_rows, anomaly_summary_columns = _summarize_anomaly(anomaly_metric_rows)
    step7_summary_rows = _load_step7_summary_rows(config)

    config.output_root.mkdir(parents=True, exist_ok=True)
    _records_to_csv(fold_metrics_path(output_root=config.output_root), columns=FOLD_METRICS_COLUMNS, records=fold_metric_rows)
    _records_to_csv(summary_path(output_root=config.output_root), columns=SUMMARY_COLUMNS, records=summary_rows)
    _records_to_csv(predictions_path(output_root=config.output_root), columns=PREDICTIONS_COLUMNS, records=_project_prediction_rows_for_csv(prediction_rows))
    _records_to_csv(model_selection_path(output_root=config.output_root), columns=MODEL_SELECTION_COLUMNS, records=model_selection_rows)
    _records_to_csv(paired_stats_path(output_root=config.output_root), columns=PAIRED_STATS_COLUMNS, records=paired_rows)
    _records_to_csv(rolling_fold_metrics_path(output_root=config.output_root), columns=ROLLING_FOLD_COLUMNS, records=rolling_fold_rows)
    _records_to_csv(rolling_summary_path(output_root=config.output_root), columns=ROLLING_SUMMARY_COLUMNS, records=rolling_summary_rows)
    _records_to_csv(anomaly_metrics_path(output_root=config.output_root), columns=ANOMALY_METRICS_COLUMNS, records=anomaly_metric_rows)
    _records_to_csv(anomaly_summary_path(output_root=config.output_root), columns=anomaly_summary_columns, records=anomaly_summary_rows)
    _records_to_csv(anomaly_predictions_path(output_root=config.output_root), columns=ANOMALY_PREDICTIONS_COLUMNS, records=anomaly_prediction_rows)

    _save_figures(
        summary_rows=summary_rows,
        rolling_summary_rows=rolling_summary_rows,
        anomaly_prediction_rows=anomaly_prediction_rows,
        step7_summary_rows=step7_summary_rows,
        figure_root=config.figure_root,
    )
    validate_outputs(
        config=config,
        primary_tables=primary_tables,
        primary_task_specs=primary_task_specs,
        rolling_task_specs=rolling_task_specs,
        fold_metric_rows=fold_metric_rows,
        summary_rows=summary_rows,
        prediction_rows=_project_prediction_rows_for_csv(prediction_rows),
        model_selection_rows=model_selection_rows,
        paired_rows=paired_rows,
        rolling_fold_rows=rolling_fold_rows,
        rolling_summary_rows=rolling_summary_rows,
        anomaly_metric_rows=anomaly_metric_rows,
        anomaly_summary_rows=anomaly_summary_rows,
        anomaly_prediction_rows=anomaly_prediction_rows,
        anomaly_summary_columns=anomaly_summary_columns,
    )
    if checkpoint_dir is not None and checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
    return Step8Summary(
        output_root=config.output_root,
        figure_root=config.figure_root,
        fold_metric_rows=len(fold_metric_rows),
        summary_rows=len(summary_rows),
        prediction_rows=len(prediction_rows),
        model_selection_rows=len(model_selection_rows),
        paired_stats_rows=len(paired_rows),
        rolling_fold_rows=len(rolling_fold_rows),
        rolling_summary_rows=len(rolling_summary_rows),
        anomaly_metrics_rows=len(anomaly_metric_rows),
        anomaly_summary_rows=len(anomaly_summary_rows),
        anomaly_prediction_rows=len(anomaly_prediction_rows),
        figure_count=len(EXPECTED_FIGURE_FILENAMES),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    default_config = Step8Config()
    parser = argparse.ArgumentParser(description="Run Step 8 GRU forecasting and anomaly detection.")
    parser.add_argument("--step4-root", type=Path, default=default_config.step4_root)
    parser.add_argument("--step5-root", type=Path, default=default_config.step5_root)
    parser.add_argument("--step6-root", type=Path, default=default_config.step6_root)
    parser.add_argument("--step7-root", type=Path, default=default_config.step7_root)
    parser.add_argument("--output-root", type=Path, default=default_config.output_root)
    parser.add_argument("--figure-root", type=Path, default=default_config.figure_root)
    parser.add_argument("--max-workers", type=int, default=default_config.max_workers)
    parser.add_argument("--reserve-cores", type=int, default=default_config.reserve_cores)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--figures-only", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Step8Config(
        step4_root=args.step4_root,
        step5_root=args.step5_root,
        step6_root=args.step6_root,
        step7_root=args.step7_root,
        output_root=args.output_root,
        figure_root=args.figure_root,
        max_workers=args.max_workers,
        reserve_cores=args.reserve_cores,
    )
    if args.figures_only:
        if args.no_resume:
            parser.error("--no-resume cannot be used with --figures-only")
        figure_count = run_step8_figures_only(config)
        logger.info("Step 8 figures-only complete: regenerated %d figure files", figure_count)
        return
    summary = run_step8(config, no_resume=args.no_resume)
    logger.info(
        "Step 8 complete: %d fold rows, %d prediction rows, %d rolling rows",
        summary.fold_metric_rows,
        summary.prediction_rows,
        summary.rolling_fold_rows,
    )


if __name__ == "__main__":
    main()
