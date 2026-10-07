"""Step 7 dual-track classical forecasting benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyriemann.utils.distance import pairwise_distance
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - scikit-learn normally provides this dependency.
    threadpool_limits = None

from src.data.forecasting_loader import (
    ForecastingTable,
    Step7Config,
    build_primary_folds,
    build_rolling_folds,
    expected_usable_rows_per_session,
    load_primary_tables,
    table_row_dict,
    target_view_values,
)
from src.data.cv_splits import group_kfold_splits
from src.data.loader import SUBJECT_IDS, expected_session_ids
from src.evaluation.forecasting_metrics import (
    PairedStatistic,
    compute_paired_statistic,
    compute_regression_metrics,
    conformal_prediction_intervals,
    conformal_quantile,
    gaussian_prediction_intervals,
    holm_correction,
)
from src.features.step6_forecasting_task import csv_float, write_csv

logger = logging.getLogger(__name__)


PRIMARY_MODELS: tuple[str, ...] = (
    "train_mean",
    "persistence",
    "moving_average_5",
    "context_score_mean",
    "autoreg_ridge_5",
    "ridge_anchor_ts",
    "elasticnet_anchor_ts",
    "pcr_context_ts",
    "pls_context_ts",
    "ridge_score5_anchor_ts",
    "pcr_score5_context_ts",
    "pls_score5_context_ts",
    "riemannian_knn_anchor",
    "pcr_context_ts_permuted_target",
)

ROLLING_MODELS: tuple[str, ...] = (
    "persistence",
    "moving_average_5",
    "context_score_mean",
    "pcr_context_ts",
    "pcr_score5_context_ts",
)

MODEL_FAMILY: dict[str, str] = {
    "train_mean": "score_only",
    "persistence": "score_only",
    "moving_average_5": "score_only",
    "context_score_mean": "score_only",
    "autoreg_ridge_5": "score_only",
    "ridge_anchor_ts": "eeg_only",
    "elasticnet_anchor_ts": "eeg_only",
    "pcr_context_ts": "eeg_only",
    "pls_context_ts": "eeg_only",
    "ridge_score5_anchor_ts": "hybrid",
    "pcr_score5_context_ts": "hybrid",
    "pls_score5_context_ts": "hybrid",
    "riemannian_knn_anchor": "geometry_control",
    "pcr_context_ts_permuted_target": "falsification_control",
}

PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("persistence_vs_moving_average_5", "persistence", "moving_average_5"),
    ("context_score_mean_vs_persistence", "context_score_mean", "persistence"),
    ("context_score_mean_vs_ridge_score5_anchor_ts", "context_score_mean", "ridge_score5_anchor_ts"),
    ("context_score_mean_vs_pcr_context_ts", "context_score_mean", "pcr_context_ts"),
    ("context_score_mean_vs_pcr_score5_context_ts", "context_score_mean", "pcr_score5_context_ts"),
    ("pcr_context_ts_vs_riemannian_knn_anchor", "pcr_context_ts", "riemannian_knn_anchor"),
    ("pcr_context_ts_vs_pcr_context_ts_permuted_target", "pcr_context_ts", "pcr_context_ts_permuted_target"),
)

PAIRED_METRICS: tuple[str, ...] = ("mae", "gaussian_nll")

FOLD_METRICS_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "horizon",
    "target_view",
    "model_name",
    "family",
    "mae",
    "rmse",
    "r2",
    "pearson_r",
    "gaussian_nll",
    "gaussian_picp_80",
    "gaussian_picp_90",
    "conformal_picp_80",
    "conformal_picp_90",
    "mean_oof_sigma",
    "n_test_rows",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "target_view",
    "model_name",
    "family",
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
    "mean_oof_sigma_mean",
    "mean_oof_sigma_std",
)

PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "subject_id",
    "session_id",
    "horizon",
    "target_view",
    "model_name",
    "family",
    "anchor_idx",
    "target_idx",
    "y_true",
    "y_pred_mean",
    "oof_sigma",
    "gaussian_lower_80",
    "gaussian_upper_80",
    "gaussian_lower_90",
    "gaussian_upper_90",
    "conformal_lower_80",
    "conformal_upper_80",
    "conformal_lower_90",
    "conformal_upper_90",
)

MODEL_SELECTION_COLUMNS: tuple[str, ...] = (
    "fold_subject_id",
    "horizon",
    "target_view",
    "model_name",
    "selected_alpha",
    "selected_n_components",
    "selected_l1_ratio",
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
    "std_delta",
    "min_detectable_delta_80",
    "better_model",
)

ROLLING_FOLD_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "train_through_session",
    "test_session",
    "horizon",
    "target_view",
    "model_name",
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
    "model_name",
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

EXPECTED_FIGURE_FILENAMES: tuple[str, ...] = (
    "step7_raw_mae_by_horizon.png",
    "step7_raw_nll_by_horizon.png",
    "step7_residual_mae_by_horizon.png",
    "step7_residual_nll_by_horizon.png",
    "step7_context_baseline_comparison.png",
    "step7_family_comparison_raw_h25.png",
    "step7_family_comparison_residual_h25.png",
    "step7_calibration_summary.png",
    "step7_control_collapse.png",
    "step7_rolling_raw_vs_residual.png",
    "step7_rolling_first_difference_summary.png",
)


@dataclass(frozen=True)
class FeatureStandardizer:
    """Train-fold-only feature standardization contract."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_train(cls, x: np.ndarray) -> "FeatureStandardizer":
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
class ConstantPredictor:
    """Constant predictor for degenerate training targets."""

    value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        row_count = 1 if x.ndim == 1 else int(x.shape[0])
        return np.full(row_count, float(self.value), dtype=np.float64)


@dataclass(frozen=True)
class StandardizedRidgeModel:
    """Minimal ridge model with train-fold-only standardization."""

    standardizer: FeatureStandardizer
    coef: np.ndarray
    y_mean: float
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        return (self.y_mean + (z @ self.coef)).astype(np.float64)


@dataclass(frozen=True)
class StandardizedElasticNetModel:
    """ElasticNet with explicit train-fold-only standardization."""

    standardizer: FeatureStandardizer
    coef: np.ndarray
    y_mean: float
    alpha: float
    l1_ratio: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        return (self.y_mean + (z @ self.coef)).astype(np.float64)


@dataclass(frozen=True)
class StandardizedPCRModel:
    """PCR using train-fold-only standardization and PCA."""

    standardizer: FeatureStandardizer
    pca: PCA
    coef: np.ndarray
    y_mean: float
    n_components: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        scores = self.pca.transform(z)
        return (self.y_mean + (scores @ self.coef)).astype(np.float64)


@dataclass(frozen=True)
class StandardizedPLSModel:
    """PLS model using train-fold-only standardization."""

    standardizer: FeatureStandardizer
    estimator: PLSRegression | ConstantPredictor
    n_components: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        return np.asarray(self.estimator.predict(z), dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class ModelResult:
    """Predictions and selection metadata for one fitted model on one fold."""

    train_pred: np.ndarray
    test_pred: np.ndarray
    calibration_residuals: np.ndarray
    selected_alpha: float | None
    selected_n_components: int | None
    selected_l1_ratio: float | None
    used_geometry_chunking: bool = False


@dataclass(frozen=True)
class Step7Summary:
    """High-level summary returned after a successful Step 7 run."""

    output_root: Path
    figure_root: Path
    fold_metric_rows: int
    summary_rows: int
    prediction_rows: int
    model_selection_rows: int
    paired_stats_rows: int
    rolling_fold_rows: int
    rolling_summary_rows: int
    figure_count: int
    geometry_chunking_used: bool


@dataclass(frozen=True)
class _PrimaryTaskSpec:
    """One independently executable primary Step 7 task (both target views)."""

    task_index: int
    horizon: int
    fold_subject_id: str
    model_name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class _PrimaryTaskResult:
    """Artifacts emitted by one primary Step 7 task (both target views)."""

    task_index: int
    fold_metric_rows: list[dict[str, Any]]
    model_selection_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    used_geometry_chunking: bool


@dataclass(frozen=True)
class _RollingTaskSpec:
    """One independently executable rolling Step 7 task (both target views)."""

    task_index: int
    horizon: int
    subject_id: str
    train_through_session: str
    test_session: str
    model_name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class _RollingTaskResult:
    """Artifacts emitted by one rolling Step 7 task (both target views)."""

    task_index: int
    rolling_fold_rows: list[dict[str, Any]]
    used_geometry_chunking: bool


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------


def _checkpoint_dir(output_root: Path) -> Path:
    """Return the checkpoint directory path for Step 7 task results."""
    return output_root / ".step7_checkpoint"


def _primary_checkpoint_key(task: _PrimaryTaskSpec) -> str:
    """Semantic key for a primary task checkpoint file."""
    return f"primary/{task.horizon}/{task.fold_subject_id}/{task.model_name}.json"


def _rolling_checkpoint_key(task: _RollingTaskSpec) -> str:
    """Semantic key for a rolling task checkpoint file."""
    return (
        f"rolling/{task.horizon}/{task.subject_id}/"
        f"{task.train_through_session}/{task.test_session}/{task.model_name}.json"
    )


def _save_checkpoint(checkpoint_dir: Path, key: str, payload: dict[str, Any]) -> None:
    """Atomically write a checkpoint file using write-to-temp + rename."""
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
    """Load a checkpoint file, returning None if missing or corrupted."""
    path = checkpoint_dir / key
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Checkpoint payload is not a dict")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning("Corrupted checkpoint %s, will recompute: %s", key, exc)
        return None


def _primary_result_to_checkpoint(result: _PrimaryTaskResult) -> dict[str, Any]:
    """Serialize a primary task result to a JSON-safe dict."""
    return {
        "fold_metric_rows": result.fold_metric_rows,
        "model_selection_rows": result.model_selection_rows,
        "prediction_rows": result.prediction_rows,
        "used_geometry_chunking": result.used_geometry_chunking,
    }


def _primary_result_from_checkpoint(task_index: int, data: dict[str, Any]) -> _PrimaryTaskResult:
    """Reconstruct a _PrimaryTaskResult from a loaded checkpoint dict."""
    return _PrimaryTaskResult(
        task_index=task_index,
        fold_metric_rows=data["fold_metric_rows"],
        model_selection_rows=data["model_selection_rows"],
        prediction_rows=data["prediction_rows"],
        used_geometry_chunking=data["used_geometry_chunking"],
    )


def _rolling_result_to_checkpoint(result: _RollingTaskResult) -> dict[str, Any]:
    """Serialize a rolling task result to a JSON-safe dict."""
    return {
        "rolling_fold_rows": result.rolling_fold_rows,
        "used_geometry_chunking": result.used_geometry_chunking,
    }


def _rolling_result_from_checkpoint(task_index: int, data: dict[str, Any]) -> _RollingTaskResult:
    """Reconstruct a _RollingTaskResult from a loaded checkpoint dict."""
    return _RollingTaskResult(
        task_index=task_index,
        rolling_fold_rows=data["rolling_fold_rows"],
        used_geometry_chunking=data["used_geometry_chunking"],
    )


def fold_metrics_path(*, output_root: Path) -> Path:
    return output_root / "step7_fold_metrics.csv"


def summary_path(*, output_root: Path) -> Path:
    return output_root / "step7_summary.csv"


def predictions_path(*, output_root: Path) -> Path:
    return output_root / "step7_predictions.csv"


def model_selection_path(*, output_root: Path) -> Path:
    return output_root / "step7_model_selection.csv"


def paired_stats_path(*, output_root: Path) -> Path:
    return output_root / "step7_paired_stats.csv"


def rolling_fold_metrics_path(*, output_root: Path) -> Path:
    return output_root / "step7_rolling_fold_metrics.csv"


def rolling_summary_path(*, output_root: Path) -> Path:
    return output_root / "step7_rolling_summary.csv"


def figure_path(*, figure_root: Path, filename: str) -> Path:
    return figure_root / filename


def model_family(model_name: str) -> str:
    if model_name not in MODEL_FAMILY:
        raise KeyError(f"Unknown Step 7 model: {model_name}")
    return MODEL_FAMILY[model_name]


def _design_matrix(table: ForecastingTable, model_name: str, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if model_name == "autoreg_ridge_5":
        return np.asarray(table.score_history_5[indices], dtype=np.float64)
    if model_name in {"ridge_anchor_ts", "elasticnet_anchor_ts"}:
        return np.asarray(table.anchor_ts[indices], dtype=np.float64)
    if model_name in {"pcr_context_ts", "pls_context_ts", "pcr_context_ts_permuted_target"}:
        return np.asarray(table.context_ts_flat[indices], dtype=np.float64)
    if model_name == "ridge_score5_anchor_ts":
        return np.concatenate(
            [
                np.asarray(table.score_history_5[indices], dtype=np.float64),
                np.asarray(table.anchor_ts[indices], dtype=np.float64),
            ],
            axis=1,
        )
    if model_name in {"pcr_score5_context_ts", "pls_score5_context_ts"}:
        return np.concatenate(
            [
                np.asarray(table.score_history_5[indices], dtype=np.float64),
                np.asarray(table.context_ts_flat[indices], dtype=np.float64),
            ],
            axis=1,
        )
    raise ValueError(f"Model {model_name} does not use a tabular design matrix")


def _fit_standardized_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> StandardizedRidgeModel | ConstantPredictor:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y row mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] == 0:
        raise ValueError("Cannot fit ridge with zero rows")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    if float(np.std(y_centered, ddof=0)) <= 1e-12:
        return ConstantPredictor(y_mean)

    standardizer = FeatureStandardizer.from_train(x)
    z = standardizer.transform(x)
    n_rows, n_features = z.shape
    if n_features <= n_rows:
        gram = z.T @ z
        coef = np.linalg.solve(gram + alpha * np.eye(n_features, dtype=np.float64), z.T @ y_centered)
    else:
        gram_dual = z @ z.T
        dual = np.linalg.solve(gram_dual + alpha * np.eye(n_rows, dtype=np.float64), y_centered)
        coef = z.T @ dual

    return StandardizedRidgeModel(
        standardizer=standardizer,
        coef=np.asarray(coef, dtype=np.float64),
        y_mean=y_mean,
        alpha=float(alpha),
    )


def _fit_standardized_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
) -> StandardizedElasticNetModel | ConstantPredictor:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not 0.0 < l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must be in (0, 1], got {l1_ratio}")
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    if float(np.std(y_centered, ddof=0)) <= 1e-12:
        return ConstantPredictor(y_mean)

    standardizer = FeatureStandardizer.from_train(x)
    z = standardizer.transform(x)
    estimator = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=False,
        max_iter=100000,
        tol=1e-4,
        selection="cyclic",
        random_state=0,
    )
    estimator.fit(z, y_centered)
    return StandardizedElasticNetModel(
        standardizer=standardizer,
        coef=np.asarray(estimator.coef_, dtype=np.float64),
        y_mean=y_mean,
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
    )


def _fit_standardized_pcr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int,
) -> StandardizedPCRModel | ConstantPredictor:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    if float(np.std(y_centered, ddof=0)) <= 1e-12:
        return ConstantPredictor(y_mean)

    standardizer = FeatureStandardizer.from_train(x)
    z = standardizer.transform(x)
    pca = PCA(n_components=int(n_components), svd_solver="full")
    scores = pca.fit_transform(z)
    coef, *_ = np.linalg.lstsq(scores, y_centered, rcond=None)
    return StandardizedPCRModel(
        standardizer=standardizer,
        pca=pca,
        coef=np.asarray(coef, dtype=np.float64),
        y_mean=y_mean,
        n_components=int(n_components),
    )


def _fit_standardized_pls(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int,
) -> StandardizedPLSModel | ConstantPredictor:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if float(np.std(y, ddof=0)) <= 1e-12:
        return ConstantPredictor(float(np.mean(y)))

    standardizer = FeatureStandardizer.from_train(x)
    z = standardizer.transform(x)
    estimator = PLSRegression(n_components=int(n_components), scale=False, max_iter=500)
    estimator.fit(z, y.reshape(-1, 1))
    return StandardizedPLSModel(
        standardizer=standardizer,
        estimator=estimator,
        n_components=int(n_components),
    )


def _rolling_forward_splits(session_ord: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    session_ord = np.asarray(session_ord, dtype=np.int64)
    unique_sessions = sorted(int(value) for value in np.unique(session_ord).tolist())
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for current_session in unique_sessions[:-1]:
        train_idx = np.flatnonzero(session_ord <= current_session).astype(np.int64)
        val_idx = np.flatnonzero(session_ord == current_session + 1).astype(np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("Rolling forward split generated an empty train/validation split")
        splits.append((train_idx, val_idx))
    return splits


def _inner_splits(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    protocol: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol == "loso":
        train_subject_id = np.asarray(table.subject_id[train_idx], dtype=object)
        unique_subject_count = len(set(train_subject_id.tolist()))
        if unique_subject_count < 2:
            return []
        return group_kfold_splits(train_subject_id, n_splits=min(5, unique_subject_count))
    if protocol == "rolling":
        train_session_ord = np.asarray(table.session_ord[train_idx], dtype=np.int64)
        unique_session_count = len(np.unique(train_session_ord))
        if unique_session_count < 2:
            return []
        return _rolling_forward_splits(train_session_ord)
    raise ValueError(f"Unknown protocol {protocol!r}")


def _permute_targets_within_sessions(
    y: np.ndarray,
    subject_id: np.ndarray,
    session_id: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    permuted = np.array(y, copy=True)
    rng = np.random.default_rng(seed)
    unique_keys = sorted(set(zip(subject_id.tolist(), session_id.tolist())))
    for key_subject, key_session in unique_keys:
        key_mask = (subject_id == key_subject) & (session_id == key_session)
        indices = np.flatnonzero(key_mask)
        if indices.size <= 1:
            continue
        permuted[indices] = permuted[indices[rng.permutation(indices.size)]]
    return permuted


def _weighted_knn_from_distances(
    distances: np.ndarray,
    train_targets: np.ndarray,
    *,
    k: int,
    epsilon: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    train_targets = np.asarray(train_targets, dtype=np.float64)
    k_effective = min(int(k), int(train_targets.shape[0]))
    nearest_idx = np.argpartition(distances, kth=k_effective - 1, axis=1)[:, :k_effective]
    nearest_dist = np.take_along_axis(distances, nearest_idx, axis=1)
    nearest_targets = train_targets[nearest_idx]
    weights = 1.0 / (nearest_dist + float(epsilon))
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.sum(weights * nearest_targets, axis=1)


def _train_knn_predictions(train_covs: np.ndarray, y_train: np.ndarray, *, config: Step7Config) -> np.ndarray:
    if y_train.shape[0] <= 1:
        return np.asarray(y_train, dtype=np.float64)
    distances = pairwise_distance(np.asarray(train_covs, dtype=np.float64), metric="riemann")
    np.fill_diagonal(distances, np.inf)
    k_effective = min(config.geometry_k, max(1, y_train.shape[0] - 1))
    return _weighted_knn_from_distances(
        distances,
        y_train,
        k=k_effective,
        epsilon=config.geometry_weight_epsilon,
    )


def _test_knn_predictions(
    train_covs: np.ndarray,
    y_train: np.ndarray,
    test_covs: np.ndarray,
    *,
    config: Step7Config,
) -> tuple[np.ndarray, bool]:
    predictions: list[np.ndarray] = []
    used_chunking = False
    for start in range(0, test_covs.shape[0], config.geometry_batch_size):
        stop = min(start + config.geometry_batch_size, test_covs.shape[0])
        if stop - start < test_covs.shape[0]:
            used_chunking = True
        batch_distances = pairwise_distance(
            np.asarray(test_covs[start:stop], dtype=np.float64),
            np.asarray(train_covs, dtype=np.float64),
            metric="riemann",
        )
        predictions.append(
            _weighted_knn_from_distances(
                batch_distances,
                y_train,
                k=config.geometry_k,
                epsilon=config.geometry_weight_epsilon,
            ),
        )
    return np.concatenate(predictions, axis=0).astype(np.float64), used_chunking


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def _feasible_component_candidates(grid: tuple[int, ...], *, x: np.ndarray) -> list[int]:
    max_feasible = max(1, min(int(x.shape[0] - 1), int(x.shape[1])))
    candidates = [int(value) for value in grid if 1 <= int(value) <= max_feasible]
    if not candidates:
        candidates = [max_feasible]
    return sorted(set(candidates))


def _select_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: Step7Config,
) -> float:
    best_alpha = float(config.alpha_grid[0])
    best_score = float("inf")
    for alpha in config.alpha_grid:
        fold_maes: list[float] = []
        for train_local_idx, val_local_idx in splits:
            model = _fit_standardized_ridge(x_train[train_local_idx], y_train[train_local_idx], alpha=float(alpha))
            val_pred = model.predict(x_train[val_local_idx])
            fold_maes.append(_mae(y_train[val_local_idx], val_pred))
        mean_mae = float(np.mean(np.asarray(fold_maes, dtype=np.float64)))
        if mean_mae < best_score - 1e-12:
            best_score = mean_mae
            best_alpha = float(alpha)
    return best_alpha


def _select_elastic_net_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: Step7Config,
) -> tuple[float, float]:
    best_alpha = float(config.elastic_net_alpha_grid[0])
    best_l1_ratio = float(config.elastic_net_l1_ratio_grid[0])
    best_score = float("inf")
    for alpha in config.elastic_net_alpha_grid:
        for l1_ratio in config.elastic_net_l1_ratio_grid:
            fold_maes: list[float] = []
            for train_local_idx, val_local_idx in splits:
                model = _fit_standardized_elastic_net(
                    x_train[train_local_idx],
                    y_train[train_local_idx],
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                )
                val_pred = model.predict(x_train[val_local_idx])
                fold_maes.append(_mae(y_train[val_local_idx], val_pred))
            mean_mae = float(np.mean(np.asarray(fold_maes, dtype=np.float64)))
            if mean_mae < best_score - 1e-12:
                best_score = mean_mae
                best_alpha = float(alpha)
                best_l1_ratio = float(l1_ratio)
    return best_alpha, best_l1_ratio


def _select_component_count(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[int],
    fit_fn: Any,
) -> int:
    best_n_components = int(candidates[0])
    best_score = float("inf")
    for n_components in candidates:
        fold_maes: list[float] = []
        for train_local_idx, val_local_idx in splits:
            model = fit_fn(x_train[train_local_idx], y_train[train_local_idx], n_components=int(n_components))
            val_pred = model.predict(x_train[val_local_idx])
            fold_maes.append(_mae(y_train[val_local_idx], val_pred))
        mean_mae = float(np.mean(np.asarray(fold_maes, dtype=np.float64)))
        if mean_mae < best_score - 1e-12:
            best_score = mean_mae
            best_n_components = int(n_components)
    return best_n_components


def _oof_residuals_from_splits(
    *,
    x_train: np.ndarray,
    fit_target_train: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fit_predict_fn: Any,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    for train_local_idx, val_local_idx in splits:
        val_pred = fit_predict_fn(
            x_train[train_local_idx],
            fit_target_train[train_local_idx],
            x_train[val_local_idx],
        )
        residuals.append(np.asarray(fit_target_train[val_local_idx] - val_pred, dtype=np.float64))
    if not residuals:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(residuals, axis=0).astype(np.float64)


def _fit_ridge_family(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    target_view: str,
    config: Step7Config,
    protocol: str,
) -> ModelResult:
    x_train = _design_matrix(table, model_name, train_idx)
    x_test = _design_matrix(table, model_name, test_idx)
    y_train = np.asarray(target_view_values(table, target_view=target_view)[train_idx], dtype=np.float64)
    splits = _inner_splits(table=table, train_idx=train_idx, protocol=protocol)
    if splits:
        selected_alpha = _select_ridge_alpha(x_train, y_train, splits=splits, config=config)
    else:
        selected_alpha = float(config.fallback_ridge_alpha)

    final_model = _fit_standardized_ridge(x_train, y_train, alpha=selected_alpha)
    train_pred = final_model.predict(x_train)
    test_pred = final_model.predict(x_test)
    calibration_residuals = _oof_residuals_from_splits(
        x_train=x_train,
        fit_target_train=y_train,
        splits=splits,
        fit_predict_fn=lambda local_x, local_y, x_val: _fit_standardized_ridge(
            local_x,
            local_y,
            alpha=selected_alpha,
        ).predict(x_val),
    )
    if calibration_residuals.size == 0:
        calibration_residuals = np.asarray(y_train - train_pred, dtype=np.float64)
    return ModelResult(
        train_pred=train_pred,
        test_pred=test_pred,
        calibration_residuals=calibration_residuals,
        selected_alpha=selected_alpha,
        selected_n_components=None,
        selected_l1_ratio=None,
    )


def _fit_elastic_net_family(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_view: str,
    config: Step7Config,
    protocol: str,
) -> ModelResult:
    x_train = _design_matrix(table, "elasticnet_anchor_ts", train_idx)
    x_test = _design_matrix(table, "elasticnet_anchor_ts", test_idx)
    y_train = np.asarray(target_view_values(table, target_view=target_view)[train_idx], dtype=np.float64)
    splits = _inner_splits(table=table, train_idx=train_idx, protocol=protocol)
    if splits:
        selected_alpha, selected_l1_ratio = _select_elastic_net_params(
            x_train,
            y_train,
            splits=splits,
            config=config,
        )
    else:
        selected_alpha = float(config.fallback_elastic_net_alpha)
        selected_l1_ratio = float(config.fallback_elastic_net_l1_ratio)

    final_model = _fit_standardized_elastic_net(
        x_train,
        y_train,
        alpha=selected_alpha,
        l1_ratio=selected_l1_ratio,
    )
    train_pred = final_model.predict(x_train)
    test_pred = final_model.predict(x_test)
    calibration_residuals = _oof_residuals_from_splits(
        x_train=x_train,
        fit_target_train=y_train,
        splits=splits,
        fit_predict_fn=lambda local_x, local_y, x_val: _fit_standardized_elastic_net(
            local_x,
            local_y,
            alpha=selected_alpha,
            l1_ratio=selected_l1_ratio,
        ).predict(x_val),
    )
    if calibration_residuals.size == 0:
        calibration_residuals = np.asarray(y_train - train_pred, dtype=np.float64)
    return ModelResult(
        train_pred=train_pred,
        test_pred=test_pred,
        calibration_residuals=calibration_residuals,
        selected_alpha=selected_alpha,
        selected_n_components=None,
        selected_l1_ratio=selected_l1_ratio,
    )


def _fit_pcr_family(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    target_view: str,
    config: Step7Config,
    protocol: str,
) -> ModelResult:
    x_train = _design_matrix(table, model_name, train_idx)
    x_test = _design_matrix(table, model_name, test_idx)
    y_train = np.asarray(target_view_values(table, target_view=target_view)[train_idx], dtype=np.float64)
    if model_name == "pcr_context_ts_permuted_target":
        fit_target_train = _permute_targets_within_sessions(
            y_train,
            np.asarray(table.subject_id[train_idx], dtype=object),
            np.asarray(table.session_id[train_idx], dtype=object),
            seed=config.permute_seed,
        )
    else:
        fit_target_train = y_train

    splits = _inner_splits(table=table, train_idx=train_idx, protocol=protocol)
    candidates = _feasible_component_candidates(config.pcr_component_grid, x=x_train)
    if splits:
        selected_n_components = _select_component_count(
            x_train,
            fit_target_train,
            splits=splits,
            candidates=candidates,
            fit_fn=lambda local_x, local_y, n_components: _fit_standardized_pcr(
                local_x,
                local_y,
                n_components=n_components,
            ),
        )
    else:
        max_feasible = max(1, min(int(x_train.shape[0] - 1), int(x_train.shape[1])))
        selected_n_components = int(min(config.fallback_component_count, max_feasible))

    final_model = _fit_standardized_pcr(x_train, fit_target_train, n_components=selected_n_components)
    train_pred = final_model.predict(x_train)
    test_pred = final_model.predict(x_test)
    calibration_residuals = _oof_residuals_from_splits(
        x_train=x_train,
        fit_target_train=fit_target_train,
        splits=splits,
        fit_predict_fn=lambda local_x, local_y, x_val: _fit_standardized_pcr(
            local_x,
            local_y,
            n_components=selected_n_components,
        ).predict(x_val),
    )
    if calibration_residuals.size == 0:
        calibration_residuals = np.asarray(fit_target_train - train_pred, dtype=np.float64)
    return ModelResult(
        train_pred=train_pred,
        test_pred=test_pred,
        calibration_residuals=calibration_residuals,
        selected_alpha=None,
        selected_n_components=selected_n_components,
        selected_l1_ratio=None,
    )


def _fit_pls_family(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    target_view: str,
    config: Step7Config,
    protocol: str,
) -> ModelResult:
    x_train = _design_matrix(table, model_name, train_idx)
    x_test = _design_matrix(table, model_name, test_idx)
    y_train = np.asarray(target_view_values(table, target_view=target_view)[train_idx], dtype=np.float64)
    splits = _inner_splits(table=table, train_idx=train_idx, protocol=protocol)
    candidates = _feasible_component_candidates(config.pls_component_grid, x=x_train)
    if splits:
        selected_n_components = _select_component_count(
            x_train,
            y_train,
            splits=splits,
            candidates=candidates,
            fit_fn=lambda local_x, local_y, n_components: _fit_standardized_pls(
                local_x,
                local_y,
                n_components=n_components,
            ),
        )
    else:
        max_feasible = max(1, min(int(x_train.shape[0] - 1), int(x_train.shape[1])))
        selected_n_components = int(min(config.fallback_component_count, max_feasible))

    final_model = _fit_standardized_pls(x_train, y_train, n_components=selected_n_components)
    train_pred = final_model.predict(x_train)
    test_pred = final_model.predict(x_test)
    calibration_residuals = _oof_residuals_from_splits(
        x_train=x_train,
        fit_target_train=y_train,
        splits=splits,
        fit_predict_fn=lambda local_x, local_y, x_val: _fit_standardized_pls(
            local_x,
            local_y,
            n_components=selected_n_components,
        ).predict(x_val),
    )
    if calibration_residuals.size == 0:
        calibration_residuals = np.asarray(y_train - train_pred, dtype=np.float64)
    return ModelResult(
        train_pred=train_pred,
        test_pred=test_pred,
        calibration_residuals=calibration_residuals,
        selected_alpha=None,
        selected_n_components=selected_n_components,
        selected_l1_ratio=None,
    )


def _exact_baseline_prediction(
    *,
    table: ForecastingTable,
    indices: np.ndarray,
    model_name: str,
    target_view: str,
) -> np.ndarray | None:
    if model_name == "persistence":
        prediction = np.asarray(table.anchor_score_raw[indices], dtype=np.float64)
        if target_view == "context_residual":
            prediction = prediction - np.asarray(table.context_score_mean_value[indices], dtype=np.float64)
        return prediction
    if model_name == "moving_average_5":
        prediction = np.mean(np.asarray(table.score_history_5[indices], dtype=np.float64), axis=1)
        if target_view == "context_residual":
            prediction = prediction - np.asarray(table.context_score_mean_value[indices], dtype=np.float64)
        return prediction
    if model_name == "context_score_mean":
        if target_view == "raw_level":
            return np.asarray(table.context_score_mean_value[indices], dtype=np.float64)
        return np.zeros(int(np.asarray(indices).shape[0]), dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Multi-view helpers: share view-independent work across target views
# ---------------------------------------------------------------------------


def _knn_predictions_from_distances(
    train_distances: np.ndarray,
    test_batch_distances: list[np.ndarray],
    y_train: np.ndarray,
    *,
    config: Step7Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted kNN predictions from pre-computed distance matrices."""
    if y_train.shape[0] <= 1:
        train_pred = np.asarray(y_train, dtype=np.float64)
    else:
        k_train = min(config.geometry_k, max(1, y_train.shape[0] - 1))
        train_pred = _weighted_knn_from_distances(
            train_distances, y_train, k=k_train, epsilon=config.geometry_weight_epsilon,
        )
    test_parts = [
        _weighted_knn_from_distances(
            bd, y_train, k=config.geometry_k, epsilon=config.geometry_weight_epsilon,
        )
        for bd in test_batch_distances
    ]
    test_pred = np.concatenate(test_parts, axis=0).astype(np.float64)
    return train_pred, test_pred


def _fit_knn_both_views(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_views: tuple[str, ...],
    config: Step7Config,
) -> dict[str, ModelResult]:
    """Riemannian kNN sharing distance matrices across target views."""
    train_covs = np.asarray(table.anchor_aligned_cov[train_idx], dtype=np.float64)
    test_covs = np.asarray(table.anchor_aligned_cov[test_idx], dtype=np.float64)

    # Compute view-independent distance matrices once.
    if train_covs.shape[0] <= 1:
        train_distances = np.zeros((train_covs.shape[0], train_covs.shape[0]), dtype=np.float64)
    else:
        train_distances = pairwise_distance(train_covs, metric="riemann")
        np.fill_diagonal(train_distances, np.inf)

    test_batch_distances: list[np.ndarray] = []
    used_chunking = False
    for start in range(0, test_covs.shape[0], config.geometry_batch_size):
        stop = min(start + config.geometry_batch_size, test_covs.shape[0])
        if stop - start < test_covs.shape[0]:
            used_chunking = True
        test_batch_distances.append(
            pairwise_distance(
                np.asarray(test_covs[start:stop], dtype=np.float64),
                train_covs,
                metric="riemann",
            )
        )

    results: dict[str, ModelResult] = {}
    for tv in target_views:
        y_train = np.asarray(target_view_values(table, target_view=tv)[train_idx], dtype=np.float64)
        train_pred, test_pred = _knn_predictions_from_distances(
            train_distances, test_batch_distances, y_train, config=config,
        )
        results[tv] = ModelResult(
            train_pred=train_pred,
            test_pred=test_pred,
            calibration_residuals=np.asarray(y_train - train_pred, dtype=np.float64),
            selected_alpha=None,
            selected_n_components=None,
            selected_l1_ratio=None,
            used_geometry_chunking=used_chunking,
        )
    return results


def _select_pcr_component_count_shared(
    x_train: np.ndarray,
    fit_targets: dict[str, np.ndarray],
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[int],
) -> dict[str, int]:
    """Select PCR component count, sharing PCA across target views.

    For each (n_components, inner fold split) the PCA is fit once; the cheap
    regression step is repeated per target view.
    """
    # Cap candidates to the smallest inner-fold training size so PCA never
    # requests more components than available rows/features.
    min_inner = min(
        min(int(train_local_idx.shape[0]), int(x_train.shape[1]))
        for train_local_idx, _ in splits
    )
    candidates = [c for c in candidates if c <= min_inner]
    if not candidates:
        candidates = [max(1, min_inner)]

    best: dict[str, tuple[int, float]] = {
        tv: (int(candidates[0]), float("inf")) for tv in fit_targets
    }
    for n_components in candidates:
        fold_maes: dict[str, list[float]] = {tv: [] for tv in fit_targets}
        for train_local_idx, val_local_idx in splits:
            local_x = x_train[train_local_idx]
            val_x = x_train[val_local_idx]
            # Shared PCA fit (view-independent).
            standardizer = FeatureStandardizer.from_train(local_x)
            z_train_local = standardizer.transform(local_x)
            z_val = standardizer.transform(val_x)
            pca = PCA(n_components=int(n_components), svd_solver="full")
            scores_train = pca.fit_transform(z_train_local)
            scores_val = pca.transform(z_val)
            # Per-view regression (cheap).
            for tv, y_fit in fit_targets.items():
                y_local = y_fit[train_local_idx]
                y_mean = float(np.mean(y_local))
                y_centered = y_local - y_mean
                if float(np.std(y_centered, ddof=0)) <= 1e-12:
                    val_pred = np.full(val_x.shape[0], y_mean, dtype=np.float64)
                else:
                    coef, *_ = np.linalg.lstsq(scores_train, y_centered, rcond=None)
                    val_pred = scores_val @ coef + y_mean
                fold_maes[tv].append(_mae(y_fit[val_local_idx], val_pred))
        for tv in fit_targets:
            mean_mae = float(np.mean(np.asarray(fold_maes[tv], dtype=np.float64)))
            if mean_mae < best[tv][1] - 1e-12:
                best[tv] = (int(n_components), mean_mae)
    return {tv: nc for tv, (nc, _) in best.items()}


def _fit_pcr_both_views(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    target_views: tuple[str, ...],
    config: Step7Config,
    protocol: str,
) -> dict[str, ModelResult]:
    """PCR models sharing PCA computation across target views."""
    x_train = _design_matrix(table, model_name, train_idx)
    x_test = _design_matrix(table, model_name, test_idx)
    splits = _inner_splits(table=table, train_idx=train_idx, protocol=protocol)
    candidates = _feasible_component_candidates(config.pcr_component_grid, x=x_train)

    # Build per-view fit targets (includes permuted-target handling).
    fit_targets: dict[str, np.ndarray] = {}
    y_targets: dict[str, np.ndarray] = {}
    for tv in target_views:
        y_train = np.asarray(target_view_values(table, target_view=tv)[train_idx], dtype=np.float64)
        y_targets[tv] = y_train
        if model_name == "pcr_context_ts_permuted_target":
            fit_targets[tv] = _permute_targets_within_sessions(
                y_train,
                np.asarray(table.subject_id[train_idx], dtype=object),
                np.asarray(table.session_id[train_idx], dtype=object),
                seed=config.permute_seed,
            )
        else:
            fit_targets[tv] = y_train

    # Select component count via shared PCA inner CV.
    if splits:
        selected_ncs = _select_pcr_component_count_shared(
            x_train, fit_targets, splits=splits, candidates=candidates,
        )
    else:
        max_feasible = max(1, min(int(x_train.shape[0] - 1), int(x_train.shape[1])))
        fallback_nc = int(min(config.fallback_component_count, max_feasible))
        selected_ncs = {tv: fallback_nc for tv in target_views}

    # Group views by selected component count to share final PCA fit.
    nc_groups: dict[int, list[str]] = {}
    for tv, nc in selected_ncs.items():
        nc_groups.setdefault(nc, []).append(tv)

    results: dict[str, ModelResult] = {}
    for nc, views_in_group in nc_groups.items():
        # Shared PCA final fit (view-independent).
        standardizer = FeatureStandardizer.from_train(x_train)
        z_train = standardizer.transform(x_train)
        z_test = standardizer.transform(x_test)
        pca = PCA(n_components=int(nc), svd_solver="full")
        scores_train = pca.fit_transform(z_train)
        scores_test = pca.transform(z_test)

        for tv in views_in_group:
            y_fit = fit_targets[tv]
            y_mean = float(np.mean(y_fit))
            y_centered = y_fit - y_mean
            if float(np.std(y_centered, ddof=0)) <= 1e-12:
                train_pred = np.full(train_idx.size, y_mean, dtype=np.float64)
                test_pred = np.full(test_idx.size, y_mean, dtype=np.float64)
            else:
                coef, *_ = np.linalg.lstsq(scores_train, y_centered, rcond=None)
                train_pred = np.asarray(scores_train @ coef + y_mean, dtype=np.float64)
                test_pred = np.asarray(scores_test @ coef + y_mean, dtype=np.float64)

            # OOF calibration residuals (PCA shared per split).
            if splits:
                oof_parts: list[np.ndarray] = []
                for train_local_idx, val_local_idx in splits:
                    local_std = FeatureStandardizer.from_train(x_train[train_local_idx])
                    z_local = local_std.transform(x_train[train_local_idx])
                    z_val = local_std.transform(x_train[val_local_idx])
                    local_pca = PCA(n_components=int(nc), svd_solver="full")
                    sc_local = local_pca.fit_transform(z_local)
                    sc_val = local_pca.transform(z_val)
                    y_local = y_fit[train_local_idx]
                    ym = float(np.mean(y_local))
                    yc = y_local - ym
                    if float(np.std(yc, ddof=0)) <= 1e-12:
                        val_pred = np.full(val_local_idx.size, ym, dtype=np.float64)
                    else:
                        c, *_ = np.linalg.lstsq(sc_local, yc, rcond=None)
                        val_pred = sc_val @ c + ym
                    oof_parts.append(np.asarray(y_fit[val_local_idx] - val_pred, dtype=np.float64))
                calibration_residuals = np.concatenate(oof_parts, axis=0).astype(np.float64)
            else:
                calibration_residuals = np.asarray(y_fit - train_pred, dtype=np.float64)

            results[tv] = ModelResult(
                train_pred=train_pred,
                test_pred=test_pred,
                calibration_residuals=calibration_residuals,
                selected_alpha=None,
                selected_n_components=nc,
                selected_l1_ratio=None,
            )
    return results


def fit_and_predict_both_views(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    target_views: tuple[str, ...],
    config: Step7Config,
    protocol: str,
) -> dict[str, ModelResult]:
    """Fit one model for all target views, sharing view-independent work.

    PCR models share PCA fits; kNN shares distance matrices.  Other models
    call the single-view ``fit_and_predict_model`` per view.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError(f"{model_name}/{protocol} received zero train rows")
    if test_idx.size == 0:
        raise ValueError(f"{model_name}/{protocol} received zero test rows")

    if model_name in {"pcr_context_ts", "pcr_score5_context_ts", "pcr_context_ts_permuted_target"}:
        return _fit_pcr_both_views(
            table=table, train_idx=train_idx, test_idx=test_idx,
            model_name=model_name, target_views=target_views,
            config=config, protocol=protocol,
        )

    if model_name == "riemannian_knn_anchor":
        return _fit_knn_both_views(
            table=table, train_idx=train_idx, test_idx=test_idx,
            target_views=target_views, config=config,
        )

    # All other models: call single-view path per view.
    return {
        tv: fit_and_predict_model(
            table=table, train_idx=train_idx, test_idx=test_idx,
            model_name=model_name, target_view=tv, config=config, protocol=protocol,
        )
        for tv in target_views
    }


def fit_and_predict_model(
    *,
    table: ForecastingTable,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    target_view: str,
    config: Step7Config,
    protocol: str,
) -> ModelResult:
    """Fit one Step 7 model on one fold and return train/test predictions."""

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError(f"{model_name}/{protocol}/{target_view} received zero train rows")
    if test_idx.size == 0:
        raise ValueError(f"{model_name}/{protocol}/{target_view} received zero test rows")

    y_train = np.asarray(target_view_values(table, target_view=target_view)[train_idx], dtype=np.float64)

    if model_name == "train_mean":
        mean_value = float(np.mean(y_train))
        train_pred = np.full(train_idx.size, mean_value, dtype=np.float64)
        test_pred = np.full(test_idx.size, mean_value, dtype=np.float64)
        return ModelResult(
            train_pred=train_pred,
            test_pred=test_pred,
            calibration_residuals=np.asarray(y_train - train_pred, dtype=np.float64),
            selected_alpha=None,
            selected_n_components=None,
            selected_l1_ratio=None,
        )

    exact_train = _exact_baseline_prediction(table=table, indices=train_idx, model_name=model_name, target_view=target_view)
    exact_test = _exact_baseline_prediction(table=table, indices=test_idx, model_name=model_name, target_view=target_view)
    if exact_train is not None and exact_test is not None:
        return ModelResult(
            train_pred=exact_train,
            test_pred=exact_test,
            calibration_residuals=np.asarray(y_train - exact_train, dtype=np.float64),
            selected_alpha=None,
            selected_n_components=None,
            selected_l1_ratio=None,
        )

    if model_name == "riemannian_knn_anchor":
        train_covs = np.asarray(table.anchor_aligned_cov[train_idx], dtype=np.float64)
        test_covs = np.asarray(table.anchor_aligned_cov[test_idx], dtype=np.float64)
        train_pred = _train_knn_predictions(train_covs, y_train, config=config)
        test_pred, used_chunking = _test_knn_predictions(train_covs, y_train, test_covs, config=config)
        return ModelResult(
            train_pred=train_pred,
            test_pred=test_pred,
            calibration_residuals=np.asarray(y_train - train_pred, dtype=np.float64),
            selected_alpha=None,
            selected_n_components=None,
            selected_l1_ratio=None,
            used_geometry_chunking=used_chunking,
        )

    if model_name in {"autoreg_ridge_5", "ridge_anchor_ts", "ridge_score5_anchor_ts"}:
        return _fit_ridge_family(
            table=table,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name=model_name,
            target_view=target_view,
            config=config,
            protocol=protocol,
        )

    if model_name == "elasticnet_anchor_ts":
        return _fit_elastic_net_family(
            table=table,
            train_idx=train_idx,
            test_idx=test_idx,
            target_view=target_view,
            config=config,
            protocol=protocol,
        )

    if model_name in {"pcr_context_ts", "pcr_score5_context_ts", "pcr_context_ts_permuted_target"}:
        return _fit_pcr_family(
            table=table,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name=model_name,
            target_view=target_view,
            config=config,
            protocol=protocol,
        )

    if model_name in {"pls_context_ts", "pls_score5_context_ts"}:
        return _fit_pls_family(
            table=table,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name=model_name,
            target_view=target_view,
            config=config,
            protocol=protocol,
        )

    raise ValueError(f"Unsupported Step 7 model {model_name}")


def _calibration_statistics(
    *,
    calibration_residuals: np.ndarray,
    sigma_floor: float,
) -> tuple[float, float, float]:
    calibration_residuals = np.asarray(calibration_residuals, dtype=np.float64)
    if calibration_residuals.ndim != 1:
        raise ValueError(f"Calibration residuals must be one-dimensional, got {calibration_residuals.shape}")
    if calibration_residuals.size == 0:
        raise ValueError("Calibration residuals must not be empty")
    if not np.all(np.isfinite(calibration_residuals)):
        raise ValueError("Calibration residuals contain non-finite values")
    oof_sigma = float(max(np.std(calibration_residuals, ddof=0), sigma_floor))
    abs_residuals = np.abs(calibration_residuals)
    q80 = float(conformal_quantile(abs_residuals, coverage=0.80))
    q90 = float(conformal_quantile(abs_residuals, coverage=0.90))
    return oof_sigma, q80, q90


def _validate_exact_baselines(
    *,
    table: ForecastingTable,
    test_idx: np.ndarray,
    model_name: str,
    target_view: str,
    test_pred: np.ndarray,
) -> None:
    expected = _exact_baseline_prediction(
        table=table,
        indices=test_idx,
        model_name=model_name,
        target_view=target_view,
    )
    if expected is None:
        return
    if not np.allclose(np.asarray(test_pred, dtype=np.float64), expected, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"{model_name}/{target_view} predictions do not match the exact Step 7 reconstruction contract",
        )


def _summarize_primary(fold_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in fold_metric_rows}):
        for target_view in sorted({str(row["target_view"]) for row in fold_metric_rows}):
            for model_name in PRIMARY_MODELS:
                model_rows = [
                    row
                    for row in fold_metric_rows
                    if int(row["horizon"]) == horizon
                    and str(row["target_view"]) == target_view
                    and str(row["model_name"]) == model_name
                ]
                if not model_rows:
                    raise ValueError(
                        f"Missing primary summary rows for horizon={horizon}, target_view={target_view}, model={model_name}",
                    )
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "target_view": target_view,
                        "model_name": model_name,
                        "family": model_family(model_name),
                        "n_folds": len(model_rows),
                        "mae_mean": float(np.mean([float(row["mae"]) for row in model_rows])),
                        "mae_std": float(np.std([float(row["mae"]) for row in model_rows], ddof=0)),
                        "gaussian_nll_mean": float(np.mean([float(row["gaussian_nll"]) for row in model_rows])),
                        "gaussian_nll_std": float(np.std([float(row["gaussian_nll"]) for row in model_rows], ddof=0)),
                        "rmse_mean": float(np.mean([float(row["rmse"]) for row in model_rows])),
                        "rmse_std": float(np.std([float(row["rmse"]) for row in model_rows], ddof=0)),
                        "r2_mean": float(np.mean([float(row["r2"]) for row in model_rows])),
                        "r2_std": float(np.std([float(row["r2"]) for row in model_rows], ddof=0)),
                        "pearson_r_mean": float(np.mean([float(row["pearson_r"]) for row in model_rows])),
                        "pearson_r_std": float(np.std([float(row["pearson_r"]) for row in model_rows], ddof=0)),
                        "gaussian_picp_80_mean": float(np.mean([float(row["gaussian_picp_80"]) for row in model_rows])),
                        "gaussian_picp_80_std": float(
                            np.std([float(row["gaussian_picp_80"]) for row in model_rows], ddof=0),
                        ),
                        "gaussian_picp_90_mean": float(np.mean([float(row["gaussian_picp_90"]) for row in model_rows])),
                        "gaussian_picp_90_std": float(
                            np.std([float(row["gaussian_picp_90"]) for row in model_rows], ddof=0),
                        ),
                        "conformal_picp_80_mean": float(
                            np.mean([float(row["conformal_picp_80"]) for row in model_rows]),
                        ),
                        "conformal_picp_80_std": float(
                            np.std([float(row["conformal_picp_80"]) for row in model_rows], ddof=0),
                        ),
                        "conformal_picp_90_mean": float(
                            np.mean([float(row["conformal_picp_90"]) for row in model_rows]),
                        ),
                        "conformal_picp_90_std": float(
                            np.std([float(row["conformal_picp_90"]) for row in model_rows], ddof=0),
                        ),
                        "mean_oof_sigma_mean": float(np.mean([float(row["mean_oof_sigma"]) for row in model_rows])),
                        "mean_oof_sigma_std": float(
                            np.std([float(row["mean_oof_sigma"]) for row in model_rows], ddof=0),
                        ),
                    },
                )
    return summary_rows


def _summarize_rolling(rolling_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for horizon in sorted({int(row["horizon"]) for row in rolling_rows}):
        for target_view in sorted({str(row["target_view"]) for row in rolling_rows}):
            for model_name in ROLLING_MODELS:
                model_rows = [
                    row
                    for row in rolling_rows
                    if int(row["horizon"]) == horizon
                    and str(row["target_view"]) == target_view
                    and str(row["model_name"]) == model_name
                ]
                if not model_rows:
                    raise ValueError(
                        f"Missing rolling summary rows for horizon={horizon}, target_view={target_view}, model={model_name}",
                    )
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "target_view": target_view,
                        "model_name": model_name,
                        "n_folds": len(model_rows),
                        "mae_mean": float(np.mean([float(row["mae"]) for row in model_rows])),
                        "mae_std": float(np.std([float(row["mae"]) for row in model_rows], ddof=0)),
                        "gaussian_nll_mean": float(np.mean([float(row["gaussian_nll"]) for row in model_rows])),
                        "gaussian_nll_std": float(np.std([float(row["gaussian_nll"]) for row in model_rows], ddof=0)),
                        "rmse_mean": float(np.mean([float(row["rmse"]) for row in model_rows])),
                        "rmse_std": float(np.std([float(row["rmse"]) for row in model_rows], ddof=0)),
                        "pearson_r_mean": float(np.mean([float(row["pearson_r"]) for row in model_rows])),
                        "pearson_r_std": float(np.std([float(row["pearson_r"]) for row in model_rows], ddof=0)),
                        "first_difference_pearson_r_mean": float(
                            np.mean([float(row["first_difference_pearson_r"]) for row in model_rows]),
                        ),
                        "first_difference_pearson_r_std": float(
                            np.std([float(row["first_difference_pearson_r"]) for row in model_rows], ddof=0),
                        ),
                    },
                )
    return summary_rows


def _paired_stats_rows(
    *,
    fold_metric_rows: list[dict[str, Any]],
    config: Step7Config,
) -> list[dict[str, Any]]:
    row_lookup: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in fold_metric_rows:
        key = (
            int(row["horizon"]),
            str(row["target_view"]),
            str(row["model_name"]),
            str(row["fold_subject_id"]),
        )
        row_lookup[key] = row

    paired_results: list[PairedStatistic] = []
    target_views = sorted({str(row["target_view"]) for row in fold_metric_rows})
    for horizon in sorted({int(row["horizon"]) for row in fold_metric_rows}):
        for target_view in target_views:
            for metric_name in PAIRED_METRICS:
                for comparison_id, model_a, model_b in PRIMARY_COMPARISONS:
                    values_a: list[float] = []
                    values_b: list[float] = []
                    for subject_id in SUBJECT_IDS:
                        key_a = (horizon, target_view, model_a, subject_id)
                        key_b = (horizon, target_view, model_b, subject_id)
                        if key_a not in row_lookup or key_b not in row_lookup:
                            raise ValueError(
                                "Missing paired-stat fold row for "
                                f"horizon={horizon}, target_view={target_view}, "
                                f"comparison={comparison_id}, subject={subject_id}",
                            )
                        values_a.append(float(row_lookup[key_a][metric_name]))
                        values_b.append(float(row_lookup[key_b][metric_name]))
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
                    if result.horizon == horizon
                    and result.target_view == target_view
                    and result.metric == metric_name
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
                "std_delta": result.std_delta,
                "min_detectable_delta_80": result.min_detectable_delta_80,
                "better_model": result.better_model,
            },
        )
        family_position[key] = offset + 1
    return finalized_rows


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _summary_value(
    summary_rows: list[dict[str, Any]],
    *,
    target_view: str,
    model_name: str,
    horizon: int,
    field: str,
) -> float:
    for row in summary_rows:
        if (
            str(row["target_view"]) == target_view
            and str(row["model_name"]) == model_name
            and int(row["horizon"]) == horizon
        ):
            return float(row[field])
    raise KeyError(f"Missing summary value for target_view={target_view}, model={model_name}, horizon={horizon}, field={field}")


def _rolling_summary_value(
    rolling_summary_rows: list[dict[str, Any]],
    *,
    target_view: str,
    model_name: str,
    horizon: int,
    field: str,
) -> float:
    for row in rolling_summary_rows:
        if (
            str(row["target_view"]) == target_view
            and str(row["model_name"]) == model_name
            and int(row["horizon"]) == horizon
        ):
            return float(row[field])
    raise KeyError(
        f"Missing rolling summary value for target_view={target_view}, model={model_name}, horizon={horizon}, field={field}",
    )


def save_figures(
    *,
    summary_rows: list[dict[str, Any]],
    rolling_summary_rows: list[dict[str, Any]],
    figure_root: Path,
) -> None:
    """Save the required Step 7 figure set."""

    horizons = sorted({int(row["horizon"]) for row in summary_rows})
    rolling_horizons = sorted({int(row["horizon"]) for row in rolling_summary_rows})

    def _line_figure(target_view: str, field: str, title: str, filename: str, ylabel: str) -> None:
        plt.figure(figsize=(11, 6))
        for model_name in PRIMARY_MODELS:
            plt.plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field=field,
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
        plt.xlabel("Horizon")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(fontsize=7, ncol=2)
        _save_figure(figure_path(figure_root=figure_root, filename=filename))

    _line_figure("raw_level", "mae_mean", "Step 7 Raw-Level MAE by Horizon", "step7_raw_mae_by_horizon.png", "MAE")
    _line_figure(
        "raw_level",
        "gaussian_nll_mean",
        "Step 7 Raw-Level Gaussian NLL by Horizon",
        "step7_raw_nll_by_horizon.png",
        "Gaussian NLL",
    )
    _line_figure(
        "context_residual",
        "mae_mean",
        "Step 7 Residual MAE by Horizon",
        "step7_residual_mae_by_horizon.png",
        "MAE",
    )
    _line_figure(
        "context_residual",
        "gaussian_nll_mean",
        "Step 7 Residual Gaussian NLL by Horizon",
        "step7_residual_nll_by_horizon.png",
        "Gaussian NLL",
    )

    comparison_models = ("context_score_mean", "persistence", "ridge_score5_anchor_ts", "pcr_score5_context_ts")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for row_index, target_view in enumerate(("raw_level", "context_residual")):
        for model_name in comparison_models:
            axes[row_index, 0].plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field="mae_mean",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
            axes[row_index, 1].plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field="gaussian_nll_mean",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
        axes[row_index, 0].set_ylabel(f"{target_view}\nmetric")
        axes[row_index, 0].set_title("MAE")
        axes[row_index, 1].set_title("Gaussian NLL")
    for axis in axes.reshape(-1):
        axis.set_xlabel("Horizon")
        axis.legend(fontsize=8)
    fig.suptitle("Context Baseline Comparison")
    _save_figure(figure_path(figure_root=figure_root, filename="step7_context_baseline_comparison.png"))

    def _family_bar(target_view: str, filename: str, title: str) -> None:
        horizon_rows = [row for row in summary_rows if int(row["horizon"]) == 25 and str(row["target_view"]) == target_view]
        horizon_rows.sort(key=lambda row: (str(row["family"]), float(row["mae_mean"]), str(row["model_name"])))
        plt.figure(figsize=(12, 6))
        plt.bar(
            [str(row["model_name"]) for row in horizon_rows],
            [float(row["mae_mean"]) for row in horizon_rows],
            color=[
                {
                    "score_only": "#1f77b4",
                    "eeg_only": "#ff7f0e",
                    "hybrid": "#2ca02c",
                    "geometry_control": "#d62728",
                    "falsification_control": "#9467bd",
                }[str(row["family"])]
                for row in horizon_rows
            ],
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("MAE")
        plt.title(title)
        _save_figure(figure_path(figure_root=figure_root, filename=filename))

    _family_bar("raw_level", "step7_family_comparison_raw_h25.png", "Step 7 Family Comparison at h=25 (Raw Level)")
    _family_bar(
        "context_residual",
        "step7_family_comparison_residual_h25.png",
        "Step 7 Family Comparison at h=25 (Context Residual)",
    )

    coverage_models = ("context_score_mean", "pcr_context_ts", "pcr_score5_context_ts", "pcr_context_ts_permuted_target")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for row_index, target_view in enumerate(("raw_level", "context_residual")):
        for model_name in coverage_models:
            axes[row_index, 0].plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field="gaussian_picp_80_mean",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
            axes[row_index, 1].plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field="conformal_picp_80_mean",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
        axes[row_index, 0].axhline(0.8, color="black", linestyle="--", linewidth=1.0)
        axes[row_index, 1].axhline(0.8, color="black", linestyle="--", linewidth=1.0)
        axes[row_index, 0].set_ylabel(target_view)
        axes[row_index, 0].set_title("Gaussian PICP 80")
        axes[row_index, 1].set_title("Conformal PICP 80")
    for axis in axes.reshape(-1):
        axis.set_xlabel("Horizon")
        axis.legend(fontsize=8)
    fig.suptitle("Calibration Summary")
    _save_figure(figure_path(figure_root=figure_root, filename="step7_calibration_summary.png"))

    control_models = ("pcr_context_ts", "pcr_context_ts_permuted_target", "riemannian_knn_anchor")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for row_index, target_view in enumerate(("raw_level", "context_residual")):
        for model_name in control_models:
            axes[row_index, 0].plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field="mae_mean",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
            axes[row_index, 1].plot(
                horizons,
                [
                    _summary_value(
                        summary_rows,
                        target_view=target_view,
                        model_name=model_name,
                        horizon=horizon,
                        field="gaussian_nll_mean",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=model_name,
            )
        axes[row_index, 0].set_ylabel(target_view)
        axes[row_index, 0].set_title("MAE")
        axes[row_index, 1].set_title("Gaussian NLL")
    for axis in axes.reshape(-1):
        axis.set_xlabel("Horizon")
        axis.legend(fontsize=8)
    fig.suptitle("Control Collapse Check")
    _save_figure(figure_path(figure_root=figure_root, filename="step7_control_collapse.png"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for row_index, target_view in enumerate(("raw_level", "context_residual")):
        for col_index, horizon in enumerate(rolling_horizons):
            horizon_rows = [
                row
                for row in rolling_summary_rows
                if int(row["horizon"]) == horizon and str(row["target_view"]) == target_view
            ]
            horizon_rows.sort(
                key=lambda row: ROLLING_MODELS.index(str(row["model_name"]))
                if str(row["model_name"]) in ROLLING_MODELS
                else 999,
            )
            axes[row_index, col_index].bar(
                [str(row["model_name"]) for row in horizon_rows],
                [float(row["mae_mean"]) for row in horizon_rows],
                color="#4c78a8",
            )
            axes[row_index, col_index].set_title(f"{target_view} rolling MAE h={horizon}")
            axes[row_index, col_index].tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("MAE")
    axes[1, 0].set_ylabel("MAE")
    fig.suptitle("Rolling Raw vs Residual")
    _save_figure(figure_path(figure_root=figure_root, filename="step7_rolling_raw_vs_residual.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, target_view in zip(axes, ("raw_level", "context_residual"), strict=True):
        width = 0.35
        x_pos = np.arange(len(ROLLING_MODELS), dtype=np.float64)
        values_h05 = [
            _rolling_summary_value(
                rolling_summary_rows,
                target_view=target_view,
                model_name=model_name,
                horizon=rolling_horizons[0],
                field="first_difference_pearson_r_mean",
            )
            for model_name in ROLLING_MODELS
        ]
        values_h25 = [
            _rolling_summary_value(
                rolling_summary_rows,
                target_view=target_view,
                model_name=model_name,
                horizon=rolling_horizons[-1],
                field="first_difference_pearson_r_mean",
            )
            for model_name in ROLLING_MODELS
        ]
        axis.bar(x_pos - width / 2.0, values_h05, width=width, label=f"h={rolling_horizons[0]}")
        axis.bar(x_pos + width / 2.0, values_h25, width=width, label=f"h={rolling_horizons[-1]}")
        axis.set_xticks(x_pos)
        axis.set_xticklabels(list(ROLLING_MODELS), rotation=45, ha="right")
        axis.set_title(target_view)
        axis.legend()
    axes[0].set_ylabel("First-difference Pearson r")
    fig.suptitle("Rolling First-Difference Summary")
    _save_figure(figure_path(figure_root=figure_root, filename="step7_rolling_first_difference_summary.png"))


def _expected_primary_row_counts(config: Step7Config) -> tuple[int, int, int, int, int]:
    total_usable_rows = sum(
        len(expected_session_ids()) * expected_usable_rows_per_session(horizon=horizon, config=config)
        for horizon in config.horizons
    )
    fold_rows = len(SUBJECT_IDS) * len(config.horizons) * len(PRIMARY_MODELS) * len(config.target_views)
    summary_rows = len(config.horizons) * len(PRIMARY_MODELS) * len(config.target_views)
    prediction_rows = total_usable_rows * len(PRIMARY_MODELS) * len(config.target_views)
    model_selection_rows = fold_rows
    paired_rows = len(config.horizons) * len(PRIMARY_COMPARISONS) * len(config.target_views) * len(PAIRED_METRICS)
    return fold_rows, summary_rows, prediction_rows, model_selection_rows, paired_rows


def _expected_rolling_row_counts(config: Step7Config) -> tuple[int, int]:
    fold_rows = len(SUBJECT_IDS) * 6 * len(config.rolling_horizons) * len(ROLLING_MODELS) * len(config.target_views)
    summary_rows = len(config.rolling_horizons) * len(ROLLING_MODELS) * len(config.target_views)
    return fold_rows, summary_rows


def validate_outputs(
    *,
    config: Step7Config,
    primary_tables: dict[int, ForecastingTable],
    fold_metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    model_selection_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    rolling_fold_rows: list[dict[str, Any]],
    rolling_summary_rows: list[dict[str, Any]],
) -> None:
    """Run the Step 7 validation gates that can be checked programmatically."""

    expected_fold_rows, expected_summary_rows, expected_prediction_rows, expected_model_rows, expected_paired_rows = (
        _expected_primary_row_counts(config)
    )
    expected_rolling_fold_rows, expected_rolling_summary_rows = _expected_rolling_row_counts(config)

    if len(fold_metric_rows) != expected_fold_rows:
        raise ValueError(f"Primary fold-metric rows {len(fold_metric_rows)} != expected {expected_fold_rows}")
    if len(summary_rows) != expected_summary_rows:
        raise ValueError(f"Primary summary rows {len(summary_rows)} != expected {expected_summary_rows}")
    if len(prediction_rows) != expected_prediction_rows:
        raise ValueError(f"Prediction rows {len(prediction_rows)} != expected {expected_prediction_rows}")
    if len(model_selection_rows) != expected_model_rows:
        raise ValueError(f"Model-selection rows {len(model_selection_rows)} != expected {expected_model_rows}")
    if len(paired_rows) != expected_paired_rows:
        raise ValueError(f"Paired-stat rows {len(paired_rows)} != expected {expected_paired_rows}")
    if len(rolling_fold_rows) != expected_rolling_fold_rows:
        raise ValueError(f"Rolling fold rows {len(rolling_fold_rows)} != expected {expected_rolling_fold_rows}")
    if len(rolling_summary_rows) != expected_rolling_summary_rows:
        raise ValueError(f"Rolling summary rows {len(rolling_summary_rows)} != expected {expected_rolling_summary_rows}")

    for horizon, table in primary_tables.items():
        expected_horizon_rows = len(expected_session_ids()) * expected_usable_rows_per_session(horizon=horizon, config=config)
        if table.n_rows != expected_horizon_rows:
            raise ValueError(f"Horizon {horizon} loaded {table.n_rows} rows, expected {expected_horizon_rows}")

    for row in fold_metric_rows:
        for field_name in (
            "mae",
            "rmse",
            "r2",
            "pearson_r",
            "gaussian_nll",
            "gaussian_picp_80",
            "gaussian_picp_90",
            "conformal_picp_80",
            "conformal_picp_90",
            "mean_oof_sigma",
        ):
            value = float(row[field_name])
            if not math.isfinite(value):
                raise ValueError(f"Primary fold metric {field_name} is non-finite: {row}")
        if int(row["n_test_rows"]) <= 0:
            raise ValueError(f"Primary fold row has non-positive n_test_rows: {row}")
        if str(row["family"]) != model_family(str(row["model_name"])):
            raise ValueError(f"Fold-metric family mismatch: {row}")

    for row in prediction_rows:
        sigma = float(row["oof_sigma"])
        if sigma < config.sigma_floor:
            raise ValueError(f"Prediction sigma {sigma} is below sigma_floor {config.sigma_floor}")
        for field_name in (
            "y_true",
            "y_pred_mean",
            "gaussian_lower_80",
            "gaussian_upper_80",
            "gaussian_lower_90",
            "gaussian_upper_90",
            "conformal_lower_80",
            "conformal_upper_80",
            "conformal_lower_90",
            "conformal_upper_90",
        ):
            if not math.isfinite(float(row[field_name])):
                raise ValueError(f"Prediction field {field_name} is non-finite: {row}")
        if float(row["gaussian_lower_80"]) > float(row["gaussian_upper_80"]):
            raise ValueError(f"Gaussian 80 interval is inverted: {row}")
        if float(row["gaussian_lower_90"]) > float(row["gaussian_upper_90"]):
            raise ValueError(f"Gaussian 90 interval is inverted: {row}")
        if float(row["conformal_lower_80"]) > float(row["conformal_upper_80"]):
            raise ValueError(f"Conformal 80 interval is inverted: {row}")
        if float(row["conformal_lower_90"]) > float(row["conformal_upper_90"]):
            raise ValueError(f"Conformal 90 interval is inverted: {row}")

    for row in rolling_fold_rows:
        for field_name in ("mae", "gaussian_nll", "rmse", "pearson_r", "first_difference_pearson_r"):
            if not math.isfinite(float(row[field_name])):
                raise ValueError(f"Rolling metric {field_name} is non-finite: {row}")
        if int(row["n_train_rows"]) <= 0 or int(row["n_test_rows"]) <= 0:
            raise ValueError(f"Rolling fold has non-positive row count: {row}")

    for horizon in config.horizons:
        persistence_predictions = [
            row
            for row in prediction_rows
            if int(row["horizon"]) == horizon
            and str(row["model_name"]) == "persistence"
            and str(row["target_view"]) == "raw_level"
        ]
        pooled_mae_step7 = float(
            np.mean([abs(float(row["y_true"]) - float(row["y_pred_mean"])) for row in persistence_predictions]),
        )
        direct_mae = float(
            np.mean(
                np.abs(
                    np.asarray(primary_tables[horizon].target_score_raw, dtype=np.float64)
                    - np.asarray(primary_tables[horizon].anchor_score_raw, dtype=np.float64),
                ),
            ),
        )
        if not np.isclose(pooled_mae_step7, direct_mae, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"Persistence MAE mismatch at horizon {horizon}: Step7={pooled_mae_step7} vs direct={direct_mae}",
            )

    for horizon in config.horizons:
        for target_view in config.target_views:
            horizon_summary = [
                row
                for row in summary_rows
                if int(row["horizon"]) == horizon and str(row["target_view"]) == target_view
            ]
            best_mae_row = min(horizon_summary, key=lambda row: (float(row["mae_mean"]), str(row["model_name"])))
            best_nll_row = min(
                horizon_summary,
                key=lambda row: (float(row["gaussian_nll_mean"]), str(row["model_name"])),
            )
            if str(best_mae_row["model_name"]) == "pcr_context_ts_permuted_target":
                raise ValueError(f"Falsification control won MAE at horizon {horizon}, target_view={target_view}")
            if str(best_nll_row["model_name"]) == "pcr_context_ts_permuted_target":
                raise ValueError(f"Falsification control won Gaussian NLL at horizon {horizon}, target_view={target_view}")

    required_figures = [figure_path(figure_root=config.figure_root, filename=filename) for filename in EXPECTED_FIGURE_FILENAMES]
    missing_figures = [path.name for path in required_figures if not path.is_file()]
    if missing_figures:
        raise FileNotFoundError(f"Missing Step 7 figures: {missing_figures}")


def _records_to_csv(path: Path, *, columns: tuple[str, ...], records: list[dict[str, Any]]) -> None:
    write_csv(path, columns=columns, records=records)


def _resolved_max_workers(config: Step7Config) -> int:
    if config.max_workers == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - config.reserve_cores)
    return int(config.max_workers)


def _math_thread_limit_context(*, max_workers: int) -> Any:
    if max_workers <= 1 or threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


def _execute_ordered_tasks(
    task_specs: list[Any],
    *,
    max_workers: int,
    run_task: Callable[[Any], Any],
) -> list[Any]:
    if not task_specs:
        return []

    with _math_thread_limit_context(max_workers=max_workers):
        if max_workers <= 1:
            return [run_task(task) for task in task_specs]

        ordered_results: dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="step7") as executor:
            futures = [executor.submit(run_task, task) for task in task_specs]
            for future in as_completed(futures):
                result = future.result()
                ordered_results[int(result.task_index)] = result
        return [ordered_results[int(task.task_index)] for task in task_specs]


def _build_primary_task_specs(
    *,
    primary_tables: dict[int, ForecastingTable],
    config: Step7Config,
) -> list[_PrimaryTaskSpec]:
    task_specs: list[_PrimaryTaskSpec] = []
    task_index = 0
    for horizon in config.horizons:
        table = primary_tables[horizon]
        primary_folds = build_primary_folds(table)
        for fold in primary_folds:
            for model_name in PRIMARY_MODELS:
                task_specs.append(
                    _PrimaryTaskSpec(
                        task_index=task_index,
                        horizon=horizon,
                        fold_subject_id=fold.fold_subject_id,
                        model_name=model_name,
                        train_idx=np.asarray(fold.train_idx, dtype=np.int64),
                        test_idx=np.asarray(fold.test_idx, dtype=np.int64),
                    ),
                )
                task_index += 1
    return task_specs


def _build_rolling_task_specs(
    *,
    primary_tables: dict[int, ForecastingTable],
    config: Step7Config,
) -> list[_RollingTaskSpec]:
    task_specs: list[_RollingTaskSpec] = []
    task_index = 0
    for horizon in config.rolling_horizons:
        table = primary_tables[horizon]
        rolling_folds = build_rolling_folds(table)
        for fold in rolling_folds:
            for model_name in ROLLING_MODELS:
                task_specs.append(
                    _RollingTaskSpec(
                        task_index=task_index,
                        horizon=horizon,
                        subject_id=fold.subject_id,
                        train_through_session=fold.train_through_session,
                        test_session=fold.test_session,
                        model_name=model_name,
                        train_idx=np.asarray(fold.train_idx, dtype=np.int64),
                        test_idx=np.asarray(fold.test_idx, dtype=np.int64),
                    ),
                )
                task_index += 1
    return task_specs


def _run_primary_task(
    task: _PrimaryTaskSpec,
    *,
    primary_tables: dict[int, ForecastingTable],
    config: Step7Config,
) -> _PrimaryTaskResult:
    table = primary_tables[task.horizon]
    logger.info(
        "Primary horizon=%d fold=%s model=%s",
        task.horizon,
        task.fold_subject_id,
        task.model_name,
    )
    view_results = fit_and_predict_both_views(
        table=table,
        train_idx=task.train_idx,
        test_idx=task.test_idx,
        model_name=task.model_name,
        target_views=config.target_views,
        config=config,
        protocol="loso",
    )
    family = model_family(task.model_name)
    fold_metric_rows: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    any_geometry_chunking = False

    for target_view in config.target_views:
        result = view_results[target_view]
        any_geometry_chunking = any_geometry_chunking or result.used_geometry_chunking
        _validate_exact_baselines(
            table=table,
            test_idx=task.test_idx,
            model_name=task.model_name,
            target_view=target_view,
            test_pred=result.test_pred,
        )
        oof_sigma, conformal_q80, conformal_q90 = _calibration_statistics(
            calibration_residuals=result.calibration_residuals,
            sigma_floor=config.sigma_floor,
        )
        y_test = np.asarray(target_view_values(table, target_view=target_view)[task.test_idx], dtype=np.float64)
        metrics = compute_regression_metrics(
            y_true=y_test,
            y_pred=result.test_pred,
            gaussian_sigma=oof_sigma,
            conformal_q80=conformal_q80,
            conformal_q90=conformal_q90,
            corr_epsilon=config.corr_epsilon,
        )
        fold_metric_rows.append({
            "fold_subject_id": task.fold_subject_id,
            "horizon": task.horizon,
            "target_view": target_view,
            "model_name": task.model_name,
            "family": family,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson_r"],
            "gaussian_nll": metrics["gaussian_nll"],
            "gaussian_picp_80": metrics["gaussian_picp_80"],
            "gaussian_picp_90": metrics["gaussian_picp_90"],
            "conformal_picp_80": metrics["conformal_picp_80"],
            "conformal_picp_90": metrics["conformal_picp_90"],
            "mean_oof_sigma": metrics["mean_oof_sigma"],
            "n_test_rows": int(metrics["n_test_rows"]),
        })
        model_selection_rows.append({
            "fold_subject_id": task.fold_subject_id,
            "horizon": task.horizon,
            "target_view": target_view,
            "model_name": task.model_name,
            "selected_alpha": "NA" if result.selected_alpha is None else float(result.selected_alpha),
            "selected_n_components": "NA" if result.selected_n_components is None else int(result.selected_n_components),
            "selected_l1_ratio": "NA" if result.selected_l1_ratio is None else float(result.selected_l1_ratio),
            "oof_residual_sigma": oof_sigma,
            "conformal_q80": conformal_q80,
            "conformal_q90": conformal_q90,
            "n_train_rows": int(task.train_idx.size),
            "n_test_rows": int(task.test_idx.size),
        })
        gaussian_intervals = gaussian_prediction_intervals(result.test_pred, oof_sigma)
        conformal_intervals = conformal_prediction_intervals(result.test_pred, conformal_q80, conformal_q90)
        for local_offset, global_row_index in enumerate(task.test_idx.tolist()):
            row_meta = table_row_dict(table, int(global_row_index))
            all_prediction_rows.append(
                {
                    "fold_subject_id": task.fold_subject_id,
                    "subject_id": row_meta["subject_id"],
                    "session_id": row_meta["session_id"],
                    "horizon": task.horizon,
                    "target_view": target_view,
                    "model_name": task.model_name,
                    "family": family,
                    "anchor_idx": row_meta["anchor_idx"],
                    "target_idx": row_meta["target_idx"],
                    "y_true": float(y_test[local_offset]),
                    "y_pred_mean": float(result.test_pred[local_offset]),
                    "oof_sigma": oof_sigma,
                    "gaussian_lower_80": float(gaussian_intervals["gaussian_lower_80"][local_offset]),
                    "gaussian_upper_80": float(gaussian_intervals["gaussian_upper_80"][local_offset]),
                    "gaussian_lower_90": float(gaussian_intervals["gaussian_lower_90"][local_offset]),
                    "gaussian_upper_90": float(gaussian_intervals["gaussian_upper_90"][local_offset]),
                    "conformal_lower_80": float(conformal_intervals["conformal_lower_80"][local_offset]),
                    "conformal_upper_80": float(conformal_intervals["conformal_upper_80"][local_offset]),
                    "conformal_lower_90": float(conformal_intervals["conformal_lower_90"][local_offset]),
                    "conformal_upper_90": float(conformal_intervals["conformal_upper_90"][local_offset]),
                },
            )
    return _PrimaryTaskResult(
        task_index=task.task_index,
        fold_metric_rows=fold_metric_rows,
        model_selection_rows=model_selection_rows,
        prediction_rows=all_prediction_rows,
        used_geometry_chunking=any_geometry_chunking,
    )


def _run_rolling_task(
    task: _RollingTaskSpec,
    *,
    primary_tables: dict[int, ForecastingTable],
    config: Step7Config,
) -> _RollingTaskResult:
    table = primary_tables[task.horizon]
    logger.info(
        "Rolling horizon=%d subject=%s train_through=%s test=%s model=%s",
        task.horizon,
        task.subject_id,
        task.train_through_session,
        task.test_session,
        task.model_name,
    )
    view_results = fit_and_predict_both_views(
        table=table,
        train_idx=task.train_idx,
        test_idx=task.test_idx,
        model_name=task.model_name,
        target_views=config.target_views,
        config=config,
        protocol="rolling",
    )
    order = np.argsort(np.asarray(table.target_idx[task.test_idx], dtype=np.int64))
    rolling_fold_rows: list[dict[str, Any]] = []
    any_geometry_chunking = False

    for target_view in config.target_views:
        result = view_results[target_view]
        any_geometry_chunking = any_geometry_chunking or result.used_geometry_chunking
        oof_sigma, conformal_q80, conformal_q90 = _calibration_statistics(
            calibration_residuals=result.calibration_residuals,
            sigma_floor=config.sigma_floor,
        )
        y_test = np.asarray(target_view_values(table, target_view=target_view)[task.test_idx], dtype=np.float64)
        ordered_y_test = y_test[order]
        ordered_y_pred = np.asarray(result.test_pred, dtype=np.float64)[order]
        metrics = compute_regression_metrics(
            y_true=ordered_y_test,
            y_pred=ordered_y_pred,
            gaussian_sigma=oof_sigma,
            conformal_q80=conformal_q80,
            conformal_q90=conformal_q90,
            corr_epsilon=config.corr_epsilon,
            include_first_difference=True,
        )
        rolling_fold_rows.append({
            "subject_id": task.subject_id,
            "train_through_session": task.train_through_session,
            "test_session": task.test_session,
            "horizon": task.horizon,
            "target_view": target_view,
            "model_name": task.model_name,
            "mae": metrics["mae"],
            "gaussian_nll": metrics["gaussian_nll"],
            "rmse": metrics["rmse"],
            "pearson_r": metrics["pearson_r"],
            "first_difference_pearson_r": metrics["first_difference_pearson_r"],
            "n_train_rows": int(task.train_idx.size),
            "n_test_rows": int(task.test_idx.size),
        })
    return _RollingTaskResult(
        task_index=task.task_index,
        rolling_fold_rows=rolling_fold_rows,
        used_geometry_chunking=any_geometry_chunking,
    )


def _execute_primary_task_specs(
    task_specs: list[_PrimaryTaskSpec],
    *,
    primary_tables: dict[int, ForecastingTable],
    config: Step7Config,
    max_workers: int,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    # Phase 1: load cached results from checkpoint directory.
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
        logger.info(
            "Primary tasks: %d cached, %d to compute (%d total)",
            len(cached_results), len(remaining_specs), len(task_specs),
        )
    else:
        remaining_specs = list(task_specs)

    # Phase 2: execute remaining tasks, saving checkpoints after each.
    def _run_and_save(task: _PrimaryTaskSpec) -> _PrimaryTaskResult:
        result = _run_primary_task(task, primary_tables=primary_tables, config=config)
        if checkpoint_dir is not None:
            _save_checkpoint(checkpoint_dir, _primary_checkpoint_key(task), _primary_result_to_checkpoint(result))
        return result

    new_results = _execute_ordered_tasks(remaining_specs, max_workers=max_workers, run_task=_run_and_save)

    # Phase 3: merge cached + new results in original task order.
    for result in new_results:
        cached_results[result.task_index] = result
    all_results = [cached_results[task.task_index] for task in task_specs]

    fold_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    geometry_chunking_used = False
    for task_result in all_results:
        fold_metric_rows.extend(task_result.fold_metric_rows)
        model_selection_rows.extend(task_result.model_selection_rows)
        prediction_rows.extend(task_result.prediction_rows)
        geometry_chunking_used = geometry_chunking_used or task_result.used_geometry_chunking
    return fold_metric_rows, prediction_rows, model_selection_rows, geometry_chunking_used


def _execute_rolling_task_specs(
    task_specs: list[_RollingTaskSpec],
    *,
    primary_tables: dict[int, ForecastingTable],
    config: Step7Config,
    max_workers: int,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    # Phase 1: load cached results from checkpoint directory.
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
        logger.info(
            "Rolling tasks: %d cached, %d to compute (%d total)",
            len(cached_results), len(remaining_specs), len(task_specs),
        )
    else:
        remaining_specs = list(task_specs)

    # Phase 2: execute remaining tasks, saving checkpoints after each.
    def _run_and_save(task: _RollingTaskSpec) -> _RollingTaskResult:
        result = _run_rolling_task(task, primary_tables=primary_tables, config=config)
        if checkpoint_dir is not None:
            _save_checkpoint(checkpoint_dir, _rolling_checkpoint_key(task), _rolling_result_to_checkpoint(result))
        return result

    new_results = _execute_ordered_tasks(remaining_specs, max_workers=max_workers, run_task=_run_and_save)

    # Phase 3: merge cached + new results in original task order.
    for result in new_results:
        cached_results[result.task_index] = result
    all_results = [cached_results[task.task_index] for task in task_specs]

    rolling_fold_rows: list[dict[str, Any]] = []
    geometry_chunking_used = False
    for task_result in all_results:
        rolling_fold_rows.extend(task_result.rolling_fold_rows)
        geometry_chunking_used = geometry_chunking_used or task_result.used_geometry_chunking
    return rolling_fold_rows, geometry_chunking_used


def run_step7(config: Step7Config | None = None, *, no_resume: bool = False) -> Step7Summary:
    """Run the full Step 7 benchmark layer.

    Args:
        config: Step 7 configuration. Uses defaults if None.
        no_resume: If True, discard any existing checkpoint data and start fresh.
    """

    config = config or Step7Config()
    primary_tables = load_primary_tables(config)
    max_workers = _resolved_max_workers(config)
    logger.info(
        "Step 7 parallel execution enabled with %d worker(s)%s",
        max_workers,
        " [auto]" if config.max_workers == 0 else "",
    )

    # Checkpoint directory setup.
    checkpoint_dir: Path | None = None
    if not no_resume:
        checkpoint_dir = _checkpoint_dir(config.output_root)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoint directory: %s", checkpoint_dir)
    else:
        stale_dir = _checkpoint_dir(config.output_root)
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)
            logger.info("Removed stale checkpoint directory: %s", stale_dir)

    primary_task_specs = _build_primary_task_specs(primary_tables=primary_tables, config=config)
    fold_metric_rows, prediction_rows, model_selection_rows, primary_geometry_chunking_used = _execute_primary_task_specs(
        primary_task_specs,
        primary_tables=primary_tables,
        config=config,
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
    )

    summary_rows = _summarize_primary(fold_metric_rows)
    paired_rows = _paired_stats_rows(fold_metric_rows=fold_metric_rows, config=config)
    rolling_task_specs = _build_rolling_task_specs(primary_tables=primary_tables, config=config)
    rolling_fold_rows, rolling_geometry_chunking_used = _execute_rolling_task_specs(
        rolling_task_specs,
        primary_tables=primary_tables,
        config=config,
        max_workers=max_workers,
        checkpoint_dir=checkpoint_dir,
    )
    geometry_chunking_used = primary_geometry_chunking_used or rolling_geometry_chunking_used

    rolling_summary_rows = _summarize_rolling(rolling_fold_rows)

    config.output_root.mkdir(parents=True, exist_ok=True)
    _records_to_csv(fold_metrics_path(output_root=config.output_root), columns=FOLD_METRICS_COLUMNS, records=fold_metric_rows)
    _records_to_csv(summary_path(output_root=config.output_root), columns=SUMMARY_COLUMNS, records=summary_rows)
    _records_to_csv(predictions_path(output_root=config.output_root), columns=PREDICTIONS_COLUMNS, records=prediction_rows)
    _records_to_csv(
        model_selection_path(output_root=config.output_root),
        columns=MODEL_SELECTION_COLUMNS,
        records=model_selection_rows,
    )
    _records_to_csv(paired_stats_path(output_root=config.output_root), columns=PAIRED_STATS_COLUMNS, records=paired_rows)
    _records_to_csv(
        rolling_fold_metrics_path(output_root=config.output_root),
        columns=ROLLING_FOLD_COLUMNS,
        records=rolling_fold_rows,
    )
    _records_to_csv(
        rolling_summary_path(output_root=config.output_root),
        columns=ROLLING_SUMMARY_COLUMNS,
        records=rolling_summary_rows,
    )

    save_figures(summary_rows=summary_rows, rolling_summary_rows=rolling_summary_rows, figure_root=config.figure_root)

    validate_outputs(
        config=config,
        primary_tables=primary_tables,
        fold_metric_rows=fold_metric_rows,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        model_selection_rows=model_selection_rows,
        paired_rows=paired_rows,
        rolling_fold_rows=rolling_fold_rows,
        rolling_summary_rows=rolling_summary_rows,
    )

    # Clean up checkpoint data after all validations pass.
    if checkpoint_dir is not None and checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
        logger.info("All validations passed. Cleaned up checkpoint directory.")

    return Step7Summary(
        output_root=config.output_root,
        figure_root=config.figure_root,
        fold_metric_rows=len(fold_metric_rows),
        summary_rows=len(summary_rows),
        prediction_rows=len(prediction_rows),
        model_selection_rows=len(model_selection_rows),
        paired_stats_rows=len(paired_rows),
        rolling_fold_rows=len(rolling_fold_rows),
        rolling_summary_rows=len(rolling_summary_rows),
        figure_count=len(EXPECTED_FIGURE_FILENAMES),
        geometry_chunking_used=geometry_chunking_used,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    default_config = Step7Config()
    parser = argparse.ArgumentParser(description="Run Step 7 dual-track classical forecasting benchmarks.")
    parser.add_argument("--step4-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step5-root", type=Path, default=Path("data/scores"))
    parser.add_argument("--step6-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--output-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/forecasting/step7"))
    parser.add_argument("--geometry-batch-size", type=int, default=default_config.geometry_batch_size)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_config.max_workers,
        help="0 uses all logical CPUs minus reserve-cores; 1 disables task-level parallelism.",
    )
    parser.add_argument(
        "--reserve-cores",
        type=int,
        default=default_config.reserve_cores,
        help="Number of CPU cores to keep free when max-workers=0 (auto). Default: 2.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Discard any existing checkpoint data and start fresh.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_step7(
        Step7Config(
            step4_root=args.step4_root,
            step5_root=args.step5_root,
            step6_root=args.step6_root,
            output_root=args.output_root,
            figure_root=args.figure_root,
            geometry_batch_size=args.geometry_batch_size,
            max_workers=args.max_workers,
            reserve_cores=args.reserve_cores,
        ),
        no_resume=args.no_resume,
    )
    logger.info(
        "Step 7 complete: %s fold rows, %s prediction rows, %s rolling rows",
        csv_float(float(summary.fold_metric_rows)),
        csv_float(float(summary.prediction_rows)),
        csv_float(float(summary.rolling_fold_rows)),
    )


if __name__ == "__main__":
    main()
