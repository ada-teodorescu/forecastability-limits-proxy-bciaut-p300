"""Step 8 anomaly-scoring helpers."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import kstest
from sklearn.metrics import average_precision_score, roc_auc_score


def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(ndtr(x), dtype=np.float64)


def raw_space_prediction(
    *,
    target_view: str,
    y_pred_mu: np.ndarray,
    y_pred_sigma: np.ndarray,
    context_score_mean_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one target-view prediction back into raw-score space."""

    y_pred_mu = np.asarray(y_pred_mu, dtype=np.float64)
    y_pred_sigma = np.asarray(y_pred_sigma, dtype=np.float64)
    context_score_mean_value = np.asarray(context_score_mean_value, dtype=np.float64)
    if y_pred_mu.shape != y_pred_sigma.shape or y_pred_mu.shape != context_score_mean_value.shape:
        raise ValueError(
            "Prediction inputs must share a shape, "
            f"got {y_pred_mu.shape}, {y_pred_sigma.shape}, and {context_score_mean_value.shape}",
        )
    if target_view == "raw_level":
        return y_pred_mu, y_pred_sigma
    if target_view == "context_residual":
        return context_score_mean_value + y_pred_mu, y_pred_sigma
    raise KeyError(f"Unknown target view: {target_view}")


def prospective_drop20_probability(
    *,
    mu_raw: np.ndarray,
    sigma_raw: np.ndarray,
    drop20_threshold_raw: np.ndarray,
) -> np.ndarray:
    """Return P(Y_raw < drop20_threshold_raw) under the predictive Gaussian."""

    mu_raw = np.asarray(mu_raw, dtype=np.float64)
    sigma_raw = np.asarray(sigma_raw, dtype=np.float64)
    drop20_threshold_raw = np.asarray(drop20_threshold_raw, dtype=np.float64)
    if mu_raw.shape != sigma_raw.shape or mu_raw.shape != drop20_threshold_raw.shape:
        raise ValueError(
            "mu_raw, sigma_raw, and drop20_threshold_raw must share a shape, "
            f"got {mu_raw.shape}, {sigma_raw.shape}, and {drop20_threshold_raw.shape}",
        )
    if np.any(sigma_raw <= 0.0):
        raise ValueError("sigma_raw must be strictly positive")
    z = (drop20_threshold_raw - mu_raw) / sigma_raw
    return _standard_normal_cdf(z).astype(np.float64)


def retrospective_z_scores(*, y_true: np.ndarray, y_pred_mu: np.ndarray, y_pred_sigma: np.ndarray) -> np.ndarray:
    """Return normalized residuals under the learned predictive sigma."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_mu = np.asarray(y_pred_mu, dtype=np.float64)
    y_pred_sigma = np.asarray(y_pred_sigma, dtype=np.float64)
    if y_true.shape != y_pred_mu.shape or y_true.shape != y_pred_sigma.shape:
        raise ValueError(
            "y_true, y_pred_mu, and y_pred_sigma must share a shape, "
            f"got {y_true.shape}, {y_pred_mu.shape}, and {y_pred_sigma.shape}",
        )
    if np.any(y_pred_sigma <= 0.0):
        raise ValueError("y_pred_sigma must be strictly positive")
    return ((y_true - y_pred_mu) / y_pred_sigma).astype(np.float64)


def binary_auc_metrics(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Return ROC-AUC and PR-AUC with NaN-safe handling for degenerate folds."""

    y_true = np.asarray(y_true, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if y_true.shape != score.shape:
        raise ValueError(f"y_true and score must share a shape, got {y_true.shape} and {score.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true and score must be one-dimensional, got ndim={y_true.ndim}")
    if y_true.size == 0:
        return float("nan"), float("nan")
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))


def standard_normal_ks(z_score: np.ndarray) -> tuple[float, float]:
    """Return the KS statistic and p-value against N(0,1)."""

    z_score = np.asarray(z_score, dtype=np.float64)
    if z_score.ndim != 1:
        raise ValueError(f"z_score must be one-dimensional, got shape {z_score.shape}")
    if z_score.size == 0:
        return float("nan"), float("nan")
    result = kstest(z_score, "norm")
    return float(result.statistic), float(result.pvalue)
