"""Metrics and inferential helpers for Step 7 forecasting benchmarks."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, t as t_dist, wilcoxon


Z_80 = 1.2815515655446004
Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class PairedStatistic:
    """One prespecified paired-statistics row."""

    comparison_id: str
    horizon: int
    target_view: str
    metric: str
    model_a: str
    model_b: str
    n_folds: int
    n_nonzero_deltas: int
    test_method: str
    zero_method: str
    p_value_method: str
    effect_size_method: str
    mean_metric_a: float
    mean_metric_b: float
    mean_delta_b_minus_a: float
    median_delta_b_minus_a: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    wilcoxon_stat: float
    p_value: float
    p_value_holm: float
    rank_biserial: float
    std_delta: float
    min_detectable_delta_80: float
    better_model: str


def safe_corr(x: np.ndarray, y: np.ndarray, *, epsilon: float) -> float:
    """Return a finite Pearson correlation, falling back to zero on degenerate inputs."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Correlation inputs must share a shape, got {x.shape} and {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"Correlation inputs must be one-dimensional, got ndim={x.ndim}")
    if x.size == 0:
        raise ValueError("Correlation input is empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Correlation inputs contain non-finite values")
    if x.size < 2:
        return 0.0
    if float(np.std(x, ddof=0)) <= epsilon or float(np.std(y, ddof=0)) <= epsilon:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not np.isfinite(corr) else corr


def safe_first_difference_corr(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float) -> float:
    """Return a finite first-difference correlation with a safe fallback."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"First-difference inputs must share a shape, got {y_true.shape} and {y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"First-difference inputs must be one-dimensional, got ndim={y_true.ndim}")
    if y_true.size < 2:
        return 0.0
    return safe_corr(np.diff(y_true), np.diff(y_pred), epsilon=epsilon)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float) -> float:
    """Return a finite R^2 with a stable fallback on degenerate targets."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"R^2 inputs must share a shape, got {y_true.shape} and {y_pred.shape}")
    residual = y_true - y_pred
    ss_res = float(np.sum(residual * residual))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered * centered))
    if ss_tot <= epsilon:
        return 1.0 if ss_res <= epsilon else 0.0
    return float(1.0 - (ss_res / ss_tot))


def gaussian_prediction_intervals(y_pred: np.ndarray, sigma: float) -> dict[str, np.ndarray]:
    """Return fixed 80% and 90% Gaussian prediction intervals."""

    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma_value}")
    return {
        "gaussian_lower_80": (y_pred - Z_80 * sigma_value).astype(np.float64),
        "gaussian_upper_80": (y_pred + Z_80 * sigma_value).astype(np.float64),
        "gaussian_lower_90": (y_pred - Z_90 * sigma_value).astype(np.float64),
        "gaussian_upper_90": (y_pred + Z_90 * sigma_value).astype(np.float64),
    }


def gaussian_prediction_intervals_vector(y_pred: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    """Return fixed 80% and 90% Gaussian intervals under per-sample sigma."""

    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if y_pred.shape != sigma.shape:
        raise ValueError(f"y_pred and sigma must share a shape, got {y_pred.shape} and {sigma.shape}")
    if y_pred.ndim != 1:
        raise ValueError(f"Expected one-dimensional inputs, got ndim={y_pred.ndim}")
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(sigma)):
        raise ValueError("Vector Gaussian interval inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("All sigma values must be positive")
    return {
        "gaussian_lower_80": (y_pred - Z_80 * sigma).astype(np.float64),
        "gaussian_upper_80": (y_pred + Z_80 * sigma).astype(np.float64),
        "gaussian_lower_90": (y_pred - Z_90 * sigma).astype(np.float64),
        "gaussian_upper_90": (y_pred + Z_90 * sigma).astype(np.float64),
    }


def conformal_quantile(abs_residuals: np.ndarray, *, coverage: float) -> float:
    """Return the standard finite-sample split-conformal quantile."""

    abs_residuals = np.sort(np.asarray(abs_residuals, dtype=np.float64))
    if abs_residuals.ndim != 1:
        raise ValueError(f"abs_residuals must be one-dimensional, got {abs_residuals.shape}")
    if abs_residuals.size == 0:
        raise ValueError("abs_residuals must not be empty")
    if not np.all(np.isfinite(abs_residuals)):
        raise ValueError("abs_residuals contain non-finite values")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    rank = int(math.ceil((abs_residuals.size + 1) * coverage) - 1)
    rank = min(max(rank, 0), int(abs_residuals.size - 1))
    return float(abs_residuals[rank])


def conformal_prediction_intervals(y_pred: np.ndarray, q80: float, q90: float) -> dict[str, np.ndarray]:
    """Return 80% and 90% split-conformal intervals."""

    y_pred = np.asarray(y_pred, dtype=np.float64)
    q80_value = float(q80)
    q90_value = float(q90)
    if q80_value < 0.0 or q90_value < 0.0:
        raise ValueError("Conformal quantiles must be non-negative")
    if q90_value < q80_value - 1e-12:
        raise ValueError(f"Expected q90 >= q80, got q80={q80_value}, q90={q90_value}")
    return {
        "conformal_lower_80": (y_pred - q80_value).astype(np.float64),
        "conformal_upper_80": (y_pred + q80_value).astype(np.float64),
        "conformal_lower_90": (y_pred - q90_value).astype(np.float64),
        "conformal_upper_90": (y_pred + q90_value).astype(np.float64),
    }


def gaussian_nll(y_true: np.ndarray, y_pred: np.ndarray, sigma: float) -> float:
    """Return the mean Gaussian negative log-likelihood under homoscedastic sigma."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma_value}")
    residual = y_true - y_pred
    variance = sigma_value * sigma_value
    nll = 0.5 * np.log(2.0 * math.pi * variance) + 0.5 * ((residual * residual) / variance)
    return float(np.mean(nll))


def gaussian_nll_vector(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    """Return the mean Gaussian negative log-likelihood under per-sample sigma."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.shape != sigma.shape:
        raise ValueError(
            "y_true, y_pred, and sigma must share a shape, "
            f"got {y_true.shape}, {y_pred.shape}, and {sigma.shape}",
        )
    if y_true.ndim != 1:
        raise ValueError(f"Expected one-dimensional inputs, got ndim={y_true.ndim}")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(sigma)):
        raise ValueError("Vector Gaussian NLL inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("All sigma values must be positive")
    residual = y_true - y_pred
    variance = sigma * sigma
    nll = 0.5 * np.log(2.0 * math.pi * variance) + 0.5 * ((residual * residual) / variance)
    return float(np.mean(nll))


def compute_regression_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gaussian_sigma: float,
    conformal_q80: float,
    conformal_q90: float,
    corr_epsilon: float,
    include_first_difference: bool = False,
) -> dict[str, float]:
    """Compute the full Step 7 metric bundle for one fold."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Metric inputs must share a shape, got {y_true.shape} and {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Metric input is empty")

    gaussian_intervals = gaussian_prediction_intervals(y_pred, gaussian_sigma)
    conformal_intervals = conformal_prediction_intervals(y_pred, conformal_q80, conformal_q90)
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": safe_r2(y_true, y_pred, epsilon=corr_epsilon),
        "pearson_r": safe_corr(y_true, y_pred, epsilon=corr_epsilon),
        "gaussian_nll": gaussian_nll(y_true, y_pred, gaussian_sigma),
        "gaussian_picp_80": float(
            np.mean(
                (y_true >= gaussian_intervals["gaussian_lower_80"])
                & (y_true <= gaussian_intervals["gaussian_upper_80"]),
            ),
        ),
        "gaussian_picp_90": float(
            np.mean(
                (y_true >= gaussian_intervals["gaussian_lower_90"])
                & (y_true <= gaussian_intervals["gaussian_upper_90"]),
            ),
        ),
        "conformal_picp_80": float(
            np.mean(
                (y_true >= conformal_intervals["conformal_lower_80"])
                & (y_true <= conformal_intervals["conformal_upper_80"]),
            ),
        ),
        "conformal_picp_90": float(
            np.mean(
                (y_true >= conformal_intervals["conformal_lower_90"])
                & (y_true <= conformal_intervals["conformal_upper_90"]),
            ),
        ),
        "mean_oof_sigma": float(gaussian_sigma),
        "n_test_rows": float(y_true.size),
    }
    if include_first_difference:
        metrics["first_difference_pearson_r"] = safe_first_difference_corr(y_true, y_pred, epsilon=corr_epsilon)
    return metrics


def compute_regression_metrics_heteroscedastic(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gaussian_sigma: np.ndarray,
    conformal_q80: float,
    conformal_q90: float,
    corr_epsilon: float,
    mean_oof_sigma: float,
    include_first_difference: bool = False,
) -> dict[str, float]:
    """Compute the Step 8 metric bundle under per-sample learned sigma."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    gaussian_sigma = np.asarray(gaussian_sigma, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.shape != gaussian_sigma.shape:
        raise ValueError(
            "Metric inputs must share a shape, "
            f"got {y_true.shape}, {y_pred.shape}, and {gaussian_sigma.shape}",
        )
    if y_true.ndim != 1:
        raise ValueError(f"Metric inputs must be one-dimensional, got ndim={y_true.ndim}")
    if y_true.size == 0:
        raise ValueError("Metric input is empty")
    if not np.all(np.isfinite(gaussian_sigma)):
        raise ValueError("gaussian_sigma contains non-finite values")
    if np.any(gaussian_sigma <= 0.0):
        raise ValueError("gaussian_sigma must be strictly positive")

    gaussian_intervals = gaussian_prediction_intervals_vector(y_pred, gaussian_sigma)
    conformal_intervals = conformal_prediction_intervals(y_pred, conformal_q80, conformal_q90)
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": safe_r2(y_true, y_pred, epsilon=corr_epsilon),
        "pearson_r": safe_corr(y_true, y_pred, epsilon=corr_epsilon),
        "gaussian_nll": gaussian_nll_vector(y_true, y_pred, gaussian_sigma),
        "gaussian_picp_80": float(
            np.mean(
                (y_true >= gaussian_intervals["gaussian_lower_80"])
                & (y_true <= gaussian_intervals["gaussian_upper_80"]),
            ),
        ),
        "gaussian_picp_90": float(
            np.mean(
                (y_true >= gaussian_intervals["gaussian_lower_90"])
                & (y_true <= gaussian_intervals["gaussian_upper_90"]),
            ),
        ),
        "conformal_picp_80": float(
            np.mean(
                (y_true >= conformal_intervals["conformal_lower_80"])
                & (y_true <= conformal_intervals["conformal_upper_80"]),
            ),
        ),
        "conformal_picp_90": float(
            np.mean(
                (y_true >= conformal_intervals["conformal_lower_90"])
                & (y_true <= conformal_intervals["conformal_upper_90"]),
            ),
        ),
        "mean_learned_sigma": float(np.mean(gaussian_sigma)),
        "mean_oof_sigma": float(mean_oof_sigma),
        "n_test_rows": float(y_true.size),
    }
    if include_first_difference:
        metrics["first_difference_pearson_r"] = safe_first_difference_corr(y_true, y_pred, epsilon=corr_epsilon)
    return metrics


def pratt_rank_sums(deltas: np.ndarray) -> tuple[float, float, float, int]:
    """Return Pratt rank sums and the induced rank-biserial effect size."""

    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1:
        raise ValueError(f"deltas must be one-dimensional, got shape {deltas.shape}")
    if deltas.size == 0:
        raise ValueError("deltas must not be empty")
    abs_deltas = np.abs(deltas)
    ranks = rankdata(abs_deltas, method="average")
    w_plus = float(np.sum(ranks[deltas > 0.0]))
    w_minus = float(np.sum(ranks[deltas < 0.0]))
    denom = w_plus + w_minus
    rank_biserial = 0.0 if denom == 0.0 else float((w_plus - w_minus) / denom)
    n_nonzero = int(np.sum(deltas != 0.0))
    return w_plus, w_minus, rank_biserial, n_nonzero


def _supports_exact_wilcoxon(deltas: np.ndarray) -> bool:
    deltas = np.asarray(deltas, dtype=np.float64)
    if np.any(deltas == 0.0):
        return False
    abs_nonzero = np.abs(deltas)
    return int(np.unique(abs_nonzero).size) == int(abs_nonzero.size)


def exhaustive_sign_permutation_pvalue(deltas: np.ndarray) -> float:
    """Return the exact Pratt-style sign-permutation p-value."""

    deltas = np.asarray(deltas, dtype=np.float64)
    abs_deltas = np.abs(deltas)
    ranks = rankdata(abs_deltas, method="average")
    nonzero_mask = deltas != 0.0
    nonzero_ranks = np.asarray(ranks[nonzero_mask], dtype=np.float64)
    if nonzero_ranks.size == 0:
        return 1.0

    observed_w_plus = float(np.sum(nonzero_ranks[deltas[nonzero_mask] > 0.0]))
    total_rank_sum = float(np.sum(nonzero_ranks))
    observed_stat = float(min(observed_w_plus, total_rank_sum - observed_w_plus))

    perm_stats: list[float] = []
    for signs in itertools.product((-1.0, 1.0), repeat=int(nonzero_ranks.size)):
        signs_arr = np.asarray(signs, dtype=np.float64)
        w_plus = float(np.sum(nonzero_ranks[signs_arr > 0.0]))
        w_minus = total_rank_sum - w_plus
        perm_stats.append(min(w_plus, w_minus))
    perm_stats_arr = np.asarray(perm_stats, dtype=np.float64)
    return float(np.mean(perm_stats_arr <= observed_stat + 1e-12))


def bootstrap_mean_delta_ci(
    *,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap CI for the mean fold delta."""

    metric_a = np.asarray(metric_a, dtype=np.float64)
    metric_b = np.asarray(metric_b, dtype=np.float64)
    if metric_a.shape != metric_b.shape:
        raise ValueError(f"Bootstrap inputs must share a shape, got {metric_a.shape} and {metric_b.shape}")
    if metric_a.ndim != 1:
        raise ValueError(f"Bootstrap inputs must be one-dimensional, got ndim={metric_a.ndim}")
    if metric_a.size == 0:
        raise ValueError("Bootstrap inputs must not be empty")
    deltas = metric_b - metric_a
    rng = np.random.default_rng(seed)
    sample_count = int(deltas.size)
    indices = rng.integers(0, sample_count, size=(int(replicates), sample_count))
    sampled_means = np.mean(deltas[indices], axis=1)
    return (
        float(np.percentile(sampled_means, 2.5)),
        float(np.percentile(sampled_means, 97.5)),
    )


def compute_paired_statistic(
    *,
    comparison_id: str,
    horizon: int,
    target_view: str,
    metric: str,
    model_a: str,
    model_b: str,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    higher_is_better: bool = False,
) -> PairedStatistic:
    """Compute one prespecified paired-statistics row."""

    metric_a = np.asarray(metric_a, dtype=np.float64)
    metric_b = np.asarray(metric_b, dtype=np.float64)
    if metric_a.shape != metric_b.shape:
        raise ValueError(f"Paired metrics must share a shape, got {metric_a.shape} and {metric_b.shape}")
    if metric_a.ndim != 1:
        raise ValueError(f"Paired metrics must be one-dimensional, got ndim={metric_a.ndim}")
    if metric_a.size == 0:
        raise ValueError("Paired metrics must not be empty")

    deltas = metric_b - metric_a
    n = int(metric_a.size)
    std_delta = float(np.std(deltas, ddof=1)) if n >= 2 else float("nan")
    if n >= 2 and std_delta > 0.0:
        df = n - 1
        t_alpha = float(t_dist.ppf(0.975, df))
        t_beta = float(t_dist.ppf(0.80, df))
        min_detectable_delta_80 = (t_alpha + t_beta) * std_delta / np.sqrt(n)
    else:
        min_detectable_delta_80 = 0.0
    mean_delta = float(np.mean(deltas))
    w_plus, w_minus, rank_biserial, n_nonzero = pratt_rank_sums(deltas)
    wilcoxon_stat = float(min(w_plus, w_minus))
    bootstrap_ci_lower, bootstrap_ci_upper = bootstrap_mean_delta_ci(
        metric_a=metric_a,
        metric_b=metric_b,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )

    if n_nonzero == 0:
        p_value = 1.0
        p_value_method = "sign_permutation_pratt"
        better_model = "tie"
    elif _supports_exact_wilcoxon(deltas):
        exact_result = wilcoxon(
            deltas,
            alternative="two-sided",
            zero_method="pratt",
            method="exact",
            correction=False,
        )
        p_value = float(exact_result.pvalue)
        p_value_method = "wilcoxon_exact"
    else:
        p_value = exhaustive_sign_permutation_pvalue(deltas)
        p_value_method = "sign_permutation_pratt"

    if mean_delta == 0.0:
        better_model = "tie"
    elif higher_is_better:
        better_model = model_b if mean_delta > 0.0 else model_a
    else:
        better_model = model_b if mean_delta < 0.0 else model_a

    return PairedStatistic(
        comparison_id=comparison_id,
        horizon=horizon,
        target_view=target_view,
        metric=metric,
        model_a=model_a,
        model_b=model_b,
        n_folds=int(metric_a.size),
        n_nonzero_deltas=n_nonzero,
        test_method="wilcoxon_signed_rank",
        zero_method="pratt",
        p_value_method=p_value_method,
        effect_size_method="rank_biserial_pratt",
        mean_metric_a=float(np.mean(metric_a)),
        mean_metric_b=float(np.mean(metric_b)),
        mean_delta_b_minus_a=mean_delta,
        median_delta_b_minus_a=float(np.median(deltas)),
        bootstrap_ci_lower=bootstrap_ci_lower,
        bootstrap_ci_upper=bootstrap_ci_upper,
        wilcoxon_stat=wilcoxon_stat,
        p_value=float(p_value),
        p_value_holm=float("nan"),
        rank_biserial=rank_biserial,
        std_delta=std_delta,
        min_detectable_delta_80=min_detectable_delta_80,
        better_model=better_model,
    )


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-corrected p-values for one family of tests."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {p_values.shape}")
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    m = sorted_p.size
    adjusted_sorted = np.empty_like(sorted_p)
    running_max = 0.0
    for rank, p_value in enumerate(sorted_p):
        adjusted = min(1.0, float((m - rank) * p_value))
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted
