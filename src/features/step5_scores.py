"""Step 5 attention score computation via a Riemannian MDM margin.

Computes per-block attention scores as the difference between Riemannian
distances to non-target and target reference covariances:

    score_raw = d_R(C_block, nontarget_ref) - d_R(C_block, target_ref)

Positive scores indicate target-like responses. Step 5 keeps this raw margin
as the canonical forecasting target, retains a global sigmoid normalization and
global threshold-relative score for exploratory cohort-level comparison, and
adds session-calibrated clinical interpretation layers derived from the same
session's social calibration blocks 10-19.

References:
    - Barachant & Congedo (2014): augmented covariance / super-trial construction
    - Zanini et al. (2018): Riemannian session alignment via recentering
    - MDM classifier margin: pyRiemann documentation
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyriemann.utils.distance import distance_riemann
from scipy.special import expit
from sklearn.covariance import OAS

from src.data.loader import expected_session_ids
from src.features.step3_covariance import (
    Step3Config,
    _validate_saved_covariance_payload,
    feature_split_path as step3_feature_split_path,
    session_id_to_ses_label,
    subject_id_to_sub_label,
)
from src.features.step4_riemannian import (
    METADATA_FIELDS,
    Step4Config,
    _validate_saved_aligned_payload,
    aligned_split_path as step4_aligned_split_path,
    load_npz_payload,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_SOURCE = "social calibration blocks 10-19"

SCORE_FIELDS: frozenset[str] = frozenset((
    "score_raw",
    "score_norm",
    "score_threshold_rel_global",
    "score_session_robust_z",
    "score_drop_rel_session",
    "d_target",
    "d_nontarget",
    "mdm_correct",
    "session_cal_mean_raw",
    "session_cal_std_raw",
    "session_cal_median_raw",
    "session_cal_mad_raw",
    "session_cal_robust_scale_raw",
    "session_cal_q10_raw",
    "session_cal_q25_raw",
    "session_cal_q50_raw",
    "session_cal_ref_count",
    "scale_degenerate_flag",
    *METADATA_FIELDS,
))

REFERENCE_NPZ_FIELDS: frozenset[str] = frozenset((
    "cov_target_ref_aligned",
    "cov_nontarget_ref_aligned",
    "ref_separation_riemann",
))

NORMALIZATION_FIELDS: frozenset[str] = frozenset((
    "mu_social_cal",
    "sigma_social_cal",
    "raw_social_calibration_scores",
    "disengagement_threshold_raw",
    "reference_subset_label",
    "n_reference_scores",
))

QC_CSV_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "session_id",
    "ref_separation_riemann",
    "online_score_raw_std",
    "online_global_below_threshold_fraction",
    "online_session_z_below0_fraction",
    "online_drop_below_10_fraction",
    "online_drop_below_20_fraction",
    "online_drop_below_30_fraction",
    "scale_degenerate_flag",
    "weak_reference_flag",
    "flat_online_flag",
    "online_mdm_accuracy",
)

DROP_QC_THRESHOLDS: tuple[float, ...] = (0.10, 0.20, 0.30)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step5Config:
    """Configuration for the Step 5 attention-score pipeline."""

    step3_root: Path = Path("data/features")
    step4_root: Path = Path("data/features")
    output_root: Path = Path("data/scores")

    # Dimensional constants (frozen from Steps 3-4)
    n_channels: int = 8
    epoch_samples: int = 128
    covariance_dim: int = 24
    calibration_blocks: int = 20
    online_blocks: int = 50
    calibration_social_start: int = 10

    # Numeric tolerances
    symmetry_tolerance: float = 1e-8
    weak_reference_threshold: float = 0.1
    flat_online_threshold: float = 1e-3
    disengagement_quantile: float = 0.25
    degenerate_scale_threshold: float = 1e-6
    minimum_relative_drop_baseline: float = 1e-6

    # K range (for validation)
    calibration_runs_per_block: int = 10
    online_k_min: int = 3
    online_k_max: int = 10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionScores:
    """In-memory raw scores and metadata for one session, before normalization."""

    subject_id: str
    session_id: str

    # Raw scores (float64, before cast-to-float32 and normalization)
    score_raw_cal: np.ndarray          # [20]
    score_raw_online: np.ndarray       # [50]
    d_target_cal: np.ndarray           # [20]
    d_nontarget_cal: np.ndarray        # [20]
    d_target_online: np.ndarray        # [50]
    d_nontarget_online: np.ndarray     # [50]
    mdm_correct_cal: np.ndarray        # int8 [20]
    mdm_correct_online: np.ndarray     # int8 [50]

    # Aligned references
    cov_target_ref_aligned: np.ndarray       # float64 [24, 24]
    cov_nontarget_ref_aligned: np.ndarray    # float64 [24, 24]
    ref_separation_riemann: float

    # Metadata pass-through from Step 4
    cal_metadata: dict[str, np.ndarray]
    online_metadata: dict[str, np.ndarray]

    # QC diagnostics
    warning_lines: tuple[str, ...]


@dataclass(frozen=True)
class NormalizationParams:
    """Global normalization parameters fitted on social calibration scores."""

    mu_social_cal: float
    sigma_social_cal: float
    disengagement_threshold_raw: float
    raw_social_calibration_scores: np.ndarray  # float64 [1050]


@dataclass(frozen=True)
class SessionCalibrationSummary:
    """Session-local calibration summary from social calibration blocks 10-19."""

    mean_raw: float
    std_raw: float
    median_raw: float
    mad_raw: float
    robust_scale_raw: float
    q10_raw: float
    q25_raw: float
    q50_raw: float
    ref_count: int
    scale_degenerate_flag: int


@dataclass(frozen=True)
class Step5Summary:
    """High-level summary of the executed Step 5 run."""

    output_root: Path
    calibration_score_file_count: int
    online_score_file_count: int
    reference_file_count: int
    total_raw_scores: int
    total_normalized_scores: int
    mu_social_cal: float
    sigma_social_cal: float
    disengagement_threshold_raw: float
    ref_separation_min: float
    ref_separation_median: float
    ref_separation_max: float
    score_raw_cal_min: float
    score_raw_cal_max: float
    score_raw_online_min: float
    score_raw_online_max: float
    score_norm_cal_min: float
    score_norm_cal_max: float
    score_norm_online_min: float
    score_norm_online_max: float
    weak_reference_count: int
    flat_online_count: int
    online_mdm_accuracy_mean: float


# ---------------------------------------------------------------------------
# Utility functions (self-contained, Step5Config-aware)
# ---------------------------------------------------------------------------


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def csv_float(value: float) -> str:
    """Serialize a float for exact roundtrip through CSV parsing."""

    return repr(float(value))


def ensure_spd(matrix: np.ndarray, *, config: Step5Config, context: str) -> np.ndarray:
    """Validate that a matrix is finite, symmetric, and strictly SPD.

    Returns:
        Eigenvalues array (ascending order).
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{context} has shape {matrix.shape}, "
            f"expected ({config.covariance_dim}, {config.covariance_dim})",
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} contains non-finite values")

    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > config.symmetry_tolerance:
        raise ValueError(f"{context} symmetry error {symmetry_error:.3e} exceeds tolerance")

    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{context} eigenvalues contain non-finite values")
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{context} is not strictly SPD; min eigenvalue={float(eigenvalues[0]):.6e}")
    return eigenvalues


def matrix_invsqrt_spd(matrix: np.ndarray, *, config: Step5Config, context: str) -> np.ndarray:
    """Compute the inverse square root of an SPD matrix via eigendecomposition."""

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{context} eigenvalues contain non-finite values")
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{context} is not strictly SPD; min eigenvalue={float(eigenvalues[0]):.6e}")

    invsqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    invsqrt = symmetrize(invsqrt)
    if not np.all(np.isfinite(invsqrt)):
        raise ValueError(f"{context} inverse square root contains non-finite values")
    return invsqrt


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


def score_split_path(*, output_root: Path, subject_id: str, session_id: str, split: str) -> Path:
    """Build the Step 5 score NPZ path for one subject/session/split."""

    return output_root / (
        f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}"
        f"_split-{split}_scores.npz"
    )


def reference_path(*, output_root: Path, subject_id: str, session_id: str) -> Path:
    """Build the Step 5 reference NPZ path for one session."""

    return output_root / (
        f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}"
        f"_references.npz"
    )


def normalization_path(*, output_root: Path) -> Path:
    """Build the Step 5 global normalization NPZ path."""

    return output_root / "normalization_params.npz"


def qc_summary_path(*, output_root: Path) -> Path:
    """Build the Step 5 QC summary CSV path."""

    return output_root / "step5_qc_summary.csv"


def step3_validation_config(config: Step5Config) -> Step3Config:
    """Build a Step 3 validation config aligned with Step 5 constants."""

    return Step3Config(
        output_root=config.step3_root,
        epoch_samples=config.epoch_samples,
        n_channels=config.n_channels,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        prototype_block_start=config.calibration_social_start,
        covariance_dim=config.covariance_dim,
        symmetry_tolerance=config.symmetry_tolerance,
    )


def step4_validation_config(config: Step5Config) -> Step4Config:
    """Build a Step 4 validation config aligned with Step 5 constants."""

    return Step4Config(
        input_root=config.step4_root,
        output_root=config.step4_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        calibration_runs_per_block=config.calibration_runs_per_block,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
        covariance_dim=config.covariance_dim,
        symmetry_tolerance=config.symmetry_tolerance,
    )


# ---------------------------------------------------------------------------
# Sub-Step 5.1 — Load and verify inputs
# ---------------------------------------------------------------------------


def load_session_inputs(
    *,
    subject_id: str,
    session_id: str,
    config: Step5Config,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Load Step 3 prototypes and Step 4 aligned covariances for one session.

    Returns:
        p_target: [8, 128] target prototype
        p_nontarget: [8, 128] non-target prototype
        aligned_covs_cal: [20, 24, 24] aligned calibration covariances
        aligned_covs_online: [50, 24, 24] aligned online covariances
        session_ref_mean: [24, 24] session Riemannian mean
        cal_metadata: dict of metadata arrays for calibration split
        online_metadata: dict of metadata arrays for online split
    """

    context = f"{subject_id}/{session_id}"
    step3_config = step3_validation_config(config)
    step4_config = step4_validation_config(config)

    # --- Step 3 calibration covariance file (for prototypes) ---
    step3_cal_path = step3_feature_split_path(
        output_root=config.step3_root,
        subject_id=subject_id,
        session_id=session_id,
        split="calibration",
    )
    step3_payload = _validate_saved_covariance_payload(
        path=step3_cal_path,
        split="calibration",
        config=step3_config,
    )

    p_target = np.asarray(step3_payload["P_target"], dtype=np.float64)
    p_nontarget = np.asarray(step3_payload["P_nontarget"], dtype=np.float64)
    if p_target.shape != (config.n_channels, config.epoch_samples):
        raise ValueError(
            f"{context} P_target shape {p_target.shape}, "
            f"expected ({config.n_channels}, {config.epoch_samples})",
        )
    if p_nontarget.shape != (config.n_channels, config.epoch_samples):
        raise ValueError(
            f"{context} P_nontarget shape {p_nontarget.shape}, "
            f"expected ({config.n_channels}, {config.epoch_samples})",
        )
    if np.array_equal(p_target, p_nontarget):
        raise ValueError(f"{context} P_target and P_nontarget are identical")
    if not np.all(np.isfinite(p_target)):
        raise ValueError(f"{context} P_target contains non-finite values")
    if not np.all(np.isfinite(p_nontarget)):
        raise ValueError(f"{context} P_nontarget contains non-finite values")

    # --- Step 4 calibration aligned file ---
    step4_cal_path = step4_aligned_split_path(
        output_root=config.step4_root,
        subject_id=subject_id,
        session_id=session_id,
        split="calibration",
    )
    step4_cal_payload = _validate_saved_aligned_payload(
        path=step4_cal_path,
        split="calibration",
        config=step4_config,
    )

    aligned_covs_cal = np.asarray(step4_cal_payload["covs"], dtype=np.float64)
    expected_cal_shape = (config.calibration_blocks, config.covariance_dim, config.covariance_dim)
    if aligned_covs_cal.shape != expected_cal_shape:
        raise ValueError(
            f"{context} calibration aligned covs shape {aligned_covs_cal.shape}, expected {expected_cal_shape}",
        )

    session_ref_mean = np.asarray(step4_cal_payload["session_ref_mean"], dtype=np.float64)
    ensure_spd(session_ref_mean, config=config, context=f"{context} session_ref_mean")

    # --- Step 4 online aligned file ---
    step4_online_path = step4_aligned_split_path(
        output_root=config.step4_root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
    )
    step4_online_payload = _validate_saved_aligned_payload(
        path=step4_online_path,
        split="online",
        config=step4_config,
    )

    aligned_covs_online = np.asarray(step4_online_payload["covs"], dtype=np.float64)
    expected_online_shape = (config.online_blocks, config.covariance_dim, config.covariance_dim)
    if aligned_covs_online.shape != expected_online_shape:
        raise ValueError(
            f"{context} online aligned covs shape {aligned_covs_online.shape}, expected {expected_online_shape}",
        )

    # Verify session_ref_mean is consistent across both Step 4 files
    online_ref_mean = np.asarray(step4_online_payload["session_ref_mean"], dtype=np.float64)
    if not np.array_equal(session_ref_mean, online_ref_mean):
        raise ValueError(f"{context} session_ref_mean differs between calibration and online Step 4 files")

    # Extract metadata dicts
    cal_metadata: dict[str, np.ndarray] = {}
    online_metadata: dict[str, np.ndarray] = {}
    for field in METADATA_FIELDS:
        if field not in step4_cal_payload:
            raise ValueError(f"{context} calibration aligned file missing metadata field {field!r}")
        if field not in step4_online_payload:
            raise ValueError(f"{context} online aligned file missing metadata field {field!r}")
        cal_metadata[field] = np.asarray(step4_cal_payload[field])
        online_metadata[field] = np.asarray(step4_online_payload[field])

    return (
        p_target,
        p_nontarget,
        aligned_covs_cal,
        aligned_covs_online,
        session_ref_mean,
        cal_metadata,
        online_metadata,
    )


# ---------------------------------------------------------------------------
# Sub-Step 5.2 — Construct target and non-target reference covariances
# ---------------------------------------------------------------------------


def construct_reference_covariances(
    p_target: np.ndarray,
    p_nontarget: np.ndarray,
    *,
    config: Step5Config,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ideal target and non-target reference covariances from prototypes.

    Augmented super-trial construction following Barachant & Congedo (2014):
        X_target_ref   = [P_target; P_nontarget; P_target]      -> [24, 128]
        X_nontarget_ref = [P_target; P_nontarget; P_nontarget]  -> [24, 128]

    OAS shrinkage covariance on X_ref.T [128, 24] produces [24, 24] SPD matrices.

    Returns:
        cov_target_ref: [24, 24] unaligned target reference covariance
        cov_nontarget_ref: [24, 24] unaligned non-target reference covariance
    """

    # Augmented super-trial for ideal target response
    x_target_ref = np.vstack([p_target, p_nontarget, p_target])       # [24, 128]
    # Augmented super-trial for ideal non-target response
    x_nontarget_ref = np.vstack([p_target, p_nontarget, p_nontarget]) # [24, 128]

    if x_target_ref.shape != (config.covariance_dim, config.epoch_samples):
        raise ValueError(
            f"{context} X_target_ref shape {x_target_ref.shape}, "
            f"expected ({config.covariance_dim}, {config.epoch_samples})",
        )
    if x_nontarget_ref.shape != (config.covariance_dim, config.epoch_samples):
        raise ValueError(
            f"{context} X_nontarget_ref shape {x_nontarget_ref.shape}, "
            f"expected ({config.covariance_dim}, {config.epoch_samples})",
        )
    if not np.all(np.isfinite(x_target_ref)):
        raise ValueError(f"{context} X_target_ref contains non-finite values")
    if not np.all(np.isfinite(x_nontarget_ref)):
        raise ValueError(f"{context} X_nontarget_ref contains non-finite values")

    # OAS on transposed data: [128, 24] -> 128 samples in 24-dim space
    cov_target_ref = OAS(store_precision=False, assume_centered=False).fit(x_target_ref.T).covariance_
    cov_target_ref = symmetrize(cov_target_ref)
    ensure_spd(cov_target_ref, config=config, context=f"{context} cov_target_ref")

    cov_nontarget_ref = OAS(store_precision=False, assume_centered=False).fit(x_nontarget_ref.T).covariance_
    cov_nontarget_ref = symmetrize(cov_nontarget_ref)
    ensure_spd(cov_nontarget_ref, config=config, context=f"{context} cov_nontarget_ref")

    return cov_target_ref, cov_nontarget_ref


# ---------------------------------------------------------------------------
# Sub-Step 5.3 — Align references and compute reference separation
# ---------------------------------------------------------------------------


def align_references(
    cov_target_ref: np.ndarray,
    cov_nontarget_ref: np.ndarray,
    session_ref_mean: np.ndarray,
    *,
    config: Step5Config,
    context: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align reference covariances using session_ref_mean and compute separation.

    Applies the same Zanini et al. (2018) recentering as Step 4:
        C_aligned = M^(-1/2) C M^(-1/2)

    Then computes the Riemannian distance between the two aligned references.

    Returns:
        cov_target_ref_aligned: [24, 24] aligned target reference
        cov_nontarget_ref_aligned: [24, 24] aligned non-target reference
        ref_separation_riemann: positive scalar (Riemannian distance)
    """

    m_invsqrt = matrix_invsqrt_spd(session_ref_mean, config=config, context=f"{context} session_ref_mean")

    cov_target_aligned = symmetrize(m_invsqrt @ cov_target_ref @ m_invsqrt)
    ensure_spd(cov_target_aligned, config=config, context=f"{context} cov_target_ref_aligned")

    cov_nontarget_aligned = symmetrize(m_invsqrt @ cov_nontarget_ref @ m_invsqrt)
    ensure_spd(cov_nontarget_aligned, config=config, context=f"{context} cov_nontarget_ref_aligned")

    ref_separation = float(distance_riemann(cov_target_aligned, cov_nontarget_aligned))

    if ref_separation <= 0.0:
        raise ValueError(f"{context} ref_separation_riemann={ref_separation:.6e} must be > 0")

    return cov_target_aligned, cov_nontarget_aligned, ref_separation


# ---------------------------------------------------------------------------
# Sub-Step 5.4 — Compute raw scores for all blocks
# ---------------------------------------------------------------------------


def compute_raw_scores(
    aligned_covs: np.ndarray,
    cov_target_ref_aligned: np.ndarray,
    cov_nontarget_ref_aligned: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw MDM margin scores for a stack of aligned covariances.

    For each block covariance C_block:
        d_target    = d_R(C_block, cov_target_ref_aligned)
        d_nontarget = d_R(C_block, cov_nontarget_ref_aligned)
        score_raw   = d_nontarget - d_target
        mdm_correct = 1 if score_raw > 0 else 0

    Positive score_raw indicates target-like response.

    Returns:
        score_raw: float64 [n_blocks]
        d_target: float64 [n_blocks]
        d_nontarget: float64 [n_blocks]
        mdm_correct: int8 [n_blocks]
    """

    n_blocks = aligned_covs.shape[0]
    score_raw = np.empty(n_blocks, dtype=np.float64)
    d_target = np.empty(n_blocks, dtype=np.float64)
    d_nontarget = np.empty(n_blocks, dtype=np.float64)
    mdm_correct = np.empty(n_blocks, dtype=np.int8)

    for i in range(n_blocks):
        c_block = np.asarray(aligned_covs[i], dtype=np.float64)
        dt = float(distance_riemann(c_block, cov_target_ref_aligned))
        dnt = float(distance_riemann(c_block, cov_nontarget_ref_aligned))

        if not (np.isfinite(dt) and np.isfinite(dnt)):
            raise ValueError(f"{context} block[{i}] distances are non-finite: d_target={dt}, d_nontarget={dnt}")

        d_target[i] = dt
        d_nontarget[i] = dnt
        score_raw[i] = dnt - dt
        mdm_correct[i] = np.int8(1) if (dnt - dt) > 0.0 else np.int8(0)

    return score_raw, d_target, d_nontarget, mdm_correct


# ---------------------------------------------------------------------------
# Session-level processing (Sub-Steps 5.1-5.4 combined)
# ---------------------------------------------------------------------------


def process_session(
    *,
    subject_id: str,
    session_id: str,
    config: Step5Config,
) -> SessionScores:
    """Process one session: load inputs, construct+align references, compute raw scores.

    Does NOT normalize or save — returns in-memory SessionScores.
    """

    context = f"{subject_id}/{session_id}"
    logger.info("Processing %s ...", context)

    # Sub-Step 5.1
    (
        p_target,
        p_nontarget,
        aligned_covs_cal,
        aligned_covs_online,
        session_ref_mean,
        cal_metadata,
        online_metadata,
    ) = load_session_inputs(
        subject_id=subject_id,
        session_id=session_id,
        config=config,
    )

    # Sub-Step 5.2
    cov_target_ref, cov_nontarget_ref = construct_reference_covariances(
        p_target,
        p_nontarget,
        config=config,
        context=context,
    )

    # Sub-Step 5.3
    cov_target_aligned, cov_nontarget_aligned, ref_separation = align_references(
        cov_target_ref,
        cov_nontarget_ref,
        session_ref_mean,
        config=config,
        context=context,
    )

    # Sub-Step 5.4 — calibration
    score_raw_cal, d_target_cal, d_nontarget_cal, mdm_correct_cal = compute_raw_scores(
        aligned_covs_cal,
        cov_target_aligned,
        cov_nontarget_aligned,
        context=f"{context} calibration",
    )

    # Sub-Step 5.4 — online
    score_raw_online, d_target_online, d_nontarget_online, mdm_correct_online = compute_raw_scores(
        aligned_covs_online,
        cov_target_aligned,
        cov_nontarget_aligned,
        context=f"{context} online",
    )

    # QC warnings
    warnings: list[str] = []
    if ref_separation < config.weak_reference_threshold:
        # Conservative first-pass heuristic — Riemannian distances have no universal scale,
        # so this threshold may need adjustment after inspecting the full 105-session distribution.
        warnings.append(
            f"weak reference separation {ref_separation:.6f} < {config.weak_reference_threshold}",
        )

    online_std = float(np.std(score_raw_online, ddof=0))
    if online_std <= config.flat_online_threshold:
        warnings.append(
            f"flat online score_raw std {online_std:.6e} <= {config.flat_online_threshold:.1e}",
        )

    if not warnings:
        warnings.append("none")

    logger.info(
        "  %s  ref_sep=%.6f  cal_score=[%.4f, %.4f]  online_score=[%.4f, %.4f]",
        context,
        ref_separation,
        float(np.min(score_raw_cal)),
        float(np.max(score_raw_cal)),
        float(np.min(score_raw_online)),
        float(np.max(score_raw_online)),
    )

    return SessionScores(
        subject_id=subject_id,
        session_id=session_id,
        score_raw_cal=score_raw_cal,
        score_raw_online=score_raw_online,
        d_target_cal=d_target_cal,
        d_nontarget_cal=d_nontarget_cal,
        d_target_online=d_target_online,
        d_nontarget_online=d_nontarget_online,
        mdm_correct_cal=mdm_correct_cal,
        mdm_correct_online=mdm_correct_online,
        cov_target_ref_aligned=cov_target_aligned,
        cov_nontarget_ref_aligned=cov_nontarget_aligned,
        ref_separation_riemann=ref_separation,
        cal_metadata=cal_metadata,
        online_metadata=online_metadata,
        warning_lines=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Sub-Step 5.5 — Fit global normalization and compute derived score families
# ---------------------------------------------------------------------------


def fit_global_normalization(
    all_session_scores: list[SessionScores],
    *,
    config: Step5Config,
) -> NormalizationParams:
    """Collect social calibration scores (float32 precision), fit mu/sigma.

    Critical: raw scores are cast to float32 (the on-disk precision) BEFORE
    extracting the social calibration subset. This ensures exact roundtrip
    reproducibility when the smoke test recomputes stats from saved files.

    Args:
        all_session_scores: All 105 session results from Phase 1.
        config: Step 5 configuration.

    Returns:
        NormalizationParams with mu, sigma, p25, and the 1050-element reference array.
    """

    reference_scores: list[np.ndarray] = []

    for session in all_session_scores:
        social_scores = extract_social_calibration_scores(
            session.score_raw_cal,
            calibration_metadata=session.cal_metadata,
            config=config,
            context=f"{session.subject_id}/{session.session_id}",
        )
        reference_scores.append(social_scores.astype(np.float64))

    all_ref = np.concatenate(reference_scores)

    expected_count = len(all_session_scores) * (config.calibration_blocks - config.calibration_social_start)
    if all_ref.shape[0] != expected_count:
        raise ValueError(
            f"Expected {expected_count} social calibration reference scores, "
            f"got {all_ref.shape[0]}",
        )

    mu = float(np.mean(all_ref))
    sigma = float(np.std(all_ref, ddof=0))
    if sigma <= 0.0:
        raise ValueError(f"Social calibration score sigma={sigma:.6e} must be > 0")

    p25 = float(np.percentile(all_ref, config.disengagement_quantile * 100, method="linear"))

    logger.info(
        "Global normalization: mu=%.6f, sigma=%.6f, p25=%.6f (from %d scores)",
        mu,
        sigma,
        p25,
        all_ref.shape[0],
    )

    return NormalizationParams(
        mu_social_cal=mu,
        sigma_social_cal=sigma,
        disengagement_threshold_raw=p25,
        raw_social_calibration_scores=all_ref,
    )


def normalize_scores(
    score_raw: np.ndarray,
    normalization: NormalizationParams,
) -> np.ndarray:
    """Apply sigmoid normalization to raw scores.

    Formula: score_norm = sigmoid((score_raw_f32 - mu) / sigma)

    Uses scipy.special.expit for numerical stability.
    Extreme z-scores may round to exactly 0.0 or 1.0 in float32.

    Returns:
        score_norm: float32 array in [0, 1].
    """

    score_raw_f32 = np.asarray(score_raw, dtype=np.float32)
    z = (score_raw_f32.astype(np.float64) - normalization.mu_social_cal) / normalization.sigma_social_cal
    score_norm = expit(z).astype(np.float32)
    if not np.all(np.isfinite(score_norm)):
        raise ValueError("score_norm contains non-finite values")
    if float(np.min(score_norm)) < 0.0 or float(np.max(score_norm)) > 1.0:
        raise ValueError(
            f"score_norm out of range [0,1]: "
            f"[{float(np.min(score_norm)):.6f}, {float(np.max(score_norm)):.6f}]",
        )
    return score_norm


def extract_social_calibration_scores(
    score_raw_cal: np.ndarray,
    *,
    calibration_metadata: dict[str, np.ndarray],
    config: Step5Config,
    context: str,
) -> np.ndarray:
    """Extract saved-precision social calibration scores from blocks 10-19."""

    score_raw_f32 = np.asarray(score_raw_cal, dtype=np.float32)
    if score_raw_f32.shape != (config.calibration_blocks,):
        raise ValueError(
            f"{context} calibration score_raw shape {score_raw_f32.shape}, "
            f"expected ({config.calibration_blocks},)",
        )

    condition = np.asarray(calibration_metadata["condition"], dtype=np.int8)
    block_ids = np.asarray(calibration_metadata["block_ids"], dtype=np.int16)
    if condition.shape != (config.calibration_blocks,) or block_ids.shape != (config.calibration_blocks,):
        raise ValueError(f"{context} calibration metadata shapes do not match expected block count")

    social_mask = (
        (condition == 1)
        & (block_ids >= config.calibration_social_start)
        & (block_ids < config.calibration_blocks)
    )
    expected_ref_count = config.calibration_blocks - config.calibration_social_start
    if int(np.sum(social_mask)) != expected_ref_count:
        raise ValueError(
            f"{context} social calibration mask selected {int(np.sum(social_mask))} blocks, "
            f"expected {expected_ref_count}",
        )
    social_scores = score_raw_f32[social_mask]
    if social_scores.shape != (expected_ref_count,):
        raise ValueError(
            f"{context} social calibration score shape {social_scores.shape}, "
            f"expected ({expected_ref_count},)",
        )
    if not np.all(np.isfinite(social_scores)):
        raise ValueError(f"{context} social calibration scores contain non-finite values")
    return social_scores


def compute_session_calibration_summary(
    score_raw_cal: np.ndarray,
    *,
    calibration_metadata: dict[str, np.ndarray],
    config: Step5Config,
    context: str,
) -> SessionCalibrationSummary:
    """Compute session-local calibration summaries from saved-precision raw scores."""

    social_scores = extract_social_calibration_scores(
        score_raw_cal,
        calibration_metadata=calibration_metadata,
        config=config,
        context=context,
    ).astype(np.float64)

    mean_raw = float(np.mean(social_scores))
    if mean_raw <= config.minimum_relative_drop_baseline:
        raise ValueError(
            f"{context} session_cal_mean_raw={mean_raw:.6e} must be > "
            f"{config.minimum_relative_drop_baseline:.1e}",
        )

    std_raw = float(np.std(social_scores, ddof=0))
    median_raw = float(np.median(social_scores))
    mad_raw = float(np.median(np.abs(social_scores - median_raw)))
    robust_scale_raw = float(1.4826 * mad_raw)
    q10_raw = float(np.percentile(social_scores, 10.0, method="linear"))
    q25_raw = float(np.percentile(social_scores, 25.0, method="linear"))
    q50_raw = float(np.percentile(social_scores, 50.0, method="linear"))
    scale_degenerate_flag = int(
        robust_scale_raw <= config.degenerate_scale_threshold
        and std_raw <= config.degenerate_scale_threshold
    )

    float_values = (
        mean_raw,
        std_raw,
        median_raw,
        mad_raw,
        robust_scale_raw,
        q10_raw,
        q25_raw,
        q50_raw,
    )
    if not np.all(np.isfinite(np.asarray(float_values, dtype=np.float64))):
        raise ValueError(f"{context} session calibration summary contains non-finite values")
    if q10_raw > q25_raw or q25_raw > q50_raw:
        raise ValueError(f"{context} session calibration quantiles are not ordered")

    return SessionCalibrationSummary(
        mean_raw=mean_raw,
        std_raw=std_raw,
        median_raw=median_raw,
        mad_raw=mad_raw,
        robust_scale_raw=robust_scale_raw,
        q10_raw=q10_raw,
        q25_raw=q25_raw,
        q50_raw=q50_raw,
        ref_count=social_scores.shape[0],
        scale_degenerate_flag=scale_degenerate_flag,
    )


def compute_global_threshold_relative_scores(
    score_raw: np.ndarray,
    *,
    disengagement_threshold_raw: float,
) -> np.ndarray:
    """Compute the legacy global threshold-relative score from saved-precision raw values."""

    score_raw_f32 = np.asarray(score_raw, dtype=np.float32)
    score_threshold_rel_global = (
        score_raw_f32.astype(np.float64) - float(disengagement_threshold_raw)
    ).astype(np.float32)
    if not np.all(np.isfinite(score_threshold_rel_global)):
        raise ValueError("score_threshold_rel_global contains non-finite values")
    return score_threshold_rel_global


def compute_session_robust_z_scores(
    score_raw: np.ndarray,
    *,
    session_summary: SessionCalibrationSummary,
    config: Step5Config,
) -> np.ndarray:
    """Compute session-local robust z scores from saved-precision raw values."""

    score_raw_f32 = np.asarray(score_raw, dtype=np.float32)
    if session_summary.scale_degenerate_flag:
        return np.zeros_like(score_raw_f32, dtype=np.float32)

    scale = session_summary.robust_scale_raw
    if scale <= config.degenerate_scale_threshold:
        scale = session_summary.std_raw
    if scale <= config.degenerate_scale_threshold:
        raise ValueError("session robust z scale is degenerate without scale_degenerate_flag")

    score_session_robust_z = (
        (score_raw_f32.astype(np.float64) - session_summary.median_raw) / float(scale)
    ).astype(np.float32)
    if not np.all(np.isfinite(score_session_robust_z)):
        raise ValueError("score_session_robust_z contains non-finite values")
    return score_session_robust_z


def compute_session_relative_drop_scores(
    score_raw: np.ndarray,
    *,
    session_cal_mean_raw: float,
    config: Step5Config,
) -> np.ndarray:
    """Compute session-relative drop scores from saved-precision raw values."""

    baseline = float(session_cal_mean_raw)
    if baseline <= config.minimum_relative_drop_baseline:
        raise ValueError(
            f"session_cal_mean_raw={baseline:.6e} must be > {config.minimum_relative_drop_baseline:.1e}",
        )

    score_raw_f32 = np.asarray(score_raw, dtype=np.float32)
    score_drop_rel_session = (
        score_raw_f32.astype(np.float64) / baseline - 1.0
    ).astype(np.float32)
    if not np.all(np.isfinite(score_drop_rel_session)):
        raise ValueError("score_drop_rel_session contains non-finite values")
    return score_drop_rel_session


# ---------------------------------------------------------------------------
# Sub-Step 5.6 — Save all files
# ---------------------------------------------------------------------------


def save_score_npz(
    *,
    path: Path,
    score_raw: np.ndarray,
    score_norm: np.ndarray,
    score_threshold_rel_global: np.ndarray,
    score_session_robust_z: np.ndarray,
    score_drop_rel_session: np.ndarray,
    d_target: np.ndarray,
    d_nontarget: np.ndarray,
    mdm_correct: np.ndarray,
    session_summary: SessionCalibrationSummary,
    metadata: dict[str, np.ndarray],
) -> None:
    """Save one Step 5 score NPZ file (calibration or online)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        score_raw=np.asarray(score_raw, dtype=np.float32),
        score_norm=np.asarray(score_norm, dtype=np.float32),
        score_threshold_rel_global=np.asarray(score_threshold_rel_global, dtype=np.float32),
        score_session_robust_z=np.asarray(score_session_robust_z, dtype=np.float32),
        score_drop_rel_session=np.asarray(score_drop_rel_session, dtype=np.float32),
        d_target=np.asarray(d_target, dtype=np.float32),
        d_nontarget=np.asarray(d_nontarget, dtype=np.float32),
        mdm_correct=np.asarray(mdm_correct, dtype=np.int8),
        session_cal_mean_raw=np.asarray(session_summary.mean_raw, dtype=np.float64),
        session_cal_std_raw=np.asarray(session_summary.std_raw, dtype=np.float64),
        session_cal_median_raw=np.asarray(session_summary.median_raw, dtype=np.float64),
        session_cal_mad_raw=np.asarray(session_summary.mad_raw, dtype=np.float64),
        session_cal_robust_scale_raw=np.asarray(session_summary.robust_scale_raw, dtype=np.float64),
        session_cal_q10_raw=np.asarray(session_summary.q10_raw, dtype=np.float64),
        session_cal_q25_raw=np.asarray(session_summary.q25_raw, dtype=np.float64),
        session_cal_q50_raw=np.asarray(session_summary.q50_raw, dtype=np.float64),
        session_cal_ref_count=np.asarray(session_summary.ref_count, dtype=np.int16),
        scale_degenerate_flag=np.asarray(session_summary.scale_degenerate_flag, dtype=np.int8),
        block_ids=np.asarray(metadata["block_ids"], dtype=np.int16),
        condition=np.asarray(metadata["condition"], dtype=np.int8),
        block_targets=np.asarray(metadata["block_targets"], dtype=np.int8),
        runs_per_block=np.asarray(metadata["runs_per_block"], dtype=np.int16),
        n_target_epochs=np.asarray(metadata["n_target_epochs"], dtype=np.int16),
        n_nontarget_epochs=np.asarray(metadata["n_nontarget_epochs"], dtype=np.int16),
    )


def save_reference_npz(
    *,
    path: Path,
    cov_target_ref_aligned: np.ndarray,
    cov_nontarget_ref_aligned: np.ndarray,
    ref_separation_riemann: float,
) -> None:
    """Save one Step 5 session reference NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cov_target_ref_aligned=np.asarray(cov_target_ref_aligned, dtype=np.float64),
        cov_nontarget_ref_aligned=np.asarray(cov_nontarget_ref_aligned, dtype=np.float64),
        ref_separation_riemann=np.asarray(ref_separation_riemann, dtype=np.float64),
    )


def save_normalization_npz(
    *,
    path: Path,
    normalization: NormalizationParams,
) -> None:
    """Save the global Step 5 normalization parameters NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mu_social_cal=np.asarray(normalization.mu_social_cal, dtype=np.float64),
        sigma_social_cal=np.asarray(normalization.sigma_social_cal, dtype=np.float64),
        raw_social_calibration_scores=np.asarray(
            normalization.raw_social_calibration_scores, dtype=np.float64,
        ),
        disengagement_threshold_raw=np.asarray(normalization.disengagement_threshold_raw, dtype=np.float64),
        reference_subset_label=np.asarray("social_calibration_only"),
        n_reference_scores=np.asarray(
            normalization.raw_social_calibration_scores.shape[0], dtype=np.int16,
        ),
    )


def save_qc_summary_csv(
    *,
    path: Path,
    all_session_scores: list[SessionScores],
    normalization: NormalizationParams,
    config: Step5Config,
) -> None:
    """Save the Step 5 QC summary CSV with one row per session."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QC_CSV_COLUMNS)
        writer.writeheader()
        for session in all_session_scores:
            online_score_raw_f32 = np.asarray(session.score_raw_online, dtype=np.float32)
            session_summary = compute_session_calibration_summary(
                session.score_raw_cal,
                calibration_metadata=session.cal_metadata,
                config=config,
                context=f"{session.subject_id}/{session.session_id}",
            )
            online_score_threshold_rel_global = compute_global_threshold_relative_scores(
                online_score_raw_f32,
                disengagement_threshold_raw=normalization.disengagement_threshold_raw,
            )
            online_score_session_z = compute_session_robust_z_scores(
                online_score_raw_f32,
                session_summary=session_summary,
                config=config,
            )
            online_score_drop_rel = compute_session_relative_drop_scores(
                online_score_raw_f32,
                session_cal_mean_raw=session_summary.mean_raw,
                config=config,
            )
            online_std = float(np.std(online_score_raw_f32, ddof=0))
            online_global_below_threshold_fraction = float(np.mean(online_score_threshold_rel_global < 0.0))
            online_session_z_below0_fraction = float(np.mean(online_score_session_z < 0.0))
            online_drop_below_10_fraction = float(np.mean(online_score_drop_rel < -DROP_QC_THRESHOLDS[0]))
            online_drop_below_20_fraction = float(np.mean(online_score_drop_rel < -DROP_QC_THRESHOLDS[1]))
            online_drop_below_30_fraction = float(np.mean(online_score_drop_rel < -DROP_QC_THRESHOLDS[2]))
            online_correct = int(np.sum(session.mdm_correct_online))
            online_total = session.mdm_correct_online.shape[0]
            online_mdm_accuracy = online_correct / online_total if online_total > 0 else 0.0

            writer.writerow({
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "ref_separation_riemann": csv_float(session.ref_separation_riemann),
                "online_score_raw_std": csv_float(online_std),
                "online_global_below_threshold_fraction": csv_float(online_global_below_threshold_fraction),
                "online_session_z_below0_fraction": csv_float(online_session_z_below0_fraction),
                "online_drop_below_10_fraction": csv_float(online_drop_below_10_fraction),
                "online_drop_below_20_fraction": csv_float(online_drop_below_20_fraction),
                "online_drop_below_30_fraction": csv_float(online_drop_below_30_fraction),
                "scale_degenerate_flag": session_summary.scale_degenerate_flag,
                "weak_reference_flag": 1 if session.ref_separation_riemann < config.weak_reference_threshold else 0,
                "flat_online_flag": 1 if online_std <= config.flat_online_threshold else 0,
                "online_mdm_accuracy": csv_float(online_mdm_accuracy),
            })


# ---------------------------------------------------------------------------
# Sub-Step 5.7 — Dataset-wide smoke tests
# ---------------------------------------------------------------------------


def _validate_saved_score_payload(
    *,
    path: Path,
    split: str,
    config: Step5Config,
) -> dict[str, np.ndarray]:
    """Load and validate one saved Step 5 score NPZ file."""

    payload = load_npz_payload(path)
    missing = sorted(SCORE_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"{path.name} missing required Step 5 fields: {missing}")
    unexpected = sorted(set(payload).difference(SCORE_FIELDS))
    if unexpected:
        raise ValueError(f"{path.name} contains unexpected Step 5 fields: {unexpected}")

    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks

    # Shape checks
    for field in (
        "score_raw",
        "score_norm",
        "score_threshold_rel_global",
        "score_session_robust_z",
        "score_drop_rel_session",
        "d_target",
        "d_nontarget",
        "mdm_correct",
    ):
        arr = np.asarray(payload[field])
        if arr.shape != (expected_blocks,):
            raise ValueError(f"{path.name} {field} shape {arr.shape}, expected ({expected_blocks},)")

    # Dtype checks
    for field in (
        "score_raw",
        "score_norm",
        "score_threshold_rel_global",
        "score_session_robust_z",
        "score_drop_rel_session",
        "d_target",
        "d_nontarget",
    ):
        if np.asarray(payload[field]).dtype != np.dtype(np.float32):
            raise ValueError(f"{path.name} {field} dtype {np.asarray(payload[field]).dtype}, expected float32")
    if np.asarray(payload["mdm_correct"]).dtype != np.dtype(np.int8):
        raise ValueError(f"{path.name} mdm_correct dtype {np.asarray(payload['mdm_correct']).dtype}, expected int8")
    scalar_dtypes = {
        "session_cal_mean_raw": np.dtype(np.float64),
        "session_cal_std_raw": np.dtype(np.float64),
        "session_cal_median_raw": np.dtype(np.float64),
        "session_cal_mad_raw": np.dtype(np.float64),
        "session_cal_robust_scale_raw": np.dtype(np.float64),
        "session_cal_q10_raw": np.dtype(np.float64),
        "session_cal_q25_raw": np.dtype(np.float64),
        "session_cal_q50_raw": np.dtype(np.float64),
        "session_cal_ref_count": np.dtype(np.int16),
        "scale_degenerate_flag": np.dtype(np.int8),
    }
    for field, expected_dtype in scalar_dtypes.items():
        arr = np.asarray(payload[field])
        if arr.shape != ():
            raise ValueError(f"{path.name} {field} shape {arr.shape}, expected scalar")
        if arr.dtype != expected_dtype:
            raise ValueError(f"{path.name} {field} dtype {arr.dtype}, expected {expected_dtype}")

    # Finiteness
    for field in (
        "score_raw",
        "score_norm",
        "score_threshold_rel_global",
        "score_session_robust_z",
        "score_drop_rel_session",
        "d_target",
        "d_nontarget",
    ):
        if not np.all(np.isfinite(payload[field])):
            raise ValueError(f"{path.name} {field} contains non-finite values")
    for field in (
        "session_cal_mean_raw",
        "session_cal_std_raw",
        "session_cal_median_raw",
        "session_cal_mad_raw",
        "session_cal_robust_scale_raw",
        "session_cal_q10_raw",
        "session_cal_q25_raw",
        "session_cal_q50_raw",
    ):
        if not np.isfinite(float(np.asarray(payload[field]))):
            raise ValueError(f"{path.name} {field} must be finite")

    # Normalized score range
    score_norm = np.asarray(payload["score_norm"], dtype=np.float32)
    if float(np.min(score_norm)) < 0.0 or float(np.max(score_norm)) > 1.0:
        raise ValueError(
            f"{path.name} score_norm out of range [0,1]: "
            f"[{float(np.min(score_norm)):.6f}, {float(np.max(score_norm)):.6f}]",
        )

    # mdm_correct consistency with score_raw sign
    score_raw = np.asarray(payload["score_raw"], dtype=np.float32)
    mdm_correct = np.asarray(payload["mdm_correct"], dtype=np.int8)
    expected_mdm = np.where(score_raw > 0.0, np.int8(1), np.int8(0))
    if not np.array_equal(mdm_correct, expected_mdm):
        mismatches = int(np.sum(mdm_correct != expected_mdm))
        raise ValueError(f"{path.name} mdm_correct has {mismatches} mismatches with score_raw sign")
    if not set(mdm_correct.tolist()).issubset({0, 1}):
        raise ValueError(f"{path.name} mdm_correct contains values outside {{0,1}}")

    session_cal_mean_raw = float(np.asarray(payload["session_cal_mean_raw"]))
    if session_cal_mean_raw <= config.minimum_relative_drop_baseline:
        raise ValueError(
            f"{path.name} session_cal_mean_raw={session_cal_mean_raw:.6e} must be > "
            f"{config.minimum_relative_drop_baseline:.1e}",
        )
    session_cal_ref_count = int(np.asarray(payload["session_cal_ref_count"]))
    expected_ref_count = config.calibration_blocks - config.calibration_social_start
    if session_cal_ref_count != expected_ref_count:
        raise ValueError(
            f"{path.name} session_cal_ref_count={session_cal_ref_count}, expected {expected_ref_count}",
        )
    scale_degenerate_flag = int(np.asarray(payload["scale_degenerate_flag"]))
    if scale_degenerate_flag not in {0, 1}:
        raise ValueError(f"{path.name} scale_degenerate_flag={scale_degenerate_flag}, expected 0 or 1")

    session_cal_std_raw = float(np.asarray(payload["session_cal_std_raw"]))
    session_cal_mad_raw = float(np.asarray(payload["session_cal_mad_raw"]))
    session_cal_robust_scale_raw = float(np.asarray(payload["session_cal_robust_scale_raw"]))
    session_cal_q10_raw = float(np.asarray(payload["session_cal_q10_raw"]))
    session_cal_q25_raw = float(np.asarray(payload["session_cal_q25_raw"]))
    session_cal_q50_raw = float(np.asarray(payload["session_cal_q50_raw"]))
    if session_cal_std_raw < 0.0 or session_cal_mad_raw < 0.0 or session_cal_robust_scale_raw < 0.0:
        raise ValueError(f"{path.name} session calibration scale summaries must be non-negative")
    if session_cal_q10_raw > session_cal_q25_raw or session_cal_q25_raw > session_cal_q50_raw:
        raise ValueError(f"{path.name} session calibration quantiles are not ordered")
    expected_scale_degenerate_flag = int(
        session_cal_robust_scale_raw <= config.degenerate_scale_threshold
        and session_cal_std_raw <= config.degenerate_scale_threshold
    )
    if scale_degenerate_flag != expected_scale_degenerate_flag:
        raise ValueError(
            f"{path.name} scale_degenerate_flag={scale_degenerate_flag}, "
            f"expected {expected_scale_degenerate_flag}",
        )

    # Metadata dtype checks
    metadata_dtypes = {
        "block_ids": np.dtype(np.int16),
        "condition": np.dtype(np.int8),
        "block_targets": np.dtype(np.int8),
        "runs_per_block": np.dtype(np.int16),
        "n_target_epochs": np.dtype(np.int16),
        "n_nontarget_epochs": np.dtype(np.int16),
    }
    for field, expected_dtype in metadata_dtypes.items():
        actual_dtype = np.asarray(payload[field]).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"{path.name} {field} dtype {actual_dtype}, expected {expected_dtype}")
        if np.asarray(payload[field]).shape != (expected_blocks,):
            raise ValueError(
                f"{path.name} {field} shape {np.asarray(payload[field]).shape}, expected ({expected_blocks},)",
            )

    return payload


def _session_summary_from_saved_payload(payload: dict[str, np.ndarray]) -> SessionCalibrationSummary:
    """Reconstruct the saved session calibration summary dataclass from one score payload."""

    return SessionCalibrationSummary(
        mean_raw=float(np.asarray(payload["session_cal_mean_raw"])),
        std_raw=float(np.asarray(payload["session_cal_std_raw"])),
        median_raw=float(np.asarray(payload["session_cal_median_raw"])),
        mad_raw=float(np.asarray(payload["session_cal_mad_raw"])),
        robust_scale_raw=float(np.asarray(payload["session_cal_robust_scale_raw"])),
        q10_raw=float(np.asarray(payload["session_cal_q10_raw"])),
        q25_raw=float(np.asarray(payload["session_cal_q25_raw"])),
        q50_raw=float(np.asarray(payload["session_cal_q50_raw"])),
        ref_count=int(np.asarray(payload["session_cal_ref_count"])),
        scale_degenerate_flag=int(np.asarray(payload["scale_degenerate_flag"])),
    )


def _validate_saved_reference_payload(
    *,
    path: Path,
    config: Step5Config,
) -> dict[str, np.ndarray]:
    """Load and validate one saved Step 5 reference NPZ file."""

    payload = load_npz_payload(path)
    missing = sorted(REFERENCE_NPZ_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"{path.name} missing required reference fields: {missing}")
    unexpected = sorted(set(payload).difference(REFERENCE_NPZ_FIELDS))
    if unexpected:
        raise ValueError(f"{path.name} contains unexpected reference fields: {unexpected}")

    for field in ("cov_target_ref_aligned", "cov_nontarget_ref_aligned"):
        cov = np.asarray(payload[field], dtype=np.float64)
        if cov.dtype != np.dtype(np.float64):
            raise ValueError(f"{path.name} {field} dtype {cov.dtype}, expected float64")
        ensure_spd(cov, config=config, context=f"{path.name} {field}")

    ref_sep_array = np.asarray(payload["ref_separation_riemann"])
    if ref_sep_array.dtype != np.dtype(np.float64):
        raise ValueError(
            f"{path.name} ref_separation_riemann dtype {ref_sep_array.dtype}, expected float64",
        )
    ref_sep = float(ref_sep_array)
    if not np.isfinite(ref_sep) or ref_sep <= 0.0:
        raise ValueError(f"{path.name} ref_separation_riemann={ref_sep:.6e} must be finite and > 0")

    return payload


def _validate_saved_normalization_payload(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the saved Step 5 normalization NPZ file."""

    payload = load_npz_payload(path)
    missing = sorted(NORMALIZATION_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"{path.name} missing required normalization fields: {missing}")
    unexpected = sorted(set(payload).difference(NORMALIZATION_FIELDS))
    if unexpected:
        raise ValueError(f"{path.name} contains unexpected normalization fields: {unexpected}")

    expected_dtypes = {
        "mu_social_cal": np.dtype(np.float64),
        "sigma_social_cal": np.dtype(np.float64),
        "raw_social_calibration_scores": np.dtype(np.float64),
        "disengagement_threshold_raw": np.dtype(np.float64),
        "n_reference_scores": np.dtype(np.int16),
    }
    for field_name, expected_dtype in expected_dtypes.items():
        actual_dtype = np.asarray(payload[field_name]).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"{path.name} {field_name} dtype {actual_dtype}, expected {expected_dtype}")

    raw_social = np.asarray(payload["raw_social_calibration_scores"], dtype=np.float64)
    if raw_social.shape != (1050,):
        raise ValueError(f"{path.name} raw_social_calibration_scores shape {raw_social.shape}, expected (1050,)")
    if not np.all(np.isfinite(raw_social)):
        raise ValueError(f"{path.name} raw_social_calibration_scores contain non-finite values")

    mu = float(np.asarray(payload["mu_social_cal"]))
    sigma = float(np.asarray(payload["sigma_social_cal"]))
    p25 = float(np.asarray(payload["disengagement_threshold_raw"]))
    if not np.isfinite(mu):
        raise ValueError(f"{path.name} mu_social_cal must be finite")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"{path.name} sigma_social_cal={sigma:.6e} must be finite and > 0")
    if not np.isfinite(p25):
        raise ValueError(f"{path.name} disengagement_threshold_raw must be finite")

    label_array = np.asarray(payload["reference_subset_label"])
    label = label_array.item() if label_array.shape == () else None
    if label != "social_calibration_only":
        raise ValueError(f"{path.name} reference_subset_label={label!r}, expected 'social_calibration_only'")

    n_reference_scores = int(np.asarray(payload["n_reference_scores"]))
    if n_reference_scores != raw_social.shape[0]:
        raise ValueError(
            f"{path.name} n_reference_scores={n_reference_scores}, expected {raw_social.shape[0]}",
        )

    return payload


def smoke_test_outputs(config: Step5Config) -> Step5Summary:
    """Run dataset-wide smoke tests across all Step 5 outputs.

    Validates all saved files: shapes, finiteness, ranges, metadata consistency,
    normalization roundtrip, total counts, and QC CSV contents.
    """

    calibration_score_file_count = 0
    online_score_file_count = 0
    reference_file_count = 0
    total_raw_scores = 0
    total_normalized_scores = 0
    total_global_threshold_scores = 0
    total_session_robust_z_scores = 0
    total_drop_rel_scores = 0

    ref_separations: list[float] = []
    score_raw_cal_values: list[float] = []
    score_raw_online_values: list[float] = []
    score_norm_cal_values: list[float] = []
    score_norm_online_values: list[float] = []
    weak_reference_count = 0
    flat_online_count = 0
    online_mdm_accuracies: list[float] = []
    weak_reference_sessions: list[str] = []
    flat_online_sessions: list[str] = []
    expected_qc_rows: dict[tuple[str, str], dict[str, str]] = {}

    # Collect social calibration scores for normalization roundtrip
    social_cal_scores_f32: list[np.ndarray] = []
    step4_config = step4_validation_config(config)
    session_pairs = expected_session_ids()
    expected_session_count = len(session_pairs)
    norm_path = normalization_path(output_root=config.output_root)
    norm_payload = _validate_saved_normalization_payload(norm_path)
    saved_mu = float(np.asarray(norm_payload["mu_social_cal"]))
    saved_sigma = float(np.asarray(norm_payload["sigma_social_cal"]))
    saved_p25 = float(np.asarray(norm_payload["disengagement_threshold_raw"]))
    saved_n = int(np.asarray(norm_payload["n_reference_scores"]))
    saved_raw_social_cal = np.asarray(norm_payload["raw_social_calibration_scores"], dtype=np.float64)

    for subject_id, session_id in session_pairs:
        context = f"{subject_id}/{session_id}"

        # --- Validate score files ---
        cal_score_path = score_split_path(
            output_root=config.output_root, subject_id=subject_id, session_id=session_id, split="calibration",
        )
        online_score_path = score_split_path(
            output_root=config.output_root, subject_id=subject_id, session_id=session_id, split="online",
        )
        ref_file_path = reference_path(
            output_root=config.output_root, subject_id=subject_id, session_id=session_id,
        )

        cal_payload = _validate_saved_score_payload(path=cal_score_path, split="calibration", config=config)
        online_payload = _validate_saved_score_payload(path=online_score_path, split="online", config=config)
        ref_payload = _validate_saved_reference_payload(path=ref_file_path, config=config)

        calibration_score_file_count += 1
        online_score_file_count += 1
        reference_file_count += 1
        total_raw_scores += cal_payload["score_raw"].shape[0] + online_payload["score_raw"].shape[0]
        total_normalized_scores += cal_payload["score_norm"].shape[0] + online_payload["score_norm"].shape[0]
        total_global_threshold_scores += (
            cal_payload["score_threshold_rel_global"].shape[0] + online_payload["score_threshold_rel_global"].shape[0]
        )
        total_session_robust_z_scores += (
            cal_payload["score_session_robust_z"].shape[0] + online_payload["score_session_robust_z"].shape[0]
        )
        total_drop_rel_scores += (
            cal_payload["score_drop_rel_session"].shape[0] + online_payload["score_drop_rel_session"].shape[0]
        )

        # Collect statistics
        ref_sep = float(np.asarray(ref_payload["ref_separation_riemann"]))
        ref_separations.append(ref_sep)
        if ref_sep < config.weak_reference_threshold:
            weak_reference_count += 1
            weak_reference_sessions.append(context)

        cal_raw = np.asarray(cal_payload["score_raw"], dtype=np.float32)
        online_raw = np.asarray(online_payload["score_raw"], dtype=np.float32)
        cal_threshold_rel_global = np.asarray(cal_payload["score_threshold_rel_global"], dtype=np.float32)
        online_threshold_rel_global = np.asarray(online_payload["score_threshold_rel_global"], dtype=np.float32)
        cal_session_robust_z = np.asarray(cal_payload["score_session_robust_z"], dtype=np.float32)
        online_session_robust_z = np.asarray(online_payload["score_session_robust_z"], dtype=np.float32)
        cal_drop_rel_session = np.asarray(cal_payload["score_drop_rel_session"], dtype=np.float32)
        online_drop_rel_session = np.asarray(online_payload["score_drop_rel_session"], dtype=np.float32)
        score_raw_cal_values.extend(cal_raw.tolist())
        score_raw_online_values.extend(online_raw.tolist())

        cal_norm = np.asarray(cal_payload["score_norm"], dtype=np.float32)
        online_norm = np.asarray(online_payload["score_norm"], dtype=np.float32)
        score_norm_cal_values.extend(cal_norm.tolist())
        score_norm_online_values.extend(online_norm.tolist())

        recomputed_session_summary = compute_session_calibration_summary(
            cal_raw,
            calibration_metadata={
                field: np.asarray(cal_payload[field])
                for field in METADATA_FIELDS
            },
            config=config,
            context=context,
        )
        saved_cal_session_summary = _session_summary_from_saved_payload(cal_payload)
        saved_online_session_summary = _session_summary_from_saved_payload(online_payload)
        if saved_cal_session_summary != recomputed_session_summary:
            raise ValueError(f"{context} calibration session summary does not match saved-precision recomputation")
        if saved_online_session_summary != recomputed_session_summary:
            raise ValueError(f"{context} online session summary does not match calibration session summary")

        expected_cal_threshold_rel_global = compute_global_threshold_relative_scores(
            cal_raw,
            disengagement_threshold_raw=saved_p25,
        )
        expected_online_threshold_rel_global = compute_global_threshold_relative_scores(
            online_raw,
            disengagement_threshold_raw=saved_p25,
        )
        if not np.array_equal(cal_threshold_rel_global, expected_cal_threshold_rel_global):
            raise ValueError(
                f"{context} calibration score_threshold_rel_global does not match score_raw - global threshold",
            )
        if not np.array_equal(online_threshold_rel_global, expected_online_threshold_rel_global):
            raise ValueError(
                f"{context} online score_threshold_rel_global does not match score_raw - global threshold",
            )

        expected_cal_session_robust_z = compute_session_robust_z_scores(
            cal_raw,
            session_summary=recomputed_session_summary,
            config=config,
        )
        expected_online_session_robust_z = compute_session_robust_z_scores(
            online_raw,
            session_summary=recomputed_session_summary,
            config=config,
        )
        if not np.array_equal(cal_session_robust_z, expected_cal_session_robust_z):
            raise ValueError(f"{context} calibration score_session_robust_z does not match recomputation")
        if not np.array_equal(online_session_robust_z, expected_online_session_robust_z):
            raise ValueError(f"{context} online score_session_robust_z does not match recomputation")

        expected_cal_drop_rel_session = compute_session_relative_drop_scores(
            cal_raw,
            session_cal_mean_raw=recomputed_session_summary.mean_raw,
            config=config,
        )
        expected_online_drop_rel_session = compute_session_relative_drop_scores(
            online_raw,
            session_cal_mean_raw=recomputed_session_summary.mean_raw,
            config=config,
        )
        if not np.array_equal(cal_drop_rel_session, expected_cal_drop_rel_session):
            raise ValueError(f"{context} calibration score_drop_rel_session does not match recomputation")
        if not np.array_equal(online_drop_rel_session, expected_online_drop_rel_session):
            raise ValueError(f"{context} online score_drop_rel_session does not match recomputation")

        online_std = float(np.std(online_raw, ddof=0))
        online_global_below_threshold_fraction = float(np.mean(online_threshold_rel_global < 0.0))
        online_session_z_below0_fraction = float(np.mean(online_session_robust_z < 0.0))
        online_drop_below_10_fraction = float(np.mean(online_drop_rel_session < -DROP_QC_THRESHOLDS[0]))
        online_drop_below_20_fraction = float(np.mean(online_drop_rel_session < -DROP_QC_THRESHOLDS[1]))
        online_drop_below_30_fraction = float(np.mean(online_drop_rel_session < -DROP_QC_THRESHOLDS[2]))
        if online_std <= config.flat_online_threshold:
            flat_online_count += 1
            flat_online_sessions.append(context)

        online_correct = int(np.sum(np.asarray(online_payload["mdm_correct"], dtype=np.int8)))
        online_total = online_payload["mdm_correct"].shape[0]
        online_mdm_accuracy = online_correct / online_total if online_total > 0 else 0.0
        online_mdm_accuracies.append(online_mdm_accuracy)
        expected_qc_rows[(subject_id, session_id)] = {
            "subject_id": subject_id,
            "session_id": session_id,
            "ref_separation_riemann": csv_float(ref_sep),
            "online_score_raw_std": csv_float(online_std),
            "online_global_below_threshold_fraction": csv_float(online_global_below_threshold_fraction),
            "online_session_z_below0_fraction": csv_float(online_session_z_below0_fraction),
            "online_drop_below_10_fraction": csv_float(online_drop_below_10_fraction),
            "online_drop_below_20_fraction": csv_float(online_drop_below_20_fraction),
            "online_drop_below_30_fraction": csv_float(online_drop_below_30_fraction),
            "scale_degenerate_flag": str(recomputed_session_summary.scale_degenerate_flag),
            "weak_reference_flag": str(1 if ref_sep < config.weak_reference_threshold else 0),
            "flat_online_flag": str(1 if online_std <= config.flat_online_threshold else 0),
            "online_mdm_accuracy": csv_float(online_mdm_accuracy),
        }

        # Metadata cross-check against Step 4
        step4_cal_path = step4_aligned_split_path(
            output_root=config.step4_root, subject_id=subject_id, session_id=session_id, split="calibration",
        )
        step4_online_path = step4_aligned_split_path(
            output_root=config.step4_root, subject_id=subject_id, session_id=session_id, split="online",
        )
        step4_cal_payload = _validate_saved_aligned_payload(
            path=step4_cal_path,
            split="calibration",
            config=step4_config,
        )
        step4_online_payload = _validate_saved_aligned_payload(
            path=step4_online_path,
            split="online",
            config=step4_config,
        )

        for field in METADATA_FIELDS:
            if not np.array_equal(cal_payload[field], step4_cal_payload[field]):
                raise ValueError(f"{context} calibration score field {field!r} does not match Step 4 metadata")
            if not np.array_equal(online_payload[field], step4_online_payload[field]):
                raise ValueError(f"{context} online score field {field!r} does not match Step 4 metadata")

        # Collect social calibration scores for normalization roundtrip
        social_cal_scores_f32.append(
            extract_social_calibration_scores(
                cal_raw,
                calibration_metadata={
                    field: np.asarray(cal_payload[field])
                    for field in METADATA_FIELDS
                },
                config=config,
                context=context,
            )
        )

    # --- File count checks ---
    expected_raw_scores = expected_session_count * (config.calibration_blocks + config.online_blocks)
    if calibration_score_file_count != expected_session_count:
        raise ValueError(
            f"Expected {expected_session_count} calibration score files, got {calibration_score_file_count}",
        )
    if online_score_file_count != expected_session_count:
        raise ValueError(f"Expected {expected_session_count} online score files, got {online_score_file_count}")
    if reference_file_count != expected_session_count:
        raise ValueError(f"Expected {expected_session_count} reference files, got {reference_file_count}")
    if total_raw_scores != expected_raw_scores:
        raise ValueError(f"Total raw scores {total_raw_scores}, expected {expected_raw_scores}")
    if total_normalized_scores != expected_raw_scores:
        raise ValueError(f"Total normalized scores {total_normalized_scores}, expected {expected_raw_scores}")
    if total_global_threshold_scores != expected_raw_scores:
        raise ValueError(f"Total global threshold scores {total_global_threshold_scores}, expected {expected_raw_scores}")
    if total_session_robust_z_scores != expected_raw_scores:
        raise ValueError(
            f"Total session robust z scores {total_session_robust_z_scores}, expected {expected_raw_scores}",
        )
    if total_drop_rel_scores != expected_raw_scores:
        raise ValueError(f"Total session relative-drop scores {total_drop_rel_scores}, expected {expected_raw_scores}")

    # --- Normalization roundtrip ---
    # Recompute from saved float32 scores
    all_social_cal_f32 = np.concatenate(social_cal_scores_f32).astype(np.float64)
    if all_social_cal_f32.shape[0] != 1050:
        raise ValueError(f"Social calibration reference count {all_social_cal_f32.shape[0]}, expected 1050")

    recomputed_mu = float(np.mean(all_social_cal_f32))
    recomputed_sigma = float(np.std(all_social_cal_f32, ddof=0))
    recomputed_p25 = float(np.percentile(all_social_cal_f32, config.disengagement_quantile * 100, method="linear"))

    if saved_mu != recomputed_mu:
        raise ValueError(f"Normalization mu mismatch: saved={saved_mu}, recomputed={recomputed_mu}")
    if saved_sigma != recomputed_sigma:
        raise ValueError(f"Normalization sigma mismatch: saved={saved_sigma}, recomputed={recomputed_sigma}")
    if saved_p25 != recomputed_p25:
        raise ValueError(f"Normalization p25 mismatch: saved={saved_p25}, recomputed={recomputed_p25}")
    if saved_n != 1050:
        raise ValueError(f"Normalization n_reference_scores={saved_n}, expected 1050")
    if not np.array_equal(saved_raw_social_cal, all_social_cal_f32):
        raise ValueError("Normalization raw_social_calibration_scores does not match recomputed saved float32 scores")

    # --- Empirical checks (reported, not failed) ---
    mean_social_cal_score = float(np.mean(all_social_cal_f32))
    logger.info(
        "Smoke test: mean social calibration score_raw = %.6f (expected positive)",
        mean_social_cal_score,
    )
    if mean_social_cal_score <= 0:
        logger.warning("Mean social calibration score_raw is NOT positive: %.6f", mean_social_cal_score)

    if flat_online_count > 0:
        logger.warning(
            "%d session(s) have flat online score_raw (std <= %.1e)",
            flat_online_count,
            config.flat_online_threshold,
        )
        logger.warning("Flat online sessions: %s", ", ".join(flat_online_sessions))
    if weak_reference_count > 0:
        logger.warning(
            "%d session(s) have weak reference separation (< %.3f)",
            weak_reference_count,
            config.weak_reference_threshold,
        )
        logger.warning("Weak reference sessions: %s", ", ".join(weak_reference_sessions))

    # --- QC CSV validation ---
    csv_path = qc_summary_path(output_root=config.output_root)
    if not csv_path.is_file():
        raise ValueError(f"QC summary CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        csv_rows = list(reader)
    if len(csv_rows) != expected_session_count:
        raise ValueError(f"QC CSV has {len(csv_rows)} rows, expected {expected_session_count}")
    if not csv_rows:
        raise ValueError("QC CSV is empty")
    for col in QC_CSV_COLUMNS:
        if col not in csv_rows[0]:
            raise ValueError(f"QC CSV missing column {col!r}")
    seen_sessions: set[tuple[str, str]] = set()
    for row in csv_rows:
        row_key = (row["subject_id"], row["session_id"])
        if row_key in seen_sessions:
            raise ValueError(f"QC CSV contains duplicate row for {row_key[0]}/{row_key[1]}")
        seen_sessions.add(row_key)
        if row_key not in expected_qc_rows:
            raise ValueError(f"QC CSV contains unexpected session {row_key[0]}/{row_key[1]}")
        expected_row = expected_qc_rows[row_key]
        for column_name, expected_value in expected_row.items():
            if row[column_name] != expected_value:
                raise ValueError(
                    f"QC CSV mismatch for {row_key[0]}/{row_key[1]} column {column_name}: "
                    f"saved={row[column_name]!r}, expected={expected_value!r}",
                )
    if seen_sessions != set(expected_qc_rows):
        missing_rows = sorted(set(expected_qc_rows).difference(seen_sessions))
        raise ValueError(f"QC CSV is missing sessions: {missing_rows}")

    # --- Build summary ---
    ref_sep_array = np.asarray(ref_separations, dtype=np.float64)
    online_mdm_acc_array = np.asarray(online_mdm_accuracies, dtype=np.float64)

    return Step5Summary(
        output_root=config.output_root,
        calibration_score_file_count=calibration_score_file_count,
        online_score_file_count=online_score_file_count,
        reference_file_count=reference_file_count,
        total_raw_scores=total_raw_scores,
        total_normalized_scores=total_normalized_scores,
        mu_social_cal=saved_mu,
        sigma_social_cal=saved_sigma,
        disengagement_threshold_raw=saved_p25,
        ref_separation_min=float(np.min(ref_sep_array)),
        ref_separation_median=float(np.median(ref_sep_array)),
        ref_separation_max=float(np.max(ref_sep_array)),
        score_raw_cal_min=float(np.min(score_raw_cal_values)),
        score_raw_cal_max=float(np.max(score_raw_cal_values)),
        score_raw_online_min=float(np.min(score_raw_online_values)),
        score_raw_online_max=float(np.max(score_raw_online_values)),
        score_norm_cal_min=float(np.min(score_norm_cal_values)),
        score_norm_cal_max=float(np.max(score_norm_cal_values)),
        score_norm_online_min=float(np.min(score_norm_online_values)),
        score_norm_online_max=float(np.max(score_norm_online_values)),
        weak_reference_count=weak_reference_count,
        flat_online_count=flat_online_count,
        online_mdm_accuracy_mean=float(np.mean(online_mdm_acc_array)),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def log_summary(summary: Step5Summary) -> None:
    """Log the headline Step 5 metrics."""

    logger.info(
        "Step 5 complete. Wrote %d calibration score files, %d online score files, %d reference files.",
        summary.calibration_score_file_count,
        summary.online_score_file_count,
        summary.reference_file_count,
    )
    logger.info(
        "Total scores: %d raw, %d normalized. Normalization: mu=%.6f, sigma=%.6f, p25=%.6f.",
        summary.total_raw_scores,
        summary.total_normalized_scores,
        summary.mu_social_cal,
        summary.sigma_social_cal,
        summary.disengagement_threshold_raw,
    )
    logger.info(
        "Reference separation: min=%.6f, median=%.6f, max=%.6f. Weak references: %d. Flat online: %d.",
        summary.ref_separation_min,
        summary.ref_separation_median,
        summary.ref_separation_max,
        summary.weak_reference_count,
        summary.flat_online_count,
    )
    logger.info(
        "Score raw range: cal=[%.4f, %.4f], online=[%.4f, %.4f]. "
        "Score norm range: cal=[%.4f, %.4f], online=[%.4f, %.4f]. "
        "Online MDM accuracy mean=%.4f.",
        summary.score_raw_cal_min,
        summary.score_raw_cal_max,
        summary.score_raw_online_min,
        summary.score_raw_online_max,
        summary.score_norm_cal_min,
        summary.score_norm_cal_max,
        summary.score_norm_online_min,
        summary.score_norm_online_max,
        summary.online_mdm_accuracy_mean,
    )


def run_step5(config: Step5Config | None = None) -> Step5Summary:
    """Run the full Step 5 attention-score pipeline.

    Three-phase orchestration:
        Phase 1: Process all 105 sessions -> collect raw scores in memory
        Phase 2: Fit global normalization from social calibration subset
        Phase 3: Normalize all scores -> save all files -> smoke test
    """

    config = config or Step5Config()
    config.output_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: Collect raw scores
    logger.info("Phase 1: Computing raw scores for all sessions ...")
    all_session_scores: list[SessionScores] = []
    for subject_id, session_id in expected_session_ids():
        session_scores = process_session(
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )
        all_session_scores.append(session_scores)

    logger.info("Phase 1 complete: %d sessions processed.", len(all_session_scores))

    # Phase 2: Global normalization
    logger.info("Phase 2: Fitting global normalization ...")
    normalization = fit_global_normalization(all_session_scores, config=config)

    # Phase 3: Normalize + Save
    logger.info("Phase 3: Normalizing and saving all files ...")
    for session in all_session_scores:
        session_summary = compute_session_calibration_summary(
            session.score_raw_cal,
            calibration_metadata=session.cal_metadata,
            config=config,
            context=f"{session.subject_id}/{session.session_id}",
        )
        score_norm_cal = normalize_scores(session.score_raw_cal, normalization)
        score_norm_online = normalize_scores(session.score_raw_online, normalization)
        score_threshold_rel_global_cal = compute_global_threshold_relative_scores(
            session.score_raw_cal,
            disengagement_threshold_raw=normalization.disengagement_threshold_raw,
        )
        score_threshold_rel_global_online = compute_global_threshold_relative_scores(
            session.score_raw_online,
            disengagement_threshold_raw=normalization.disengagement_threshold_raw,
        )
        score_session_robust_z_cal = compute_session_robust_z_scores(
            session.score_raw_cal,
            session_summary=session_summary,
            config=config,
        )
        score_session_robust_z_online = compute_session_robust_z_scores(
            session.score_raw_online,
            session_summary=session_summary,
            config=config,
        )
        score_drop_rel_session_cal = compute_session_relative_drop_scores(
            session.score_raw_cal,
            session_cal_mean_raw=session_summary.mean_raw,
            config=config,
        )
        score_drop_rel_session_online = compute_session_relative_drop_scores(
            session.score_raw_online,
            session_cal_mean_raw=session_summary.mean_raw,
            config=config,
        )

        save_score_npz(
            path=score_split_path(
                output_root=config.output_root,
                subject_id=session.subject_id,
                session_id=session.session_id,
                split="calibration",
            ),
            score_raw=session.score_raw_cal,
            score_norm=score_norm_cal,
            score_threshold_rel_global=score_threshold_rel_global_cal,
            score_session_robust_z=score_session_robust_z_cal,
            score_drop_rel_session=score_drop_rel_session_cal,
            d_target=session.d_target_cal,
            d_nontarget=session.d_nontarget_cal,
            mdm_correct=session.mdm_correct_cal,
            session_summary=session_summary,
            metadata=session.cal_metadata,
        )
        save_score_npz(
            path=score_split_path(
                output_root=config.output_root,
                subject_id=session.subject_id,
                session_id=session.session_id,
                split="online",
            ),
            score_raw=session.score_raw_online,
            score_norm=score_norm_online,
            score_threshold_rel_global=score_threshold_rel_global_online,
            score_session_robust_z=score_session_robust_z_online,
            score_drop_rel_session=score_drop_rel_session_online,
            d_target=session.d_target_online,
            d_nontarget=session.d_nontarget_online,
            mdm_correct=session.mdm_correct_online,
            session_summary=session_summary,
            metadata=session.online_metadata,
        )
        save_reference_npz(
            path=reference_path(
                output_root=config.output_root,
                subject_id=session.subject_id,
                session_id=session.session_id,
            ),
            cov_target_ref_aligned=session.cov_target_ref_aligned,
            cov_nontarget_ref_aligned=session.cov_nontarget_ref_aligned,
            ref_separation_riemann=session.ref_separation_riemann,
        )

    save_normalization_npz(
        path=normalization_path(output_root=config.output_root),
        normalization=normalization,
    )
    save_qc_summary_csv(
        path=qc_summary_path(output_root=config.output_root),
        all_session_scores=all_session_scores,
        normalization=normalization,
        config=config,
    )

    logger.info("Phase 3 complete. Running smoke tests ...")
    summary = smoke_test_outputs(config)
    log_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Step 5."""

    parser = argparse.ArgumentParser(description="Run Step 5 attention score computation.")
    parser.add_argument("--step3-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step4-root", type=Path, default=Path("data/features"))
    parser.add_argument("--output-root", type=Path, default=Path("data/scores"))
    return parser


def main() -> None:
    """CLI entrypoint for Step 5."""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_step5(
        Step5Config(
            step3_root=args.step3_root,
            step4_root=args.step4_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
