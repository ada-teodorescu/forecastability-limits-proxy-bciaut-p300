"""Step 4 Riemannian alignment and tangent-space projection."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyriemann.utils.mean import mean_riemann

from src.data.loader import expected_session_ids
from src.features.step3_covariance import (
    Step3Config,
    _validate_saved_covariance_payload,
    feature_split_path as step3_feature_split_path,
    session_id_to_ses_label,
    subject_id_to_sub_label,
)

logger = logging.getLogger(__name__)

SQRT_TWO = float(np.sqrt(2.0))
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
REFERENCE_SOURCE = "all calibration blocks 0-19"
METADATA_FIELDS: tuple[str, ...] = (
    "block_ids",
    "condition",
    "block_targets",
    "runs_per_block",
    "n_target_epochs",
    "n_nontarget_epochs",
)
ALIGNED_FIELDS: frozenset[str] = frozenset(("covs", *METADATA_FIELDS, "session_ref_mean"))
TANGENT_FIELDS: frozenset[str] = frozenset(("features", *METADATA_FIELDS, "session_ref_mean"))


@dataclass(frozen=True)
class Step4Config:
    """Configuration for the Step 4 alignment and tangent pipeline."""

    input_root: Path = Path("data/features")
    output_root: Path = Path("data/features")
    calibration_blocks: int = 20
    online_blocks: int = 50
    calibration_social_start: int = 10
    calibration_runs_per_block: int = 10
    online_k_min: int = 3
    online_k_max: int = 10
    covariance_dim: int = 24
    tangent_dim: int = 300
    symmetry_tolerance: float = 1e-8
    mean_riemann_tol: float = 1e-8
    mean_riemann_maxiter: int = 50
    identity_error_tolerance: float = 1e-6
    min_feature_std: float = 1e-6


@dataclass(frozen=True)
class SessionOutputs:
    """Session-level diagnostics returned by Step 4."""

    subject_id: str
    session_id: str
    session_k: int
    session_ref_mean: np.ndarray
    identity_error: float
    session_ref_eigen_min: float
    session_ref_eigen_max: float
    calibration_aligned_min: float
    calibration_aligned_max: float
    online_aligned_min: float
    online_aligned_max: float
    calibration_feature_min: float
    calibration_feature_max: float
    online_feature_min: float
    online_feature_max: float
    warning_lines: tuple[str, ...]


@dataclass(frozen=True)
class Step4Summary:
    """High-level summary of the executed Step 4 run."""

    output_root: Path
    calibration_aligned_file_count: int
    online_aligned_file_count: int
    calibration_tangent_file_count: int
    online_tangent_file_count: int
    alignment_log_count: int
    total_aligned_matrices: int
    total_tangent_vectors: int
    identity_error_mean: float
    identity_error_median: float
    identity_error_max: float
    session_ref_eigen_min_min: float
    session_ref_eigen_min_median: float
    session_ref_eigen_min_max: float
    session_ref_eigen_max_min: float
    session_ref_eigen_max_median: float
    session_ref_eigen_max_max: float
    calibration_feature_min: float
    calibration_feature_max: float
    online_feature_min: float
    online_feature_max: float
    zero_variance_dimensions: int
    warning_session_count: int


def aligned_split_path(*, output_root: Path, subject_id: str, session_id: str, split: str) -> Path:
    """Build the Step 4 aligned-SPD NPZ path for one subject/session/split."""

    return output_root / (
        f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_split-{split}_aligned.npz"
    )


def ts_split_path(*, output_root: Path, subject_id: str, session_id: str, split: str) -> Path:
    """Build the Step 4 tangent-feature NPZ path for one subject/session/split."""

    return output_root / f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_split-{split}_ts.npz"


def alignment_log_path(*, output_root: Path, subject_id: str, session_id: str) -> Path:
    """Build the Step 4 alignment-log path."""

    return output_root / f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_alignment_log.txt"


def format_float(value: float) -> str:
    """Format a float deterministically for logs and comparisons."""

    return f"{float(value):.6f}"


def format_scientific(value: float) -> str:
    """Format a float in scientific notation for logs."""

    return f"{float(value):.1e}"


def rounded_float(value: float) -> float:
    """Round a float the same way the log formatter does."""

    return float(format_float(value))


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def ensure_spd(matrix: np.ndarray, *, config: Step4Config, context: str) -> np.ndarray:
    """Validate that a matrix is finite, symmetric, and strictly SPD."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{context} has shape {matrix.shape}, expected ({config.covariance_dim}, {config.covariance_dim})",
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


def matrix_invsqrt_spd(matrix: np.ndarray, *, config: Step4Config, context: str) -> np.ndarray:
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


def matrix_log_spd(matrix: np.ndarray, *, config: Step4Config, context: str) -> np.ndarray:
    """Compute the matrix logarithm of an SPD matrix via eigendecomposition."""

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{context} eigenvalues contain non-finite values")
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{context} is not strictly SPD; min eigenvalue={float(eigenvalues[0]):.6e}")

    tangent = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    tangent = symmetrize(tangent)
    if not np.all(np.isfinite(tangent)):
        raise ValueError(f"{context} logm contains non-finite values")
    return tangent


def vectorize_upper_triangle(tangent_matrix: np.ndarray, *, config: Step4Config) -> np.ndarray:
    """Vectorize a symmetric tangent matrix with sqrt(2) off-diagonal scaling."""

    tangent_matrix = np.asarray(tangent_matrix, dtype=np.float64)
    if tangent_matrix.shape != (config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"Tangent matrix shape {tangent_matrix.shape}, expected ({config.covariance_dim}, {config.covariance_dim})",
        )

    row_indices, col_indices = np.triu_indices(config.covariance_dim)
    vector = tangent_matrix[row_indices, col_indices].astype(np.float64, copy=True)
    off_diagonal_mask = row_indices != col_indices
    vector[off_diagonal_mask] *= SQRT_TWO
    if vector.shape != (config.tangent_dim,):
        raise ValueError(f"Tangent feature vector shape {vector.shape}, expected ({config.tangent_dim},)")
    return vector


def load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ payload without coercing dtypes."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing NPZ file: {path}")
    with np.load(path, allow_pickle=False) as npz_file:
        return {name: np.asarray(npz_file[name]) for name in npz_file.files}


def expected_condition_array(*, split: str, config: Step4Config) -> np.ndarray:
    """Return the expected condition metadata for one split."""

    if split == "calibration":
        return np.concatenate(
            [
                np.zeros(config.calibration_social_start, dtype=np.int8),
                np.ones(config.calibration_blocks - config.calibration_social_start, dtype=np.int8),
            ]
        )
    if split == "online":
        return np.ones(config.online_blocks, dtype=np.int8)
    raise ValueError(f"Unsupported split {split!r}")


def validate_common_metadata_arrays(
    *,
    payload: dict[str, np.ndarray],
    split: str,
    config: Step4Config,
    path_name: str,
) -> None:
    """Validate the shared block-level metadata contract for Step 4 outputs."""

    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    for field_name in METADATA_FIELDS:
        field = np.asarray(payload[field_name])
        if field.shape != (expected_blocks,):
            raise ValueError(f"{path_name} field {field_name} has shape {field.shape}, expected ({expected_blocks},)")

    block_ids = np.asarray(payload["block_ids"], dtype=np.int16)
    condition = np.asarray(payload["condition"], dtype=np.int8)
    block_targets = np.asarray(payload["block_targets"], dtype=np.int8)
    runs_per_block = np.asarray(payload["runs_per_block"], dtype=np.int16)
    n_target_epochs = np.asarray(payload["n_target_epochs"], dtype=np.int16)
    n_nontarget_epochs = np.asarray(payload["n_nontarget_epochs"], dtype=np.int16)

    if not np.array_equal(block_ids, np.arange(expected_blocks, dtype=np.int16)):
        raise ValueError(f"{path_name} block_ids mismatch: {block_ids.tolist()}")
    if not np.array_equal(condition, expected_condition_array(split=split, config=config)):
        raise ValueError(f"{path_name} {split} condition metadata mismatch")
    if not set(block_targets.tolist()).issubset(set(range(1, 9))):
        raise ValueError(f"{path_name} block_targets values {sorted(set(block_targets.tolist()))} fall outside 1..8")

    if split == "calibration":
        if not np.all(runs_per_block == config.calibration_runs_per_block):
            raise ValueError(f"{path_name} calibration runs_per_block must all equal {config.calibration_runs_per_block}")
    else:
        unique_runs = np.unique(runs_per_block)
        if unique_runs.size != 1:
            raise ValueError(f"{path_name} online runs_per_block must be constant within file")
        session_k = int(unique_runs[0])
        if session_k < config.online_k_min or session_k > config.online_k_max:
            raise ValueError(
                f"{path_name} online runs_per_block={session_k}, expected {config.online_k_min}..{config.online_k_max}",
            )

    if not np.array_equal(n_target_epochs, runs_per_block):
        raise ValueError(f"{path_name} must satisfy n_target_epochs == runs_per_block")
    if not np.array_equal(n_nontarget_epochs, 7 * runs_per_block):
        raise ValueError(f"{path_name} must satisfy n_nontarget_epochs == 7 * runs_per_block")


def load_step3_covariance_payload(
    *,
    path: Path,
    split: str,
    config: Step4Config,
) -> dict[str, np.ndarray]:
    """Load and validate one Step 3 covariance NPZ file for Step 4 consumption."""

    step3_config = Step3Config(output_root=config.input_root)
    payload = _validate_saved_covariance_payload(path=path, split=split, config=step3_config)

    covariances = np.asarray(payload["covs"], dtype=np.float64)
    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    if covariances.shape != (expected_blocks, config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{path.name} covariances shape {covariances.shape}, expected "
            f"({expected_blocks}, {config.covariance_dim}, {config.covariance_dim})",
        )
    validate_common_metadata_arrays(payload=payload, split=split, config=config, path_name=path.name)
    for index, covariance in enumerate(covariances):
        ensure_spd(covariance, config=config, context=f"{path.name} covs[{index}]")
    return payload


def compute_session_reference(
    calibration_payload: dict[str, np.ndarray],
    config: Step4Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the session Riemannian mean from all 20 calibration blocks."""

    reference_covariances = np.asarray(calibration_payload["covs"], dtype=np.float64)
    session_ref_mean = mean_riemann(
        reference_covariances,
        tol=config.mean_riemann_tol,
        maxiter=config.mean_riemann_maxiter,
    )
    session_ref_mean = symmetrize(session_ref_mean)
    eigenvalues = ensure_spd(session_ref_mean, config=config, context="session_ref_mean")
    session_ref_invsqrt = matrix_invsqrt_spd(session_ref_mean, config=config, context="session_ref_mean")
    return session_ref_mean, session_ref_invsqrt, eigenvalues


def recenter_covariances(
    covariances: np.ndarray,
    *,
    session_ref_invsqrt: np.ndarray,
    config: Step4Config,
    context: str,
) -> np.ndarray:
    """Recenter a stack of covariance matrices to the identity."""

    covariances = np.asarray(covariances, dtype=np.float64)
    if covariances.ndim != 3 or covariances.shape[1:] != (config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{context} covariances shape {covariances.shape}, expected (*, {config.covariance_dim}, {config.covariance_dim})",
        )

    aligned = np.zeros_like(covariances, dtype=np.float64)
    for index, covariance in enumerate(covariances):
        recentered = session_ref_invsqrt @ covariance @ session_ref_invsqrt
        recentered = symmetrize(recentered)
        ensure_spd(recentered, config=config, context=f"{context} aligned covariance[{index}]")
        aligned[index] = recentered
    return aligned


def compute_identity_error(aligned_calibration_covariances: np.ndarray, config: Step4Config) -> float:
    """Compute the aligned-calibration mean error relative to the identity."""

    aligned_reference_mean = mean_riemann(
        np.asarray(aligned_calibration_covariances, dtype=np.float64),
        tol=config.mean_riemann_tol,
        maxiter=config.mean_riemann_maxiter,
    )
    aligned_reference_mean = symmetrize(aligned_reference_mean)
    ensure_spd(aligned_reference_mean, config=config, context="aligned_calibration_mean")
    identity = np.eye(config.covariance_dim, dtype=np.float64)
    return float(np.linalg.norm(aligned_reference_mean - identity, ord="fro"))


def tangent_features_from_covariances(
    aligned_covariances: np.ndarray,
    *,
    config: Step4Config,
    context: str,
) -> np.ndarray:
    """Project aligned covariances to tangent space and vectorize them."""

    aligned_covariances = np.asarray(aligned_covariances, dtype=np.float64)
    features = np.zeros((aligned_covariances.shape[0], config.tangent_dim), dtype=np.float32)
    for index, covariance in enumerate(aligned_covariances):
        tangent = matrix_log_spd(covariance, config=config, context=f"{context} aligned covariance[{index}]")
        features[index] = vectorize_upper_triangle(tangent, config=config).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{context} tangent features contain non-finite values")
    return features


def value_range(array: np.ndarray) -> tuple[float, float]:
    """Return the global min/max of an array as floats."""

    array = np.asarray(array)
    return float(np.min(array)), float(np.max(array))


def session_k_from_payload(payload: dict[str, np.ndarray], *, split: str, config: Step4Config) -> int:
    """Extract the session K value from a split payload."""

    runs_per_block = np.asarray(payload["runs_per_block"], dtype=np.int16)
    if split == "calibration":
        unique_runs = np.unique(runs_per_block)
        if unique_runs.size != 1 or int(unique_runs[0]) != config.calibration_runs_per_block:
            raise ValueError("Calibration runs_per_block does not match the Step 4 contract")
        return int(unique_runs[0])

    unique_runs = np.unique(runs_per_block)
    if unique_runs.size != 1:
        raise ValueError("Online runs_per_block is not constant within file")
    session_k = int(unique_runs[0])
    if session_k < config.online_k_min or session_k > config.online_k_max:
        raise ValueError(f"Online session K={session_k}, expected {config.online_k_min}..{config.online_k_max}")
    return session_k


def save_aligned_npz(
    *,
    path: Path,
    covariances: np.ndarray,
    step3_payload: dict[str, np.ndarray],
    session_ref_mean: np.ndarray,
) -> None:
    """Save one Step 4 aligned-SPD NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        covs=np.asarray(covariances, dtype=np.float64),
        block_ids=np.asarray(step3_payload["block_ids"], dtype=np.int16),
        condition=np.asarray(step3_payload["condition"], dtype=np.int8),
        block_targets=np.asarray(step3_payload["block_targets"], dtype=np.int8),
        runs_per_block=np.asarray(step3_payload["runs_per_block"], dtype=np.int16),
        n_target_epochs=np.asarray(step3_payload["n_target_epochs"], dtype=np.int16),
        n_nontarget_epochs=np.asarray(step3_payload["n_nontarget_epochs"], dtype=np.int16),
        session_ref_mean=np.asarray(session_ref_mean, dtype=np.float64),
    )


def save_tangent_npz(
    *,
    path: Path,
    features: np.ndarray,
    step3_payload: dict[str, np.ndarray],
    session_ref_mean: np.ndarray,
) -> None:
    """Save one Step 4 tangent-feature NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        block_ids=np.asarray(step3_payload["block_ids"], dtype=np.int16),
        condition=np.asarray(step3_payload["condition"], dtype=np.int8),
        block_targets=np.asarray(step3_payload["block_targets"], dtype=np.int8),
        runs_per_block=np.asarray(step3_payload["runs_per_block"], dtype=np.int16),
        n_target_epochs=np.asarray(step3_payload["n_target_epochs"], dtype=np.int16),
        n_nontarget_epochs=np.asarray(step3_payload["n_nontarget_epochs"], dtype=np.int16),
        session_ref_mean=np.asarray(session_ref_mean, dtype=np.float64),
    )


def write_alignment_log(
    *,
    path: Path,
    session_output: SessionOutputs,
    config: Step4Config,
) -> None:
    """Write the session-level Step 4 alignment log."""

    lines = [
        f"Session: {session_output.subject_id}/{session_output.session_id}",
        f"Reference source: {REFERENCE_SOURCE}",
        f"Session K: {session_output.session_k}",
        f"mean_riemann tol: {format_scientific(config.mean_riemann_tol)}",
        f"mean_riemann maxiter: {config.mean_riemann_maxiter}",
        (
            "Session ref mean eigenvalue range: "
            f"{format_float(session_output.session_ref_eigen_min)} to {format_float(session_output.session_ref_eigen_max)}"
        ),
        f"Identity-centering error: {format_float(session_output.identity_error)}",
        (
            "Aligned calibration value range: "
            f"{format_float(session_output.calibration_aligned_min)} to {format_float(session_output.calibration_aligned_max)}"
        ),
        (
            "Aligned online value range: "
            f"{format_float(session_output.online_aligned_min)} to {format_float(session_output.online_aligned_max)}"
        ),
        (
            "Tangent calibration value range: "
            f"{format_float(session_output.calibration_feature_min)} to {format_float(session_output.calibration_feature_max)}"
        ),
        (
            "Tangent online value range: "
            f"{format_float(session_output.online_feature_min)} to {format_float(session_output.online_feature_max)}"
        ),
        "Warnings:",
    ]
    if session_output.warning_lines:
        lines.extend(f"- {warning}" for warning in session_output.warning_lines)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_alignment_log_contents(path: Path) -> dict[str, object]:
    """Validate the Step 4 alignment-log schema and numeric domains."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 4 alignment log: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()

    def get_line(prefix: str) -> str:
        line = next((candidate for candidate in lines if candidate.startswith(prefix)), None)
        if line is None:
            raise ValueError(f"{path.name} missing required log line starting with {prefix!r}")
        return line

    session_line = get_line("Session:")
    expected_session_match = re.fullmatch(r"sub-(\d{2})_ses-(\d{2})_alignment_log\.txt", path.name)
    session_match = re.fullmatch(r"Session:\s+(SBJ\d{2})/(S\d{2})", session_line)
    if session_match is None:
        raise ValueError(f"{path.name} has invalid Session line: {session_line!r}")
    if expected_session_match is not None:
        expected_subject = f"SBJ{expected_session_match.group(1)}"
        expected_session = f"S{expected_session_match.group(2)}"
        if session_match.group(1) != expected_subject or session_match.group(2) != expected_session:
            raise ValueError(
                f"{path.name} Session line {session_match.group(1)}/{session_match.group(2)} "
                f"does not match filename {expected_subject}/{expected_session}",
            )

    reference_line = get_line("Reference source:")
    if reference_line != f"Reference source: {REFERENCE_SOURCE}":
        raise ValueError(f"{path.name} has unexpected reference source line: {reference_line!r}")

    session_k_line = get_line("Session K:")
    session_k_match = re.fullmatch(r"Session K:\s+(\d+)", session_k_line)
    if session_k_match is None:
        raise ValueError(f"{path.name} has invalid Session K line: {session_k_line!r}")
    session_k = int(session_k_match.group(1))
    if session_k < 3 or session_k > 10:
        raise ValueError(f"{path.name} has invalid Session K value {session_k}")

    tol_line = get_line("mean_riemann tol:")
    tol_match = re.fullmatch(rf"mean_riemann tol:\s+({FLOAT_PATTERN})", tol_line)
    if tol_match is None:
        raise ValueError(f"{path.name} has invalid mean_riemann tol line: {tol_line!r}")
    tol_value = float(tol_match.group(1))
    if not np.isfinite(tol_value) or tol_value <= 0.0:
        raise ValueError(f"{path.name} has invalid mean_riemann tol {tol_value}")

    maxiter_line = get_line("mean_riemann maxiter:")
    maxiter_match = re.fullmatch(r"mean_riemann maxiter:\s+(\d+)", maxiter_line)
    if maxiter_match is None:
        raise ValueError(f"{path.name} has invalid mean_riemann maxiter line: {maxiter_line!r}")
    maxiter_value = int(maxiter_match.group(1))
    if maxiter_value <= 0:
        raise ValueError(f"{path.name} has invalid mean_riemann maxiter {maxiter_value}")

    range_prefixes = (
        "Session ref mean eigenvalue range:",
        "Aligned calibration value range:",
        "Aligned online value range:",
        "Tangent calibration value range:",
        "Tangent online value range:",
    )
    ranges: dict[str, tuple[float, float]] = {}
    for prefix in range_prefixes:
        line = get_line(prefix)
        range_match = re.fullmatch(rf"{re.escape(prefix)}\s+({FLOAT_PATTERN}) to ({FLOAT_PATTERN})", line)
        if range_match is None:
            raise ValueError(f"{path.name} has invalid range line: {line!r}")
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if not (np.isfinite(low) and np.isfinite(high)) or low > high:
            raise ValueError(f"{path.name} has invalid range {low} to {high} for {prefix!r}")
        if prefix == "Session ref mean eigenvalue range:" and low <= 0.0:
            raise ValueError(f"{path.name} session_ref_mean eigenvalues must be strictly positive")
        ranges[prefix] = (low, high)

    identity_error_line = get_line("Identity-centering error:")
    identity_error_match = re.fullmatch(rf"Identity-centering error:\s+({FLOAT_PATTERN})", identity_error_line)
    if identity_error_match is None:
        raise ValueError(f"{path.name} has invalid identity error line: {identity_error_line!r}")
    identity_error = float(identity_error_match.group(1))
    if not np.isfinite(identity_error) or identity_error < 0.0:
        raise ValueError(f"{path.name} has invalid identity error {identity_error}")

    warnings_index = next((index for index, line in enumerate(lines) if line == "Warnings:"), None)
    if warnings_index is None or warnings_index == len(lines) - 1:
        raise ValueError(f"{path.name} warnings section must contain at least one bullet line")
    warning_lines = lines[warnings_index + 1 :]
    if not all(line.startswith("- ") for line in warning_lines):
        raise ValueError(f"{path.name} warnings section must contain only '- ' bullet lines after the header")

    return {
        "subject_id": session_match.group(1),
        "session_id": session_match.group(2),
        "reference_source": REFERENCE_SOURCE,
        "session_k": session_k,
        "mean_riemann_tol": tol_value,
        "mean_riemann_maxiter": maxiter_value,
        "ranges": ranges,
        "identity_error": identity_error,
        "warnings": tuple(line[2:] for line in warning_lines),
    }


def process_session(
    *,
    subject_id: str,
    session_id: str,
    config: Step4Config,
) -> SessionOutputs:
    """Process one subject/session pair into Step 4 aligned and tangent outputs."""

    calibration_payload = load_step3_covariance_payload(
        path=step3_feature_split_path(
            output_root=config.input_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        ),
        split="calibration",
        config=config,
    )
    online_payload = load_step3_covariance_payload(
        path=step3_feature_split_path(
            output_root=config.input_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        ),
        split="online",
        config=config,
    )

    session_ref_mean, session_ref_invsqrt, ref_eigenvalues = compute_session_reference(calibration_payload, config)

    aligned_calibration = recenter_covariances(
        np.asarray(calibration_payload["covs"], dtype=np.float64),
        session_ref_invsqrt=session_ref_invsqrt,
        config=config,
        context=f"{subject_id}/{session_id} calibration",
    )
    aligned_online = recenter_covariances(
        np.asarray(online_payload["covs"], dtype=np.float64),
        session_ref_invsqrt=session_ref_invsqrt,
        config=config,
        context=f"{subject_id}/{session_id} online",
    )

    identity_error = compute_identity_error(aligned_calibration, config)
    if identity_error > config.identity_error_tolerance:
        raise ValueError(
            f"{subject_id}/{session_id} aligned calibration mean identity error "
            f"{identity_error:.6e} exceeds tolerance {config.identity_error_tolerance:.1e}",
        )

    calibration_features = tangent_features_from_covariances(
        aligned_calibration,
        config=config,
        context=f"{subject_id}/{session_id} calibration",
    )
    online_features = tangent_features_from_covariances(
        aligned_online,
        config=config,
        context=f"{subject_id}/{session_id} online",
    )

    save_aligned_npz(
        path=aligned_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        ),
        covariances=aligned_calibration,
        step3_payload=calibration_payload,
        session_ref_mean=session_ref_mean,
    )
    save_aligned_npz(
        path=aligned_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        ),
        covariances=aligned_online,
        step3_payload=online_payload,
        session_ref_mean=session_ref_mean,
    )
    save_tangent_npz(
        path=ts_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        ),
        features=calibration_features,
        step3_payload=calibration_payload,
        session_ref_mean=session_ref_mean,
    )
    save_tangent_npz(
        path=ts_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        ),
        features=online_features,
        step3_payload=online_payload,
        session_ref_mean=session_ref_mean,
    )

    calibration_aligned_min, calibration_aligned_max = value_range(aligned_calibration)
    online_aligned_min, online_aligned_max = value_range(aligned_online)
    calibration_feature_min, calibration_feature_max = value_range(calibration_features)
    online_feature_min, online_feature_max = value_range(online_features)
    session_output = SessionOutputs(
        subject_id=subject_id,
        session_id=session_id,
        session_k=session_k_from_payload(online_payload, split="online", config=config),
        session_ref_mean=session_ref_mean,
        identity_error=identity_error,
        session_ref_eigen_min=float(ref_eigenvalues[0]),
        session_ref_eigen_max=float(ref_eigenvalues[-1]),
        calibration_aligned_min=calibration_aligned_min,
        calibration_aligned_max=calibration_aligned_max,
        online_aligned_min=online_aligned_min,
        online_aligned_max=online_aligned_max,
        calibration_feature_min=calibration_feature_min,
        calibration_feature_max=calibration_feature_max,
        online_feature_min=online_feature_min,
        online_feature_max=online_feature_max,
        warning_lines=tuple(),
    )
    write_alignment_log(
        path=alignment_log_path(output_root=config.output_root, subject_id=subject_id, session_id=session_id),
        session_output=session_output,
        config=config,
    )
    return session_output


def _validate_saved_aligned_payload(
    *,
    path: Path,
    split: str,
    config: Step4Config,
) -> dict[str, np.ndarray]:
    """Load and validate one saved Step 4 aligned-SPD NPZ file."""

    payload = load_npz_payload(path)
    missing = sorted(ALIGNED_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"{path.name} missing required Step 4 fields: {missing}")
    unexpected = sorted(set(payload).difference(ALIGNED_FIELDS))
    if unexpected:
        raise ValueError(f"{path.name} contains unexpected Step 4 fields: {unexpected}")

    expected_dtypes = {
        "covs": np.dtype(np.float64),
        "block_ids": np.dtype(np.int16),
        "condition": np.dtype(np.int8),
        "block_targets": np.dtype(np.int8),
        "runs_per_block": np.dtype(np.int16),
        "n_target_epochs": np.dtype(np.int16),
        "n_nontarget_epochs": np.dtype(np.int16),
        "session_ref_mean": np.dtype(np.float64),
    }
    for field_name, expected_dtype in expected_dtypes.items():
        actual_dtype = np.asarray(payload[field_name]).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"{path.name} field {field_name} has dtype {actual_dtype}, expected {expected_dtype}")

    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    covariances = np.asarray(payload["covs"], dtype=np.float64)
    if covariances.shape != (expected_blocks, config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{path.name} covs shape {covariances.shape}, expected ({expected_blocks}, {config.covariance_dim}, {config.covariance_dim})",
        )
    validate_common_metadata_arrays(payload=payload, split=split, config=config, path_name=path.name)
    for index, covariance in enumerate(covariances):
        ensure_spd(covariance, config=config, context=f"{path.name} covs[{index}]")

    session_ref_mean = np.asarray(payload["session_ref_mean"], dtype=np.float64)
    ensure_spd(session_ref_mean, config=config, context=f"{path.name} session_ref_mean")
    return payload


def _validate_saved_tangent_payload(
    *,
    path: Path,
    split: str,
    config: Step4Config,
) -> dict[str, np.ndarray]:
    """Load and validate one saved Step 4 tangent-feature NPZ file."""

    payload = load_npz_payload(path)
    missing = sorted(TANGENT_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"{path.name} missing required Step 4 fields: {missing}")
    unexpected = sorted(set(payload).difference(TANGENT_FIELDS))
    if unexpected:
        raise ValueError(f"{path.name} contains unexpected Step 4 fields: {unexpected}")

    expected_dtypes = {
        "features": np.dtype(np.float32),
        "block_ids": np.dtype(np.int16),
        "condition": np.dtype(np.int8),
        "block_targets": np.dtype(np.int8),
        "runs_per_block": np.dtype(np.int16),
        "n_target_epochs": np.dtype(np.int16),
        "n_nontarget_epochs": np.dtype(np.int16),
        "session_ref_mean": np.dtype(np.float64),
    }
    for field_name, expected_dtype in expected_dtypes.items():
        actual_dtype = np.asarray(payload[field_name]).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"{path.name} field {field_name} has dtype {actual_dtype}, expected {expected_dtype}")

    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    features = np.asarray(payload["features"], dtype=np.float32)
    if features.shape != (expected_blocks, config.tangent_dim):
        raise ValueError(
            f"{path.name} features shape {features.shape}, expected ({expected_blocks}, {config.tangent_dim})",
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{path.name} features contain non-finite values")

    validate_common_metadata_arrays(payload=payload, split=split, config=config, path_name=path.name)

    session_ref_mean = np.asarray(payload["session_ref_mean"], dtype=np.float64)
    ensure_spd(session_ref_mean, config=config, context=f"{path.name} session_ref_mean")
    return payload


def smoke_test_outputs(config: Step4Config) -> Step4Summary:
    """Run dataset-wide smoke tests across all Step 4 outputs."""

    calibration_aligned_file_count = 0
    online_aligned_file_count = 0
    calibration_tangent_file_count = 0
    online_tangent_file_count = 0
    alignment_log_count = 0
    total_aligned_matrices = 0
    total_tangent_vectors = 0
    identity_errors: list[float] = []
    session_ref_eigen_mins: list[float] = []
    session_ref_eigen_maxs: list[float] = []
    calibration_feature_mins: list[float] = []
    calibration_feature_maxs: list[float] = []
    online_feature_mins: list[float] = []
    online_feature_maxs: list[float] = []
    all_features: list[np.ndarray] = []
    warning_session_count = 0

    for subject_id, session_id in expected_session_ids():
        calibration_aligned_path = aligned_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        )
        online_aligned_path = aligned_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        )
        calibration_ts_path = ts_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        )
        online_ts_path = ts_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        )
        log_path = alignment_log_path(output_root=config.output_root, subject_id=subject_id, session_id=session_id)

        calibration_aligned_payload = _validate_saved_aligned_payload(
            path=calibration_aligned_path,
            split="calibration",
            config=config,
        )
        online_aligned_payload = _validate_saved_aligned_payload(
            path=online_aligned_path,
            split="online",
            config=config,
        )
        calibration_ts_payload = _validate_saved_tangent_payload(
            path=calibration_ts_path,
            split="calibration",
            config=config,
        )
        online_ts_payload = _validate_saved_tangent_payload(
            path=online_ts_path,
            split="online",
            config=config,
        )
        log_payload = validate_alignment_log_contents(log_path)

        calibration_aligned_file_count += 1
        online_aligned_file_count += 1
        calibration_tangent_file_count += 1
        online_tangent_file_count += 1
        alignment_log_count += 1
        total_aligned_matrices += int(calibration_aligned_payload["covs"].shape[0] + online_aligned_payload["covs"].shape[0])
        total_tangent_vectors += int(calibration_ts_payload["features"].shape[0] + online_ts_payload["features"].shape[0])

        if log_payload["subject_id"] != subject_id or log_payload["session_id"] != session_id:
            raise ValueError(f"{subject_id}/{session_id} log session identity does not match the current artifacts")
        if log_payload["reference_source"] != REFERENCE_SOURCE:
            raise ValueError(f"{subject_id}/{session_id} log reference source does not match Step 4 contract")
        if float(log_payload["mean_riemann_tol"]) != config.mean_riemann_tol:
            raise ValueError(
                f"{subject_id}/{session_id} log mean_riemann tol {float(log_payload['mean_riemann_tol'])} "
                f"does not match config {config.mean_riemann_tol}",
            )
        if int(log_payload["mean_riemann_maxiter"]) != config.mean_riemann_maxiter:
            raise ValueError(
                f"{subject_id}/{session_id} log mean_riemann maxiter {int(log_payload['mean_riemann_maxiter'])} "
                f"does not match config {config.mean_riemann_maxiter}",
            )

        session_ref_mean = np.asarray(calibration_aligned_payload["session_ref_mean"], dtype=np.float64)
        for payload_name, payload in (
            ("online aligned", online_aligned_payload),
            ("calibration tangent", calibration_ts_payload),
            ("online tangent", online_ts_payload),
        ):
            if not np.array_equal(session_ref_mean, np.asarray(payload["session_ref_mean"], dtype=np.float64)):
                raise ValueError(f"{subject_id}/{session_id} {payload_name} session_ref_mean arrays do not match")

        step3_calibration_payload = load_step3_covariance_payload(
            path=step3_feature_split_path(
                output_root=config.input_root,
                subject_id=subject_id,
                session_id=session_id,
                split="calibration",
            ),
            split="calibration",
            config=config,
        )
        step3_online_payload = load_step3_covariance_payload(
            path=step3_feature_split_path(
                output_root=config.input_root,
                subject_id=subject_id,
                session_id=session_id,
                split="online",
            ),
            split="online",
            config=config,
        )
        for field_name in METADATA_FIELDS:
            if not np.array_equal(calibration_aligned_payload[field_name], step3_calibration_payload[field_name]):
                raise ValueError(
                    f"{subject_id}/{session_id} calibration aligned field {field_name} does not match Step 3 metadata",
                )
            if not np.array_equal(online_aligned_payload[field_name], step3_online_payload[field_name]):
                raise ValueError(
                    f"{subject_id}/{session_id} online aligned field {field_name} does not match Step 3 metadata",
                )
            if not np.array_equal(calibration_ts_payload[field_name], step3_calibration_payload[field_name]):
                raise ValueError(
                    f"{subject_id}/{session_id} calibration tangent field {field_name} does not match Step 3 metadata",
                )
            if not np.array_equal(online_ts_payload[field_name], step3_online_payload[field_name]):
                raise ValueError(
                    f"{subject_id}/{session_id} online tangent field {field_name} does not match Step 3 metadata",
                )

        recomputed_ref_mean, _, _ = compute_session_reference(step3_calibration_payload, config)
        if not np.allclose(session_ref_mean, recomputed_ref_mean, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{subject_id}/{session_id} saved session_ref_mean does not match recomputed calibration mean")

        session_k = session_k_from_payload(online_aligned_payload, split="online", config=config)
        if int(log_payload["session_k"]) != session_k:
            raise ValueError(
                f"{subject_id}/{session_id} log Session K {int(log_payload['session_k'])} "
                f"does not match saved runs_per_block={session_k}",
            )

        ref_eigenvalues = np.linalg.eigvalsh(session_ref_mean)
        actual_ref_range = (
            rounded_float(float(ref_eigenvalues[0])),
            rounded_float(float(ref_eigenvalues[-1])),
        )
        if tuple(log_payload["ranges"]["Session ref mean eigenvalue range:"]) != actual_ref_range:
            raise ValueError(
                f"{subject_id}/{session_id} log ref-eigenvalue range "
                f"{tuple(log_payload['ranges']['Session ref mean eigenvalue range:'])} does not match saved session_ref_mean "
                f"{actual_ref_range}",
            )

        expected_ranges = {
            "Aligned calibration value range:": (
                rounded_float(value_range(calibration_aligned_payload["covs"])[0]),
                rounded_float(value_range(calibration_aligned_payload["covs"])[1]),
            ),
            "Aligned online value range:": (
                rounded_float(value_range(online_aligned_payload["covs"])[0]),
                rounded_float(value_range(online_aligned_payload["covs"])[1]),
            ),
            "Tangent calibration value range:": (
                rounded_float(value_range(calibration_ts_payload["features"])[0]),
                rounded_float(value_range(calibration_ts_payload["features"])[1]),
            ),
            "Tangent online value range:": (
                rounded_float(value_range(online_ts_payload["features"])[0]),
                rounded_float(value_range(online_ts_payload["features"])[1]),
            ),
        }
        for prefix, expected_range in expected_ranges.items():
            if tuple(log_payload["ranges"][prefix]) != expected_range:
                raise ValueError(
                    f"{subject_id}/{session_id} log range {prefix!r}={tuple(log_payload['ranges'][prefix])} "
                    f"does not match saved arrays {expected_range}",
                )

        actual_identity_error = compute_identity_error(
            np.asarray(calibration_aligned_payload["covs"], dtype=np.float64),
            config,
        )
        identity_errors.append(actual_identity_error)
        if actual_identity_error > config.identity_error_tolerance:
            raise ValueError(
                f"{subject_id}/{session_id} aligned calibration mean identity error "
                f"{actual_identity_error:.6e} exceeds tolerance {config.identity_error_tolerance:.1e}",
            )
        rounded_identity_error = rounded_float(actual_identity_error)
        if float(log_payload["identity_error"]) != rounded_identity_error:
            raise ValueError(
                f"{subject_id}/{session_id} log identity error {float(log_payload['identity_error'])} "
                f"does not match recomputed rounded value {rounded_identity_error}",
            )

        if tuple(log_payload["warnings"]) != ("none",):
            warning_session_count += 1

        session_ref_eigen_mins.append(float(ref_eigenvalues[0]))
        session_ref_eigen_maxs.append(float(ref_eigenvalues[-1]))
        calibration_feature_mins.append(float(np.min(calibration_ts_payload["features"])))
        calibration_feature_maxs.append(float(np.max(calibration_ts_payload["features"])))
        online_feature_mins.append(float(np.min(online_ts_payload["features"])))
        online_feature_maxs.append(float(np.max(online_ts_payload["features"])))
        all_features.append(np.asarray(calibration_ts_payload["features"], dtype=np.float32))
        all_features.append(np.asarray(online_ts_payload["features"], dtype=np.float32))

    if (
        calibration_aligned_file_count != 105
        or online_aligned_file_count != 105
        or calibration_tangent_file_count != 105
        or online_tangent_file_count != 105
        or alignment_log_count != 105
    ):
        raise ValueError(
            "Step 4 output counts mismatch: expected 105 calibration aligned files, 105 online aligned files, "
            "105 calibration tangent files, 105 online tangent files, and 105 logs",
        )
    if total_aligned_matrices != 7350:
        raise ValueError(f"Total aligned-matrix count {total_aligned_matrices}, expected 7350")
    if total_tangent_vectors != 7350:
        raise ValueError(f"Total tangent-vector count {total_tangent_vectors}, expected 7350")

    feature_matrix = np.vstack(all_features).astype(np.float64, copy=False)
    feature_std = np.std(feature_matrix, axis=0)
    zero_variance_dimensions = int(np.sum(feature_std < config.min_feature_std))
    if zero_variance_dimensions != 0:
        raise ValueError(
            f"Found {zero_variance_dimensions} tangent dimensions with std < {config.min_feature_std:.1e}",
        )

    identity_array = np.asarray(identity_errors, dtype=np.float64)
    ref_eigen_min_array = np.asarray(session_ref_eigen_mins, dtype=np.float64)
    ref_eigen_max_array = np.asarray(session_ref_eigen_maxs, dtype=np.float64)
    return Step4Summary(
        output_root=config.output_root,
        calibration_aligned_file_count=calibration_aligned_file_count,
        online_aligned_file_count=online_aligned_file_count,
        calibration_tangent_file_count=calibration_tangent_file_count,
        online_tangent_file_count=online_tangent_file_count,
        alignment_log_count=alignment_log_count,
        total_aligned_matrices=total_aligned_matrices,
        total_tangent_vectors=total_tangent_vectors,
        identity_error_mean=float(np.mean(identity_array)),
        identity_error_median=float(np.median(identity_array)),
        identity_error_max=float(np.max(identity_array)),
        session_ref_eigen_min_min=float(np.min(ref_eigen_min_array)),
        session_ref_eigen_min_median=float(np.median(ref_eigen_min_array)),
        session_ref_eigen_min_max=float(np.max(ref_eigen_min_array)),
        session_ref_eigen_max_min=float(np.min(ref_eigen_max_array)),
        session_ref_eigen_max_median=float(np.median(ref_eigen_max_array)),
        session_ref_eigen_max_max=float(np.max(ref_eigen_max_array)),
        calibration_feature_min=float(np.min(np.asarray(calibration_feature_mins, dtype=np.float64))),
        calibration_feature_max=float(np.max(np.asarray(calibration_feature_maxs, dtype=np.float64))),
        online_feature_min=float(np.min(np.asarray(online_feature_mins, dtype=np.float64))),
        online_feature_max=float(np.max(np.asarray(online_feature_maxs, dtype=np.float64))),
        zero_variance_dimensions=zero_variance_dimensions,
        warning_session_count=warning_session_count,
    )


def log_summary(summary: Step4Summary) -> None:
    """Log the headline Step 4 metrics."""

    logger.info(
        "Step 4 complete. Wrote %d calibration aligned files, %d online aligned files, "
        "%d calibration tangent files, %d online tangent files, and %d logs.",
        summary.calibration_aligned_file_count,
        summary.online_aligned_file_count,
        summary.calibration_tangent_file_count,
        summary.online_tangent_file_count,
        summary.alignment_log_count,
    )
    logger.info(
        "Total aligned matrices %d, total tangent vectors %d. Identity error mean %.6e, median %.6e, max %.6e.",
        summary.total_aligned_matrices,
        summary.total_tangent_vectors,
        summary.identity_error_mean,
        summary.identity_error_median,
        summary.identity_error_max,
    )
    logger.info(
        "Session-ref min eigen range [%.6f, %.6f]; max eigen range [%.6f, %.6f]. "
        "Calibration tangent range [%.6f, %.6f]; online tangent range [%.6f, %.6f]. "
        "Tangent dimensions with std < %.1e: %d. Warning sessions: %d.",
        summary.session_ref_eigen_min_min,
        summary.session_ref_eigen_min_max,
        summary.session_ref_eigen_max_min,
        summary.session_ref_eigen_max_max,
        summary.calibration_feature_min,
        summary.calibration_feature_max,
        summary.online_feature_min,
        summary.online_feature_max,
        Step4Config.min_feature_std,
        summary.zero_variance_dimensions,
        summary.warning_session_count,
    )


def run_step4(config: Step4Config | None = None) -> Step4Summary:
    """Run the full Step 4 alignment and tangent-feature pipeline."""

    config = config or Step4Config()
    config.output_root.mkdir(parents=True, exist_ok=True)

    for subject_id, session_id in expected_session_ids():
        process_session(subject_id=subject_id, session_id=session_id, config=config)

    summary = smoke_test_outputs(config)
    log_summary(summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Step 4."""

    parser = argparse.ArgumentParser(description="Run Step 4 Riemannian alignment and tangent projection.")
    parser.add_argument("--input-root", type=Path, default=Path("data/features"))
    parser.add_argument("--output-root", type=Path, default=Path("data/features"))
    return parser


def main() -> None:
    """CLI entrypoint for Step 4."""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_step4(
        Step4Config(
            input_root=args.input_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
