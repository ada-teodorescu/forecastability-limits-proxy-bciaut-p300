"""Step 3 covariance-matrix computation."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import OAS

from src.data.loader import CHANNEL_NAMES, expected_session_ids

logger = logging.getLogger(__name__)

REQUIRED_NPZ_FIELDS: tuple[str, ...] = (
    "epochs",
    "labels",
    "block_ids",
    "run_ids",
    "flash_ids",
    "block_targets",
    "condition",
    "fs",
    "ch_names",
)


@dataclass(frozen=True)
class Step3Config:
    """Configuration for the Step 3 covariance pipeline."""

    processed_root: Path = Path("data/processed")
    output_root: Path = Path("data/features")
    inventory_path: Path = Path("data/processed/dataset_inventory.txt")
    output_fs: int = 128
    epoch_samples: int = 128
    n_channels: int = 8
    calibration_blocks: int = 20
    online_blocks: int = 50
    calibration_runs_per_block: int = 10
    events_per_run: int = 8
    prototype_block_start: int = 10
    prototype_block_end: int = 19
    covariance_dim: int = 24
    symmetry_tolerance: float = 1e-8
    condition_number_warning: float = 1e6
    dominance_ratio_warning: float = 100.0
    raw_min_eigenvalue_warning: float = 1e-10


@dataclass(frozen=True)
class BlockAverages:
    """Per-block averaged target responses plus saved metadata."""

    target_averages: np.ndarray
    block_ids: np.ndarray
    condition: np.ndarray
    block_targets: np.ndarray
    runs_per_block: np.ndarray
    n_target_epochs: np.ndarray
    n_nontarget_epochs: np.ndarray


@dataclass(frozen=True)
class CovarianceMetrics:
    """Spectral QC metrics for one SPD matrix."""

    min_eigenvalue: float
    condition_number: float
    dominance_ratio: float
    symmetry_error: float


@dataclass(frozen=True)
class SessionOutputs:
    """Session-level diagnostics returned by the Step 3 loop."""

    subject_id: str
    session_id: str
    session_k: int
    prototype_separation_norm: float
    calibration_alphas: np.ndarray
    online_alphas: np.ndarray
    calibration_condition_numbers: np.ndarray
    online_condition_numbers: np.ndarray
    calibration_dominance_ratios: np.ndarray
    online_dominance_ratios: np.ndarray
    warning_lines: tuple[str, ...]


@dataclass(frozen=True)
class Step3Summary:
    """High-level summary of the executed Step 3 run."""

    output_root: Path
    calibration_file_count: int
    online_file_count: int
    session_log_count: int
    total_covariance_matrices: int
    prototype_separation_mean: float
    prototype_separation_median: float
    prototype_separation_min: float
    prototype_separation_max: float
    shrinkage_alpha_mean: float
    shrinkage_alpha_median: float
    shrinkage_alpha_min: float
    shrinkage_alpha_max: float
    spectral_outlier_blocks: int
    near_singular_sessions: int


def subject_id_to_sub_label(subject_id: str) -> str:
    """Convert SBJxx to sub-xx."""

    return f"sub-{subject_id[-2:]}"


def session_id_to_ses_label(session_id: str) -> str:
    """Convert Sxx to ses-xx."""

    return f"ses-{session_id[-2:]}"


def processed_split_path(*, processed_root: Path, subject_id: str, session_id: str, split: str) -> Path:
    """Build the processed Step 1 NPZ path for one subject/session/split."""

    return processed_root / f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_split-{split}.npz"


def feature_split_path(*, output_root: Path, subject_id: str, session_id: str, split: str) -> Path:
    """Build the Step 3 covariance NPZ path for one subject/session/split."""

    return output_root / f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_split-{split}_cov.npz"


def session_log_path(*, output_root: Path, subject_id: str, session_id: str) -> Path:
    """Build the Step 3 session-log path."""

    return output_root / f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_cov_log.txt"


def format_float(value: float) -> str:
    """Format a float deterministically for logs."""

    return f"{float(value):.6f}"


def parse_k_values_from_inventory(path: Path) -> dict[tuple[str, str], int]:
    """Parse all 105 K values from Step 1's dataset inventory text file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    direct_pattern = re.compile(r"^\s*(SBJ\d{2})/(S\d{2})\s*:\s*K=(\d+)\s*$")
    direct_matches: dict[tuple[str, str], int] = {}
    for line in lines:
        match = direct_pattern.match(line)
        if match:
            subject_id, session_id, k_value = match.groups()
            direct_matches[(subject_id, session_id)] = int(k_value)
    if direct_matches:
        if len(direct_matches) != 105:
            raise ValueError(f"Expected 105 K values in {path}, found {len(direct_matches)}")
        return direct_matches

    table_header_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("Subject")),
        None,
    )
    if table_header_index is None:
        raise ValueError(f"Could not find K-value table header in {path}")

    header_sessions = re.findall(r"S\d{2}", lines[table_header_index])
    if header_sessions != [f"S{i:02d}" for i in range(1, 8)]:
        raise ValueError(f"Unexpected session header in {path}: {header_sessions}")

    k_values: dict[tuple[str, str], int] = {}
    for line in lines[table_header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if k_values:
                break
            continue
        parts = stripped.split()
        if len(parts) == 8 and re.fullmatch(r"SBJ\d{2}", parts[0]):
            subject_id = parts[0]
            for session_id, value_text in zip(header_sessions, parts[1:], strict=True):
                k_values[(subject_id, session_id)] = int(value_text)
    if len(k_values) != 105:
        raise ValueError(f"Expected 105 K values in {path}, found {len(k_values)}")
    return k_values


def validate_run_level_target_contract(
    *,
    labels: np.ndarray,
    flash_ids: np.ndarray,
    block_ids: np.ndarray,
    run_ids: np.ndarray,
    block_targets: np.ndarray,
    expected_blocks: int,
    expected_run_ids: np.ndarray,
    context: str,
    config: Step3Config,
) -> None:
    """Validate one-target-per-run and flash/target metadata consistency."""

    expected_flash_ids = np.arange(1, config.events_per_run + 1, dtype=np.int16)
    labels = np.asarray(labels, dtype=np.int8)
    flash_ids = np.asarray(flash_ids, dtype=np.int16)
    block_ids = np.asarray(block_ids, dtype=np.int16)
    run_ids = np.asarray(run_ids, dtype=np.int16)
    block_targets = np.asarray(block_targets, dtype=np.int16)

    for block_id in range(expected_blocks):
        target_flash = int(block_targets[block_id])
        if target_flash not in expected_flash_ids.tolist():
            raise ValueError(f"{context} block {block_id} has invalid target flash {target_flash}")
        for run_id in expected_run_ids.tolist():
            run_mask = (block_ids == block_id) & (run_ids == run_id)
            n_run_epochs = int(np.sum(run_mask))
            if n_run_epochs != config.events_per_run:
                raise ValueError(
                    f"{context} block {block_id} run {run_id} has {n_run_epochs} epochs, expected {config.events_per_run}",
                )

            run_flash_ids = flash_ids[run_mask]
            run_labels = labels[run_mask]
            if not np.array_equal(np.sort(run_flash_ids), expected_flash_ids):
                raise ValueError(
                    f"{context} block {block_id} run {run_id} has flash_ids {run_flash_ids.tolist()}, expected one each of 1..8",
                )

            expected_labels = (run_flash_ids == target_flash).astype(np.int8)
            if not np.array_equal(run_labels, expected_labels):
                raise ValueError(
                    f"{context} block {block_id} run {run_id} labels/flash_ids disagree with block_targets[{block_id}]={target_flash}",
                )


def load_processed_npz(path: Path, *, split: str, config: Step3Config) -> dict[str, np.ndarray]:
    """Load and validate one Step 1 NPZ file."""

    if not path.is_file():
        raise FileNotFoundError(f"Processed NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as npz_file:
        payload = {name: np.asarray(npz_file[name]) for name in npz_file.files}

    missing_fields = [name for name in REQUIRED_NPZ_FIELDS if name not in payload]
    if missing_fields:
        raise ValueError(f"{path.name} missing required fields: {missing_fields}")

    epochs = np.asarray(payload["epochs"])
    if epochs.ndim != 3 or epochs.shape[1:] != (config.n_channels, config.epoch_samples):
        raise ValueError(
            f"{path.name} has unexpected epochs shape {epochs.shape}; expected (*, {config.n_channels}, {config.epoch_samples})",
        )

    n_epochs = int(epochs.shape[0])
    for field_name in ("labels", "block_ids", "run_ids", "flash_ids", "condition"):
        field = np.asarray(payload[field_name])
        if field.shape != (n_epochs,):
            raise ValueError(f"{path.name} field {field_name} has shape {field.shape}, expected ({n_epochs},)")

    labels = np.asarray(payload["labels"], dtype=np.int8)
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError(f"{path.name} has unexpected label values {np.unique(labels).tolist()}")

    fs_value = int(np.asarray(payload["fs"]).item())
    if fs_value != config.output_fs:
        raise ValueError(f"{path.name} has fs={fs_value}, expected {config.output_fs}")

    ch_names = [str(item) for item in np.asarray(payload["ch_names"]).tolist()]
    if ch_names != list(CHANNEL_NAMES):
        raise ValueError(f"{path.name} has unexpected channel names {ch_names}")

    block_ids = np.asarray(payload["block_ids"], dtype=np.int16)
    run_ids = np.asarray(payload["run_ids"], dtype=np.int16)
    flash_ids = np.asarray(payload["flash_ids"], dtype=np.int16)
    block_targets = np.asarray(payload["block_targets"], dtype=np.int16)
    conditions = np.asarray(payload["condition"], dtype=np.int8)

    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    expected_block_ids = np.arange(expected_blocks, dtype=np.int16)
    unique_block_ids = np.unique(block_ids)
    if not np.array_equal(unique_block_ids, expected_block_ids):
        raise ValueError(f"{path.name} has block_ids {unique_block_ids.tolist()}, expected {expected_block_ids.tolist()}")
    if block_targets.shape != (expected_blocks,):
        raise ValueError(f"{path.name} block_targets shape {block_targets.shape}, expected ({expected_blocks},)")
    if not set(block_targets.tolist()).issubset(set(range(1, config.events_per_run + 1))):
        raise ValueError(f"{path.name} has out-of-range block_targets {np.unique(block_targets).tolist()}")

    if split == "calibration":
        expected_epochs = config.calibration_blocks * config.calibration_runs_per_block * config.events_per_run
        if n_epochs != expected_epochs:
            raise ValueError(f"{path.name} has {n_epochs} epochs, expected {expected_epochs}")
        unique_runs = np.unique(run_ids)
        expected_runs = np.arange(config.calibration_runs_per_block, dtype=np.int16)
        if not np.array_equal(unique_runs, expected_runs):
            raise ValueError(f"{path.name} run_ids {unique_runs.tolist()}, expected {expected_runs.tolist()}")
    else:
        unique_runs = np.unique(run_ids)
        if unique_runs.size < 3 or unique_runs.size > 10:
            raise ValueError(f"{path.name} derived K={unique_runs.size}, expected 3..10")
        expected_epochs = config.online_blocks * unique_runs.size * config.events_per_run
        if n_epochs != expected_epochs:
            raise ValueError(f"{path.name} has {n_epochs} epochs, expected {expected_epochs} for K={unique_runs.size}")

    for block_id in range(expected_blocks):
        block_mask = block_ids == block_id
        if not np.any(block_mask):
            raise ValueError(f"{path.name} missing block {block_id}")
        block_condition_values = np.unique(conditions[block_mask])
        if block_condition_values.size != 1:
            raise ValueError(f"{path.name} block {block_id} has mixed condition values {block_condition_values.tolist()}")
        expected_condition = 1 if split == "online" or block_id >= config.prototype_block_start else 0
        if int(block_condition_values[0]) != expected_condition:
            raise ValueError(
                f"{path.name} block {block_id} has condition={int(block_condition_values[0])}, expected {expected_condition}",
            )

    validate_run_level_target_contract(
        labels=labels,
        flash_ids=flash_ids,
        block_ids=block_ids,
        run_ids=run_ids,
        block_targets=block_targets,
        expected_blocks=expected_blocks,
        expected_run_ids=unique_runs,
        context=path.name,
        config=config,
    )

    return payload


def derive_online_k_from_payload(payload: dict[str, np.ndarray], config: Step3Config) -> int:
    """Derive and validate K from the online split run_ids."""

    run_ids = np.asarray(payload["run_ids"], dtype=np.int16)
    block_ids = np.asarray(payload["block_ids"], dtype=np.int16)
    unique_runs = np.unique(run_ids)
    derived_k = int(unique_runs.size)
    expected_runs = np.arange(derived_k, dtype=np.int16)
    if not np.array_equal(unique_runs, expected_runs):
        raise ValueError(f"Online run_ids are {unique_runs.tolist()}, expected contiguous 0..{derived_k - 1}")
    if derived_k < 3 or derived_k > 10:
        raise ValueError(f"Derived online K={derived_k}, expected 3..10")

    for block_id in range(config.online_blocks):
        block_run_ids = np.unique(run_ids[block_ids == block_id])
        if not np.array_equal(block_run_ids, expected_runs):
            raise ValueError(
                f"Online block {block_id} has run_ids {block_run_ids.tolist()}, expected {expected_runs.tolist()}",
            )
    return derived_k


def build_session_prototypes(
    calibration_payload: dict[str, np.ndarray],
    config: Step3Config,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute P_target and P_nontarget from the social calibration subset only."""

    epochs = np.asarray(calibration_payload["epochs"], dtype=np.float64)
    labels = np.asarray(calibration_payload["labels"], dtype=np.int8)
    block_ids = np.asarray(calibration_payload["block_ids"], dtype=np.int16)
    conditions = np.asarray(calibration_payload["condition"], dtype=np.int8)

    social_mask = (
        (conditions == 1)
        & (block_ids >= config.prototype_block_start)
        & (block_ids <= config.prototype_block_end)
    )
    if int(np.sum(social_mask)) != 10 * config.calibration_runs_per_block * config.events_per_run:
        raise ValueError(
            "Social calibration subset must contain exactly 800 epochs "
            f"(found {int(np.sum(social_mask))})",
        )

    target_epochs = epochs[social_mask & (labels == 1)]
    nontarget_epochs = epochs[social_mask & (labels == 0)]
    if target_epochs.shape != (100, config.n_channels, config.epoch_samples):
        raise ValueError(f"Expected 100 social target epochs, got shape {target_epochs.shape}")
    if nontarget_epochs.shape != (700, config.n_channels, config.epoch_samples):
        raise ValueError(f"Expected 700 social non-target epochs, got shape {nontarget_epochs.shape}")

    p_target = target_epochs.mean(axis=0)
    p_nontarget = nontarget_epochs.mean(axis=0)
    separation = float(np.linalg.norm(p_target - p_nontarget, ord="fro"))
    return p_target, p_nontarget, separation


def aggregate_blocks(
    *,
    payload: dict[str, np.ndarray],
    split: str,
    runs_per_block: int,
    config: Step3Config,
) -> BlockAverages:
    """Compute one block-level target average plus saved metadata per block."""

    epochs = np.asarray(payload["epochs"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int8)
    block_ids = np.asarray(payload["block_ids"], dtype=np.int16)
    run_ids = np.asarray(payload["run_ids"], dtype=np.int16)
    flash_ids = np.asarray(payload["flash_ids"], dtype=np.int16)
    conditions = np.asarray(payload["condition"], dtype=np.int8)
    source_block_targets = np.asarray(payload["block_targets"], dtype=np.int16)

    n_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    validate_run_level_target_contract(
        labels=labels,
        flash_ids=flash_ids,
        block_ids=block_ids,
        run_ids=run_ids,
        block_targets=source_block_targets,
        expected_blocks=n_blocks,
        expected_run_ids=np.arange(runs_per_block, dtype=np.int16),
        context=f"{split} aggregation payload",
        config=config,
    )

    target_averages = np.zeros((n_blocks, config.n_channels, config.epoch_samples), dtype=np.float64)
    condition = np.zeros(n_blocks, dtype=np.int8)
    block_targets = np.zeros(n_blocks, dtype=np.int8)
    n_target_epochs = np.zeros(n_blocks, dtype=np.int16)
    n_nontarget_epochs = np.zeros(n_blocks, dtype=np.int16)
    runs_per_block_array = np.full(n_blocks, runs_per_block, dtype=np.int16)

    expected_target_count = runs_per_block
    expected_nontarget_count = 7 * runs_per_block
    expected_epoch_count = expected_target_count + expected_nontarget_count

    for block_id in range(n_blocks):
        block_mask = block_ids == block_id
        if int(np.sum(block_mask)) != expected_epoch_count:
            raise ValueError(
                f"{split} block {block_id} has {int(np.sum(block_mask))} epochs, expected {expected_epoch_count}",
            )
        block_conditions = np.unique(conditions[block_mask])
        if block_conditions.size != 1:
            raise ValueError(f"{split} block {block_id} has mixed conditions {block_conditions.tolist()}")

        target_mask = block_mask & (labels == 1)
        nontarget_mask = block_mask & (labels == 0)
        if int(np.sum(target_mask)) != expected_target_count:
            raise ValueError(
                f"{split} block {block_id} has {int(np.sum(target_mask))} target epochs, expected {expected_target_count}",
            )
        if int(np.sum(nontarget_mask)) != expected_nontarget_count:
            raise ValueError(
                f"{split} block {block_id} has {int(np.sum(nontarget_mask))} non-target epochs, expected {expected_nontarget_count}",
            )

        target_averages[block_id] = epochs[target_mask].mean(axis=0)
        condition[block_id] = int(block_conditions[0])
        block_targets[block_id] = int(source_block_targets[block_id])
        n_target_epochs[block_id] = int(np.sum(target_mask))
        n_nontarget_epochs[block_id] = int(np.sum(nontarget_mask))

    return BlockAverages(
        target_averages=target_averages,
        block_ids=np.arange(n_blocks, dtype=np.int16),
        condition=condition,
        block_targets=block_targets,
        runs_per_block=runs_per_block_array,
        n_target_epochs=n_target_epochs,
        n_nontarget_epochs=n_nontarget_epochs,
    )


def build_super_trial(p_target: np.ndarray, p_nontarget: np.ndarray, x_target_avg: np.ndarray) -> np.ndarray:
    """Construct the 24 x 128 augmented super-trial."""

    x_aug = np.vstack(
        [
            np.asarray(p_target, dtype=np.float64),
            np.asarray(p_nontarget, dtype=np.float64),
            np.asarray(x_target_avg, dtype=np.float64),
        ]
    )
    if x_aug.shape != (24, 128):
        raise ValueError(f"Super-trial shape {x_aug.shape}, expected (24, 128)")
    return x_aug


def estimate_oas_covariance(
    x_aug: np.ndarray,
    config: Step3Config,
) -> tuple[np.ndarray, float, float]:
    """Estimate the centered OAS covariance and a raw pre-shrinkage diagnostic."""

    x_aug = np.asarray(x_aug, dtype=np.float64)
    centered = x_aug - x_aug.mean(axis=1, keepdims=True)
    raw_covariance = centered @ centered.T / float(config.epoch_samples - 1)
    raw_min_eigenvalue = float(np.linalg.eigvalsh(raw_covariance).min())

    estimator = OAS(store_precision=False, assume_centered=False)
    estimator.fit(x_aug.T)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    shrinkage_alpha = float(estimator.shrinkage_)
    return covariance, shrinkage_alpha, raw_min_eigenvalue


def spectral_qc_metrics(covariance: np.ndarray, config: Step3Config) -> CovarianceMetrics:
    """Verify symmetry / SPD and compute spectral QC metrics."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"Covariance matrix shape {covariance.shape}, expected ({config.covariance_dim}, {config.covariance_dim})",
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance matrix contains non-finite values")

    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    if symmetry_error > config.symmetry_tolerance:
        raise ValueError(f"Covariance matrix symmetry error {symmetry_error:.3e} exceeds tolerance")

    eigenvalues = np.linalg.eigvalsh(covariance)
    min_eigenvalue = float(eigenvalues[0])
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Covariance eigenvalues contain non-finite values")
    if min_eigenvalue <= 0.0:
        raise ValueError(f"Covariance matrix is not strictly SPD; min eigenvalue={min_eigenvalue:.6e}")

    lambda_max = float(eigenvalues[-1])
    lambda_median = float(np.median(eigenvalues))
    condition_number = lambda_max / min_eigenvalue
    dominance_ratio = lambda_max / lambda_median
    return CovarianceMetrics(
        min_eigenvalue=min_eigenvalue,
        condition_number=float(condition_number),
        dominance_ratio=float(dominance_ratio),
        symmetry_error=symmetry_error,
    )


def compute_covariances_for_blocks(
    *,
    block_averages: BlockAverages,
    p_target: np.ndarray,
    p_nontarget: np.ndarray,
    config: Step3Config,
    split: str,
    subject_id: str,
    session_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build one SPD covariance matrix per block plus QC arrays."""

    n_blocks = int(block_averages.block_ids.size)
    covariances = np.zeros((n_blocks, config.covariance_dim, config.covariance_dim), dtype=np.float64)
    shrinkage_alphas = np.zeros(n_blocks, dtype=np.float64)
    condition_numbers = np.zeros(n_blocks, dtype=np.float64)
    dominance_ratios = np.zeros(n_blocks, dtype=np.float64)
    warning_lines: list[str] = []

    for block_index, block_id in enumerate(block_averages.block_ids.tolist()):
        x_aug = build_super_trial(p_target, p_nontarget, block_averages.target_averages[block_index])
        covariance, alpha, raw_min_eigenvalue = estimate_oas_covariance(x_aug, config)
        metrics = spectral_qc_metrics(covariance, config)

        covariances[block_index] = covariance
        shrinkage_alphas[block_index] = alpha
        condition_numbers[block_index] = metrics.condition_number
        dominance_ratios[block_index] = metrics.dominance_ratio

        if raw_min_eigenvalue <= config.raw_min_eigenvalue_warning:
            warning_lines.append(
                f"{subject_id}/{session_id} {split} block {block_id}: raw centered covariance min eigenvalue "
                f"{raw_min_eigenvalue:.6e} <= {config.raw_min_eigenvalue_warning:.1e}"
            )
        if metrics.condition_number > config.condition_number_warning:
            warning_lines.append(
                f"{subject_id}/{session_id} {split} block {block_id}: condition number "
                f"{metrics.condition_number:.6e} > {config.condition_number_warning:.1e}"
            )
        if metrics.dominance_ratio > config.dominance_ratio_warning:
            warning_lines.append(
                f"{subject_id}/{session_id} {split} block {block_id}: dominance ratio "
                f"{metrics.dominance_ratio:.6e} > {config.dominance_ratio_warning:.1f}"
            )

    return covariances, shrinkage_alphas, condition_numbers, dominance_ratios, tuple(warning_lines)


def save_covariance_npz(
    *,
    path: Path,
    covariances: np.ndarray,
    block_averages: BlockAverages,
    p_target: np.ndarray,
    p_nontarget: np.ndarray,
    shrinkage_alphas: np.ndarray,
    condition_numbers: np.ndarray,
    dominance_ratios: np.ndarray,
) -> None:
    """Save one Step 3 covariance NPZ file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        covs=np.asarray(covariances, dtype=np.float64),
        block_ids=np.asarray(block_averages.block_ids, dtype=np.int16),
        condition=np.asarray(block_averages.condition, dtype=np.int8),
        block_targets=np.asarray(block_averages.block_targets, dtype=np.int8),
        runs_per_block=np.asarray(block_averages.runs_per_block, dtype=np.int16),
        n_target_epochs=np.asarray(block_averages.n_target_epochs, dtype=np.int16),
        n_nontarget_epochs=np.asarray(block_averages.n_nontarget_epochs, dtype=np.int16),
        P_target=np.asarray(p_target, dtype=np.float64),
        P_nontarget=np.asarray(p_nontarget, dtype=np.float64),
        shrinkage_alphas=np.asarray(shrinkage_alphas, dtype=np.float64),
        condition_numbers=np.asarray(condition_numbers, dtype=np.float64),
        lambda_max_over_median=np.asarray(dominance_ratios, dtype=np.float64),
    )


def write_session_log(
    *,
    path: Path,
    session_output: SessionOutputs,
) -> None:
    """Write the session-level Step 3 diagnostics log."""

    lines = [
        f"Session: {session_output.subject_id}/{session_output.session_id}",
        "Prototype source: social calibration blocks 10-19",
        "Prototype epoch counts: 100 target, 700 non-target",
        f"Prototype separation Frobenius norm: {format_float(session_output.prototype_separation_norm)}",
        f"Session K: {session_output.session_k}",
        (
            "Calibration shrinkage alpha range: "
            f"{format_float(float(np.min(session_output.calibration_alphas)))} to "
            f"{format_float(float(np.max(session_output.calibration_alphas)))}"
        ),
        (
            "Online shrinkage alpha range: "
            f"{format_float(float(np.min(session_output.online_alphas)))} to "
            f"{format_float(float(np.max(session_output.online_alphas)))}"
        ),
        (
            "Calibration condition-number range: "
            f"{format_float(float(np.min(session_output.calibration_condition_numbers)))} to "
            f"{format_float(float(np.max(session_output.calibration_condition_numbers)))}"
        ),
        (
            "Online condition-number range: "
            f"{format_float(float(np.min(session_output.online_condition_numbers)))} to "
            f"{format_float(float(np.max(session_output.online_condition_numbers)))}"
        ),
        (
            "Calibration dominance-ratio range: "
            f"{format_float(float(np.min(session_output.calibration_dominance_ratios)))} to "
            f"{format_float(float(np.max(session_output.calibration_dominance_ratios)))}"
        ),
        (
            "Online dominance-ratio range: "
            f"{format_float(float(np.min(session_output.online_dominance_ratios)))} to "
            f"{format_float(float(np.max(session_output.online_dominance_ratios)))}"
        ),
        "Warnings:",
    ]
    if session_output.warning_lines:
        lines.extend(f"- {warning}" for warning in session_output.warning_lines)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_session_log_contents(path: Path) -> dict[str, object]:
    """Validate a Step 3 session log and return its parsed contents."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 3 session log: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()

    def get_line(prefix: str) -> str:
        line = next((candidate for candidate in lines if candidate.startswith(prefix)), None)
        if line is None:
            raise ValueError(f"{path.name} missing required log line starting with {prefix!r}")
        return line

    session_line = get_line("Session:")
    expected_session_match = re.fullmatch(r"sub-(\d{2})_ses-(\d{2})_cov_log\.txt", path.name)
    session_match = re.fullmatch(r"Session:\s+(SBJ\d{2})/(S\d{2})", session_line)
    if session_match is None:
        raise ValueError(f"{path.name} has invalid Session line: {session_line!r}")
    if expected_session_match is not None:
        expected_subject = f"SBJ{expected_session_match.group(1)}"
        expected_session = f"S{expected_session_match.group(2)}"
        if session_match.group(1) != expected_subject or session_match.group(2) != expected_session:
            raise ValueError(
                f"{path.name} Session line {session_match.group(1)}/{session_match.group(2)} "
                f"does not match filename {expected_subject}/{expected_session}"
            )

    prototype_source_line = get_line("Prototype source:")
    if prototype_source_line != "Prototype source: social calibration blocks 10-19":
        raise ValueError(f"{path.name} has unexpected prototype source line: {prototype_source_line!r}")

    prototype_counts_line = get_line("Prototype epoch counts:")
    if prototype_counts_line != "Prototype epoch counts: 100 target, 700 non-target":
        raise ValueError(f"{path.name} has unexpected prototype epoch counts line: {prototype_counts_line!r}")

    separation_line = get_line("Prototype separation Frobenius norm:")
    separation_match = re.fullmatch(r"Prototype separation Frobenius norm:\s+([-+]?\d+(?:\.\d+)?)", separation_line)
    if separation_match is None:
        raise ValueError(f"{path.name} has invalid prototype separation line: {separation_line!r}")
    separation_value = float(separation_match.group(1))
    if not np.isfinite(separation_value) or separation_value <= 0.0:
        raise ValueError(f"{path.name} has invalid prototype separation value {separation_value}")

    session_k_line = get_line("Session K:")
    session_k_match = re.fullmatch(r"Session K:\s+(\d+)", session_k_line)
    if session_k_match is None:
        raise ValueError(f"{path.name} has invalid Session K line: {session_k_line!r}")
    session_k = int(session_k_match.group(1))
    if session_k < 3 or session_k > 10:
        raise ValueError(f"{path.name} has Session K={session_k}, expected 3..10")

    parsed_ranges: dict[str, tuple[float, float]] = {}
    range_specs = (
        ("Calibration shrinkage alpha range:", 0.0, 1.0),
        ("Online shrinkage alpha range:", 0.0, 1.0),
        ("Calibration condition-number range:", 0.0, None),
        ("Online condition-number range:", 0.0, None),
        ("Calibration dominance-ratio range:", 0.0, None),
        ("Online dominance-ratio range:", 0.0, None),
    )
    for prefix, lower_bound, upper_bound in range_specs:
        line = get_line(prefix)
        range_match = re.fullmatch(
            rf"{re.escape(prefix)}\s+([-+]?\d+(?:\.\d+)?) to ([-+]?\d+(?:\.\d+)?)",
            line,
        )
        if range_match is None:
            raise ValueError(f"{path.name} has invalid range line: {line!r}")
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(f"{path.name} has non-finite range values in line: {line!r}")
        if low > high:
            raise ValueError(f"{path.name} has decreasing range in line: {line!r}")
        if low < lower_bound or high < lower_bound:
            raise ValueError(f"{path.name} has out-of-domain range in line: {line!r}")
        if upper_bound is not None and (low > upper_bound or high > upper_bound):
            raise ValueError(f"{path.name} has out-of-domain range in line: {line!r}")
        parsed_ranges[prefix] = (low, high)

    warnings_index = next((index for index, line in enumerate(lines) if line.startswith("Warnings:")), None)
    if warnings_index is None or warnings_index == len(lines) - 1:
        raise ValueError(f"{path.name} warnings section must contain at least one bullet line")
    if not all(line.startswith("- ") for line in lines[warnings_index + 1 :]):
        raise ValueError(f"{path.name} warnings section must contain only '- ' bullet lines after the header")
    return {
        "subject_id": session_match.group(1),
        "session_id": session_match.group(2),
        "prototype_separation": separation_value,
        "session_k": session_k,
        "ranges": parsed_ranges,
    }


def process_session(
    *,
    subject_id: str,
    session_id: str,
    session_k: int,
    config: Step3Config,
) -> SessionOutputs:
    """Process one subject/session pair into Step 3 covariance outputs."""

    calibration_payload = load_processed_npz(
        processed_split_path(
            processed_root=config.processed_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        ),
        split="calibration",
        config=config,
    )
    online_payload = load_processed_npz(
        processed_split_path(
            processed_root=config.processed_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        ),
        split="online",
        config=config,
    )

    derived_k = derive_online_k_from_payload(online_payload, config)
    if derived_k != session_k:
        raise ValueError(f"{subject_id}/{session_id} inventory K={session_k}, derived online K={derived_k}")

    p_target, p_nontarget, prototype_separation = build_session_prototypes(calibration_payload, config)
    calibration_blocks = aggregate_blocks(
        payload=calibration_payload,
        split="calibration",
        runs_per_block=config.calibration_runs_per_block,
        config=config,
    )
    online_blocks = aggregate_blocks(
        payload=online_payload,
        split="online",
        runs_per_block=session_k,
        config=config,
    )

    (
        calibration_covariances,
        calibration_alphas,
        calibration_condition_numbers,
        calibration_dominance_ratios,
        calibration_warnings,
    ) = compute_covariances_for_blocks(
        block_averages=calibration_blocks,
        p_target=p_target,
        p_nontarget=p_nontarget,
        config=config,
        split="calibration",
        subject_id=subject_id,
        session_id=session_id,
    )
    (
        online_covariances,
        online_alphas,
        online_condition_numbers,
        online_dominance_ratios,
        online_warnings,
    ) = compute_covariances_for_blocks(
        block_averages=online_blocks,
        p_target=p_target,
        p_nontarget=p_nontarget,
        config=config,
        split="online",
        subject_id=subject_id,
        session_id=session_id,
    )

    save_covariance_npz(
        path=feature_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        ),
        covariances=calibration_covariances,
        block_averages=calibration_blocks,
        p_target=p_target,
        p_nontarget=p_nontarget,
        shrinkage_alphas=calibration_alphas,
        condition_numbers=calibration_condition_numbers,
        dominance_ratios=calibration_dominance_ratios,
    )
    save_covariance_npz(
        path=feature_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        ),
        covariances=online_covariances,
        block_averages=online_blocks,
        p_target=p_target,
        p_nontarget=p_nontarget,
        shrinkage_alphas=online_alphas,
        condition_numbers=online_condition_numbers,
        dominance_ratios=online_dominance_ratios,
    )

    session_outputs = SessionOutputs(
        subject_id=subject_id,
        session_id=session_id,
        session_k=session_k,
        prototype_separation_norm=prototype_separation,
        calibration_alphas=calibration_alphas,
        online_alphas=online_alphas,
        calibration_condition_numbers=calibration_condition_numbers,
        online_condition_numbers=online_condition_numbers,
        calibration_dominance_ratios=calibration_dominance_ratios,
        online_dominance_ratios=online_dominance_ratios,
        warning_lines=tuple([*calibration_warnings, *online_warnings]),
    )
    write_session_log(
        path=session_log_path(output_root=config.output_root, subject_id=subject_id, session_id=session_id),
        session_output=session_outputs,
    )
    return session_outputs


def _validate_saved_covariance_payload(
    *,
    path: Path,
    split: str,
    config: Step3Config,
) -> dict[str, np.ndarray]:
    """Load and validate one saved Step 3 covariance NPZ file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 3 output: {path}")
    with np.load(path, allow_pickle=False) as npz_file:
        payload = {name: np.asarray(npz_file[name]) for name in npz_file.files}

    required_fields = {
        "covs",
        "block_ids",
        "condition",
        "block_targets",
        "runs_per_block",
        "n_target_epochs",
        "n_nontarget_epochs",
        "P_target",
        "P_nontarget",
        "shrinkage_alphas",
        "condition_numbers",
        "lambda_max_over_median",
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise ValueError(f"{path.name} missing required Step 3 fields: {missing}")

    expected_dtypes = {
        "covs": np.dtype(np.float64),
        "block_ids": np.dtype(np.int16),
        "condition": np.dtype(np.int8),
        "block_targets": np.dtype(np.int8),
        "runs_per_block": np.dtype(np.int16),
        "n_target_epochs": np.dtype(np.int16),
        "n_nontarget_epochs": np.dtype(np.int16),
        "P_target": np.dtype(np.float64),
        "P_nontarget": np.dtype(np.float64),
        "shrinkage_alphas": np.dtype(np.float64),
        "condition_numbers": np.dtype(np.float64),
        "lambda_max_over_median": np.dtype(np.float64),
    }
    for field_name, expected_dtype in expected_dtypes.items():
        actual_dtype = np.asarray(payload[field_name]).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"{path.name} field {field_name} has dtype {actual_dtype}, expected {expected_dtype}",
            )

    expected_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    covs = np.asarray(payload["covs"], dtype=np.float64)
    if covs.shape != (expected_blocks, config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{path.name} has covs shape {covs.shape}, expected ({expected_blocks}, {config.covariance_dim}, {config.covariance_dim})",
        )
    for field_name in (
        "block_ids",
        "condition",
        "block_targets",
        "runs_per_block",
        "n_target_epochs",
        "n_nontarget_epochs",
        "shrinkage_alphas",
        "condition_numbers",
        "lambda_max_over_median",
    ):
        field = np.asarray(payload[field_name])
        if field.shape != (expected_blocks,):
            raise ValueError(f"{path.name} field {field_name} has shape {field.shape}, expected ({expected_blocks},)")

    p_target = np.asarray(payload["P_target"], dtype=np.float64)
    p_nontarget = np.asarray(payload["P_nontarget"], dtype=np.float64)
    if p_target.shape != (config.n_channels, config.epoch_samples):
        raise ValueError(f"{path.name} P_target shape {p_target.shape}, expected ({config.n_channels}, {config.epoch_samples})")
    if p_nontarget.shape != (config.n_channels, config.epoch_samples):
        raise ValueError(
            f"{path.name} P_nontarget shape {p_nontarget.shape}, expected ({config.n_channels}, {config.epoch_samples})",
        )
    if np.allclose(p_target, p_nontarget):
        raise ValueError(f"{path.name} has identical P_target and P_nontarget prototypes")

    block_ids = np.asarray(payload["block_ids"], dtype=np.int16)
    if not np.array_equal(block_ids, np.arange(expected_blocks, dtype=np.int16)):
        raise ValueError(f"{path.name} block_ids mismatch: {block_ids.tolist()}")

    condition = np.asarray(payload["condition"], dtype=np.int8)
    block_targets = np.asarray(payload["block_targets"], dtype=np.int8)
    runs_per_block = np.asarray(payload["runs_per_block"], dtype=np.int16)
    n_target_epochs = np.asarray(payload["n_target_epochs"], dtype=np.int16)
    n_nontarget_epochs = np.asarray(payload["n_nontarget_epochs"], dtype=np.int16)
    if not set(block_targets.tolist()).issubset(set(range(1, config.events_per_run + 1))):
        raise ValueError(
            f"{path.name} block_targets values {sorted(set(block_targets.tolist()))} fall outside 1..{config.events_per_run}",
        )

    if split == "calibration":
        expected_condition = np.concatenate(
            [
                np.zeros(config.prototype_block_start, dtype=np.int8),
                np.ones(config.calibration_blocks - config.prototype_block_start, dtype=np.int8),
            ]
        )
        if not np.array_equal(condition, expected_condition):
            raise ValueError(f"{path.name} calibration condition metadata mismatch")
        if not np.all(runs_per_block == config.calibration_runs_per_block):
            raise ValueError(f"{path.name} calibration runs_per_block must all equal {config.calibration_runs_per_block}")
    else:
        if not np.all(condition == 1):
            raise ValueError(f"{path.name} online condition metadata must be all 1")
        unique_runs = np.unique(runs_per_block)
        if unique_runs.size != 1:
            raise ValueError(f"{path.name} online runs_per_block must be constant within file")
        if int(unique_runs[0]) < 3 or int(unique_runs[0]) > 10:
            raise ValueError(f"{path.name} online runs_per_block={int(unique_runs[0])}, expected 3..10")

    if not np.array_equal(n_target_epochs, runs_per_block):
        raise ValueError(f"{path.name} must satisfy n_target_epochs == runs_per_block")
    if not np.array_equal(n_nontarget_epochs, 7 * runs_per_block):
        raise ValueError(f"{path.name} must satisfy n_nontarget_epochs == 7 * runs_per_block")

    shrinkage_alphas = np.asarray(payload["shrinkage_alphas"], dtype=np.float64)
    condition_numbers = np.asarray(payload["condition_numbers"], dtype=np.float64)
    dominance_ratios = np.asarray(payload["lambda_max_over_median"], dtype=np.float64)
    if not np.all(np.isfinite(shrinkage_alphas)):
        raise ValueError(f"{path.name} shrinkage_alphas contain non-finite values")
    if np.any((shrinkage_alphas < 0.0) | (shrinkage_alphas > 1.0)):
        raise ValueError(f"{path.name} shrinkage_alphas fall outside [0, 1]")
    if not np.all(np.isfinite(condition_numbers)) or np.any(condition_numbers <= 0.0):
        raise ValueError(f"{path.name} condition_numbers must be finite and positive")
    if not np.all(np.isfinite(dominance_ratios)) or np.any(dominance_ratios <= 0.0):
        raise ValueError(f"{path.name} lambda_max_over_median must be finite and positive")

    for covariance in covs:
        spectral_qc_metrics(covariance, config)

    return payload


def smoke_test_outputs(config: Step3Config) -> Step3Summary:
    """Run dataset-wide smoke tests across all Step 3 outputs."""

    calibration_file_count = 0
    online_file_count = 0
    session_log_count = 0
    total_covariance_matrices = 0
    prototype_separations: list[float] = []
    shrinkage_alphas: list[float] = []
    spectral_outlier_blocks = 0
    near_singular_sessions = 0

    def rounded_float(value: float) -> float:
        return float(format_float(value))

    for subject_id, session_id in expected_session_ids():
        calibration_path = feature_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="calibration",
        )
        online_path = feature_split_path(
            output_root=config.output_root,
            subject_id=subject_id,
            session_id=session_id,
            split="online",
        )
        log_path = session_log_path(output_root=config.output_root, subject_id=subject_id, session_id=session_id)

        calibration_payload = _validate_saved_covariance_payload(path=calibration_path, split="calibration", config=config)
        online_payload = _validate_saved_covariance_payload(path=online_path, split="online", config=config)
        log_payload = validate_session_log_contents(log_path)

        calibration_file_count += 1
        online_file_count += 1
        session_log_count += 1
        total_covariance_matrices += int(calibration_payload["covs"].shape[0] + online_payload["covs"].shape[0])

        calibration_p_target = np.asarray(calibration_payload["P_target"], dtype=np.float64)
        calibration_p_nontarget = np.asarray(calibration_payload["P_nontarget"], dtype=np.float64)
        online_p_target = np.asarray(online_payload["P_target"], dtype=np.float64)
        online_p_nontarget = np.asarray(online_payload["P_nontarget"], dtype=np.float64)
        if not np.array_equal(calibration_p_target, online_p_target):
            raise ValueError(f"{subject_id}/{session_id} calibration and online P_target arrays do not match")
        if not np.array_equal(calibration_p_nontarget, online_p_nontarget):
            raise ValueError(f"{subject_id}/{session_id} calibration and online P_nontarget arrays do not match")

        prototype_separations.append(
            float(
                np.linalg.norm(
                    calibration_p_target - calibration_p_nontarget,
                    ord="fro",
                )
            )
        )
        shrinkage_alphas.extend(np.asarray(calibration_payload["shrinkage_alphas"], dtype=np.float64).tolist())
        shrinkage_alphas.extend(np.asarray(online_payload["shrinkage_alphas"], dtype=np.float64).tolist())

        session_condition_numbers = np.concatenate(
            [
                np.asarray(calibration_payload["condition_numbers"], dtype=np.float64),
                np.asarray(online_payload["condition_numbers"], dtype=np.float64),
            ]
        )
        session_dominance = np.concatenate(
            [
                np.asarray(calibration_payload["lambda_max_over_median"], dtype=np.float64),
                np.asarray(online_payload["lambda_max_over_median"], dtype=np.float64),
            ]
        )
        spectral_outlier_blocks += int(np.sum(session_dominance > config.dominance_ratio_warning))
        if np.any(session_condition_numbers > config.condition_number_warning):
            near_singular_sessions += 1

        online_runs = np.asarray(online_payload["runs_per_block"], dtype=np.int16)
        if np.unique(online_runs).size != 1:
            raise ValueError(f"{subject_id}/{session_id} online runs_per_block is not constant")
        actual_session_k = int(online_runs[0])
        if log_payload["subject_id"] != subject_id or log_payload["session_id"] != session_id:
            raise ValueError(f"{subject_id}/{session_id} log session identity does not match the current artifacts")
        if int(log_payload["session_k"]) != actual_session_k:
            raise ValueError(
                f"{subject_id}/{session_id} log Session K={int(log_payload['session_k'])} "
                f"does not match saved runs_per_block={actual_session_k}",
            )

        actual_prototype_separation = rounded_float(
            float(np.linalg.norm(calibration_p_target - calibration_p_nontarget, ord="fro"))
        )
        if float(log_payload["prototype_separation"]) != actual_prototype_separation:
            raise ValueError(
                f"{subject_id}/{session_id} log prototype separation {float(log_payload['prototype_separation'])} "
                f"does not match saved prototypes {actual_prototype_separation}",
            )

        log_ranges = log_payload["ranges"]
        expected_ranges = {
            "Calibration shrinkage alpha range:": (
                rounded_float(float(np.min(calibration_payload["shrinkage_alphas"]))),
                rounded_float(float(np.max(calibration_payload["shrinkage_alphas"]))),
            ),
            "Online shrinkage alpha range:": (
                rounded_float(float(np.min(online_payload["shrinkage_alphas"]))),
                rounded_float(float(np.max(online_payload["shrinkage_alphas"]))),
            ),
            "Calibration condition-number range:": (
                rounded_float(float(np.min(calibration_payload["condition_numbers"]))),
                rounded_float(float(np.max(calibration_payload["condition_numbers"]))),
            ),
            "Online condition-number range:": (
                rounded_float(float(np.min(online_payload["condition_numbers"]))),
                rounded_float(float(np.max(online_payload["condition_numbers"]))),
            ),
            "Calibration dominance-ratio range:": (
                rounded_float(float(np.min(calibration_payload["lambda_max_over_median"]))),
                rounded_float(float(np.max(calibration_payload["lambda_max_over_median"]))),
            ),
            "Online dominance-ratio range:": (
                rounded_float(float(np.min(online_payload["lambda_max_over_median"]))),
                rounded_float(float(np.max(online_payload["lambda_max_over_median"]))),
            ),
        }
        for prefix, expected_range in expected_ranges.items():
            if tuple(log_ranges[prefix]) != expected_range:
                raise ValueError(
                    f"{subject_id}/{session_id} log range {prefix!r}={tuple(log_ranges[prefix])} "
                    f"does not match saved values {expected_range}",
                )

    if calibration_file_count != 105 or online_file_count != 105 or session_log_count != 105:
        raise ValueError(
            "Step 3 output counts mismatch: expected 105 calibration NPZs, "
            "105 online NPZs, and 105 logs"
        )
    if total_covariance_matrices != 7350:
        raise ValueError(f"Total covariance-matrix count {total_covariance_matrices}, expected 7350")

    prototype_array = np.asarray(prototype_separations, dtype=np.float64)
    alpha_array = np.asarray(shrinkage_alphas, dtype=np.float64)
    return Step3Summary(
        output_root=config.output_root,
        calibration_file_count=calibration_file_count,
        online_file_count=online_file_count,
        session_log_count=session_log_count,
        total_covariance_matrices=total_covariance_matrices,
        prototype_separation_mean=float(np.mean(prototype_array)),
        prototype_separation_median=float(np.median(prototype_array)),
        prototype_separation_min=float(np.min(prototype_array)),
        prototype_separation_max=float(np.max(prototype_array)),
        shrinkage_alpha_mean=float(np.mean(alpha_array)),
        shrinkage_alpha_median=float(np.median(alpha_array)),
        shrinkage_alpha_min=float(np.min(alpha_array)),
        shrinkage_alpha_max=float(np.max(alpha_array)),
        spectral_outlier_blocks=spectral_outlier_blocks,
        near_singular_sessions=near_singular_sessions,
    )


def log_summary(summary: Step3Summary) -> None:
    """Log the headline Step 3 metrics."""

    logger.info(
        "Step 3 complete. Wrote %d calibration files, %d online files, %d logs, and %d covariance matrices.",
        summary.calibration_file_count,
        summary.online_file_count,
        summary.session_log_count,
        summary.total_covariance_matrices,
    )
    logger.info(
        "Prototype separation mean %.6f, median %.6f, range [%.6f, %.6f].",
        summary.prototype_separation_mean,
        summary.prototype_separation_median,
        summary.prototype_separation_min,
        summary.prototype_separation_max,
    )
    logger.info(
        "Shrinkage alpha mean %.6f, median %.6f, range [%.6f, %.6f]. Spectral outlier blocks: %d. Near-singular warning sessions: %d.",
        summary.shrinkage_alpha_mean,
        summary.shrinkage_alpha_median,
        summary.shrinkage_alpha_min,
        summary.shrinkage_alpha_max,
        summary.spectral_outlier_blocks,
        summary.near_singular_sessions,
    )


def run_step3(config: Step3Config | None = None) -> Step3Summary:
    """Run the full Step 3 covariance-computation pipeline."""

    config = config or Step3Config()
    config.output_root.mkdir(parents=True, exist_ok=True)
    k_values = parse_k_values_from_inventory(config.inventory_path)

    for subject_id, session_id in expected_session_ids():
        process_session(
            subject_id=subject_id,
            session_id=session_id,
            session_k=k_values[(subject_id, session_id)],
            config=config,
        )

    summary = smoke_test_outputs(config)
    log_summary(summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Step 3."""

    parser = argparse.ArgumentParser(description="Run Step 3 covariance computation.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("data/features"))
    parser.add_argument("--inventory-path", type=Path, default=Path("data/processed/dataset_inventory.txt"))
    return parser


def main() -> None:
    """CLI entrypoint for Step 3."""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_step3(
        Step3Config(
            processed_root=args.processed_root,
            output_root=args.output_root,
            inventory_path=args.inventory_path,
        )
    )


if __name__ == "__main__":
    main()
