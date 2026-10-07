"""Step 1 preprocessing pipeline for BCIAUT-P300.

Processing order per session (calibration and online kept separate):
  transpose → crop → hard-clip → baseline-correct →
  compute winsorizing thresholds (calibration only) →
  apply winsorizing → downsample → save NPZ
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.data.loader import (
    CHANNEL_NAMES,
    DEFAULT_ACCEPTED_RAW_TIMEPOINTS,
    CALIBRATION_RUNS_PER_BLOCK,
    EVENTS_PER_RUN,
    N_CALIBRATION_BLOCKS,
    N_CALIBRATION_EPOCHS,
    N_CHANNELS,
    N_ONLINE_BLOCKS,
    DatasetInventory,
    LoadedSession,
    LoadedSplit,
    SessionMetadataRepair,
    build_dataset_inventory,
    expected_session_ids,
    format_dataset_inventory,
    load_session,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_FLASH_IDS: np.ndarray = np.arange(1, 9, dtype=np.int8)

CH_NAMES_ARRAY: np.ndarray = np.array(CHANNEL_NAMES, dtype="U4")

# Manual override entries for downstream QC (sub-step 1.9)
MANUAL_OVERRIDES: list[dict[str, str]] = [
    {
        "subject": "SBJ03", "session": "S07", "split": "calibration",
        "flag_type": "manual_override", "channel": "POz",
        "details": "POz baseline std anomalous - expert override; does not meet >=4-channel session_level threshold",
    },
    {
        "subject": "SBJ03", "session": "S07", "split": "online",
        "flag_type": "manual_override", "channel": "POz",
        "details": "POz baseline std anomalous - expert override; does not meet >=4-channel session_level threshold",
    },
    {
        "subject": "SBJ05", "session": "S07", "split": "online",
        "flag_type": "manual_override", "channel": "POz",
        "details": "POz baseline std marginally out of range - expert override",
    },
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for Step 1 preprocessing."""

    raw_root: Path = Path("data/raw")
    processed_root: Path = Path("data/processed")
    crop_samples: int = 250
    baseline_samples_raw: int = 50
    baseline_samples_downsampled: int = 26
    original_fs: int = 250
    output_fs: int = 128
    downsample_up: int = 64
    downsample_down: int = 125
    winsor_low_percentile: float = 1.0
    winsor_high_percentile: float = 99.0
    hard_clip_uv: float = 500.0
    baseline_std_warn_low_uv: float = 1.0
    baseline_std_warn_high_uv: float = 15.0
    p300_window_start_sample: int = 64  # 300 ms at 128 Hz with -200 ms offset
    p300_window_end_sample: int = 90    # 500 ms at 128 Hz with -200 ms offset
    p300_pz_index: int = 5              # Pz channel index
    p300_min_peak_uv: float = 0.5
    smoke_report_filename: str = "step1_smoke_test_report.txt"
    downstream_qc_filename: str = "step1_downstream_qc_flags.csv"
    accepted_raw_timepoints: tuple[int, ...] = DEFAULT_ACCEPTED_RAW_TIMEPOINTS
    run_smoke_tests: bool = True


@dataclass(frozen=True)
class SplitQcSummary:
    """QC summary for one split after downsampling."""

    baseline_std_uv: np.ndarray           # [n_channels]
    noisy_channel_indices: tuple[int, ...]
    noisy_channel_names: tuple[str, ...]


@dataclass(frozen=True)
class SessionQcSummary:
    """QC summary for one subject/session after preprocessing."""

    calibration: SplitQcSummary
    online: SplitQcSummary
    noisy_channel_indices: tuple[int, ...]  # union of both splits
    noisy_channel_names: tuple[str, ...]
    qc_status: str  # "clean", "channel_level_warning", "session_level_warning"


@dataclass(frozen=True)
class SmokeTestWarning:
    """Structured warning emitted by the post-run smoke tests."""

    file_path: Path
    warning_type: str
    channel_indices: tuple[int, ...]
    channel_names: tuple[str, ...]
    observed_values_uv: tuple[float, ...]
    lower_bound_uv: float
    upper_bound_uv: float
    message: str


@dataclass(frozen=True)
class SmokeValidationResult:
    """Validation result for one saved NPZ file."""

    file_path: Path
    passed: bool
    warnings: list[SmokeTestWarning] = field(default_factory=list)


@dataclass(frozen=True)
class DownstreamQcFlag:
    """One row in the downstream QC flags CSV (sub-step 1.9)."""

    subject: str
    session: str
    split: str
    flag_type: str    # "session_level_warning", "channel_level_warning", "manual_override"
    channel: str      # channel name or "ALL"
    details: str


def _format_repair_operations(repair: SessionMetadataRepair) -> str:
    """Format one repair as compact text for logs and reports."""
    operations = ", ".join(
        f"{op.index}:{op.old_value}->{op.new_value}" for op in repair.operations
    )
    return (
        f"{repair.subject_id}/{repair.session_id} {repair.split} {repair.field} "
        f"[{operations}]"
    )


# ---------------------------------------------------------------------------
# Pure processing functions (Phase 2)
# ---------------------------------------------------------------------------


def transpose_and_crop(epochs: np.ndarray, crop_samples: int) -> np.ndarray:
    """Transpose [C, T, N] → [N, C, T] and crop time axis (sub-step 1.2).

    Args:
        epochs: Raw array with shape [channels, timepoints, n_epochs].
        crop_samples: Number of time samples to keep from the start.

    Returns:
        Array with shape [n_epochs, channels, crop_samples].
    """
    # Move last axis (epochs) to front: [C, T, N] → [N, C, T]
    transposed = np.moveaxis(epochs, -1, 0)
    return transposed[:, :, :crop_samples]


def compute_block_and_run_ids(
    n_epochs: int, runs_per_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-epoch block_id and run_id from sequential ordering (sub-step 1.3).

    Args:
        n_epochs: Total number of epochs.
        runs_per_block: K (runs per block).

    Returns:
        (block_ids, run_ids) both as int16 arrays of length n_epochs.
    """
    epoch_indices = np.arange(n_epochs)
    epochs_per_block = EVENTS_PER_RUN * runs_per_block
    block_ids = (epoch_indices // epochs_per_block).astype(np.int16)
    run_ids = ((epoch_indices // EVENTS_PER_RUN) % runs_per_block).astype(np.int16)
    return block_ids, run_ids


def hard_clip(
    epochs: np.ndarray, clip_uv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply symmetric hard clipping (sub-step 1.4).

    Args:
        epochs: Array [n_epochs, channels, timepoints].
        clip_uv: Symmetric clipping threshold in µV.

    Returns:
        (clipped_epochs, counts_per_channel) where counts_per_channel has
        shape [channels] and reports the number of clipped samples per channel.
    """
    clipped = np.clip(epochs, -clip_uv, clip_uv)
    was_clipped = epochs != clipped
    # Sum over epochs and time axes → per-channel count
    counts_per_channel = was_clipped.sum(axis=(0, 2))
    return clipped, counts_per_channel


def baseline_correct(epochs: np.ndarray, baseline_samples: int) -> np.ndarray:
    """Subtract per-epoch, per-channel baseline mean (sub-step 1.5).

    Args:
        epochs: Array [n_epochs, channels, timepoints].
        baseline_samples: Number of samples in the pre-stimulus baseline window.

    Returns:
        Baseline-corrected array (same shape).
    """
    baseline_mean = epochs[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    return epochs - baseline_mean


def compute_winsor_thresholds(
    epochs: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel winsorizing thresholds (sub-step 1.5).

    Args:
        epochs: Array [n_epochs, channels, timepoints] — must be calibration only.
        low_percentile: Lower percentile (e.g. 1.0).
        high_percentile: Upper percentile (e.g. 99.0).

    Returns:
        (lower_thresholds, upper_thresholds) each with shape [channels].
    """
    n_channels = epochs.shape[1]
    lower = np.empty(n_channels, dtype=np.float64)
    upper = np.empty(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        ch_data = epochs[:, ch, :].ravel()
        lower[ch] = np.percentile(ch_data, low_percentile)
        upper[ch] = np.percentile(ch_data, high_percentile)
    return lower, upper


def apply_winsorizing(
    epochs: np.ndarray,
    lower_thresholds: np.ndarray,
    upper_thresholds: np.ndarray,
) -> np.ndarray:
    """Apply per-channel winsorizing (sub-step 1.6).

    Args:
        epochs: Array [n_epochs, channels, timepoints].
        lower_thresholds: Per-channel lower bounds [channels].
        upper_thresholds: Per-channel upper bounds [channels].

    Returns:
        Winsorized array (same shape).
    """
    result = epochs.copy()
    for ch in range(epochs.shape[1]):
        result[:, ch, :] = np.clip(
            result[:, ch, :], lower_thresholds[ch], upper_thresholds[ch],
        )
    return result


def downsample_epochs(epochs: np.ndarray, up: int, down: int) -> np.ndarray:
    """Downsample along the time axis using polyphase resampling (sub-step 1.7).

    Args:
        epochs: Array [n_epochs, channels, timepoints].
        up: Upsampling factor (64 for 250→128 Hz).
        down: Downsampling factor (125 for 250→128 Hz).

    Returns:
        Resampled array with new time-axis length.
    """
    # axis=2 is critical — default axis=0 would resample across epochs
    return resample_poly(epochs, up, down, axis=2)


def build_npz_payload(
    epochs: np.ndarray,
    labels: np.ndarray,
    block_ids: np.ndarray,
    run_ids: np.ndarray,
    flash_ids: np.ndarray,
    block_targets: np.ndarray,
    condition: np.ndarray,
    fs: int,
) -> dict[str, np.ndarray]:
    """Build the dict of arrays to save as NPZ (sub-step 1.7).

    Args:
        epochs: Preprocessed EEG [n_epochs, 8, 128], float32.
        labels: Target labels [n_epochs], int8.
        block_ids: [n_epochs], int16.
        run_ids: [n_epochs], int16.
        flash_ids: [n_epochs], int8.
        block_targets: [n_blocks], int8.
        condition: [n_epochs], int8.
        fs: Sampling rate (128).

    Returns:
        Dict ready for np.savez_compressed.
    """
    return {
        "epochs": epochs.astype(np.float32),
        "labels": labels.astype(np.int8),
        "block_ids": block_ids.astype(np.int16),
        "run_ids": run_ids.astype(np.int16),
        "flash_ids": flash_ids.astype(np.int8),
        "block_targets": block_targets.astype(np.int8),
        "condition": condition.astype(np.int8),
        "fs": np.int16(fs),
        "ch_names": CH_NAMES_ARRAY,
    }


# ---------------------------------------------------------------------------
# Validation functions (Phase 2 cont.)
# ---------------------------------------------------------------------------


def check_sequential_flash_structure(flash_ids: np.ndarray) -> bool:
    """Check that every consecutive 8-epoch group is a permutation of {1..8}.

    Args:
        flash_ids: 1-D array of flash IDs.

    Returns:
        True if all groups pass.
    """
    n_groups = len(flash_ids) // EVENTS_PER_RUN
    for i in range(n_groups):
        group = flash_ids[i * EVENTS_PER_RUN : (i + 1) * EVENTS_PER_RUN]
        if set(group.tolist()) != set(range(1, 9)):
            return False
    return True


def find_nonsequential_flash_groups(flash_ids: np.ndarray) -> list[int]:
    """Return indices of 8-epoch groups that are not permutations of {1..8}.

    Args:
        flash_ids: 1-D array of flash IDs.

    Returns:
        List of group indices that fail the permutation check.
    """
    bad: list[int] = []
    n_groups = len(flash_ids) // EVENTS_PER_RUN
    for i in range(n_groups):
        group = flash_ids[i * EVENTS_PER_RUN : (i + 1) * EVENTS_PER_RUN]
        if set(group.tolist()) != set(range(1, 9)):
            bad.append(i)
    return bad


def check_block_sizes(
    block_ids: np.ndarray,
    runs_per_block: int,
    expected_n_blocks: int | None = None,
) -> list[tuple[int, int, int]]:
    """Verify each block contains the expected number of epochs.

    Args:
        block_ids: Per-epoch block IDs.
        runs_per_block: Expected K.
        expected_n_blocks: Optional expected number of unique blocks.

    Returns:
        List of (block_id, actual_count, expected_count) for failing blocks.
    """
    expected = EVENTS_PER_RUN * runs_per_block
    failures: list[tuple[int, int, int]] = []
    observed_blocks = sorted(set(int(x) for x in block_ids.tolist()))
    blocks_to_check = observed_blocks
    if expected_n_blocks is not None:
        expected_blocks = list(range(expected_n_blocks))
        extra_blocks = [bid for bid in observed_blocks if bid not in expected_blocks]
        blocks_to_check = expected_blocks + extra_blocks
    for bid in blocks_to_check:
        count = int((block_ids == bid).sum())
        if count != expected:
            failures.append((bid, count, expected))
    return failures


def check_target_flash_match(
    flash_ids: np.ndarray,
    labels: np.ndarray,
    block_ids: np.ndarray,
    run_ids: np.ndarray,
    block_targets: np.ndarray,
    runs_per_block: int,
) -> list[str]:
    """Verify target epochs match block_targets (sub-step 1.3-23/24).

    For each run within each block, exactly one epoch should have label=1,
    and its flash_id should equal block_targets[block_id].

    Args:
        flash_ids: Per-epoch flash IDs [n_epochs].
        labels: Per-epoch target labels [n_epochs], 0/1.
        block_ids: Per-epoch block IDs [n_epochs].
        run_ids: Per-epoch run IDs [n_epochs].
        block_targets: Per-block target flash ID [n_blocks].
        runs_per_block: K.

    Returns:
        List of mismatch messages (empty if all pass).
    """
    mismatches: list[str] = []
    unique_blocks = sorted(set(block_ids.tolist()))

    for bid in unique_blocks:
        block_mask = block_ids == bid
        expected_target = int(block_targets[bid])

        for rid in range(runs_per_block):
            run_mask = block_mask & (run_ids == rid)
            run_labels = labels[run_mask]
            run_flash = flash_ids[run_mask]

            target_count = int(run_labels.sum())
            if target_count != 1:
                mismatches.append(
                    f"block={bid} run={rid}: expected 1 target, got {target_count}"
                )
                continue

            target_flash = int(run_flash[run_labels == 1][0])
            if target_flash != expected_target:
                mismatches.append(
                    f"block={bid} run={rid}: target flash={target_flash}, "
                    f"expected={expected_target}"
                )
    return mismatches


def validate_split_metadata(
    name: str,
    labels: np.ndarray,
    flash_ids: np.ndarray,
    block_ids: np.ndarray,
    run_ids: np.ndarray,
    block_targets: np.ndarray,
    runs_per_block: int,
    expected_targets: int,
    expected_nontargets: int,
) -> dict[str, object]:
    """Run all ordering and structural checks on one split (sub-step 1.3).

    Args:
        name: "calibration" or "online".
        labels: Target labels [n_epochs], 0/1.
        flash_ids: Flash IDs [n_epochs], 1-8.
        block_ids: Block IDs [n_epochs].
        run_ids: Run IDs [n_epochs].
        block_targets: Block-level target labels.
        runs_per_block: K.
        expected_targets: Expected number of target epochs.
        expected_nontargets: Expected number of non-target epochs.

    Returns:
        Dict with check results: flash_ok, block_size_ok, target_flash_ok,
        target_count, nontarget_count, and any failure details.
    """
    result: dict[str, object] = {}

    # Target/non-target counts (1.3-13 through 1.3-15)
    target_count = int(labels.sum())
    nontarget_count = int(len(labels) - target_count)
    result["target_count"] = target_count
    result["nontarget_count"] = nontarget_count
    result["target_count_ok"] = target_count == expected_targets
    result["nontarget_count_ok"] = nontarget_count == expected_nontargets
    result["count_ok"] = bool(result["target_count_ok"] and result["nontarget_count_ok"])
    if not result["count_ok"]:
        logger.error(
            (
                "%s: TARGET_COUNT_MISMATCH — got %d target / %d non-target, "
                "expected %d / %d"
            ),
            name,
            target_count,
            nontarget_count,
            expected_targets,
            expected_nontargets,
        )

    # Flash-ID permutation check (1.3-21)
    result["flash_ok"] = check_sequential_flash_structure(flash_ids)
    if not result["flash_ok"]:
        bad_groups = find_nonsequential_flash_groups(flash_ids)
        result["flash_bad_groups"] = bad_groups
        logger.error("%s: EPOCH_ORDER_FAIL — bad flash groups: %s", name, bad_groups)

    # Block-size check (1.3-22)
    block_failures = check_block_sizes(
        block_ids,
        runs_per_block,
        expected_n_blocks=len(block_targets),
    )
    result["block_size_ok"] = len(block_failures) == 0
    if block_failures:
        result["block_size_failures"] = block_failures
        logger.warning("%s: BLOCK_SIZE_FAIL — %s", name, block_failures)

    # Target-flash match check (1.3-23/24)
    mismatches = check_target_flash_match(
        flash_ids, labels, block_ids, run_ids, block_targets, runs_per_block,
    )
    result["target_flash_ok"] = len(mismatches) == 0
    if mismatches:
        result["target_flash_mismatches"] = mismatches
        logger.warning("%s: TARGET_FLASH_MISMATCH — %d issues", name, len(mismatches))

    return result


# ---------------------------------------------------------------------------
# QC functions
# ---------------------------------------------------------------------------


def compute_baseline_std_per_channel(
    epochs: np.ndarray, baseline_samples: int,
) -> np.ndarray:
    """Compute per-channel baseline standard deviation (sub-step 1.8-20).

    Args:
        epochs: Preprocessed array [n_epochs, channels, timepoints].
        baseline_samples: Number of baseline samples.

    Returns:
        Per-channel std array [channels].
    """
    baseline = epochs[:, :, :baseline_samples]
    # std across epochs and time for each channel
    return baseline.std(axis=(0, 2))


def find_noisy_channel_indices(
    baseline_std: np.ndarray,
    low_uv: float,
    high_uv: float,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Find channels with baseline std outside the QC band.

    Args:
        baseline_std: Per-channel baseline std [channels].
        low_uv: Lower bound of acceptable range.
        high_uv: Upper bound of acceptable range.

    Returns:
        (indices, names) of noisy channels.
    """
    noisy = []
    for i, std_val in enumerate(baseline_std):
        if std_val < low_uv or std_val > high_uv:
            noisy.append(i)
    indices = tuple(noisy)
    names = tuple(CHANNEL_NAMES[i] for i in indices)
    return indices, names


def summarize_split_qc(
    epochs: np.ndarray,
    config: PreprocessingConfig,
) -> SplitQcSummary:
    """Compute QC summary for one split.

    Args:
        epochs: Downsampled array [n_epochs, channels, timepoints].
        config: Preprocessing configuration.

    Returns:
        SplitQcSummary.
    """
    baseline_std = compute_baseline_std_per_channel(
        epochs, config.baseline_samples_downsampled,
    )
    indices, names = find_noisy_channel_indices(
        baseline_std, config.baseline_std_warn_low_uv, config.baseline_std_warn_high_uv,
    )
    return SplitQcSummary(
        baseline_std_uv=baseline_std,
        noisy_channel_indices=indices,
        noisy_channel_names=names,
    )


def summarize_session_qc(
    cal_qc: SplitQcSummary,
    online_qc: SplitQcSummary,
) -> SessionQcSummary:
    """Combine split QC summaries into a session-level QC status.

    Args:
        cal_qc: Calibration split QC.
        online_qc: Online split QC.

    Returns:
        SessionQcSummary with overall status.
    """
    all_noisy = set(cal_qc.noisy_channel_indices) | set(online_qc.noisy_channel_indices)
    noisy_indices = tuple(sorted(all_noisy))
    noisy_names = tuple(CHANNEL_NAMES[i] for i in noisy_indices)

    if len(all_noisy) >= 4:
        status = "session_level_warning"
    elif len(all_noisy) > 0:
        status = "channel_level_warning"
    else:
        status = "clean"

    return SessionQcSummary(
        calibration=cal_qc,
        online=online_qc,
        noisy_channel_indices=noisy_indices,
        noisy_channel_names=noisy_names,
        qc_status=status,
    )


# ---------------------------------------------------------------------------
# Structure log formatting
# ---------------------------------------------------------------------------


def format_structure_log(
    session: LoadedSession,
    config: PreprocessingConfig,
    cal_validation: dict[str, object],
    online_validation: dict[str, object],
    cal_clip_counts: np.ndarray,
    online_clip_counts: np.ndarray,
    winsor_lower: np.ndarray,
    winsor_upper: np.ndarray,
    session_qc: SessionQcSummary,
    cal_flash_ids: np.ndarray,
    online_flash_ids: np.ndarray,
    cal_block_ids: np.ndarray,
    online_block_ids: np.ndarray,
    cal_raw_min: float,
    cal_raw_max: float,
    online_raw_min: float,
    online_raw_max: float,
) -> str:
    """Format the per-session structure log (sub-step 1.3-10).

    Args:
        session: The loaded session.
        config: Preprocessing config.
        cal_validation: Calibration validation results.
        online_validation: Online validation results.
        cal_clip_counts: Hard-clip counts per channel for calibration.
        online_clip_counts: Hard-clip counts per channel for online.
        winsor_lower: Per-channel lower winsorizing thresholds.
        winsor_upper: Per-channel upper winsorizing thresholds.
        session_qc: Session QC summary.
        cal_flash_ids: Observed calibration flash IDs.
        online_flash_ids: Observed online flash IDs.
        cal_block_ids: Computed calibration block IDs.
        online_block_ids: Computed online block IDs.
        cal_raw_min: Min raw amplitude (calibration, before clipping).
        cal_raw_max: Max raw amplitude (calibration, before clipping).
        online_raw_min: Min raw amplitude (online, before clipping).
        online_raw_max: Max raw amplitude (online, before clipping).

    Returns:
        Multi-line string for the structure log file.
    """
    lines: list[str] = []
    lines.append(f"Structure Log: {session.subject_id}/{session.session_id}")
    lines.append("=" * 60)
    lines.append(f"Subject: {session.subject_id} ({session.sub_label})")
    lines.append(f"Session: {session.session_id} ({session.ses_label})")
    lines.append(f"K value: {session.k_value}")
    repairs = getattr(session, "repairs", ())
    if repairs:
        lines.append("Applied metadata repairs:")
        for repair in repairs:
            lines.append(f"  {_format_repair_operations(repair)}")
            lines.append(f"    Reason: {repair.reason}")
    lines.append("")

    # Shapes (1.3-11)
    lines.append("Shapes after transpose and crop:")
    lines.append(f"  Calibration: [1600, 8, {config.crop_samples}]")
    n_online = N_ONLINE_BLOCKS * EVENTS_PER_RUN * session.k_value
    lines.append(f"  Online: [{n_online}, 8, {config.crop_samples}]")
    lines.append("")

    # Target/non-target counts (1.3-13/14/15)
    lines.append("Target / non-target counts:")
    lines.append(
        f"  Calibration: {cal_validation['target_count']} target / "
        f"{cal_validation['nontarget_count']} non-target"
    )
    lines.append(
        f"  Online: {online_validation['target_count']} target / "
        f"{online_validation['nontarget_count']} non-target"
    )
    lines.append("")

    # Flash IDs — observed values, not hardcoded (1.3-16)
    cal_unique_flash = sorted(set(int(x) for x in np.unique(cal_flash_ids)))
    online_unique_flash = sorted(set(int(x) for x in np.unique(online_flash_ids)))
    lines.append(f"Unique flash IDs (calibration): {set(cal_unique_flash)}")
    lines.append(f"Unique flash IDs (online): {set(online_unique_flash)}")
    lines.append("")

    # Block IDs — observed unique values (1.3-17/18)
    cal_unique_blocks = sorted(set(int(x) for x in np.unique(cal_block_ids)))
    online_unique_blocks = sorted(set(int(x) for x in np.unique(online_block_ids)))
    lines.append(f"Unique block IDs (calibration): {cal_unique_blocks}")
    lines.append(f"Unique block IDs (online): {online_unique_blocks}")
    lines.append("")

    # Raw amplitude range (1.3-19)
    lines.append("Raw amplitude range (before hard clip):")
    lines.append(f"  Calibration: min={cal_raw_min:.4f}, max={cal_raw_max:.4f}")
    lines.append(f"  Online: min={online_raw_min:.4f}, max={online_raw_max:.4f}")
    lines.append("")

    # Ordering checks (1.3-20)
    lines.append("Ordering checks:")
    lines.append(
        f"  Target-count exactness (cal): {'PASS' if cal_validation['count_ok'] else 'FAIL'}"
    )
    lines.append(
        f"  Target-count exactness (online): {'PASS' if online_validation['count_ok'] else 'FAIL'}"
    )
    lines.append(f"  Flash-ID permutation (cal): {'PASS' if cal_validation['flash_ok'] else 'FAIL'}")
    lines.append(f"  Flash-ID permutation (online): {'PASS' if online_validation['flash_ok'] else 'FAIL'}")
    lines.append(f"  Block-size (cal): {'PASS' if cal_validation['block_size_ok'] else 'FAIL'}")
    lines.append(f"  Block-size (online): {'PASS' if online_validation['block_size_ok'] else 'FAIL'}")
    lines.append(f"  Target-flash match (cal): {'PASS' if cal_validation['target_flash_ok'] else 'FAIL'}")
    lines.append(f"  Target-flash match (online): {'PASS' if online_validation['target_flash_ok'] else 'FAIL'}")
    lines.append("")

    # Hard-clip counts (1.4-02/04)
    lines.append("Hard-clip counts per channel:")
    for i, ch in enumerate(CHANNEL_NAMES):
        lines.append(f"  {ch}: cal={int(cal_clip_counts[i])}, online={int(online_clip_counts[i])}")
    lines.append("")

    # Winsorizing thresholds (1.5-07)
    lines.append("Winsorizing thresholds (source: calibration_only):")
    for i, ch in enumerate(CHANNEL_NAMES):
        lines.append(f"  {ch}: lower={winsor_lower[i]:.4f}, upper={winsor_upper[i]:.4f}")
    lines.append("")

    # Baseline std (1.8-25)
    lines.append("Baseline std per channel (downsampled, µV):")
    lines.append("  Calibration:")
    for i, ch in enumerate(CHANNEL_NAMES):
        lines.append(f"    {ch}: {session_qc.calibration.baseline_std_uv[i]:.4f}")
    lines.append("  Online:")
    for i, ch in enumerate(CHANNEL_NAMES):
        lines.append(f"    {ch}: {session_qc.online.baseline_std_uv[i]:.4f}")
    lines.append("")

    # QC status
    lines.append(f"QC status: {session_qc.qc_status}")
    if session_qc.noisy_channel_names:
        lines.append(f"Noisy channels: {', '.join(session_qc.noisy_channel_names)}")

    return "\n".join(lines)


def format_aborted_structure_log(
    session: LoadedSession,
    config: PreprocessingConfig,
    cal_validation: dict[str, object],
    online_validation: dict[str, object],
    cal_flash_ids: np.ndarray,
    online_flash_ids: np.ndarray,
    cal_block_ids: np.ndarray,
    online_block_ids: np.ndarray,
    cal_raw_min: float,
    cal_raw_max: float,
    online_raw_min: float,
    online_raw_max: float,
    abort_reason: str,
) -> str:
    """Format a partial structure log for sessions aborted by EPOCH_ORDER_FAIL.

    Args:
        session: The loaded session.
        config: Preprocessing config.
        cal_validation: Calibration validation results.
        online_validation: Online validation results.
        cal_flash_ids: Observed calibration flash IDs.
        online_flash_ids: Observed online flash IDs.
        cal_block_ids: Computed calibration block IDs.
        online_block_ids: Computed online block IDs.
        cal_raw_min: Min raw amplitude (calibration).
        cal_raw_max: Max raw amplitude (calibration).
        online_raw_min: Min raw amplitude (online).
        online_raw_max: Max raw amplitude (online).
        abort_reason: Human-readable abort code.

    Returns:
        Multi-line string for the structure log file.
    """
    lines: list[str] = []
    lines.append(f"Structure Log: {session.subject_id}/{session.session_id}")
    lines.append("=" * 60)
    lines.append(f"STATUS: ABORTED — {abort_reason}")
    lines.append("")
    lines.append(f"Subject: {session.subject_id} ({session.sub_label})")
    lines.append(f"Session: {session.session_id} ({session.ses_label})")
    lines.append(f"K value: {session.k_value}")
    repairs = getattr(session, "repairs", ())
    if repairs:
        lines.append("Applied metadata repairs:")
        for repair in repairs:
            lines.append(f"  {_format_repair_operations(repair)}")
            lines.append(f"    Reason: {repair.reason}")
    lines.append("")

    # Shapes
    lines.append("Shapes after transpose and crop:")
    lines.append(f"  Calibration: [1600, 8, {config.crop_samples}]")
    n_online = N_ONLINE_BLOCKS * EVENTS_PER_RUN * session.k_value
    lines.append(f"  Online: [{n_online}, 8, {config.crop_samples}]")
    lines.append("")

    # Target/non-target counts
    lines.append("Target / non-target counts:")
    lines.append(
        f"  Calibration: {cal_validation['target_count']} target / "
        f"{cal_validation['nontarget_count']} non-target"
    )
    lines.append(
        f"  Online: {online_validation['target_count']} target / "
        f"{online_validation['nontarget_count']} non-target"
    )
    lines.append("")

    # Observed flash IDs
    cal_unique_flash = sorted(set(int(x) for x in np.unique(cal_flash_ids)))
    online_unique_flash = sorted(set(int(x) for x in np.unique(online_flash_ids)))
    lines.append(f"Unique flash IDs (calibration): {set(cal_unique_flash)}")
    lines.append(f"Unique flash IDs (online): {set(online_unique_flash)}")
    lines.append("")

    # Block IDs
    cal_unique_blocks = sorted(set(int(x) for x in np.unique(cal_block_ids)))
    online_unique_blocks = sorted(set(int(x) for x in np.unique(online_block_ids)))
    lines.append(f"Unique block IDs (calibration): {cal_unique_blocks}")
    lines.append(f"Unique block IDs (online): {online_unique_blocks}")
    lines.append("")

    # Raw amplitude range
    lines.append("Raw amplitude range (before hard clip):")
    lines.append(f"  Calibration: min={cal_raw_min:.4f}, max={cal_raw_max:.4f}")
    lines.append(f"  Online: min={online_raw_min:.4f}, max={online_raw_max:.4f}")
    lines.append("")

    # Ordering checks
    lines.append("Ordering checks:")
    lines.append(
        f"  Target-count exactness (cal): {'PASS' if cal_validation['count_ok'] else 'FAIL'}"
    )
    lines.append(
        f"  Target-count exactness (online): {'PASS' if online_validation['count_ok'] else 'FAIL'}"
    )
    lines.append(f"  Flash-ID permutation (cal): {'PASS' if cal_validation['flash_ok'] else 'FAIL'}")
    lines.append(f"  Flash-ID permutation (online): {'PASS' if online_validation['flash_ok'] else 'FAIL'}")
    lines.append(f"  Block-size (cal): {'PASS' if cal_validation['block_size_ok'] else 'FAIL'}")
    lines.append(f"  Block-size (online): {'PASS' if online_validation['block_size_ok'] else 'FAIL'}")
    lines.append(f"  Target-flash match (cal): {'PASS' if cal_validation['target_flash_ok'] else 'FAIL'}")
    lines.append(f"  Target-flash match (online): {'PASS' if online_validation['target_flash_ok'] else 'FAIL'}")
    lines.append("")

    # Note: no hard-clip, winsorizing, baseline, or QC sections since processing was aborted
    lines.append("NOTE: Processing aborted before hard-clip/baseline/winsorizing/downsample stages.")
    lines.append("No NPZ files were generated for this session.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# process_session — the corrected per-session pipeline
# ---------------------------------------------------------------------------


@dataclass
class ProcessSessionResult:
    """Result of processing one session, including validation details."""

    qc: SessionQcSummary
    cal_validation: dict[str, object]
    online_validation: dict[str, object]
    aborted: bool = False


def process_session(
    session: LoadedSession,
    config: PreprocessingConfig,
) -> ProcessSessionResult:
    """Process one subject × session and save calibration + online NPZ files.

    Implements the corrected processing order:
    transpose → crop → hard-clip → baseline-correct →
    compute winsor thresholds (calibration only) → apply winsorizing →
    downsample → save.

    Args:
        session: Loaded session data.
        config: Preprocessing configuration.

    Returns:
        ProcessSessionResult with QC summary and validation dicts.
    """
    # Step 1.2 — Transpose and crop (SEPARATE)
    cal_epochs = transpose_and_crop(session.calibration.epochs, config.crop_samples)
    online_epochs = transpose_and_crop(session.online.epochs, config.crop_samples)

    # Capture raw amplitude stats BEFORE hard clipping (1.3-19)
    cal_raw_min = float(cal_epochs.min())
    cal_raw_max = float(cal_epochs.max())
    online_raw_min = float(online_epochs.min())
    online_raw_max = float(online_epochs.max())

    # Step 1.3 — Compute block/run IDs and condition
    cal_block_ids, cal_run_ids = compute_block_and_run_ids(
        cal_epochs.shape[0], CALIBRATION_RUNS_PER_BLOCK,
    )
    online_block_ids, online_run_ids = compute_block_and_run_ids(
        online_epochs.shape[0], session.k_value,
    )

    # Condition labels (1.3-08/09)
    cal_condition = np.where(cal_block_ids < 10, 0, 1).astype(np.int8)
    online_condition = np.ones(online_epochs.shape[0], dtype=np.int8)

    # Validate metadata (1.3-10 through 1.3-24)
    cal_validation = validate_split_metadata(
        name=f"{session.subject_id}/{session.session_id}/calibration",
        labels=session.calibration.labels,
        flash_ids=session.calibration.flash_ids,
        block_ids=cal_block_ids,
        run_ids=cal_run_ids,
        block_targets=session.calibration.block_targets,
        runs_per_block=CALIBRATION_RUNS_PER_BLOCK,
        expected_targets=200,
        expected_nontargets=1400,
    )
    online_validation = validate_split_metadata(
        name=f"{session.subject_id}/{session.session_id}/online",
        labels=session.online.labels,
        flash_ids=session.online.flash_ids,
        block_ids=online_block_ids,
        run_ids=online_run_ids,
        block_targets=session.online.block_targets,
        runs_per_block=session.k_value,
        expected_targets=50 * session.k_value,
        expected_nontargets=350 * session.k_value,
    )

    if not cal_validation["count_ok"] or not online_validation["count_ok"]:
        log_text = format_aborted_structure_log(
            session=session,
            config=config,
            cal_validation=cal_validation,
            online_validation=online_validation,
            cal_flash_ids=session.calibration.flash_ids,
            online_flash_ids=session.online.flash_ids,
            cal_block_ids=cal_block_ids,
            online_block_ids=online_block_ids,
            cal_raw_min=cal_raw_min,
            cal_raw_max=cal_raw_max,
            online_raw_min=online_raw_min,
            online_raw_max=online_raw_max,
            abort_reason="TARGET_COUNT_MISMATCH",
        )
        log_path = config.processed_root / f"{session.sub_label}_{session.ses_label}_structure_log.txt"
        log_path.write_text(log_text, encoding="utf-8")
        raise ValueError(
            (
                f"TARGET_COUNT_MISMATCH: {session.subject_id}/{session.session_id} "
                f"cal={cal_validation['target_count']}/{cal_validation['nontarget_count']} "
                f"online={online_validation['target_count']}/{online_validation['nontarget_count']}"
            )
        )

    # If flash ordering fails, write a partial structure log and abort
    if not cal_validation["flash_ok"] or not online_validation["flash_ok"]:
        log_text = format_aborted_structure_log(
            session=session,
            config=config,
            cal_validation=cal_validation,
            online_validation=online_validation,
            cal_flash_ids=session.calibration.flash_ids,
            online_flash_ids=session.online.flash_ids,
            cal_block_ids=cal_block_ids,
            online_block_ids=online_block_ids,
            cal_raw_min=cal_raw_min,
            cal_raw_max=cal_raw_max,
            online_raw_min=online_raw_min,
            online_raw_max=online_raw_max,
            abort_reason="EPOCH_ORDER_FAIL",
        )
        log_path = config.processed_root / f"{session.sub_label}_{session.ses_label}_structure_log.txt"
        log_path.write_text(log_text, encoding="utf-8")
        logger.error(
            "EPOCH_ORDER_FAIL: %s/%s — structure log written, session aborted",
            session.subject_id, session.session_id,
        )
        return ProcessSessionResult(
            qc=SessionQcSummary(
                calibration=SplitQcSummary(
                    baseline_std_uv=np.zeros(N_CHANNELS),
                    noisy_channel_indices=(), noisy_channel_names=(),
                ),
                online=SplitQcSummary(
                    baseline_std_uv=np.zeros(N_CHANNELS),
                    noisy_channel_indices=(), noisy_channel_names=(),
                ),
                noisy_channel_indices=(), noisy_channel_names=(),
                qc_status="EPOCH_ORDER_FAIL",
            ),
            cal_validation=cal_validation,
            online_validation=online_validation,
            aborted=True,
        )

    # Step 1.4 — Hard clip (SEPARATE, before baseline correction)
    cal_epochs, cal_clip_counts = hard_clip(cal_epochs, config.hard_clip_uv)
    online_epochs, online_clip_counts = hard_clip(online_epochs, config.hard_clip_uv)

    # Step 1.5 — Baseline correct (SEPARATE, before winsorizing)
    cal_epochs = baseline_correct(cal_epochs, config.baseline_samples_raw)
    online_epochs = baseline_correct(online_epochs, config.baseline_samples_raw)

    # Step 1.5 cont. — Compute winsorizing thresholds (CALIBRATION ONLY)
    winsor_lower, winsor_upper = compute_winsor_thresholds(
        cal_epochs, config.winsor_low_percentile, config.winsor_high_percentile,
    )

    # Step 1.6 — Apply winsorizing (same thresholds to both)
    cal_epochs = apply_winsorizing(cal_epochs, winsor_lower, winsor_upper)
    online_epochs = apply_winsorizing(online_epochs, winsor_lower, winsor_upper)

    # Step 1.7 — Downsample (SEPARATE)
    cal_epochs = downsample_epochs(cal_epochs, config.downsample_up, config.downsample_down)
    online_epochs = downsample_epochs(online_epochs, config.downsample_up, config.downsample_down)

    # QC summary (on downsampled data)
    cal_qc = summarize_split_qc(cal_epochs, config)
    online_qc = summarize_split_qc(online_epochs, config)
    session_qc = summarize_session_qc(cal_qc, online_qc)

    # Build NPZ payloads
    cal_payload = build_npz_payload(
        epochs=cal_epochs,
        labels=session.calibration.labels,
        block_ids=cal_block_ids,
        run_ids=cal_run_ids,
        flash_ids=session.calibration.flash_ids,
        block_targets=session.calibration.block_targets,
        condition=cal_condition,
        fs=config.output_fs,
    )
    online_payload = build_npz_payload(
        epochs=online_epochs,
        labels=session.online.labels,
        block_ids=online_block_ids,
        run_ids=online_run_ids,
        flash_ids=session.online.flash_ids,
        block_targets=session.online.block_targets,
        condition=online_condition,
        fs=config.output_fs,
    )

    # Save NPZ files
    cal_path = config.processed_root / f"{session.sub_label}_{session.ses_label}_split-calibration.npz"
    online_path = config.processed_root / f"{session.sub_label}_{session.ses_label}_split-online.npz"
    np.savez_compressed(str(cal_path), **cal_payload)
    np.savez_compressed(str(online_path), **online_payload)
    logger.info("Saved: %s, %s", cal_path.name, online_path.name)

    # Write structure log (Fix 4: pass observed values, not hardcoded)
    log_text = format_structure_log(
        session=session,
        config=config,
        cal_validation=cal_validation,
        online_validation=online_validation,
        cal_clip_counts=cal_clip_counts,
        online_clip_counts=online_clip_counts,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        session_qc=session_qc,
        cal_flash_ids=session.calibration.flash_ids,
        online_flash_ids=session.online.flash_ids,
        cal_block_ids=cal_block_ids,
        online_block_ids=online_block_ids,
        cal_raw_min=cal_raw_min,
        cal_raw_max=cal_raw_max,
        online_raw_min=online_raw_min,
        online_raw_max=online_raw_max,
    )
    log_path = config.processed_root / f"{session.sub_label}_{session.ses_label}_structure_log.txt"
    log_path.write_text(log_text, encoding="utf-8")

    return ProcessSessionResult(
        qc=session_qc,
        cal_validation=cal_validation,
        online_validation=online_validation,
    )


# ---------------------------------------------------------------------------
# Smoke tests (sub-step 1.8)
# ---------------------------------------------------------------------------


def run_smoke_tests(
    config: PreprocessingConfig,
    inventory: DatasetInventory,
) -> list[SmokeTestWarning]:
    """Run post-processing smoke tests on all 210 NPZ files (sub-step 1.8).

    Args:
        config: Preprocessing configuration.
        inventory: Dataset inventory (for K values).

    Returns:
        List of all warnings emitted.
    """
    all_warnings: list[SmokeTestWarning] = []
    report_lines: list[str] = []
    report_lines.append("Step 1 Smoke Test Report")
    report_lines.append("=" * 60)

    files_checked = 0
    files_with_warnings = 0

    for subj_id, sess_id in expected_session_ids():
        entry = inventory.entries.get((subj_id, sess_id))
        if entry is None:
            continue

        subj_num = int(subj_id.replace("SBJ", ""))
        sess_num = int(sess_id.replace("S", ""))
        sub_label = f"sub-{subj_num:02d}"
        ses_label = f"ses-{sess_num:02d}"

        for split, expected_epochs, expected_bt_len, k_val in [
            ("calibration", N_CALIBRATION_EPOCHS, N_CALIBRATION_BLOCKS, CALIBRATION_RUNS_PER_BLOCK),
            ("online", N_ONLINE_BLOCKS * EVENTS_PER_RUN * entry.k_value, N_ONLINE_BLOCKS, entry.k_value),
        ]:
            file_path = config.processed_root / f"{sub_label}_{ses_label}_split-{split}.npz"
            if not file_path.is_file():
                report_lines.append(f"SKIPPED (not generated): {file_path.name}")
                continue
            result = _validate_saved_file(
                file_path=file_path,
                expected_epochs=expected_epochs,
                expected_bt_len=expected_bt_len,
                k_value=k_val,
                split=split,
                config=config,
            )
            files_checked += 1
            if result.warnings:
                files_with_warnings += 1
                all_warnings.extend(result.warnings)
                for w in result.warnings:
                    report_lines.append(f"WARNING: {w.message}")

    report_lines.append("")
    report_lines.append(f"Files checked: {files_checked}")
    report_lines.append(f"Files with warnings: {files_with_warnings}")
    report_lines.append(f"Total warnings: {len(all_warnings)}")

    # P300 peak check on calibration files (1.8-27)
    p300_results = _check_p300_peaks(config, inventory)
    report_lines.append("")
    report_lines.append("P300 peak check (calibration, Pz, 300-500 ms):")
    for subj_sess, peak_val in sorted(p300_results.items()):
        status = "PASS" if peak_val > config.p300_min_peak_uv else "WARN"
        report_lines.append(f"  {subj_sess}: peak={peak_val:.3f} µV [{status}]")

    report_path = config.processed_root / config.smoke_report_filename
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Smoke test report written to %s", report_path)

    return all_warnings


def _validate_saved_file(
    file_path: Path,
    expected_epochs: int,
    expected_bt_len: int,
    k_value: int,
    split: str,
    config: PreprocessingConfig,
) -> SmokeValidationResult:
    """Validate one saved NPZ file (sub-step 1.8 checks).

    Args:
        file_path: Path to the NPZ file.
        expected_epochs: Expected number of epochs.
        expected_bt_len: Expected length of block_targets.
        k_value: Runs per block for this session.
        split: "calibration" or "online".
        config: Preprocessing config.

    Returns:
        SmokeValidationResult.
    """
    warnings: list[SmokeTestWarning] = []

    # 1.8-01/02: Load without pickle
    try:
        data = np.load(str(file_path), allow_pickle=False)
    except Exception as exc:
        return SmokeValidationResult(
            file_path=file_path,
            passed=False,
            warnings=[SmokeTestWarning(
                file_path=file_path,
                warning_type="load_failed",
                channel_indices=(),
                channel_names=(),
                observed_values_uv=(),
                lower_bound_uv=0.0,
                upper_bound_uv=0.0,
                message=f"Failed to load {file_path.name}: {exc}",
            )],
        )

    # 1.8-03: epochs shape and dtype
    epochs = data["epochs"]
    if epochs.shape != (expected_epochs, N_CHANNELS, config.output_fs):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="shape_mismatch",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: epochs shape {epochs.shape} != expected ({expected_epochs}, {N_CHANNELS}, {config.output_fs})",
        ))
    if epochs.dtype != np.float32:
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="dtype_mismatch",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: epochs dtype {epochs.dtype} != float32",
        ))

    # 1.8-04/05: label dtypes
    for arr_name, expected_dtype in [
        ("labels", np.int8), ("condition", np.int8),
        ("flash_ids", np.int8), ("block_targets", np.int8),
        ("block_ids", np.int16), ("run_ids", np.int16),
    ]:
        if data[arr_name].dtype != expected_dtype:
            warnings.append(SmokeTestWarning(
                file_path=file_path,
                warning_type="dtype_mismatch",
                channel_indices=(), channel_names=(),
                observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
                message=f"{file_path.name}: {arr_name} dtype {data[arr_name].dtype} != {expected_dtype}",
            ))

    expected_epoch_shape = (expected_epochs,)
    for arr_name in ["labels", "condition", "flash_ids", "block_ids", "run_ids"]:
        if data[arr_name].shape != expected_epoch_shape:
            warnings.append(SmokeTestWarning(
                file_path=file_path,
                warning_type="metadata_shape",
                channel_indices=(), channel_names=(),
                observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
                message=(
                    f"{file_path.name}: {arr_name} shape {data[arr_name].shape} "
                    f"!= {expected_epoch_shape}"
                ),
            ))

    # 1.8-06: fs
    fs_val = data["fs"]
    if int(fs_val) != config.output_fs:
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="fs_mismatch",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: fs={int(fs_val)} != {config.output_fs}",
        ))

    # 1.8-07: ch_names shape/dtype and exact content
    ch = data["ch_names"]
    if ch.dtype.kind != "U" or ch.shape != (N_CHANNELS,):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="ch_names_mismatch",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: ch_names dtype={ch.dtype}, shape={ch.shape}",
        ))
    elif not np.array_equal(ch, CH_NAMES_ARRAY):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="ch_names_content",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: ch_names content {list(ch)} != {list(CH_NAMES_ARRAY)}",
        ))

    # 1.8-08: labels values
    labels = data["labels"]
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="label_values",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: labels contain values other than 0/1",
        ))

    # 1.8-12: Target ratio 1:7
    n_targets = int(labels.sum())
    n_nontargets = int(len(labels) - n_targets)
    if n_targets > 0 and n_nontargets != 7 * n_targets:
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="ratio_mismatch",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: ratio {n_targets}:{n_nontargets} != 1:7",
        ))

    # 1.8-10/11: condition values
    cond = data["condition"]
    if split == "calibration":
        if not set(np.unique(cond).tolist()).issubset({0, 1}):
            warnings.append(SmokeTestWarning(
                file_path=file_path,
                warning_type="condition_values",
                channel_indices=(), channel_names=(),
                observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
                message=f"{file_path.name}: calibration condition contains values other than 0/1",
            ))
    elif split == "online":
        if not np.all(cond == 1):
            warnings.append(SmokeTestWarning(
                file_path=file_path,
                warning_type="condition_values",
                channel_indices=(), channel_names=(),
                observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
                message=f"{file_path.name}: online condition contains values other than 1",
            ))

    # 1.8-13/14: epoch count and block_targets shape
    bt = data["block_targets"]
    if bt.shape != (expected_bt_len,):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="block_targets_shape",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: block_targets shape {bt.shape} != ({expected_bt_len},)",
        ))

    # 1.8-09: flash_ids and block_targets value range {1..8}
    flash_ids = data["flash_ids"]
    flash_unique = set(np.unique(flash_ids).tolist())
    valid_flash_range = set(range(1, 9))
    if not flash_unique.issubset(valid_flash_range):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="flash_ids_range",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: flash_ids values {sorted(flash_unique)} not within {{1..8}}",
        ))
    bt_unique = set(np.unique(bt).tolist())
    if not bt_unique.issubset(valid_flash_range):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="block_targets_range",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: block_targets values {sorted(bt_unique)} not within {{1..8}}",
        ))

    # 1.8-15: online epoch count sanity band [1200, 4000]
    if split == "online":
        n_epochs = epochs.shape[0]
        if n_epochs < 1200 or n_epochs > 4000:
            warnings.append(SmokeTestWarning(
                file_path=file_path,
                warning_type="epoch_count_sanity",
                channel_indices=(), channel_names=(),
                observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
                message=f"{file_path.name}: online epoch count {n_epochs} outside sanity band [1200, 4000]",
            ))

    # 1.8-16/18: contiguous block_ids
    block_ids = data["block_ids"]
    if split == "calibration":
        expected_blocks = list(range(N_CALIBRATION_BLOCKS))
    else:
        expected_blocks = list(range(N_ONLINE_BLOCKS))
    observed_blocks = sorted(set(int(x) for x in np.unique(block_ids)))
    if observed_blocks != expected_blocks:
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="block_ids_not_contiguous",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: block_ids {observed_blocks} != expected {expected_blocks[:3]}...{expected_blocks[-1]}",
        ))

    # 1.8-17/19: per-block run_ids (calibration: 0..9, online: 0..K-1)
    run_ids = data["run_ids"]
    expected_max_run = (CALIBRATION_RUNS_PER_BLOCK if split == "calibration" else k_value) - 1
    if block_ids.shape == expected_epoch_shape and run_ids.shape == expected_epoch_shape:
        for blk in observed_blocks:
            blk_mask = block_ids == blk
            blk_runs = sorted(set(int(x) for x in np.unique(run_ids[blk_mask])))
            expected_runs = list(range(expected_max_run + 1))
            if blk_runs != expected_runs:
                warnings.append(SmokeTestWarning(
                    file_path=file_path,
                    warning_type="run_ids_per_block",
                    channel_indices=(), channel_names=(),
                    observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
                    message=f"{file_path.name}: block {blk} run_ids {blk_runs} != expected {expected_runs}",
                ))
                break  # One warning is enough — don't flood

    # 1.8-29: NaN/Inf check
    if np.any(~np.isfinite(epochs)):
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="nan_inf",
            channel_indices=(), channel_names=(),
            observed_values_uv=(), lower_bound_uv=0.0, upper_bound_uv=0.0,
            message=f"{file_path.name}: contains NaN or Inf values",
        ))

    # 1.8-20 through 1.8-25: Baseline std QC
    baseline_std = compute_baseline_std_per_channel(
        epochs, config.baseline_samples_downsampled,
    )
    noisy_idx, noisy_names = find_noisy_channel_indices(
        baseline_std, config.baseline_std_warn_low_uv, config.baseline_std_warn_high_uv,
    )
    if noisy_idx:
        warnings.append(SmokeTestWarning(
            file_path=file_path,
            warning_type="baseline_std_out_of_range",
            channel_indices=noisy_idx,
            channel_names=noisy_names,
            observed_values_uv=tuple(float(baseline_std[i]) for i in noisy_idx),
            lower_bound_uv=config.baseline_std_warn_low_uv,
            upper_bound_uv=config.baseline_std_warn_high_uv,
            message=(
                f"{file_path.name}: baseline std out of range for channels "
                f"{', '.join(noisy_names)}: "
                f"{[f'{baseline_std[i]:.2f}' for i in noisy_idx]}"
            ),
        ))

    passed = len(warnings) == 0
    return SmokeValidationResult(file_path=file_path, passed=passed, warnings=warnings)


def _check_p300_peaks(
    config: PreprocessingConfig,
    inventory: DatasetInventory,
) -> dict[str, float]:
    """Check P300 peak on calibration target epochs at Pz (sub-step 1.8-27/28).

    Args:
        config: Preprocessing config.
        inventory: Dataset inventory.

    Returns:
        Dict mapping "SBJxx/Syy" to peak amplitude in µV.
    """
    results: dict[str, float] = {}
    pz = config.p300_pz_index
    start = config.p300_window_start_sample
    end = config.p300_window_end_sample

    for subj_id, sess_id in expected_session_ids():
        entry = inventory.entries.get((subj_id, sess_id))
        if entry is None:
            continue

        subj_num = int(subj_id.replace("SBJ", ""))
        sess_num = int(sess_id.replace("S", ""))
        sub_label = f"sub-{subj_num:02d}"
        ses_label = f"ses-{sess_num:02d}"

        cal_path = config.processed_root / f"{sub_label}_{ses_label}_split-calibration.npz"
        if not cal_path.is_file():
            continue

        data = np.load(str(cal_path), allow_pickle=False)
        epochs = data["epochs"]
        labels = data["labels"]

        # Pool target epochs, extract Pz, find max in P300 window
        target_mask = labels == 1
        if target_mask.sum() == 0:
            results[f"{subj_id}/{sess_id}"] = 0.0
            continue

        target_pz = epochs[target_mask, pz, :]
        mean_target_pz = target_pz.mean(axis=0)
        peak = float(mean_target_pz[start:end].max())
        results[f"{subj_id}/{sess_id}"] = peak

    return results


# ---------------------------------------------------------------------------
# Downstream QC flags (sub-step 1.9)
# ---------------------------------------------------------------------------


def build_downstream_qc_flags(
    warnings: list[SmokeTestWarning],
    validation_flags: dict[str, list[str]] | None = None,
    applied_repairs: list[SessionMetadataRepair] | None = None,
) -> list[DownstreamQcFlag]:
    """Build downstream QC flag rows from smoke-test warnings + manual overrides.

    Args:
        warnings: All smoke-test warnings.
        validation_flags: Optional dict mapping "SBJxx/Syy" to list of flag
            names (e.g. "BLOCK_SIZE_FAIL", "TARGET_FLASH_MISMATCH").
        applied_repairs: Metadata repairs applied during loading.

    Returns:
        List of DownstreamQcFlag rows.
    """
    flags: list[DownstreamQcFlag] = []

    # Manual overrides first (1.9-08)
    for mo in MANUAL_OVERRIDES:
        flags.append(DownstreamQcFlag(**mo))

    if applied_repairs:
        for repair in applied_repairs:
            flags.append(DownstreamQcFlag(
                subject=repair.subject_id,
                session=repair.session_id,
                split=repair.split,
                flag_type="manual_override",
                channel="ALL",
                details=(
                    f"Deterministic metadata repair applied to {repair.field}: "
                    f"{_format_repair_operations(repair)}"
                ),
            ))

    # Automated flags from baseline std warnings
    for w in warnings:
        if w.warning_type != "baseline_std_out_of_range":
            continue

        parts = _parse_npz_filename(w.file_path)
        if parts is None:
            continue
        subject, session, split = parts

        n_noisy = len(w.channel_indices)
        if n_noisy >= 4:
            flags.append(DownstreamQcFlag(
                subject=subject, session=session, split=split,
                flag_type="session_level_warning", channel="ALL",
                details=f"Baseline std outside [{w.lower_bound_uv}-{w.upper_bound_uv} µV] for {n_noisy} channels: {', '.join(w.channel_names)}",
            ))
        elif n_noisy > 0:
            for i, ch_idx in enumerate(w.channel_indices):
                flags.append(DownstreamQcFlag(
                    subject=subject, session=session, split=split,
                    flag_type="channel_level_warning",
                    channel=w.channel_names[i],
                    details=f"Baseline std {w.observed_values_uv[i]:.2f} µV outside [{w.lower_bound_uv}-{w.upper_bound_uv} µV]",
                ))

    # Validation flags (BLOCK_SIZE_FAIL, TARGET_FLASH_MISMATCH) — per-split rows (Fix 1)
    if validation_flags:
        for key, flag_names in validation_flags.items():
            parts = key.split("/")
            if len(parts) != 2:
                continue
            subject, session = parts
            for flag_name in flag_names:
                # Flag names are suffixed with _calibration or _online
                # e.g. "BLOCK_SIZE_FAIL_calibration", "TARGET_FLASH_MISMATCH_online"
                for split_name in ("calibration", "online"):
                    if flag_name.endswith(f"_{split_name}"):
                        base_flag = flag_name.rsplit(f"_{split_name}", 1)[0]
                        flags.append(DownstreamQcFlag(
                            subject=subject, session=session, split=split_name,
                            flag_type="session_level_warning", channel="ALL",
                            details=base_flag,
                        ))
                        break

    return flags


def format_downstream_qc_flags_csv(flags: list[DownstreamQcFlag]) -> str:
    """Format QC flags as CSV (sub-step 1.9-01/02).

    Args:
        flags: List of QC flag rows.

    Returns:
        CSV string with header.
    """
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["subject", "session", "split", "flag_type", "channel", "details"])
    for f in flags:
        writer.writerow([f.subject, f.session, f.split, f.flag_type, f.channel, f.details])
    return output.getvalue()


def format_step1_report(
    config: PreprocessingConfig,
    inventory: DatasetInventory,
    processed_sessions: list[str],
    aborted_sessions: list[str],
    warnings: list[SmokeTestWarning],
    flags: list[DownstreamQcFlag],
    applied_repairs: list[SessionMetadataRepair],
) -> str:
    """Format the end-of-run Step 1 implementation report."""
    expected_sessions = len(expected_session_ids())
    smoke_status = "enabled" if config.run_smoke_tests else "disabled"
    validation_flag_rows = sum(
        1 for flag in flags if flag.flag_type == "session_level_warning" and flag.details in {
            "BLOCK_SIZE_FAIL",
            "TARGET_FLASH_MISMATCH",
        }
    )

    lines: list[str] = []
    lines.append("Step 1 Report")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Inventory")
    lines.append(f"  Expected sessions: {expected_sessions}")
    lines.append(f"  Inventory entries: {len(inventory.entries)}")
    lines.append(f"  Blocking issues: {len(inventory.blocking_issues)}")
    lines.append(f"  Deviations: {len(inventory.deviations)}")
    lines.append("")
    lines.append("Execution")
    lines.append(f"  Processed sessions: {len(processed_sessions)}")
    lines.append(f"  Aborted sessions: {len(aborted_sessions)}")
    lines.append(f"  Saved NPZ files: {len(processed_sessions) * 2}")
    lines.append(f"  Structure logs written: {len(processed_sessions) + len(aborted_sessions)}")
    if aborted_sessions:
        lines.append(f"  Aborted session IDs: {', '.join(aborted_sessions)}")
    lines.append("")
    lines.append("Repairs")
    lines.append(f"  Applied metadata repairs: {len(applied_repairs)}")
    for repair in applied_repairs:
        lines.append(f"  {_format_repair_operations(repair)}")
        lines.append(f"    Reason: {repair.reason}")
    lines.append("")
    lines.append("QC Outputs")
    lines.append(f"  Smoke tests: {smoke_status}")
    lines.append(f"  Smoke-test warnings: {len(warnings)}")
    lines.append(f"  Downstream QC rows: {len(flags)}")
    lines.append(f"  Validation flag rows: {validation_flag_rows}")
    lines.append("")
    lines.append("Artifacts")
    lines.append(f"  Inventory: {config.processed_root / 'dataset_inventory.txt'}")
    lines.append(f"  Smoke report: {config.processed_root / config.smoke_report_filename}")
    lines.append(f"  Downstream QC CSV: {config.processed_root / config.downstream_qc_filename}")
    lines.append(f"  Step 1 report: {config.processed_root / 'step1_report.txt'}")
    return "\n".join(lines)


def _parse_npz_filename(path: Path) -> tuple[str, str, str] | None:
    """Parse subject, session, split from NPZ filename.

    Expected format: sub-XX_ses-YY_split-{calibration|online}.npz

    Returns:
        (subject, session, split) as ("SBJxx", "Syy", "calibration"/"online"),
        or None if parsing fails.
    """
    stem = path.stem  # e.g. "sub-01_ses-01_split-calibration"
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    try:
        sub_num = int(parts[0].replace("sub-", ""))
        ses_num = int(parts[1].replace("ses-", ""))
        split = parts[2].replace("split-", "")
        return f"SBJ{sub_num:02d}", f"S{ses_num:02d}", split
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_step1(config: PreprocessingConfig | None = None) -> None:
    """Run the full Step 1 pipeline and write all output files.

    Args:
        config: Optional preprocessing configuration. Uses defaults if None.
    """
    config = config or PreprocessingConfig()
    config.processed_root.mkdir(parents=True, exist_ok=True)

    # Sub-step 1.1a — Build and save inventory
    logger.info("Building dataset inventory...")
    inventory = build_dataset_inventory(
        raw_root=config.raw_root,
        accepted_raw_timepoints=config.accepted_raw_timepoints,
    )
    inv_path = config.processed_root / "dataset_inventory.txt"
    inv_path.write_text(format_dataset_inventory(inventory), encoding="utf-8")
    logger.info("Inventory saved to %s", inv_path)

    if not inventory.is_complete():
        raise ValueError(
            (
                "Dataset inventory is incomplete; refusing to start Step 1. "
                f"Sessions found={len(inventory.entries)}, "
                f"blocking issues={len(inventory.blocking_issues)}"
            )
        )

    # Collect validation flags across sessions
    validation_flags: dict[str, list[str]] = {}
    processed_sessions: list[str] = []
    aborted_sessions: list[str] = []
    applied_repairs: list[SessionMetadataRepair] = []

    # Sub-steps 1.1 through 1.7 — Process each session
    for subject_id, session_id in expected_session_ids():
        if (subject_id, session_id) not in inventory.entries:
            logger.warning("Skipping %s/%s: not in inventory", subject_id, session_id)
            continue

        logger.info("Processing %s/%s...", subject_id, session_id)
        session = load_session(
            inventory=inventory,
            subject_id=subject_id,
            session_id=session_id,
            accepted_raw_timepoints=config.accepted_raw_timepoints,
        )
        applied_repairs.extend(session.repairs)
        result = process_session(session=session, config=config)

        # Populate validation_flags from returned validation dicts (Fix 1)
        session_flags: list[str] = []
        for split_name, val_dict in [
            ("calibration", result.cal_validation),
            ("online", result.online_validation),
        ]:
            if not val_dict.get("block_size_ok", True):
                session_flags.append(f"BLOCK_SIZE_FAIL_{split_name}")
            if not val_dict.get("target_flash_ok", True):
                session_flags.append(f"TARGET_FLASH_MISMATCH_{split_name}")
        if session_flags:
            validation_flags[f"{subject_id}/{session_id}"] = session_flags

        if result.aborted:
            logger.error("Skipping %s/%s: EPOCH_ORDER_FAIL", subject_id, session_id)
            aborted_sessions.append(f"{subject_id}/{session_id}")
            continue
        processed_sessions.append(f"{subject_id}/{session_id}")

    # Sub-step 1.8 — Smoke tests
    all_warnings: list[SmokeTestWarning] = []
    if config.run_smoke_tests:
        logger.info("Running smoke tests...")
        all_warnings = run_smoke_tests(config=config, inventory=inventory)

    # Sub-step 1.9 — Downstream QC flags
    logger.info("Writing downstream QC flags...")
    flags = build_downstream_qc_flags(all_warnings, validation_flags, applied_repairs)
    qc_csv = format_downstream_qc_flags_csv(flags)
    qc_path = config.processed_root / config.downstream_qc_filename
    qc_path.write_text(qc_csv, encoding="utf-8")
    logger.info("QC flags saved to %s", qc_path)

    report_text = format_step1_report(
        config=config,
        inventory=inventory,
        processed_sessions=processed_sessions,
        aborted_sessions=aborted_sessions,
        warnings=all_warnings,
        flags=flags,
        applied_repairs=applied_repairs,
    )
    report_path = config.processed_root / "step1_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Step 1 report saved to %s", report_path)

    logger.info("Step 1 complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> PreprocessingConfig:
    """Parse command-line arguments into a PreprocessingConfig."""
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Data Loading and Preprocessing")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--no-smoke-tests", action="store_true")
    args = parser.parse_args()

    return PreprocessingConfig(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        run_smoke_tests=not args.no_smoke_tests,
    )


def main() -> None:
    """CLI entry point for Step 1."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = parse_args()
    run_step1(config)


if __name__ == "__main__":
    main()
