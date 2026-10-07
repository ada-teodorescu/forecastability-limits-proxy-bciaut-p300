"""Step 2 ERP exploration and visualization."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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
SPECIAL_POZ_SUBJECTS: tuple[str, ...] = ("SBJ03", "SBJ05")


@dataclass(frozen=True)
class PeakMetrics:
    """Peak amplitude/latency inside an inclusive sample window."""

    amplitude_uV: float
    latency_ms: float
    sample_index: int


@dataclass(frozen=True)
class SubjectP300Summary:
    """Per-subject pooled calibration ERP metrics at Pz."""

    subject_id: str
    target_waveform_pz: np.ndarray
    diff_wave_peak_amplitude_uV: float
    diff_wave_peak_latency_ms: float
    target_peak_amplitude_uV: float
    target_peak_latency_ms: float


@dataclass(frozen=True)
class RepresentativeSubject:
    """Deterministic representative subject selection."""

    subject_id: str
    category: str
    diff_wave_peak_amplitude_uV: float


@dataclass(frozen=True)
class Step2Summary:
    """High-level summary of the executed Step 2 run."""

    output_root: Path
    grand_average_target_peak: PeakMetrics
    grand_average_diff_peak: PeakMetrics
    diff_wave_amplitude_cv: float
    diff_wave_latency_cv: float
    target_group_latency_r: float
    target_group_latency_p: float
    diff_group_latency_r: float
    diff_group_latency_p: float
    snr_mean: float
    snr_median: float
    snr_min: float
    snr_max: float
    flagged_subjects: tuple[str, ...]
    representative_subjects: tuple[str, ...]


@dataclass(frozen=True)
class Step2Config:
    """Configuration for the Step 2 ERP exploration pipeline."""

    processed_root: Path = Path("data/processed")
    output_root: Path = Path("results/figures/step2")
    inventory_path: Path = Path("data/processed/dataset_inventory.txt")
    output_fs: int = 128
    epoch_samples: int = 128
    time_offset_ms: float = -200.0
    p300_window: tuple[int, int] = (64, 90)
    peak_search_window: tuple[int, int] = (58, 102)
    p200_window: tuple[int, int] = (45, 58)
    pz_channel_index: int = 5
    poz_channel_index: int = 7
    representative_target_ks: tuple[int, int] = (5, 6)


def processed_split_path(*, processed_root: Path, subject_id: str, session_id: str, split: str) -> Path:
    """Build the processed NPZ path for one subject/session/split."""

    return processed_root / f"{subject_id_to_sub_label(subject_id)}_{session_id_to_ses_label(session_id)}_split-{split}.npz"


def load_processed_npz(path: Path, *, split: str, config: Step2Config) -> dict[str, np.ndarray]:
    """Load and validate one Step 1 NPZ file."""

    if not path.is_file():
        raise FileNotFoundError(f"Processed NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as npz_file:
        payload = {name: np.asarray(npz_file[name]) for name in npz_file.files}

    missing_fields = [name for name in REQUIRED_NPZ_FIELDS if name not in payload]
    if missing_fields:
        raise ValueError(f"{path.name} missing required fields: {missing_fields}")

    epochs = np.asarray(payload["epochs"])
    if epochs.shape[1:] != (len(CHANNEL_NAMES), config.epoch_samples):
        raise ValueError(
            f"{path.name} has unexpected epochs shape {epochs.shape}; expected (*, {len(CHANNEL_NAMES)}, {config.epoch_samples})",
        )
    fs_value = int(np.asarray(payload["fs"]).item())
    if fs_value != config.output_fs:
        raise ValueError(f"{path.name} has fs={fs_value}, expected {config.output_fs}")

    ch_names = [str(item) for item in np.asarray(payload["ch_names"]).tolist()]
    if ch_names != list(CHANNEL_NAMES):
        raise ValueError(f"{path.name} has unexpected channel names {ch_names}")

    condition_values = set(int(item) for item in np.unique(payload["condition"]).tolist())
    if split == "online" and condition_values != {1}:
        raise ValueError(f"{path.name} has online condition values {condition_values}, expected {{1}}")
    if split == "calibration" and not condition_values.issubset({0, 1}):
        raise ValueError(f"{path.name} has calibration condition values {condition_values}, expected subset of {{0, 1}}")
    return payload


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


def build_time_axis_ms(*, n_samples: int, fs: int, time_offset_ms: float) -> np.ndarray:
    """Build the post-resampling epoch time axis in milliseconds."""

    return (np.arange(n_samples, dtype=np.float64) * 1000.0 / float(fs)) + float(time_offset_ms)


def compute_peak_metrics(
    *,
    waveform: np.ndarray,
    window: tuple[int, int],
    fs: int,
    time_offset_ms: float,
) -> PeakMetrics:
    """Compute the maximum amplitude and latency inside an inclusive sample window."""

    waveform = np.asarray(waveform, dtype=np.float64)
    start, end = window
    window_values = waveform[start : end + 1]
    if window_values.size == 0:
        raise ValueError(f"Peak window is empty: {window}")
    relative_index = int(np.argmax(window_values))
    sample_index = start + relative_index
    latency_ms = (sample_index * 1000.0 / float(fs)) + float(time_offset_ms)
    return PeakMetrics(
        amplitude_uV=float(window_values[relative_index]),
        latency_ms=float(latency_ms),
        sample_index=sample_index,
    )


def compute_window_mean_amplitude(waveform: np.ndarray, window: tuple[int, int]) -> float:
    """Compute the mean amplitude inside an inclusive sample window."""

    start, end = window
    return float(np.mean(np.asarray(waveform, dtype=np.float64)[start : end + 1]))


def compute_epoch_window_amplitudes(
    *,
    epochs: np.ndarray,
    channel_index: int,
    window: tuple[int, int],
) -> np.ndarray:
    """Compute per-epoch mean amplitudes over the configured inclusive window."""

    start, end = window
    return np.asarray(epochs[:, channel_index, start : end + 1], dtype=np.float64).mean(axis=1)


def compute_channel_peak_amplitudes(waveform: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Compute channel-wise maximum amplitudes inside the inclusive P300 window."""

    start, end = window
    return np.max(np.asarray(waveform, dtype=np.float64)[:, start : end + 1], axis=1)


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation with deterministic NaN handling for constant input."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def coefficient_of_variation(values: np.ndarray) -> float:
    """Compute coefficient of variation as std / mean."""

    values = np.asarray(values, dtype=np.float64)
    mean_value = float(np.mean(values))
    if math.isclose(mean_value, 0.0, abs_tol=0.0):
        raise ValueError("Coefficient of variation is undefined for mean 0")
    return float(np.std(values, ddof=0) / mean_value)


def divide_sum_by_count(summed_waveform: np.ndarray, count: int) -> np.ndarray:
    """Convert an accumulated waveform sum to a mean waveform."""

    if count <= 0:
        raise ValueError(f"Count must be positive, got {count}")
    return np.asarray(summed_waveform, dtype=np.float64) / float(count)


def mean_waveform(epochs_2d: np.ndarray, *, context: str) -> np.ndarray:
    """Compute a mean waveform from an [n_epochs, n_samples] array."""

    epochs_2d = np.asarray(epochs_2d, dtype=np.float64)
    if epochs_2d.ndim != 2 or epochs_2d.shape[0] == 0:
        raise ValueError(f"{context}: expected non-empty 2-D array, got shape {epochs_2d.shape}")
    return epochs_2d.mean(axis=0)


def build_class_balance_row(
    *,
    subject_id: str,
    session: str,
    split: str,
    n_target: int,
    n_nontarget: int,
) -> dict[str, str]:
    """Build one class-balance CSV row."""

    return {
        "subject_id": subject_id,
        "session": session,
        "split": split,
        "n_target": str(n_target),
        "n_nontarget": str(n_nontarget),
        "ratio": format_ratio(n_target=n_target, n_nontarget=n_nontarget),
    }


def format_ratio(*, n_target: int, n_nontarget: int) -> str:
    """Format the target-to-non-target ratio in lowest terms."""

    if n_target <= 0 or n_nontarget <= 0:
        raise ValueError(f"Counts must be positive, got target={n_target}, non-target={n_nontarget}")
    divisor = math.gcd(int(n_target), int(n_nontarget))
    return f"{n_target // divisor}:{n_nontarget // divisor}"


def write_csv_rows(
    *,
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, str]],
    comment_lines: Iterable[str] | None = None,
) -> None:
    """Write deterministic CSV rows with optional leading comment lines."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        for comment in comment_lines or ():
            handle.write(f"# {comment}\n")
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows while ignoring Step 2 comment lines."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        filtered_lines = [line for line in handle if not line.startswith("#")]
    if not filtered_lines:
        return []
    reader = csv.DictReader(filtered_lines)
    return list(reader)


def format_float(value: float) -> str:
    """Format a float for deterministic CSV output."""

    return f"{float(value):.6f}"


def format_float_or_nan(value: float) -> str:
    """Format a float, preserving NaN text for undefined correlations."""

    if math.isnan(value):
        return "NaN"
    return format_float(value)


def session_sort_key(session_id: str) -> int:
    """Convert Sxx to an integer for deterministic ordering."""

    return int(session_id[1:])


def subject_id_to_sub_label(subject_id: str) -> str:
    """Convert SBJxx to sub-xx."""

    return f"sub-{subject_id[-2:]}"


def session_id_to_ses_label(session_id: str) -> str:
    """Convert Sxx to ses-xx."""

    return f"ses-{session_id[-2:]}"


def process_calibration_payload(
    *,
    payload: dict[str, np.ndarray],
    subject_id: str,
    session_id: str,
    config: Step2Config,
    pooled_target_sum: np.ndarray,
    pooled_nontarget_sum: np.ndarray,
    pooled_target_count_ref: list[int],
    pooled_nontarget_count_ref: list[int],
    social_target_sum_pz: np.ndarray,
    nonsocial_target_sum_pz: np.ndarray,
    social_target_count_ref: list[int],
    nonsocial_target_count_ref: list[int],
    per_subject_target_sum_pz: dict[str, np.ndarray],
    per_subject_target_count_pz: dict[str, int],
    per_subject_nontarget_sum_pz: dict[str, np.ndarray],
    per_subject_nontarget_count_pz: dict[str, int],
    session_social_target_pz: dict[str, dict[str, np.ndarray]],
    session_social_nontarget_pz: dict[str, dict[str, np.ndarray]],
    calibration_poz_by_subject: dict[str, dict[str, np.ndarray]],
    class_balance_rows: list[dict[str, str]],
) -> None:
    """Accumulate one calibration payload into the Step 2 aggregates."""

    epochs = np.asarray(payload["epochs"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int8)
    conditions = np.asarray(payload["condition"], dtype=np.int8)

    target_mask = labels == 1
    nontarget_mask = labels == 0
    social_mask = conditions == 1
    nonsocial_mask = conditions == 0

    pooled_target_sum += epochs[target_mask].sum(axis=0)
    pooled_nontarget_sum += epochs[nontarget_mask].sum(axis=0)
    pooled_target_count_ref[0] += int(target_mask.sum())
    pooled_nontarget_count_ref[0] += int(nontarget_mask.sum())

    social_target_mask = target_mask & social_mask
    nonsocial_target_mask = target_mask & nonsocial_mask
    social_nontarget_mask = nontarget_mask & social_mask

    social_target_sum_pz += epochs[social_target_mask, config.pz_channel_index, :].sum(axis=0)
    nonsocial_target_sum_pz += epochs[nonsocial_target_mask, config.pz_channel_index, :].sum(axis=0)
    social_target_count_ref[0] += int(social_target_mask.sum())
    nonsocial_target_count_ref[0] += int(nonsocial_target_mask.sum())

    per_subject_target_sum_pz[subject_id] += epochs[target_mask, config.pz_channel_index, :].sum(axis=0)
    per_subject_target_count_pz[subject_id] += int(target_mask.sum())
    per_subject_nontarget_sum_pz[subject_id] += epochs[nontarget_mask, config.pz_channel_index, :].sum(axis=0)
    per_subject_nontarget_count_pz[subject_id] += int(nontarget_mask.sum())

    session_social_target_pz[subject_id][session_id] = mean_waveform(
        epochs[social_target_mask, config.pz_channel_index, :],
        context=f"{subject_id}/{session_id} social target Pz",
    )
    session_social_nontarget_pz[subject_id][session_id] = mean_waveform(
        epochs[social_nontarget_mask, config.pz_channel_index, :],
        context=f"{subject_id}/{session_id} social non-target Pz",
    )

    if subject_id in SPECIAL_POZ_SUBJECTS:
        calibration_poz_by_subject[subject_id][session_id] = mean_waveform(
            epochs[target_mask, config.poz_channel_index, :],
            context=f"{subject_id}/{session_id} calibration target POz",
        )

    class_balance_rows.append(
        build_class_balance_row(
            subject_id=subject_id,
            session=session_id,
            split="calibration",
            n_target=int(target_mask.sum()),
            n_nontarget=int(nontarget_mask.sum()),
        )
    )


def process_online_payload(
    *,
    payload: dict[str, np.ndarray],
    subject_id: str,
    session_id: str,
    config: Step2Config,
    pooled_online_target_sum_pz: np.ndarray,
    online_poz_by_subject: dict[str, dict[str, np.ndarray]],
    class_balance_rows: list[dict[str, str]],
) -> int:
    """Accumulate one online payload into the Step 2 aggregates."""

    epochs = np.asarray(payload["epochs"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int8)
    target_mask = labels == 1
    nontarget_mask = labels == 0

    pooled_online_target_sum_pz += epochs[target_mask, config.pz_channel_index, :].sum(axis=0)
    if subject_id in SPECIAL_POZ_SUBJECTS:
        online_poz_by_subject[subject_id][session_id] = mean_waveform(
            epochs[target_mask, config.poz_channel_index, :],
            context=f"{subject_id}/{session_id} online target POz",
        )

    class_balance_rows.append(
        build_class_balance_row(
            subject_id=subject_id,
            session=session_id,
            split="online",
            n_target=int(target_mask.sum()),
            n_nontarget=int(nontarget_mask.sum()),
        )
    )
    return int(target_mask.sum())


def build_subject_p300_summaries(
    *,
    per_subject_target_sum_pz: dict[str, np.ndarray],
    per_subject_target_count_pz: dict[str, int],
    per_subject_nontarget_sum_pz: dict[str, np.ndarray],
    per_subject_nontarget_count_pz: dict[str, int],
    config: Step2Config,
) -> list[SubjectP300Summary]:
    """Build pooled calibration P300 summaries for all subjects."""

    summaries: list[SubjectP300Summary] = []
    for subject_id in sorted(per_subject_target_sum_pz):
        target_waveform = divide_sum_by_count(
            per_subject_target_sum_pz[subject_id],
            per_subject_target_count_pz[subject_id],
        )
        nontarget_waveform = divide_sum_by_count(
            per_subject_nontarget_sum_pz[subject_id],
            per_subject_nontarget_count_pz[subject_id],
        )
        diff_waveform = target_waveform - nontarget_waveform
        diff_peak = compute_peak_metrics(
            waveform=diff_waveform,
            window=config.peak_search_window,
            fs=config.output_fs,
            time_offset_ms=config.time_offset_ms,
        )
        target_peak = compute_peak_metrics(
            waveform=target_waveform,
            window=config.peak_search_window,
            fs=config.output_fs,
            time_offset_ms=config.time_offset_ms,
        )
        summaries.append(
            SubjectP300Summary(
                subject_id=subject_id,
                target_waveform_pz=target_waveform,
                diff_wave_peak_amplitude_uV=diff_peak.amplitude_uV,
                diff_wave_peak_latency_ms=diff_peak.latency_ms,
                target_peak_amplitude_uV=target_peak.amplitude_uV,
                target_peak_latency_ms=target_peak.latency_ms,
            )
        )
    return summaries


def build_per_subject_summary_comments(subject_summaries: list[SubjectP300Summary]) -> list[str]:
    """Build CSV header comments for the per-subject summary table."""

    amplitude_cv = coefficient_of_variation(
        np.asarray([row.diff_wave_peak_amplitude_uV for row in subject_summaries], dtype=np.float64),
    )
    latency_cv = coefficient_of_variation(
        np.asarray([row.diff_wave_peak_latency_ms for row in subject_summaries], dtype=np.float64),
    )
    return [
        f"diff_wave_peak_amplitude_cv={format_float(amplitude_cv)}",
        f"diff_wave_peak_latency_cv={format_float(latency_cv)}",
    ]


def select_representative_subjects(subject_summaries: list[SubjectP300Summary]) -> list[RepresentativeSubject]:
    """Select strong, average, and weakest subjects deterministically."""

    if len(subject_summaries) != 15:
        raise ValueError(f"Expected 15 subject summaries, got {len(subject_summaries)}")

    ranked = sorted(
        subject_summaries,
        key=lambda row: (row.diff_wave_peak_amplitude_uV, row.subject_id),
    )
    selected = [
        ("strong", ranked[-1]),
        ("average", ranked[len(ranked) // 2]),
        ("weakest", ranked[0]),
    ]
    return [
        RepresentativeSubject(
            subject_id=row.subject_id,
            category=category,
            diff_wave_peak_amplitude_uV=row.diff_wave_peak_amplitude_uV,
        )
        for category, row in selected
    ]


def select_representative_session(
    *,
    subject_id: str,
    k_values: dict[tuple[str, str], int],
    preferred_ks: tuple[int, int],
) -> tuple[str, int]:
    """Choose one moderate-K session for the representative subject."""

    subject_sessions = sorted(
        (
            session_id,
            k_value,
        )
        for (current_subject_id, session_id), k_value in k_values.items()
        if current_subject_id == subject_id
    )
    if len(subject_sessions) != 7:
        raise ValueError(f"Expected 7 sessions for {subject_id}, got {len(subject_sessions)}")

    exact_matches = [item for item in subject_sessions if item[1] in preferred_ks]
    if exact_matches:
        return min(exact_matches, key=lambda item: session_sort_key(item[0]))

    target_k = float(sum(preferred_ks)) / float(len(preferred_ks))
    fallback = min(
        subject_sessions,
        key=lambda item: (abs(float(item[1]) - target_k), session_sort_key(item[0])),
    )
    logger.warning(
        "No K in %s for %s; using closest session %s (K=%d)",
        preferred_ks,
        subject_id,
        fallback[0],
        fallback[1],
    )
    return fallback


def build_session_latency_rows(
    *,
    session_social_target_pz: dict[str, dict[str, np.ndarray]],
    session_social_nontarget_pz: dict[str, dict[str, np.ndarray]],
    config: Step2Config,
) -> list[dict[str, str | float]]:
    """Build all 15 x 7 session latency rows for both latency methods."""

    rows: list[dict[str, str | float]] = []
    for subject_id in sorted(session_social_target_pz):
        session_ids = sorted(session_social_target_pz[subject_id], key=session_sort_key)
        if len(session_ids) != 7:
            raise ValueError(f"{subject_id} does not have 7 session waveforms")
        for session_id in session_ids:
            target_waveform = np.asarray(session_social_target_pz[subject_id][session_id], dtype=np.float64)
            nontarget_waveform = np.asarray(session_social_nontarget_pz[subject_id][session_id], dtype=np.float64)
            diff_waveform = target_waveform - nontarget_waveform
            target_peak = compute_peak_metrics(
                waveform=target_waveform,
                window=config.peak_search_window,
                fs=config.output_fs,
                time_offset_ms=config.time_offset_ms,
            )
            diff_peak = compute_peak_metrics(
                waveform=diff_waveform,
                window=config.peak_search_window,
                fs=config.output_fs,
                time_offset_ms=config.time_offset_ms,
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "session": session_id,
                    "target_peak_latency_ms": target_peak.latency_ms,
                    "diff_wave_peak_latency_ms": diff_peak.latency_ms,
                }
            )
    return rows


def build_per_subject_latency_summary_rows(
    session_latency_rows: list[dict[str, str | float]],
) -> tuple[list[dict[str, str | float]], list[str]]:
    """Summarize per-subject difference-wave latency evolution."""

    by_subject: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in session_latency_rows:
        subject_id = str(row["subject_id"])
        session_number = session_sort_key(str(row["session"]))
        latency_ms = float(row["diff_wave_peak_latency_ms"])
        by_subject[subject_id].append((session_number, latency_ms))

    summary_rows: list[dict[str, str | float]] = []
    flagged_subjects: list[str] = []
    for subject_id in sorted(by_subject):
        points = sorted(by_subject[subject_id], key=lambda item: item[0])
        if len(points) != 7:
            raise ValueError(f"{subject_id} does not have 7 latency rows")
        session_numbers = np.asarray([item[0] for item in points], dtype=np.float64)
        latencies = np.asarray([item[1] for item in points], dtype=np.float64)
        pearson_r, pearson_p = safe_pearsonr(session_numbers, latencies)
        mean_latency = float(np.mean(latencies))
        if mean_latency < 300.0 or mean_latency > 550.0:
            flagged_subjects.append(subject_id)
        summary_rows.append(
            {
                "subject_id": subject_id,
                "mean_latency_ms": mean_latency,
                "std_latency_ms": float(np.std(latencies, ddof=0)),
                "latency_change_s1_to_s7_ms": float(latencies[-1] - latencies[0]),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
            }
        )
    return summary_rows, flagged_subjects


def compute_group_level_latency_correlation(
    session_latency_rows: list[dict[str, str | float]],
    *,
    latency_key: str,
) -> tuple[float, float]:
    """Compute the 7-point group-level latency correlation."""

    session_to_values: dict[int, list[float]] = defaultdict(list)
    for row in session_latency_rows:
        session_to_values[session_sort_key(str(row["session"]))].append(float(row[latency_key]))

    session_numbers = np.asarray(sorted(session_to_values), dtype=np.float64)
    group_means = np.asarray(
        [float(np.mean(session_to_values[int(session)])) for session in session_numbers],
        dtype=np.float64,
    )
    return safe_pearsonr(session_numbers, group_means)


def compute_within_session_variability(
    *,
    payload: dict[str, np.ndarray],
    subject_id: str,
    session_id: str,
    config: Step2Config,
) -> tuple[dict[str, str], float]:
    """Compute block SD, noise floor, and SNR for one online session."""

    epochs = np.asarray(payload["epochs"], dtype=np.float64)
    labels = np.asarray(payload["labels"], dtype=np.int8)
    block_ids = np.asarray(payload["block_ids"], dtype=np.int16)
    target_epoch_amplitudes = compute_epoch_window_amplitudes(
        epochs=epochs,
        channel_index=config.pz_channel_index,
        window=config.p300_window,
    )

    block_means: list[float] = []
    block_noise: list[float] = []
    for block_id in range(50):
        mask = (block_ids == block_id) & (labels == 1)
        block_amplitudes = target_epoch_amplitudes[mask]
        if block_amplitudes.size == 0:
            raise ValueError(f"{subject_id}/{session_id} online block {block_id} has no target epochs")
        block_means.append(float(np.mean(block_amplitudes)))
        block_noise.append(float(np.std(block_amplitudes, ddof=0)))

    block_sd = float(np.std(np.asarray(block_means, dtype=np.float64), ddof=0))
    noise_floor = float(np.mean(np.asarray(block_noise, dtype=np.float64)))
    if math.isclose(noise_floor, 0.0, abs_tol=0.0):
        raise ValueError(
            f"{subject_id}/{session_id} has noise_floor_uV == 0. Step 2 validation requires failure here.",
        )
    snr = block_sd / noise_floor
    if not (np.isfinite(block_sd) and np.isfinite(noise_floor) and np.isfinite(snr)):
        raise ValueError(f"{subject_id}/{session_id} produced non-finite within-session variability metrics")

    row = {
        "subject_id": subject_id,
        "session": session_id,
        "block_sd_uV": format_float(block_sd),
        "noise_floor_uV": format_float(noise_floor),
        "snr": format_float(snr),
    }
    return row, snr


def compute_block_target_amplitudes(
    *,
    epochs: np.ndarray,
    labels: np.ndarray,
    block_ids: np.ndarray,
    expected_blocks: int,
    channel_index: int,
    window: tuple[int, int],
) -> np.ndarray:
    """Compute one mean target amplitude per block using the Step 2 definition."""

    epoch_amplitudes = compute_epoch_window_amplitudes(
        epochs=epochs,
        channel_index=channel_index,
        window=window,
    )
    block_means = np.zeros(expected_blocks, dtype=np.float64)
    for block_id in range(expected_blocks):
        mask = (block_ids == block_id) & (labels == 1)
        block_values = epoch_amplitudes[mask]
        if block_values.size == 0:
            raise ValueError(f"Block {block_id} has no target epochs")
        block_means[block_id] = float(np.mean(block_values))
    return block_means


def build_block_trajectory(
    *,
    calibration_payload: dict[str, np.ndarray],
    online_payload: dict[str, np.ndarray],
    config: Step2Config,
) -> np.ndarray:
    """Build the full 70-block target-amplitude trajectory for one session."""

    calibration_means = compute_block_target_amplitudes(
        epochs=np.asarray(calibration_payload["epochs"], dtype=np.float64),
        labels=np.asarray(calibration_payload["labels"], dtype=np.int8),
        block_ids=np.asarray(calibration_payload["block_ids"], dtype=np.int16),
        expected_blocks=20,
        channel_index=config.pz_channel_index,
        window=config.p300_window,
    )
    online_means = compute_block_target_amplitudes(
        epochs=np.asarray(online_payload["epochs"], dtype=np.float64),
        labels=np.asarray(online_payload["labels"], dtype=np.int8),
        block_ids=np.asarray(online_payload["block_ids"], dtype=np.int16),
        expected_blocks=50,
        channel_index=config.pz_channel_index,
        window=config.p300_window,
    )
    return np.concatenate([calibration_means, online_means]).astype(np.float64)


def plot_grand_average_all_channels(
    *,
    target_waveform: np.ndarray,
    nontarget_waveform: np.ndarray,
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot pooled calibration target vs non-target ERPs for all channels."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    shade_start = time_ms[config.p300_window[0]]
    shade_end = time_ms[config.p300_window[1]]
    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        ax = axes[channel_index]
        ax.plot(time_ms, target_waveform[channel_index], color="tab:blue", linewidth=2.0, label="Target")
        ax.plot(time_ms, nontarget_waveform[channel_index], color="0.5", linewidth=2.0, label="Non-target")
        ax.axvspan(shade_start, shade_end, color="tab:orange", alpha=0.10)
        ax.set_title(channel_name)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Grand-average calibration ERP by channel")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pz_target_vs_nontarget(
    *,
    target_waveform: np.ndarray,
    nontarget_waveform: np.ndarray,
    diff_waveform: np.ndarray,
    diff_peak: PeakMetrics,
    target_peak: PeakMetrics,
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot grand-average Pz target, non-target, and difference-wave ERPs."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_ms, target_waveform, label="Target", linewidth=2.4, color="tab:blue")
    ax.plot(time_ms, nontarget_waveform, label="Non-target", linewidth=2.2, color="0.5")
    ax.plot(time_ms, diff_waveform, label="Difference (target - non-target)", linewidth=2.4, color="tab:red")
    ax.axvspan(
        time_ms[config.p300_window[0]],
        time_ms[config.p300_window[1]],
        color="tab:orange",
        alpha=0.12,
        label="P300 window",
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude ($\\mu$V)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pz_condition_comparison(
    *,
    social_waveform: np.ndarray,
    nonsocial_waveform: np.ndarray,
    social_peak: PeakMetrics,
    nonsocial_peak: PeakMetrics,
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot grand-average target Pz waveforms for social vs non-social calibration."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_ms, social_waveform, label="Social (blocks 10-19)", linewidth=2.4, color="tab:green")
    ax.plot(time_ms, nonsocial_waveform, label="Non-social (blocks 0-9)", linewidth=2.4, color="tab:purple")
    ax.axvspan(time_ms[config.p300_window[0]], time_ms[config.p300_window[1]], color="tab:orange", alpha=0.12)
    title = (
        "Target ERP at Pz: social vs non-social\n"
        f"Social peak {social_peak.amplitude_uV:.2f} uV @ {social_peak.latency_ms:.1f} ms; "
        f"non-social peak {nonsocial_peak.amplitude_uV:.2f} uV @ {nonsocial_peak.latency_ms:.1f} ms"
    )
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_subject_pz(
    *,
    subject_summaries: list[SubjectP300Summary],
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot all 15 pooled subject ERPs at Pz."""

    fig, ax = plt.subplots(figsize=(13, 8))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(subject_summaries)))
    for color, row in zip(colors, sorted(subject_summaries, key=lambda item: item.subject_id), strict=True):
        ax.plot(time_ms, row.target_waveform_pz, linewidth=1.8, color=color, label=row.subject_id)
    ax.axvspan(time_ms[config.p300_window[0]], time_ms[config.p300_window[1]], color="tab:orange", alpha=0.10)
    ax.set_title("Per-subject pooled calibration target ERP at Pz")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_poz_session_diagnostic(
    *,
    subject_id: str,
    calibration_waveforms: dict[str, np.ndarray],
    online_waveforms: dict[str, np.ndarray],
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot session-level POz diagnostics for one flagged subject."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    session_ids = sorted(calibration_waveforms, key=session_sort_key)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(session_ids)))
    for axis, split_name, waveforms in (
        (axes[0], "Calibration", calibration_waveforms),
        (axes[1], "Online", online_waveforms),
    ):
        for color, session_id in zip(colors, session_ids, strict=True):
            axis.plot(time_ms, waveforms[session_id], linewidth=1.8, color=color, label=session_id)
        axis.axvspan(time_ms[config.p300_window[0]], time_ms[config.p300_window[1]], color="tab:orange", alpha=0.10)
        axis.set_title(f"{split_name} target ERP at POz")
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Amplitude (uV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{subject_id} session-level POz diagnostic")
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_session_evolution(
    *,
    subject_id: str,
    category: str,
    session_waveforms: dict[str, np.ndarray],
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot social-condition calibration target Pz ERP across sessions."""

    fig, ax = plt.subplots(figsize=(11, 7))
    session_ids = sorted(session_waveforms, key=session_sort_key)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(session_ids)))
    for color, session_id in zip(colors, session_ids, strict=True):
        ax.plot(time_ms, session_waveforms[session_id], linewidth=2.0, color=color, label=session_id)
    ax.axvspan(time_ms[config.p300_window[0]], time_ms[config.p300_window[1]], color="tab:orange", alpha=0.10)
    ax.set_title(f"{subject_id} social-calibration session evolution at Pz ({category})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spatial_distribution(*, channel_peak_amplitudes: np.ndarray, output_path: Path) -> None:
    """Plot channel-wise grand-average target P300 peak amplitudes."""

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "tab:blue" if channel_name in {"C3", "Cz", "C4", "CPz"} else "tab:green"
        for channel_name in CHANNEL_NAMES
    ]
    bars = ax.bar(CHANNEL_NAMES, channel_peak_amplitudes, color=colors)
    ax.set_title("Grand-average target P300 spatial distribution")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Peak amplitude (uV)")
    for bar, value in zip(bars, channel_peak_amplitudes, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.03, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_calibration_social_vs_online(
    *,
    calibration_social_waveform: np.ndarray,
    online_waveform: np.ndarray,
    calibration_peak: PeakMetrics,
    online_peak: PeakMetrics,
    time_ms: np.ndarray,
    config: Step2Config,
    output_path: Path,
) -> None:
    """Plot pooled calibration-social vs pooled online target ERP at Pz."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_ms, calibration_social_waveform, linewidth=2.4, color="tab:green", label="Calibration (social)")
    ax.plot(time_ms, online_waveform, linewidth=2.4, color="tab:red", label="Online")
    ax.axvspan(time_ms[config.p300_window[0]], time_ms[config.p300_window[1]], color="tab:orange", alpha=0.12)
    title = (
        "Target ERP at Pz: calibration social vs online\n"
        f"Calibration peak {calibration_peak.amplitude_uV:.2f} uV @ {calibration_peak.latency_ms:.1f} ms; "
        f"online peak {online_peak.amplitude_uV:.2f} uV @ {online_peak.latency_ms:.1f} ms"
    )
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_block_trajectory(
    *,
    subject_id: str,
    session_id: str,
    session_k: int,
    category: str,
    amplitudes: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the 70-block target-amplitude trajectory for one selected session."""

    x_values = np.arange(70, dtype=np.int16)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, amplitudes, marker="o", linewidth=1.8, markersize=4, color="tab:blue")
    ax.axvline(10.0, color="tab:gray", linestyle="--", linewidth=1.2)
    ax.axvline(20.0, color="tab:red", linestyle="--", linewidth=1.2)
    peak_y = float(np.max(amplitudes))
    ax.text(4.5, peak_y, "Non-social cal", ha="center", va="bottom", fontsize=9)
    ax.text(14.5, peak_y, "Social cal", ha="center", va="bottom", fontsize=9)
    ax.text(44.5, peak_y, "Online", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{subject_id} {session_id} block trajectory at Pz ({category}, K={session_k})")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Mean P300-window amplitude (uV)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_snr_histogram(*, snr_values: np.ndarray, output_path: Path) -> None:
    """Plot the within-session SNR distribution."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(snr_values, bins=15, color="tab:blue", edgecolor="white")
    ax.axvline(1.0, color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_title("Within-session SNR distribution")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Session count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_k_histogram(*, k_values: np.ndarray, output_path: Path) -> None:
    """Plot the K-value distribution with count labels."""

    values, counts = np.unique(k_values.astype(np.int16), return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(values.astype(int), counts.astype(int), color="tab:purple")
    ax.set_title("Distribution of K values across sessions")
    ax.set_xlabel("K")
    ax.set_ylabel("Session count")
    ax.set_xticks(values.astype(int))
    for bar, count in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, int(count) + 0.2, str(int(count)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def validate_step2_outputs(*, config: Step2Config, summary: Step2Summary) -> None:
    """Validate the core Step 2 output contract after generation."""

    expected_paths = [
        config.output_root / "grand_average_target_vs_nontarget_all_channels.png",
        config.output_root / "grand_average_pz_target_vs_nontarget.png",
        config.output_root / "pz_social_vs_nonsocial_target.png",
        config.output_root / "per_subject_pz_target_erp.png",
        config.output_root / "sub-03_poz_session_diagnostic.png",
        config.output_root / "sub-05_poz_session_diagnostic.png",
        config.output_root / "representative_subjects.csv",
        config.output_root / "session_latency_evolution.csv",
        config.output_root / "p300_spatial_distribution.png",
        config.output_root / "pz_calibration_social_vs_online.png",
        config.output_root / "within_session_variability.csv",
        config.output_root / "snr_histogram.png",
        config.output_root / "k_value_histogram.png",
        config.output_root / "class_balance_summary.csv",
        config.output_root / "per_subject_latency_summary.csv",
        config.output_root / "per_subject_p300_summary.csv",
    ]
    expected_paths.extend(
        config.output_root / f"representative_{subject_id_to_sub_label(subject_id)}_pz_session_evolution.png"
        for subject_id in summary.representative_subjects
    )
    expected_paths.extend(
        config.output_root / f"representative_{subject_id_to_sub_label(subject_id)}_block_trajectory.png"
        for subject_id in summary.representative_subjects
    )
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected Step 2 outputs: {missing}")

    if summary.grand_average_target_peak.amplitude_uV <= 0.0:
        raise ValueError("Grand-average target peak at Pz must be > 0 uV")

    if len(read_csv_rows(config.output_root / "per_subject_p300_summary.csv")) != 15:
        raise ValueError("per_subject_p300_summary.csv must have 15 data rows")
    if len(read_csv_rows(config.output_root / "representative_subjects.csv")) != 3:
        raise ValueError("representative_subjects.csv must have 3 rows")
    if len(read_csv_rows(config.output_root / "session_latency_evolution.csv")) != 105:
        raise ValueError("session_latency_evolution.csv must have 105 rows")
    within_rows = read_csv_rows(config.output_root / "within_session_variability.csv")
    if len(within_rows) != 105:
        raise ValueError("within_session_variability.csv must have 105 rows")
    class_balance_rows = read_csv_rows(config.output_root / "class_balance_summary.csv")
    if len(class_balance_rows) != 210:
        raise ValueError("class_balance_summary.csv must have 210 rows")
    if len(read_csv_rows(config.output_root / "per_subject_latency_summary.csv")) != 15:
        raise ValueError("per_subject_latency_summary.csv must have 15 rows")
    for row in class_balance_rows:
        if row["ratio"] != "1:7":
            raise ValueError(f"Class balance ratio mismatch: {row}")
    for row in within_rows:
        block_sd = float(row["block_sd_uV"])
        noise_floor = float(row["noise_floor_uV"])
        snr = float(row["snr"])
        if not (np.isfinite(block_sd) and np.isfinite(noise_floor) and np.isfinite(snr)):
            raise ValueError(f"Non-finite within-session variability row: {row}")
        if math.isclose(noise_floor, 0.0, abs_tol=0.0):
            raise ValueError(f"Zero noise floor detected: {row}")


def log_summary(summary: Step2Summary) -> None:
    """Log the headline Step 2 metrics."""

    logger.info(
        "Step 2 complete. Grand-average target peak %.3f uV @ %.1f ms; diff peak %.3f uV @ %.1f ms.",
        summary.grand_average_target_peak.amplitude_uV,
        summary.grand_average_target_peak.latency_ms,
        summary.grand_average_diff_peak.amplitude_uV,
        summary.grand_average_diff_peak.latency_ms,
    )
    logger.info(
        "Diff-wave amplitude CV %.3f, latency CV %.3f. Group target r=%.3f (p=%.4f), diff r=%.3f (p=%.4f).",
        summary.diff_wave_amplitude_cv,
        summary.diff_wave_latency_cv,
        summary.target_group_latency_r,
        summary.target_group_latency_p,
        summary.diff_group_latency_r,
        summary.diff_group_latency_p,
    )
    logger.info(
        "SNR mean %.3f, median %.3f, range [%.3f, %.3f]. Representatives: %s. Flagged latency subjects: %s.",
        summary.snr_mean,
        summary.snr_median,
        summary.snr_min,
        summary.snr_max,
        ", ".join(summary.representative_subjects),
        ", ".join(summary.flagged_subjects) if summary.flagged_subjects else "none",
    )


def run_step2(config: Step2Config | None = None) -> Step2Summary:
    """Run the full Step 2 ERP exploration pipeline."""

    config = config or Step2Config()
    config.output_root.mkdir(parents=True, exist_ok=True)

    time_ms = build_time_axis_ms(
        n_samples=config.epoch_samples,
        fs=config.output_fs,
        time_offset_ms=config.time_offset_ms,
    )
    k_values = parse_k_values_from_inventory(config.inventory_path)

    subjects = sorted({subject_id for subject_id, _ in expected_session_ids()})
    n_channels = len(CHANNEL_NAMES)
    n_samples = config.epoch_samples

    pooled_target_sum = np.zeros((n_channels, n_samples), dtype=np.float64)
    pooled_nontarget_sum = np.zeros((n_channels, n_samples), dtype=np.float64)
    pooled_target_count = [0]
    pooled_nontarget_count = [0]

    social_target_sum_pz = np.zeros(n_samples, dtype=np.float64)
    nonsocial_target_sum_pz = np.zeros(n_samples, dtype=np.float64)
    social_target_count_pz = [0]
    nonsocial_target_count_pz = [0]

    per_subject_target_sum_pz = {subject_id: np.zeros(n_samples, dtype=np.float64) for subject_id in subjects}
    per_subject_target_count_pz = {subject_id: 0 for subject_id in subjects}
    per_subject_nontarget_sum_pz = {subject_id: np.zeros(n_samples, dtype=np.float64) for subject_id in subjects}
    per_subject_nontarget_count_pz = {subject_id: 0 for subject_id in subjects}

    session_social_target_pz: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    session_social_nontarget_pz: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    calibration_poz_by_subject: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    online_poz_by_subject: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    pooled_online_target_sum_pz = np.zeros(n_samples, dtype=np.float64)
    pooled_online_target_count_pz = 0

    class_balance_rows: list[dict[str, str]] = []
    within_session_rows: list[dict[str, str]] = []
    snr_values: list[float] = []

    for subject_id, session_id in expected_session_ids():
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
        process_calibration_payload(
            payload=calibration_payload,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
            pooled_target_sum=pooled_target_sum,
            pooled_nontarget_sum=pooled_nontarget_sum,
            pooled_target_count_ref=pooled_target_count,
            pooled_nontarget_count_ref=pooled_nontarget_count,
            social_target_sum_pz=social_target_sum_pz,
            nonsocial_target_sum_pz=nonsocial_target_sum_pz,
            social_target_count_ref=social_target_count_pz,
            nonsocial_target_count_ref=nonsocial_target_count_pz,
            per_subject_target_sum_pz=per_subject_target_sum_pz,
            per_subject_target_count_pz=per_subject_target_count_pz,
            per_subject_nontarget_sum_pz=per_subject_nontarget_sum_pz,
            per_subject_nontarget_count_pz=per_subject_nontarget_count_pz,
            session_social_target_pz=session_social_target_pz,
            session_social_nontarget_pz=session_social_nontarget_pz,
            calibration_poz_by_subject=calibration_poz_by_subject,
            class_balance_rows=class_balance_rows,
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
        pooled_online_target_count_pz += process_online_payload(
            payload=online_payload,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
            pooled_online_target_sum_pz=pooled_online_target_sum_pz,
            online_poz_by_subject=online_poz_by_subject,
            class_balance_rows=class_balance_rows,
        )
        within_row, snr_value = compute_within_session_variability(
            payload=online_payload,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )
        within_session_rows.append(within_row)
        snr_values.append(snr_value)

    pooled_target_waveform = divide_sum_by_count(pooled_target_sum, pooled_target_count[0])
    pooled_nontarget_waveform = divide_sum_by_count(pooled_nontarget_sum, pooled_nontarget_count[0])
    pooled_target_pz = pooled_target_waveform[config.pz_channel_index]
    pooled_nontarget_pz = pooled_nontarget_waveform[config.pz_channel_index]
    pooled_diff_pz = pooled_target_pz - pooled_nontarget_pz

    social_target_pz = divide_sum_by_count(social_target_sum_pz, social_target_count_pz[0])
    nonsocial_target_pz = divide_sum_by_count(nonsocial_target_sum_pz, nonsocial_target_count_pz[0])
    online_target_pz = divide_sum_by_count(pooled_online_target_sum_pz, pooled_online_target_count_pz)

    subject_summaries = build_subject_p300_summaries(
        per_subject_target_sum_pz=per_subject_target_sum_pz,
        per_subject_target_count_pz=per_subject_target_count_pz,
        per_subject_nontarget_sum_pz=per_subject_nontarget_sum_pz,
        per_subject_nontarget_count_pz=per_subject_nontarget_count_pz,
        config=config,
    )
    representative_subjects = select_representative_subjects(subject_summaries)

    write_csv_rows(
        path=config.output_root / "per_subject_p300_summary.csv",
        fieldnames=(
            "subject_id",
            "diff_wave_peak_amplitude_uV",
            "diff_wave_peak_latency_ms",
            "target_peak_amplitude_uV",
            "target_peak_latency_ms",
        ),
        rows=(
            {
                "subject_id": row.subject_id,
                "diff_wave_peak_amplitude_uV": format_float(row.diff_wave_peak_amplitude_uV),
                "diff_wave_peak_latency_ms": format_float(row.diff_wave_peak_latency_ms),
                "target_peak_amplitude_uV": format_float(row.target_peak_amplitude_uV),
                "target_peak_latency_ms": format_float(row.target_peak_latency_ms),
            }
            for row in subject_summaries
        ),
        comment_lines=build_per_subject_summary_comments(subject_summaries),
    )
    write_csv_rows(
        path=config.output_root / "representative_subjects.csv",
        fieldnames=("subject_id", "category", "diff_wave_peak_amplitude_uV"),
        rows=(
            {
                "subject_id": row.subject_id,
                "category": row.category,
                "diff_wave_peak_amplitude_uV": format_float(row.diff_wave_peak_amplitude_uV),
            }
            for row in representative_subjects
        ),
    )

    session_latency_rows = build_session_latency_rows(
        session_social_target_pz=session_social_target_pz,
        session_social_nontarget_pz=session_social_nontarget_pz,
        config=config,
    )
    write_csv_rows(
        path=config.output_root / "session_latency_evolution.csv",
        fieldnames=("subject_id", "session", "target_peak_latency_ms", "diff_wave_peak_latency_ms"),
        rows=(
            {
                "subject_id": row["subject_id"],
                "session": row["session"],
                "target_peak_latency_ms": format_float(float(row["target_peak_latency_ms"])),
                "diff_wave_peak_latency_ms": format_float(float(row["diff_wave_peak_latency_ms"])),
            }
            for row in session_latency_rows
        ),
    )

    per_subject_latency_rows, flagged_subjects = build_per_subject_latency_summary_rows(session_latency_rows)
    write_csv_rows(
        path=config.output_root / "per_subject_latency_summary.csv",
        fieldnames=(
            "subject_id",
            "mean_latency_ms",
            "std_latency_ms",
            "latency_change_s1_to_s7_ms",
            "pearson_r",
            "pearson_p",
        ),
        rows=(
            {
                "subject_id": row["subject_id"],
                "mean_latency_ms": format_float(float(row["mean_latency_ms"])),
                "std_latency_ms": format_float(float(row["std_latency_ms"])),
                "latency_change_s1_to_s7_ms": format_float(float(row["latency_change_s1_to_s7_ms"])),
                "pearson_r": format_float_or_nan(float(row["pearson_r"])),
                "pearson_p": format_float_or_nan(float(row["pearson_p"])),
            }
            for row in per_subject_latency_rows
        ),
    )

    write_csv_rows(
        path=config.output_root / "within_session_variability.csv",
        fieldnames=("subject_id", "session", "block_sd_uV", "noise_floor_uV", "snr"),
        rows=within_session_rows,
    )
    write_csv_rows(
        path=config.output_root / "class_balance_summary.csv",
        fieldnames=("subject_id", "session", "split", "n_target", "n_nontarget", "ratio"),
        rows=sorted(class_balance_rows, key=lambda row: (row["subject_id"], row["session"], row["split"])),
    )

    grand_average_target_peak = compute_peak_metrics(
        waveform=pooled_target_pz,
        window=config.p300_window,
        fs=config.output_fs,
        time_offset_ms=config.time_offset_ms,
    )
    grand_average_diff_peak = compute_peak_metrics(
        waveform=pooled_diff_pz,
        window=config.p300_window,
        fs=config.output_fs,
        time_offset_ms=config.time_offset_ms,
    )
    social_p300_peak = compute_peak_metrics(
        waveform=social_target_pz,
        window=config.p300_window,
        fs=config.output_fs,
        time_offset_ms=config.time_offset_ms,
    )
    nonsocial_p300_peak = compute_peak_metrics(
        waveform=nonsocial_target_pz,
        window=config.p300_window,
        fs=config.output_fs,
        time_offset_ms=config.time_offset_ms,
    )
    online_p300_peak = compute_peak_metrics(
        waveform=online_target_pz,
        window=config.p300_window,
        fs=config.output_fs,
        time_offset_ms=config.time_offset_ms,
    )
    group_target_latency_r, group_target_latency_p = compute_group_level_latency_correlation(
        session_latency_rows,
        latency_key="target_peak_latency_ms",
    )
    group_diff_latency_r, group_diff_latency_p = compute_group_level_latency_correlation(
        session_latency_rows,
        latency_key="diff_wave_peak_latency_ms",
    )

    plot_grand_average_all_channels(
        target_waveform=pooled_target_waveform,
        nontarget_waveform=pooled_nontarget_waveform,
        time_ms=time_ms,
        config=config,
        output_path=config.output_root / "grand_average_target_vs_nontarget_all_channels.png",
    )
    plot_pz_target_vs_nontarget(
        target_waveform=pooled_target_pz,
        nontarget_waveform=pooled_nontarget_pz,
        diff_waveform=pooled_diff_pz,
        diff_peak=grand_average_diff_peak,
        target_peak=grand_average_target_peak,
        time_ms=time_ms,
        config=config,
        output_path=config.output_root / "grand_average_pz_target_vs_nontarget.png",
    )
    plot_pz_condition_comparison(
        social_waveform=social_target_pz,
        nonsocial_waveform=nonsocial_target_pz,
        social_peak=social_p300_peak,
        nonsocial_peak=nonsocial_p300_peak,
        time_ms=time_ms,
        config=config,
        output_path=config.output_root / "pz_social_vs_nonsocial_target.png",
    )
    plot_per_subject_pz(
        subject_summaries=subject_summaries,
        time_ms=time_ms,
        config=config,
        output_path=config.output_root / "per_subject_pz_target_erp.png",
    )
    for subject_id in SPECIAL_POZ_SUBJECTS:
        plot_poz_session_diagnostic(
            subject_id=subject_id,
            calibration_waveforms=calibration_poz_by_subject[subject_id],
            online_waveforms=online_poz_by_subject[subject_id],
            time_ms=time_ms,
            config=config,
            output_path=config.output_root / f"{subject_id_to_sub_label(subject_id)}_poz_session_diagnostic.png",
        )
    for row in representative_subjects:
        plot_session_evolution(
            subject_id=row.subject_id,
            category=row.category,
            session_waveforms=session_social_target_pz[row.subject_id],
            time_ms=time_ms,
            config=config,
            output_path=config.output_root / f"representative_{subject_id_to_sub_label(row.subject_id)}_pz_session_evolution.png",
        )

    plot_spatial_distribution(
        channel_peak_amplitudes=compute_channel_peak_amplitudes(
            waveform=pooled_target_waveform,
            window=config.p300_window,
        ),
        output_path=config.output_root / "p300_spatial_distribution.png",
    )
    plot_calibration_social_vs_online(
        calibration_social_waveform=social_target_pz,
        online_waveform=online_target_pz,
        calibration_peak=social_p300_peak,
        online_peak=online_p300_peak,
        time_ms=time_ms,
        config=config,
        output_path=config.output_root / "pz_calibration_social_vs_online.png",
    )
    for row in representative_subjects:
        session_id, session_k = select_representative_session(
            subject_id=row.subject_id,
            k_values=k_values,
            preferred_ks=config.representative_target_ks,
        )
        calibration_payload = load_processed_npz(
            processed_split_path(
                processed_root=config.processed_root,
                subject_id=row.subject_id,
                session_id=session_id,
                split="calibration",
            ),
            split="calibration",
            config=config,
        )
        online_payload = load_processed_npz(
            processed_split_path(
                processed_root=config.processed_root,
                subject_id=row.subject_id,
                session_id=session_id,
                split="online",
            ),
            split="online",
            config=config,
        )
        plot_block_trajectory(
            subject_id=row.subject_id,
            session_id=session_id,
            session_k=session_k,
            category=row.category,
            amplitudes=build_block_trajectory(
                calibration_payload=calibration_payload,
                online_payload=online_payload,
                config=config,
            ),
            output_path=config.output_root / f"representative_{subject_id_to_sub_label(row.subject_id)}_block_trajectory.png",
        )

    plot_snr_histogram(
        snr_values=np.asarray(snr_values, dtype=np.float64),
        output_path=config.output_root / "snr_histogram.png",
    )
    plot_k_histogram(
        k_values=np.asarray(list(k_values.values()), dtype=np.int16),
        output_path=config.output_root / "k_value_histogram.png",
    )

    summary = Step2Summary(
        output_root=config.output_root,
        grand_average_target_peak=grand_average_target_peak,
        grand_average_diff_peak=grand_average_diff_peak,
        diff_wave_amplitude_cv=coefficient_of_variation(
            np.asarray([row.diff_wave_peak_amplitude_uV for row in subject_summaries], dtype=np.float64),
        ),
        diff_wave_latency_cv=coefficient_of_variation(
            np.asarray([row.diff_wave_peak_latency_ms for row in subject_summaries], dtype=np.float64),
        ),
        target_group_latency_r=group_target_latency_r,
        target_group_latency_p=group_target_latency_p,
        diff_group_latency_r=group_diff_latency_r,
        diff_group_latency_p=group_diff_latency_p,
        snr_mean=float(np.mean(snr_values)),
        snr_median=float(np.median(snr_values)),
        snr_min=float(np.min(snr_values)),
        snr_max=float(np.max(snr_values)),
        flagged_subjects=tuple(flagged_subjects),
        representative_subjects=tuple(row.subject_id for row in representative_subjects),
    )
    validate_step2_outputs(config=config, summary=summary)
    log_summary(summary)
    return summary
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Step 2 CLI parser."""

    parser = argparse.ArgumentParser(description="Run Step 2 ERP exploration and visualization.")
    parser.add_argument("--processed-root", type=Path, default=Step2Config.processed_root)
    parser.add_argument("--output-root", type=Path, default=Step2Config.output_root)
    parser.add_argument("--inventory-path", type=Path, default=Step2Config.inventory_path)
    return parser


def main() -> None:
    """Run the Step 2 CLI entrypoint."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    run_step2(
        Step2Config(
            processed_root=args.processed_root,
            output_root=args.output_root,
            inventory_path=args.inventory_path,
        )
    )
