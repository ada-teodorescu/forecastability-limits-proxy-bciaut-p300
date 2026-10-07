"""Step 6 forecasting task definition and trajectory exploration.

Builds a research-grade forecasting layer on top of frozen Step 4 tangent
features and Step 5 score trajectories. The canonical downstream task remains
future `score_raw` regression, while the manifests also carry research labels
for future raw change and clinical decline onset.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import expected_session_ids
from src.features.step4_riemannian import (
    METADATA_FIELDS,
    Step4Config,
    _validate_saved_tangent_payload,
    ts_split_path,
)
from src.features.step5_scores import (
    Step5Config,
    _validate_saved_score_payload,
    score_split_path,
)

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS: tuple[str, ...] = (
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
    "delta_score_raw",
    "target_score_drop_rel_session",
    "target_drop20_state",
    "future_drop20_onset",
)

SESSION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "session_id",
    "n_online_blocks",
    "online_runs_per_block",
    "online_score_raw_mean",
    "online_score_raw_std",
    "online_score_raw_min",
    "online_score_raw_max",
    "online_score_drop_rel_session_mean",
    "online_score_drop_rel_session_std",
    "online_score_drop_rel_session_min",
    "online_score_drop_rel_session_max",
    "online_drop20_prevalence",
)

EXPLORATION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "n_sessions",
    "n_online_blocks_total",
    "online_score_raw_mean",
    "online_score_raw_std",
    "online_score_raw_min",
    "online_score_raw_max",
    "online_score_drop_rel_session_mean",
    "online_score_drop_rel_session_std",
    "online_score_drop_rel_session_min",
    "online_score_drop_rel_session_max",
    "online_drop20_prevalence",
    "n_valid_pairs",
    "lagged_raw_corr",
    "persistence_mae",
    "anchor_delta_corr",
    "delta_score_raw_mean",
    "delta_score_raw_std",
    "future_drop20_state_prevalence",
    "future_drop20_onset_prevalence",
)

BLOCK_POSITION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "online_block_pos",
    "n_sessions",
    "score_raw_mean",
    "score_raw_std",
    "score_drop_rel_session_mean",
    "score_drop_rel_session_std",
    "drop20_prevalence",
)

TASK_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "n_sessions",
    "manifest_row_count",
    "manifest_rows_per_session",
    "default_context_length",
    "default_context_usable_row_count",
    "default_context_usable_rows_per_session",
    "target_score_raw_mean",
    "target_score_raw_std",
    "target_score_drop_rel_session_mean",
    "target_score_drop_rel_session_std",
    "delta_score_raw_mean",
    "delta_score_raw_std",
    "target_drop20_state_prevalence",
    "future_drop20_onset_prevalence",
)

EXPECTED_FIGURE_FILENAMES: tuple[str, ...] = (
    "step6_mean_trajectories.png",
    "step6_horizon_diagnostics.png",
    "step6_block_position_summary.png",
)


@dataclass(frozen=True)
class Step6Config:
    """Configuration for the Step 6 forecasting-task pipeline."""

    step4_root: Path = Path("data/features")
    step5_root: Path = Path("data/scores")
    output_root: Path = Path("data/forecasting")
    figure_root: Path = Path("outputs/forecasting/step6")
    horizons: tuple[int, ...] = (5, 10, 15, 20, 25)
    online_blocks: int = 50
    calibration_blocks: int = 20
    calibration_social_start: int = 10
    tangent_dim: int = 300
    default_context_length: int = 10
    drop_threshold: float = -0.20
    online_k_min: int = 3
    online_k_max: int = 10
    corr_epsilon: float = 1e-12


@dataclass(frozen=True)
class OnlineSessionInputs:
    """Validated Step 4/5 online artifacts for one subject-session pair."""

    subject_id: str
    session_id: str
    session_k: int
    features: np.ndarray
    score_raw: np.ndarray
    score_drop_rel_session: np.ndarray
    block_ids: np.ndarray
    condition: np.ndarray
    block_targets: np.ndarray
    runs_per_block: np.ndarray
    n_target_epochs: np.ndarray
    n_nontarget_epochs: np.ndarray


@dataclass(frozen=True)
class Step6Summary:
    """High-level summary of the executed Step 6 run."""

    output_root: Path
    figure_root: Path
    session_count: int
    session_summary_rows: int
    block_position_rows: int
    exploration_rows: int
    task_summary_rows: int
    manifest_file_count: int
    total_manifest_rows: int
    total_usable_rows_default_context: int
    figure_count: int
    session_k_min: int
    session_k_max: int
    horizons: tuple[int, ...]
    manifest_rows_by_horizon: tuple[int, ...]
    usable_rows_by_horizon: tuple[int, ...]


def session_summary_path(*, output_root: Path) -> Path:
    """Build the Step 6 session-summary CSV path."""

    return output_root / "step6_session_summary.csv"


def exploration_summary_path(*, output_root: Path) -> Path:
    """Build the Step 6 exploration-summary CSV path."""

    return output_root / "step6_exploration_summary.csv"


def block_position_summary_path(*, output_root: Path) -> Path:
    """Build the Step 6 block-position summary CSV path."""

    return output_root / "step6_block_position_summary.csv"


def task_summary_path(*, output_root: Path) -> Path:
    """Build the Step 6 task-summary CSV path."""

    return output_root / "step6_task_summary.csv"


def manifest_path(*, output_root: Path, horizon: int) -> Path:
    """Build the Step 6 per-horizon manifest CSV path."""

    return output_root / f"step6_h{int(horizon):02d}_manifest.csv"


def figure_path(*, figure_root: Path, filename: str) -> Path:
    """Build a Step 6 figure path."""

    return figure_root / filename


def csv_float(value: float) -> str:
    """Serialize a float deterministically for exact CSV roundtrips."""

    return repr(float(value))


def serialize_csv_value(value: Any) -> str:
    """Serialize one scalar value for CSV output."""

    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return csv_float(float(value))
    raise TypeError(f"Unsupported CSV value type: {type(value)!r}")


def serialize_records(records: list[dict[str, Any]], columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Serialize typed row dictionaries to string-only CSV rows."""

    serialized_rows: list[dict[str, str]] = []
    for record in records:
        missing = [column for column in columns if column not in record]
        if missing:
            raise ValueError(f"Record is missing required columns: {missing}")
        unexpected = sorted(set(record).difference(columns))
        if unexpected:
            raise ValueError(f"Record contains unexpected columns: {unexpected}")
        serialized_rows.append({column: serialize_csv_value(record[column]) for column in columns})
    return serialized_rows


def write_csv(path: Path, *, columns: tuple[str, ...], records: list[dict[str, Any]]) -> None:
    """Write a deterministic CSV with a fixed column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = serialize_records(records, columns)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV file and return its header and rows."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def validate_config(config: Step6Config) -> None:
    """Validate the frozen Step 6 configuration."""

    if config.online_blocks <= 0:
        raise ValueError("online_blocks must be positive")
    if config.default_context_length <= 0:
        raise ValueError("default_context_length must be positive")
    if not config.horizons:
        raise ValueError("At least one horizon is required")
    if len(set(config.horizons)) != len(config.horizons):
        raise ValueError(f"Horizons must be unique: {config.horizons}")
    for horizon in config.horizons:
        if horizon <= 0 or horizon >= config.online_blocks:
            raise ValueError(
                f"Horizon {horizon} is invalid for online_blocks={config.online_blocks}; expected 1..{config.online_blocks - 1}",
            )


def step4_validation_config(config: Step6Config) -> Step4Config:
    """Build a Step 4 validation config aligned with the Step 6 contract."""

    return Step4Config(
        output_root=config.step4_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        tangent_dim=config.tangent_dim,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
    )


def step5_validation_config(config: Step6Config) -> Step5Config:
    """Build a Step 5 validation config aligned with the Step 6 contract."""

    return Step5Config(
        output_root=config.step5_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
    )


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
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= epsilon or y_std <= epsilon:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        raise ValueError("Correlation result is non-finite")
    return corr


def compute_drop20_state(value: float, *, threshold: float) -> int:
    """Return the endpoint decline state under the Step 6 threshold rule."""

    return int(float(value) < float(threshold))


def compute_future_drop20_onset(
    score_drop_rel_session: np.ndarray,
    *,
    anchor_idx: int,
    target_idx: int,
    threshold: float,
) -> int:
    """Return 1 iff the first threshold crossing occurs after the anchor and within horizon."""

    scores = np.asarray(score_drop_rel_session, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"score_drop_rel_session must be 1-D, got shape {scores.shape}")
    if anchor_idx < 0 or target_idx < 0:
        raise ValueError("anchor_idx and target_idx must be non-negative")
    if anchor_idx > target_idx:
        raise ValueError(f"anchor_idx={anchor_idx} cannot exceed target_idx={target_idx}")
    if target_idx >= scores.shape[0]:
        raise ValueError(
            f"target_idx={target_idx} is out of range for score_drop_rel_session length {scores.shape[0]}",
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_drop_rel_session contains non-finite values")

    anchor_value = float(scores[anchor_idx])
    if anchor_value < float(threshold):
        return 0
    future_window = scores[anchor_idx + 1 : target_idx + 1]
    return int(np.any(future_window < float(threshold)))


def expected_manifest_rows_per_session(*, horizon: int, config: Step6Config) -> int:
    """Return the exact Step 6 manifest rows per session for one horizon."""

    return config.online_blocks - int(horizon)


def expected_default_usable_rows_per_session(*, horizon: int, config: Step6Config) -> int:
    """Return the default-context usable rows per session for one horizon."""

    usable = config.online_blocks - int(horizon) - config.default_context_length + 1
    return max(0, usable)


def count_default_usable_rows(records: list[dict[str, Any]], *, config: Step6Config) -> int:
    """Count manifest rows usable under the default context length."""

    min_anchor_idx = config.default_context_length - 1
    return sum(int(record["anchor_idx"]) >= min_anchor_idx for record in records)


def load_online_session_inputs(*, subject_id: str, session_id: str, config: Step6Config) -> OnlineSessionInputs:
    """Load and validate one Step 4 tangent file and one Step 5 score file for the online split."""

    step4_config = step4_validation_config(config)
    step5_config = step5_validation_config(config)
    step4_path = ts_split_path(output_root=config.step4_root, subject_id=subject_id, session_id=session_id, split="online")
    step5_path = score_split_path(output_root=config.step5_root, subject_id=subject_id, session_id=session_id, split="online")

    step4_payload = _validate_saved_tangent_payload(path=step4_path, split="online", config=step4_config)
    step5_payload = _validate_saved_score_payload(path=step5_path, split="online", config=step5_config)

    for field_name in METADATA_FIELDS:
        if not np.array_equal(step4_payload[field_name], step5_payload[field_name]):
            raise ValueError(
                f"{subject_id}/{session_id} Step 4 and Step 5 online metadata mismatch for field {field_name!r}",
            )

    runs_per_block = np.asarray(step4_payload["runs_per_block"], dtype=np.int16)
    unique_runs = np.unique(runs_per_block)
    if unique_runs.size != 1:
        raise ValueError(f"{subject_id}/{session_id} runs_per_block is not constant within the online split")
    session_k = int(unique_runs[0])
    if session_k < config.online_k_min or session_k > config.online_k_max:
        raise ValueError(
            f"{subject_id}/{session_id} session_k={session_k} outside expected {config.online_k_min}..{config.online_k_max}",
        )

    features = np.asarray(step4_payload["features"], dtype=np.float32)
    score_raw = np.asarray(step5_payload["score_raw"], dtype=np.float32)
    score_drop_rel_session = np.asarray(step5_payload["score_drop_rel_session"], dtype=np.float32)
    if features.shape != (config.online_blocks, config.tangent_dim):
        raise ValueError(
            f"{subject_id}/{session_id} features shape {features.shape}, expected ({config.online_blocks}, {config.tangent_dim})",
        )
    if score_raw.shape != (config.online_blocks,):
        raise ValueError(f"{subject_id}/{session_id} score_raw shape {score_raw.shape}, expected ({config.online_blocks},)")
    if score_drop_rel_session.shape != (config.online_blocks,):
        raise ValueError(
            f"{subject_id}/{session_id} score_drop_rel_session shape {score_drop_rel_session.shape}, expected ({config.online_blocks},)",
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{subject_id}/{session_id} features contain non-finite values")
    if not np.all(np.isfinite(score_raw)):
        raise ValueError(f"{subject_id}/{session_id} score_raw contains non-finite values")
    if not np.all(np.isfinite(score_drop_rel_session)):
        raise ValueError(f"{subject_id}/{session_id} score_drop_rel_session contains non-finite values")

    return OnlineSessionInputs(
        subject_id=subject_id,
        session_id=session_id,
        session_k=session_k,
        features=features,
        score_raw=score_raw,
        score_drop_rel_session=score_drop_rel_session,
        block_ids=np.asarray(step4_payload["block_ids"], dtype=np.int16),
        condition=np.asarray(step4_payload["condition"], dtype=np.int8),
        block_targets=np.asarray(step4_payload["block_targets"], dtype=np.int8),
        runs_per_block=runs_per_block,
        n_target_epochs=np.asarray(step4_payload["n_target_epochs"], dtype=np.int16),
        n_nontarget_epochs=np.asarray(step4_payload["n_nontarget_epochs"], dtype=np.int16),
    )


def build_session_summary_records(
    sessions: list[OnlineSessionInputs],
    *,
    config: Step6Config,
) -> list[dict[str, Any]]:
    """Build one descriptive summary row per subject-session pair."""

    records: list[dict[str, Any]] = []
    for session in sessions:
        score_raw = np.asarray(session.score_raw, dtype=np.float64)
        score_drop = np.asarray(session.score_drop_rel_session, dtype=np.float64)
        records.append(
            {
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "n_online_blocks": score_raw.shape[0],
                "online_runs_per_block": session.session_k,
                "online_score_raw_mean": float(np.mean(score_raw)),
                "online_score_raw_std": float(np.std(score_raw, ddof=0)),
                "online_score_raw_min": float(np.min(score_raw)),
                "online_score_raw_max": float(np.max(score_raw)),
                "online_score_drop_rel_session_mean": float(np.mean(score_drop)),
                "online_score_drop_rel_session_std": float(np.std(score_drop, ddof=0)),
                "online_score_drop_rel_session_min": float(np.min(score_drop)),
                "online_score_drop_rel_session_max": float(np.max(score_drop)),
                "online_drop20_prevalence": float(np.mean(score_drop < float(config.drop_threshold))),
            }
        )
    return records


def build_block_position_summary_records(
    sessions: list[OnlineSessionInputs],
    *,
    config: Step6Config,
) -> list[dict[str, Any]]:
    """Build pooled descriptive summaries for each online block position."""

    records: list[dict[str, Any]] = []
    raw_matrix = np.stack([np.asarray(session.score_raw, dtype=np.float64) for session in sessions], axis=0)
    drop_matrix = np.stack([np.asarray(session.score_drop_rel_session, dtype=np.float64) for session in sessions], axis=0)
    n_sessions = len(sessions)
    for block_pos in range(config.online_blocks):
        raw_values = raw_matrix[:, block_pos]
        drop_values = drop_matrix[:, block_pos]
        records.append(
            {
                "online_block_pos": block_pos,
                "n_sessions": n_sessions,
                "score_raw_mean": float(np.mean(raw_values)),
                "score_raw_std": float(np.std(raw_values, ddof=0)),
                "score_drop_rel_session_mean": float(np.mean(drop_values)),
                "score_drop_rel_session_std": float(np.std(drop_values, ddof=0)),
                "drop20_prevalence": float(np.mean(drop_values < float(config.drop_threshold))),
            }
        )
    return records


def build_manifest_records_for_session(
    session: OnlineSessionInputs,
    *,
    horizon: int,
    config: Step6Config,
) -> list[dict[str, Any]]:
    """Build the Step 6 per-horizon manifest rows for one session."""

    if horizon <= 0 or horizon >= config.online_blocks:
        raise ValueError(f"Horizon {horizon} is invalid for online_blocks={config.online_blocks}")
    if session.score_raw.shape[0] != config.online_blocks:
        raise ValueError(
            f"{session.subject_id}/{session.session_id} score_raw length {session.score_raw.shape[0]}, expected {config.online_blocks}",
        )
    if session.score_drop_rel_session.shape[0] != config.online_blocks:
        raise ValueError(
            f"{session.subject_id}/{session.session_id} score_drop_rel_session length {session.score_drop_rel_session.shape[0]}, expected {config.online_blocks}",
        )

    records: list[dict[str, Any]] = []
    for target_idx in range(horizon, config.online_blocks):
        anchor_idx = target_idx - horizon
        anchor_score_raw = float(np.asarray(session.score_raw[anchor_idx], dtype=np.float32))
        target_score_raw = float(np.asarray(session.score_raw[target_idx], dtype=np.float32))
        target_score_drop_rel_session = float(np.asarray(session.score_drop_rel_session[target_idx], dtype=np.float32))
        records.append(
            {
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "session_k": session.session_k,
                "horizon": horizon,
                "anchor_idx": anchor_idx,
                "target_idx": target_idx,
                "target_block_id": int(np.asarray(session.block_ids[target_idx], dtype=np.int16)),
                "elapsed_runs": int(horizon * session.session_k),
                "anchor_score_raw": anchor_score_raw,
                "target_score_raw": target_score_raw,
                "delta_score_raw": float(target_score_raw - anchor_score_raw),
                "target_score_drop_rel_session": target_score_drop_rel_session,
                "target_drop20_state": compute_drop20_state(target_score_drop_rel_session, threshold=config.drop_threshold),
                "future_drop20_onset": compute_future_drop20_onset(
                    session.score_drop_rel_session,
                    anchor_idx=anchor_idx,
                    target_idx=target_idx,
                    threshold=config.drop_threshold,
                ),
            }
        )
    return records


def build_manifest_records_by_horizon(
    sessions: list[OnlineSessionInputs],
    *,
    config: Step6Config,
) -> dict[int, list[dict[str, Any]]]:
    """Build all per-horizon manifest rows across the dataset."""

    manifests: dict[int, list[dict[str, Any]]] = {}
    for horizon in config.horizons:
        records: list[dict[str, Any]] = []
        for session in sessions:
            records.extend(build_manifest_records_for_session(session, horizon=horizon, config=config))
        manifests[horizon] = records
    return manifests


def build_task_summary_records(
    manifest_records_by_horizon: dict[int, list[dict[str, Any]]],
    *,
    n_sessions: int,
    config: Step6Config,
) -> list[dict[str, Any]]:
    """Build one task-summary row per horizon."""

    records: list[dict[str, Any]] = []
    for horizon in config.horizons:
        manifest_records = manifest_records_by_horizon[horizon]
        target_score_raw = np.asarray([record["target_score_raw"] for record in manifest_records], dtype=np.float64)
        target_score_drop = np.asarray(
            [record["target_score_drop_rel_session"] for record in manifest_records],
            dtype=np.float64,
        )
        delta_score_raw = np.asarray([record["delta_score_raw"] for record in manifest_records], dtype=np.float64)
        target_drop20_state = np.asarray([record["target_drop20_state"] for record in manifest_records], dtype=np.int8)
        future_drop20_onset = np.asarray([record["future_drop20_onset"] for record in manifest_records], dtype=np.int8)
        default_usable = count_default_usable_rows(manifest_records, config=config)
        records.append(
            {
                "horizon": horizon,
                "n_sessions": n_sessions,
                "manifest_row_count": len(manifest_records),
                "manifest_rows_per_session": expected_manifest_rows_per_session(horizon=horizon, config=config),
                "default_context_length": config.default_context_length,
                "default_context_usable_row_count": default_usable,
                "default_context_usable_rows_per_session": expected_default_usable_rows_per_session(
                    horizon=horizon,
                    config=config,
                ),
                "target_score_raw_mean": float(np.mean(target_score_raw)),
                "target_score_raw_std": float(np.std(target_score_raw, ddof=0)),
                "target_score_drop_rel_session_mean": float(np.mean(target_score_drop)),
                "target_score_drop_rel_session_std": float(np.std(target_score_drop, ddof=0)),
                "delta_score_raw_mean": float(np.mean(delta_score_raw)),
                "delta_score_raw_std": float(np.std(delta_score_raw, ddof=0)),
                "target_drop20_state_prevalence": float(np.mean(target_drop20_state)),
                "future_drop20_onset_prevalence": float(np.mean(future_drop20_onset)),
            }
        )
    return records


def build_exploration_summary_records(
    sessions: list[OnlineSessionInputs],
    *,
    manifest_records_by_horizon: dict[int, list[dict[str, Any]]],
    config: Step6Config,
) -> list[dict[str, Any]]:
    """Build pooled descriptive and horizon-aware exploration rows."""

    all_score_raw = np.concatenate([np.asarray(session.score_raw, dtype=np.float64) for session in sessions], axis=0)
    all_score_drop = np.concatenate(
        [np.asarray(session.score_drop_rel_session, dtype=np.float64) for session in sessions],
        axis=0,
    )
    n_sessions = len(sessions)
    records: list[dict[str, Any]] = []
    for horizon in config.horizons:
        manifest_records = manifest_records_by_horizon[horizon]
        anchor_score_raw = np.asarray([record["anchor_score_raw"] for record in manifest_records], dtype=np.float64)
        target_score_raw = np.asarray([record["target_score_raw"] for record in manifest_records], dtype=np.float64)
        delta_score_raw = np.asarray([record["delta_score_raw"] for record in manifest_records], dtype=np.float64)
        target_drop20_state = np.asarray([record["target_drop20_state"] for record in manifest_records], dtype=np.int8)
        future_drop20_onset = np.asarray([record["future_drop20_onset"] for record in manifest_records], dtype=np.int8)
        persistence_mae = float(np.mean(np.abs(target_score_raw - anchor_score_raw)))
        records.append(
            {
                "horizon": horizon,
                "n_sessions": n_sessions,
                "n_online_blocks_total": all_score_raw.shape[0],
                "online_score_raw_mean": float(np.mean(all_score_raw)),
                "online_score_raw_std": float(np.std(all_score_raw, ddof=0)),
                "online_score_raw_min": float(np.min(all_score_raw)),
                "online_score_raw_max": float(np.max(all_score_raw)),
                "online_score_drop_rel_session_mean": float(np.mean(all_score_drop)),
                "online_score_drop_rel_session_std": float(np.std(all_score_drop, ddof=0)),
                "online_score_drop_rel_session_min": float(np.min(all_score_drop)),
                "online_score_drop_rel_session_max": float(np.max(all_score_drop)),
                "online_drop20_prevalence": float(np.mean(all_score_drop < float(config.drop_threshold))),
                "n_valid_pairs": len(manifest_records),
                "lagged_raw_corr": safe_corr(anchor_score_raw, target_score_raw, epsilon=config.corr_epsilon),
                "persistence_mae": persistence_mae,
                "anchor_delta_corr": safe_corr(anchor_score_raw, delta_score_raw, epsilon=config.corr_epsilon),
                "delta_score_raw_mean": float(np.mean(delta_score_raw)),
                "delta_score_raw_std": float(np.std(delta_score_raw, ddof=0)),
                "future_drop20_state_prevalence": float(np.mean(target_drop20_state)),
                "future_drop20_onset_prevalence": float(np.mean(future_drop20_onset)),
            }
        )
    return records


def plot_mean_trajectories(sessions: list[OnlineSessionInputs], *, config: Step6Config) -> None:
    """Save pooled mean-trajectory plots for score_raw and score_drop_rel_session."""

    raw_matrix = np.stack([np.asarray(session.score_raw, dtype=np.float64) for session in sessions], axis=0)
    drop_matrix = np.stack([np.asarray(session.score_drop_rel_session, dtype=np.float64) for session in sessions], axis=0)
    x = np.arange(config.online_blocks, dtype=np.int16)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    raw_mean = np.mean(raw_matrix, axis=0)
    raw_std = np.std(raw_matrix, axis=0, ddof=0)
    axes[0].plot(x, raw_mean, color="tab:blue", linewidth=2)
    axes[0].fill_between(x, raw_mean - raw_std, raw_mean + raw_std, color="tab:blue", alpha=0.2)
    axes[0].set_ylabel("score_raw")
    axes[0].set_title("Pooled Online score_raw Trajectory")
    axes[0].grid(alpha=0.25)

    drop_mean = np.mean(drop_matrix, axis=0)
    drop_std = np.std(drop_matrix, axis=0, ddof=0)
    axes[1].plot(x, drop_mean, color="tab:red", linewidth=2)
    axes[1].fill_between(x, drop_mean - drop_std, drop_mean + drop_std, color="tab:red", alpha=0.2)
    axes[1].axhline(config.drop_threshold, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Online block position")
    axes[1].set_ylabel("score_drop_rel_session")
    axes[1].set_title("Pooled Online Relative-Drop Trajectory")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = figure_path(figure_root=config.figure_root, filename="step6_mean_trajectories.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_horizon_diagnostics(
    exploration_summary_records: list[dict[str, Any]],
    *,
    config: Step6Config,
) -> None:
    """Save horizon-diagnostic plots for the canonical and research labels."""

    horizons = np.asarray([record["horizon"] for record in exploration_summary_records], dtype=np.int16)
    lagged_raw_corr = np.asarray([record["lagged_raw_corr"] for record in exploration_summary_records], dtype=np.float64)
    persistence_mae = np.asarray([record["persistence_mae"] for record in exploration_summary_records], dtype=np.float64)
    anchor_delta_corr = np.asarray([record["anchor_delta_corr"] for record in exploration_summary_records], dtype=np.float64)
    future_drop20_state_prev = np.asarray(
        [record["future_drop20_state_prevalence"] for record in exploration_summary_records],
        dtype=np.float64,
    )
    future_drop20_onset_prev = np.asarray(
        [record["future_drop20_onset_prevalence"] for record in exploration_summary_records],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(horizons, lagged_raw_corr, marker="o", color="tab:blue")
    axes[0, 0].set_title("Endpoint Correlation")
    axes[0, 0].set_xlabel("Future-block horizon h")
    axes[0, 0].set_ylabel("corr(anchor_raw, target_raw)")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(horizons, persistence_mae, marker="o", color="tab:orange")
    axes[0, 1].set_title("Persistence Baseline Error")
    axes[0, 1].set_xlabel("Future-block horizon h")
    axes[0, 1].set_ylabel("MAE(target_raw - anchor_raw)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(horizons, anchor_delta_corr, marker="o", color="tab:green")
    axes[1, 0].set_title("Delta Predictability")
    axes[1, 0].set_xlabel("Future-block horizon h")
    axes[1, 0].set_ylabel("corr(anchor_raw, delta_raw)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(horizons, future_drop20_state_prev, marker="o", color="tab:red", label="endpoint state")
    axes[1, 1].plot(horizons, future_drop20_onset_prev, marker="s", color="tab:purple", label="onset")
    axes[1, 1].set_title("Clinical Event Prevalence")
    axes[1, 1].set_xlabel("Future-block horizon h")
    axes[1, 1].set_ylabel("Prevalence")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Step 6 Block-Based Horizon Diagnostics", y=1.02)
    fig.tight_layout()
    path = figure_path(figure_root=config.figure_root, filename="step6_horizon_diagnostics.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_block_position_summary(
    block_position_records: list[dict[str, Any]],
    *,
    config: Step6Config,
) -> None:
    """Save a pooled block-position summary plot."""

    block_pos = np.asarray([record["online_block_pos"] for record in block_position_records], dtype=np.int16)
    score_raw_mean = np.asarray([record["score_raw_mean"] for record in block_position_records], dtype=np.float64)
    score_drop_mean = np.asarray(
        [record["score_drop_rel_session_mean"] for record in block_position_records],
        dtype=np.float64,
    )
    drop20_prev = np.asarray([record["drop20_prevalence"] for record in block_position_records], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(block_pos, score_raw_mean, color="tab:blue", linewidth=2, label="score_raw mean")
    axes[0].plot(block_pos, score_drop_mean, color="tab:red", linewidth=2, label="drop mean")
    axes[0].axhline(config.drop_threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean value")
    axes[0].set_title("Pooled Block-Position Means")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(block_pos, drop20_prev, color="tab:purple", linewidth=2)
    axes[1].set_xlabel("Online block position")
    axes[1].set_ylabel("drop20 prevalence")
    axes[1].set_title("Pooled Clinical Event Prevalence")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = figure_path(figure_root=config.figure_root, filename="step6_block_position_summary.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_figures(
    *,
    sessions: list[OnlineSessionInputs],
    block_position_records: list[dict[str, Any]],
    exploration_summary_records: list[dict[str, Any]],
    config: Step6Config,
) -> None:
    """Save all required Step 6 figures."""

    plot_mean_trajectories(sessions, config=config)
    plot_horizon_diagnostics(exploration_summary_records, config=config)
    plot_block_position_summary(block_position_records, config=config)


def validate_csv_against_expected(
    *,
    path: Path,
    columns: tuple[str, ...],
    expected_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Validate one saved CSV exactly against expected records."""

    fieldnames, rows = load_csv_rows(path)
    if fieldnames != list(columns):
        raise ValueError(f"{path.name} columns {fieldnames}, expected {list(columns)}")
    expected_rows = serialize_records(expected_records, columns)
    if rows != expected_rows:
        raise ValueError(f"{path.name} contents do not match exact recomputation")
    return rows


def smoke_test_outputs(config: Step6Config) -> Step6Summary:
    """Run dataset-wide smoke tests across all Step 6 outputs."""

    validate_config(config)
    session_pairs = expected_session_ids()
    sessions = [
        load_online_session_inputs(subject_id=subject_id, session_id=session_id, config=config)
        for subject_id, session_id in session_pairs
    ]
    n_sessions = len(sessions)

    session_summary_records = build_session_summary_records(sessions, config=config)
    block_position_records = build_block_position_summary_records(sessions, config=config)
    manifest_records_by_horizon = build_manifest_records_by_horizon(sessions, config=config)
    task_summary_records = build_task_summary_records(
        manifest_records_by_horizon,
        n_sessions=n_sessions,
        config=config,
    )
    exploration_summary_records = build_exploration_summary_records(
        sessions,
        manifest_records_by_horizon=manifest_records_by_horizon,
        config=config,
    )

    session_summary_rows = validate_csv_against_expected(
        path=session_summary_path(output_root=config.output_root),
        columns=SESSION_SUMMARY_COLUMNS,
        expected_records=session_summary_records,
    )
    block_position_rows = validate_csv_against_expected(
        path=block_position_summary_path(output_root=config.output_root),
        columns=BLOCK_POSITION_SUMMARY_COLUMNS,
        expected_records=block_position_records,
    )
    exploration_rows = validate_csv_against_expected(
        path=exploration_summary_path(output_root=config.output_root),
        columns=EXPLORATION_SUMMARY_COLUMNS,
        expected_records=exploration_summary_records,
    )
    task_summary_rows = validate_csv_against_expected(
        path=task_summary_path(output_root=config.output_root),
        columns=TASK_SUMMARY_COLUMNS,
        expected_records=task_summary_records,
    )

    manifest_rows_by_horizon: list[int] = []
    usable_rows_by_horizon: list[int] = []
    total_manifest_rows = 0
    total_usable_rows_default_context = 0
    for horizon in config.horizons:
        records = manifest_records_by_horizon[horizon]
        rows = validate_csv_against_expected(
            path=manifest_path(output_root=config.output_root, horizon=horizon),
            columns=MANIFEST_COLUMNS,
            expected_records=records,
        )
        expected_rows = expected_manifest_rows_per_session(horizon=horizon, config=config) * n_sessions
        if len(rows) != expected_rows:
            raise ValueError(f"h={horizon} manifest has {len(rows)} rows, expected {expected_rows}")
        usable_rows = sum(int(row["anchor_idx"]) >= config.default_context_length - 1 for row in rows)
        expected_usable_rows = expected_default_usable_rows_per_session(horizon=horizon, config=config) * n_sessions
        if usable_rows != expected_usable_rows:
            raise ValueError(
                f"h={horizon} usable rows for default context length {config.default_context_length}: "
                f"{usable_rows}, expected {expected_usable_rows}",
            )
        manifest_rows_by_horizon.append(len(rows))
        usable_rows_by_horizon.append(usable_rows)
        total_manifest_rows += len(rows)
        total_usable_rows_default_context += usable_rows

    figure_count = 0
    for filename in EXPECTED_FIGURE_FILENAMES:
        path = figure_path(figure_root=config.figure_root, filename=filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step 6 figure: {path}")
        figure_count += 1

    session_ks = [session.session_k for session in sessions]
    return Step6Summary(
        output_root=config.output_root,
        figure_root=config.figure_root,
        session_count=n_sessions,
        session_summary_rows=len(session_summary_rows),
        block_position_rows=len(block_position_rows),
        exploration_rows=len(exploration_rows),
        task_summary_rows=len(task_summary_rows),
        manifest_file_count=len(config.horizons),
        total_manifest_rows=total_manifest_rows,
        total_usable_rows_default_context=total_usable_rows_default_context,
        figure_count=figure_count,
        session_k_min=min(session_ks),
        session_k_max=max(session_ks),
        horizons=tuple(int(h) for h in config.horizons),
        manifest_rows_by_horizon=tuple(manifest_rows_by_horizon),
        usable_rows_by_horizon=tuple(usable_rows_by_horizon),
    )


def log_summary(summary: Step6Summary) -> None:
    """Log the headline Step 6 metrics."""

    horizon_counts = ", ".join(
        f"h={h}:{rows} rows/{usable} usable"
        for h, rows, usable in zip(
            summary.horizons,
            summary.manifest_rows_by_horizon,
            summary.usable_rows_by_horizon,
            strict=True,
        )
    )
    logger.info(
        "Step 6 complete. Sessions=%d, manifests=%d, total rows=%d, default-context usable rows=%d.",
        summary.session_count,
        summary.manifest_file_count,
        summary.total_manifest_rows,
        summary.total_usable_rows_default_context,
    )
    logger.info(
        "Session K range=%d..%d. Session summary rows=%d, block summary rows=%d, exploration rows=%d, task rows=%d.",
        summary.session_k_min,
        summary.session_k_max,
        summary.session_summary_rows,
        summary.block_position_rows,
        summary.exploration_rows,
        summary.task_summary_rows,
    )
    logger.info("Per-horizon row counts: %s.", horizon_counts)


def run_step6(config: Step6Config | None = None) -> Step6Summary:
    """Run the full Step 6 forecasting-task pipeline."""

    config = config or Step6Config()
    validate_config(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.figure_root.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: Loading and validating Step 4/5 online inputs ...")
    sessions = [
        load_online_session_inputs(subject_id=subject_id, session_id=session_id, config=config)
        for subject_id, session_id in expected_session_ids()
    ]
    logger.info("Phase 1 complete: %d sessions loaded.", len(sessions))

    logger.info("Phase 2: Building summaries and manifests ...")
    session_summary_records = build_session_summary_records(sessions, config=config)
    block_position_records = build_block_position_summary_records(sessions, config=config)
    manifest_records_by_horizon = build_manifest_records_by_horizon(sessions, config=config)
    task_summary_records = build_task_summary_records(
        manifest_records_by_horizon,
        n_sessions=len(sessions),
        config=config,
    )
    exploration_summary_records = build_exploration_summary_records(
        sessions,
        manifest_records_by_horizon=manifest_records_by_horizon,
        config=config,
    )

    write_csv(
        session_summary_path(output_root=config.output_root),
        columns=SESSION_SUMMARY_COLUMNS,
        records=session_summary_records,
    )
    write_csv(
        block_position_summary_path(output_root=config.output_root),
        columns=BLOCK_POSITION_SUMMARY_COLUMNS,
        records=block_position_records,
    )
    write_csv(
        exploration_summary_path(output_root=config.output_root),
        columns=EXPLORATION_SUMMARY_COLUMNS,
        records=exploration_summary_records,
    )
    write_csv(
        task_summary_path(output_root=config.output_root),
        columns=TASK_SUMMARY_COLUMNS,
        records=task_summary_records,
    )
    for horizon in config.horizons:
        write_csv(
            manifest_path(output_root=config.output_root, horizon=horizon),
            columns=MANIFEST_COLUMNS,
            records=manifest_records_by_horizon[horizon],
        )
    save_figures(
        sessions=sessions,
        block_position_records=block_position_records,
        exploration_summary_records=exploration_summary_records,
        config=config,
    )

    logger.info("Phase 2 complete. Running smoke tests ...")
    summary = smoke_test_outputs(config)
    log_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Sub-Step 6.2b — Variance decomposition and detrended diagnostics (addendum)
# ---------------------------------------------------------------------------

VARIANCE_DECOMPOSITION_COLUMNS: tuple[str, ...] = (
    "var_total",
    "var_between_subject",
    "var_between_session_within_subject",
    "var_within_session",
    "share_between_subject",
    "share_between_session_within_subject",
    "share_within_session",
    "icc_subject",
)

DETRENDED_EXPLORATION_COLUMNS: tuple[str, ...] = (
    "horizon",
    "lagged_raw_corr",
    "detrended_lagged_corr",
    "persistence_mae",
    "detrended_persistence_mae",
)


def variance_decomposition_path(*, output_root: Path) -> Path:
    """Build the Step 6 variance-decomposition CSV path."""
    return output_root / "step6_variance_decomposition.csv"


def detrended_exploration_path(*, output_root: Path) -> Path:
    """Build the Step 6 detrended-exploration CSV path."""
    return output_root / "step6_detrended_exploration.csv"


def compute_variance_decomposition(
    sessions: list[OnlineSessionInputs],
) -> dict[str, float]:
    """Three-level nested variance decomposition of online score_raw.

    Partitions total variance into:
    - Between-subject: variance of subject-level means
    - Between-session-within-subject: mean of within-subject session-mean variances
    - Within-session: mean of within-session block-score variances

    Returns:
        Dict with var_total, three component variances, three shares, and ICC(1).
    """
    # Group sessions by subject
    subject_sessions: dict[str, list[OnlineSessionInputs]] = {}
    for session in sessions:
        subject_sessions.setdefault(session.subject_id, []).append(session)

    all_scores = np.concatenate(
        [np.asarray(s.score_raw, dtype=np.float64) for s in sessions], axis=0
    )
    var_total = float(np.var(all_scores, ddof=0))

    # Between-subject: variance of subject means
    subject_means = np.array(
        [
            np.mean(
                np.concatenate(
                    [np.asarray(s.score_raw, dtype=np.float64) for s in sess_list]
                )
            )
            for sess_list in subject_sessions.values()
        ],
        dtype=np.float64,
    )
    var_between_subject = float(np.var(subject_means, ddof=0))

    # Between-session-within-subject: for each subject, variance of session means
    within_subject_session_vars: list[float] = []
    for sess_list in subject_sessions.values():
        session_means = np.array(
            [float(np.mean(np.asarray(s.score_raw, dtype=np.float64))) for s in sess_list],
            dtype=np.float64,
        )
        within_subject_session_vars.append(float(np.var(session_means, ddof=0)))
    var_between_session_within_subject = float(np.mean(within_subject_session_vars))

    # Within-session: for each session, variance of block scores
    within_session_vars: list[float] = []
    for s in sessions:
        within_session_vars.append(
            float(np.var(np.asarray(s.score_raw, dtype=np.float64), ddof=0))
        )
    var_within_session = float(np.mean(within_session_vars))

    # Shares (normalize to sum to 1)
    var_sum = var_between_subject + var_between_session_within_subject + var_within_session
    share_between_subject = var_between_subject / var_sum if var_sum > 0 else 0.0
    share_between_session = var_between_session_within_subject / var_sum if var_sum > 0 else 0.0
    share_within_session = var_within_session / var_sum if var_sum > 0 else 0.0

    icc_subject = var_between_subject / var_total if var_total > 0 else 0.0

    return {
        "var_total": var_total,
        "var_between_subject": var_between_subject,
        "var_between_session_within_subject": var_between_session_within_subject,
        "var_within_session": var_within_session,
        "share_between_subject": share_between_subject,
        "share_between_session_within_subject": share_between_session,
        "share_within_session": share_within_session,
        "icc_subject": icc_subject,
    }


def compute_detrended_lagged_correlations(
    sessions: list[OnlineSessionInputs],
    *,
    config: Step6Config,
) -> list[dict[str, Any]]:
    """Recompute lagged correlations and persistence MAE on session-detrended scores.

    For each session, subtracts the session mean from score_raw.
    Then pools all (anchor, target) pairs per horizon and computes
    detrended_lagged_corr and detrended_persistence_mae.

    Also loads the original exploration summary to include original values
    for side-by-side comparison.

    Args:
        sessions: All 105 validated online sessions.
        config: Step 6 configuration.

    Returns:
        List of 5 dicts (one per horizon) with original and detrended diagnostics.
    """
    # Load original exploration summary for comparison values
    orig_path = exploration_summary_path(output_root=config.output_root)
    _, orig_rows = load_csv_rows(orig_path)
    orig_by_horizon: dict[int, dict[str, str]] = {
        int(row["horizon"]): row for row in orig_rows
    }

    records: list[dict[str, Any]] = []
    for horizon in config.horizons:
        # Collect detrended anchor/target pairs across all sessions
        anchors: list[float] = []
        targets: list[float] = []
        for session in sessions:
            raw = np.asarray(session.score_raw, dtype=np.float64)
            session_mean = float(np.mean(raw))
            detrended = raw - session_mean
            for target_idx in range(horizon, config.online_blocks):
                anchor_idx = target_idx - horizon
                anchors.append(float(detrended[anchor_idx]))
                targets.append(float(detrended[target_idx]))

        anchor_arr = np.array(anchors, dtype=np.float64)
        target_arr = np.array(targets, dtype=np.float64)
        detrended_lagged_corr = safe_corr(anchor_arr, target_arr, epsilon=config.corr_epsilon)
        detrended_persistence_mae = float(np.mean(np.abs(target_arr - anchor_arr)))

        orig_row = orig_by_horizon[horizon]
        records.append({
            "horizon": horizon,
            "lagged_raw_corr": float(orig_row["lagged_raw_corr"]),
            "detrended_lagged_corr": detrended_lagged_corr,
            "persistence_mae": float(orig_row["persistence_mae"]),
            "detrended_persistence_mae": detrended_persistence_mae,
        })
    return records


def plot_variance_decomposition(
    decomposition: dict[str, float],
    *,
    config: Step6Config,
) -> None:
    """Save a stacked bar chart of the three variance components."""
    shares = [
        decomposition["share_between_subject"],
        decomposition["share_between_session_within_subject"],
        decomposition["share_within_session"],
    ]
    labels = ["Between-subject", "Between-session\n(within-subject)", "Within-session"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bottom = 0.0
    for share, label, color in zip(shares, labels, colors):
        ax.bar("score_raw", share, bottom=bottom, label=label, color=color, width=0.5)
        if share > 0.03:
            ax.text(0, bottom + share / 2, f"{share:.1%}", ha="center", va="center", fontsize=10)
        bottom += share

    ax.set_ylabel("Share of total variance")
    ax.set_title("Variance Decomposition of Online score_raw")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = figure_path(figure_root=config.figure_root, filename="step6_variance_decomposition.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_detrended_vs_original_corr(
    detrended_records: list[dict[str, Any]],
    *,
    config: Step6Config,
) -> None:
    """Save a paired line plot of original vs detrended lagged correlations."""
    horizons = [r["horizon"] for r in detrended_records]
    original = [r["lagged_raw_corr"] for r in detrended_records]
    detrended = [r["detrended_lagged_corr"] for r in detrended_records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(horizons, original, "o-", color="tab:blue", linewidth=2, markersize=8, label="Original lagged corr")
    ax.plot(horizons, detrended, "s--", color="tab:red", linewidth=2, markersize=8, label="Detrended lagged corr")

    for h, o, d in zip(horizons, original, detrended):
        ax.annotate(f"{o:.3f}", (h, o), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        ax.annotate(f"{d:.3f}", (h, d), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)

    ax.set_xlabel("Horizon (blocks ahead)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Original vs Session-Detrended Lagged Correlations")
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = figure_path(figure_root=config.figure_root, filename="step6_detrended_vs_original_corr.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def run_step6_addendum(config: Step6Config | None = None) -> dict[str, Any]:
    """Run the Step 6.2b addendum: variance decomposition and detrended diagnostics.

    This can be run independently from the main Step 6 pipeline.
    It reuses the same session-loading logic from Sub-Step 6.1.

    Returns:
        Dict summarizing the addendum outputs.
    """
    config = config or Step6Config()
    validate_config(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.figure_root.mkdir(parents=True, exist_ok=True)

    logger.info("Step 6.2b addendum: Loading sessions ...")
    sessions = [
        load_online_session_inputs(subject_id=subject_id, session_id=session_id, config=config)
        for subject_id, session_id in expected_session_ids()
    ]
    logger.info("Loaded %d sessions.", len(sessions))

    # Variance decomposition
    logger.info("Computing variance decomposition ...")
    decomposition = compute_variance_decomposition(sessions)
    var_shares_sum = (
        decomposition["share_between_subject"]
        + decomposition["share_between_session_within_subject"]
        + decomposition["share_within_session"]
    )
    assert abs(var_shares_sum - 1.0) < 1e-10, (
        f"Variance shares must sum to 1.0, got {var_shares_sum}"
    )
    write_csv(
        variance_decomposition_path(output_root=config.output_root),
        columns=VARIANCE_DECOMPOSITION_COLUMNS,
        records=[decomposition],
    )
    logger.info(
        "Variance decomposition: between-subject=%.1f%%, between-session=%.1f%%, within-session=%.1f%%, ICC(1)=%.3f",
        decomposition["share_between_subject"] * 100,
        decomposition["share_between_session_within_subject"] * 100,
        decomposition["share_within_session"] * 100,
        decomposition["icc_subject"],
    )

    # Detrended lagged correlations
    logger.info("Computing detrended lagged correlations ...")
    detrended_records = compute_detrended_lagged_correlations(sessions, config=config)
    write_csv(
        detrended_exploration_path(output_root=config.output_root),
        columns=DETRENDED_EXPLORATION_COLUMNS,
        records=detrended_records,
    )
    for rec in detrended_records:
        logger.info(
            "  h=%d: original_corr=%.3f, detrended_corr=%.3f, original_mae=%.3f, detrended_mae=%.3f",
            rec["horizon"],
            rec["lagged_raw_corr"],
            rec["detrended_lagged_corr"],
            rec["persistence_mae"],
            rec["detrended_persistence_mae"],
        )

    # Figures
    logger.info("Saving addendum figures ...")
    plot_variance_decomposition(decomposition, config=config)
    plot_detrended_vs_original_corr(detrended_records, config=config)

    logger.info("Step 6.2b addendum complete.")
    return {
        "decomposition": decomposition,
        "detrended_records": detrended_records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Step 6."""

    parser = argparse.ArgumentParser(description="Run Step 6 forecasting-task definition and exploration.")
    parser.add_argument("--step4-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step5-root", type=Path, default=Path("data/scores"))
    parser.add_argument("--output-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/forecasting/step6"))
    parser.add_argument("--default-context-length", type=int, default=Step6Config().default_context_length)
    parser.add_argument("--drop-threshold", type=float, default=Step6Config().drop_threshold)
    parser.add_argument("--horizons", type=int, nargs="*", default=list(Step6Config().horizons))
    return parser


def main() -> None:
    """CLI entrypoint for Step 6."""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_step6(
        Step6Config(
            step4_root=args.step4_root,
            step5_root=args.step5_root,
            output_root=args.output_root,
            figure_root=args.figure_root,
            default_context_length=args.default_context_length,
            drop_threshold=args.drop_threshold,
            horizons=tuple(int(horizon) for horizon in args.horizons),
        )
    )


if __name__ == "__main__":
    main()
