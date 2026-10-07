"""Step 6c upstream audit for BCIAUT target formation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann

from src.data.loader import expected_session_ids
from src.features.step3_covariance import (
    Step3Config,
    _validate_saved_covariance_payload,
    feature_split_path as step3_feature_split_path,
)
from src.features.step4_riemannian import (
    METADATA_FIELDS,
    Step4Config,
    _validate_saved_aligned_payload,
    _validate_saved_tangent_payload,
    aligned_split_path,
    matrix_log_spd,
    ts_split_path,
    vectorize_upper_triangle,
)
from src.features.step5_scores import (
    Step5Config,
    _validate_saved_reference_payload,
    _validate_saved_score_payload,
    reference_path,
    score_split_path,
)
from src.features.step6_forecasting_task import (
    Step6Config,
    exploration_summary_path,
    load_csv_rows,
    safe_corr,
    variance_decomposition_path,
    write_csv,
)

logger = logging.getLogger(__name__)

TARGET_VIEWS: tuple[str, ...] = ("raw_level", "session_mean_centered", "first_difference")
STEP4_DISPERSION_COLUMNS: tuple[str, ...] = (
    "representation",
    "var_total",
    "var_between_subject",
    "var_between_session_within_subject",
    "var_within_session",
    "share_between_subject",
    "share_between_session_within_subject",
    "share_within_session",
    "session_identity_share",
)
STEP4_SESSION_REFERENCE_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "session_id",
    "session_k",
    "raw_online_mean_to_session_ref",
    "aligned_online_mean_to_identity",
    "raw_online_block_dispersion",
    "aligned_online_block_dispersion",
    "raw_online_feature_var",
    "aligned_online_feature_var",
    "aligned_calibration_identity_error",
)
STEP5_SESSION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "session_id",
    "session_k",
    "ref_separation_riemann",
    "session_cal_mean_raw",
    "cal_mean_score_raw",
    "online_mean_score_raw",
    "cal_std_score_raw",
    "online_std_score_raw",
    "cal_mean_d_target",
    "online_mean_d_target",
    "cal_mean_d_nontarget",
    "online_mean_d_nontarget",
    "online_mdm_accuracy",
    "score_roundtrip_error_max",
    "corr_centered_score_vs_centered_d_target",
    "corr_centered_score_vs_centered_d_nontarget",
)
STEP5_SCORE_DECOMPOSITION_COLUMNS: tuple[str, ...] = (
    "max_score_roundtrip_error",
    "corr_online_mean_score_vs_cal_mean_score",
    "corr_online_mean_score_vs_session_cal_mean_raw",
    "corr_online_mean_score_vs_ref_separation",
    "corr_online_mean_score_vs_online_mdm_accuracy",
    "corr_online_mean_score_vs_session_k",
    "corr_online_std_score_vs_session_cal_mean_raw",
    "corr_online_std_score_vs_ref_separation",
    "corr_online_std_score_vs_online_mdm_accuracy",
    "corr_online_std_score_vs_session_k",
    "corr_cal_mean_d_target_vs_online_mean_d_target",
    "corr_cal_mean_d_nontarget_vs_online_mean_d_nontarget",
    "corr_online_mean_score_vs_online_mean_d_target",
    "corr_online_mean_score_vs_online_mean_d_nontarget",
    "corr_centered_score_vs_centered_d_target",
    "corr_centered_score_vs_centered_d_nontarget",
)
STEP6_TARGET_VARIANCE_COLUMNS: tuple[str, ...] = (
    "target_view",
    "var_total",
    "var_between_subject",
    "var_between_session_within_subject",
    "var_within_session",
    "share_between_subject",
    "share_between_session_within_subject",
    "share_within_session",
    "subject_separability",
    "session_separability",
)
STEP6_TARGET_LAGGED_COLUMNS: tuple[str, ...] = (
    "target_view",
    "horizon",
    "lagged_corr",
    "persistence_mae",
)
HYPOTHESIS_COLUMNS: tuple[str, ...] = (
    "hypothesis",
    "judgment",
    "primary_evidence",
    "secondary_evidence",
)


@dataclass(frozen=True)
class Step6AuditConfig:
    """Configuration for the read-only upstream target audit."""

    step3_root: Path = Path("data/features")
    step4_root: Path = Path("data/features")
    step5_root: Path = Path("data/scores")
    step6_root: Path = Path("data/forecasting")
    output_root: Path = Path("data/forecasting")
    figure_root: Path = Path("outputs/forecasting/step6c")
    horizons: tuple[int, ...] = Step6Config().horizons
    calibration_blocks: int = Step6Config().calibration_blocks
    online_blocks: int = Step6Config().online_blocks
    calibration_social_start: int = Step6Config().calibration_social_start
    covariance_dim: int = 24
    tangent_dim: int = 300
    corr_epsilon: float = Step6Config().corr_epsilon
    step4_identity_share_increase_threshold: float = 0.10
    step4_within_share_drop_threshold: float = 0.10
    step5_calibration_link_threshold: float = 0.80
    step5_reference_link_threshold: float = 0.60
    step6_raw_corr_threshold: float = 0.60
    step6_centered_corr_ceiling: float = 0.15
    step6_session_identity_share_threshold: float = 0.70


@dataclass(frozen=True)
class AuditSessionInputs:
    """All validated Step 3-5 artifacts needed for one subject-session audit."""

    subject_id: str
    session_id: str
    session_k: int
    step3_online_covs: np.ndarray
    step4_calibration_aligned_covs: np.ndarray
    step4_online_aligned_covs: np.ndarray
    step4_online_features: np.ndarray
    session_ref_mean: np.ndarray
    score_raw_cal: np.ndarray
    score_raw_online: np.ndarray
    d_target_cal: np.ndarray
    d_nontarget_cal: np.ndarray
    d_target_online: np.ndarray
    d_nontarget_online: np.ndarray
    online_mdm_accuracy: float
    session_cal_mean_raw: float
    ref_separation_riemann: float


@dataclass(frozen=True)
class Step6AuditSummary:
    """High-level summary returned by the audit runner."""

    session_count: int
    step4_dispersion_rows: int
    step5_session_rows: int
    step6_target_variance_rows: int
    step6_target_lagged_rows: int
    hypothesis_rows: int
    conclusion_path: Path


def step4_dispersion_path(*, output_root: Path) -> Path:
    return output_root / "step6c_step4_dispersion_summary.csv"


def step4_session_reference_path(*, output_root: Path) -> Path:
    return output_root / "step6c_step4_session_reference.csv"


def step5_session_diagnostic_path(*, output_root: Path) -> Path:
    return output_root / "step6c_step5_session_diagnostics.csv"


def step5_score_decomposition_path(*, output_root: Path) -> Path:
    return output_root / "step6c_step5_score_decomposition.csv"


def step6_target_variance_path(*, output_root: Path) -> Path:
    return output_root / "step6c_step6_target_variance.csv"


def step6_target_lagged_path(*, output_root: Path) -> Path:
    return output_root / "step6c_step6_target_lagged_metrics.csv"


def hypothesis_judgment_path(*, output_root: Path) -> Path:
    return output_root / "step6c_hypothesis_judgments.csv"


def conclusion_markdown_path(*, figure_root: Path) -> Path:
    return figure_root / "step6c_upstream_target_audit.md"


def validate_config(config: Step6AuditConfig) -> None:
    """Validate the fixed audit configuration."""

    if not config.horizons:
        raise ValueError("At least one horizon is required")
    if len(set(config.horizons)) != len(config.horizons):
        raise ValueError(f"Duplicate horizons are not allowed: {config.horizons}")
    for horizon in config.horizons:
        if horizon <= 0 or horizon >= config.online_blocks:
            raise ValueError(
                f"Horizon {horizon} is invalid for online_blocks={config.online_blocks}; "
                f"expected 1..{config.online_blocks - 1}",
            )
    if config.corr_epsilon <= 0.0:
        raise ValueError("corr_epsilon must be positive")


def step3_validation_config(config: Step6AuditConfig) -> Step3Config:
    return Step3Config(output_root=config.step3_root)


def step4_validation_config(config: Step6AuditConfig) -> Step4Config:
    return Step4Config(
        input_root=config.step3_root,
        output_root=config.step4_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        covariance_dim=config.covariance_dim,
        tangent_dim=config.tangent_dim,
    )


def step5_validation_config(config: Step6AuditConfig) -> Step5Config:
    return Step5Config(
        output_root=config.step5_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
    )


def _metadata_match(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    context: str,
) -> None:
    for field_name in METADATA_FIELDS:
        if not np.array_equal(np.asarray(left[field_name]), np.asarray(right[field_name])):
            raise ValueError(f"{context} metadata mismatch for field {field_name!r}")


def _session_k_from_online_metadata(payload: dict[str, np.ndarray]) -> int:
    runs_per_block = np.asarray(payload["runs_per_block"], dtype=np.int16)
    unique_runs = np.unique(runs_per_block)
    if unique_runs.size != 1:
        raise ValueError("Online runs_per_block is not constant within session")
    return int(unique_runs[0])


def _max_score_roundtrip_error(
    score_raw: np.ndarray,
    d_target: np.ndarray,
    d_nontarget: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(score_raw, dtype=np.float64)
                - (np.asarray(d_nontarget, dtype=np.float64) - np.asarray(d_target, dtype=np.float64))
            )
        )
    )


def load_audit_sessions(config: Step6AuditConfig) -> list[AuditSessionInputs]:
    """Load all validated session artifacts required by the audit."""

    sessions: list[AuditSessionInputs] = []
    step3_config = step3_validation_config(config)
    step4_config = step4_validation_config(config)
    step5_config = step5_validation_config(config)

    for subject_id, session_id in expected_session_ids():
        step3_online_payload = _validate_saved_covariance_payload(
            path=step3_feature_split_path(
                output_root=config.step3_root,
                subject_id=subject_id,
                session_id=session_id,
                split="online",
            ),
            split="online",
            config=step3_config,
        )
        step4_cal_payload = _validate_saved_aligned_payload(
            path=aligned_split_path(
                output_root=config.step4_root,
                subject_id=subject_id,
                session_id=session_id,
                split="calibration",
            ),
            split="calibration",
            config=step4_config,
        )
        step4_online_payload = _validate_saved_aligned_payload(
            path=aligned_split_path(
                output_root=config.step4_root,
                subject_id=subject_id,
                session_id=session_id,
                split="online",
            ),
            split="online",
            config=step4_config,
        )
        step4_tangent_payload = _validate_saved_tangent_payload(
            path=ts_split_path(
                output_root=config.step4_root,
                subject_id=subject_id,
                session_id=session_id,
                split="online",
            ),
            split="online",
            config=step4_config,
        )
        step5_cal_payload = _validate_saved_score_payload(
            path=score_split_path(
                output_root=config.step5_root,
                subject_id=subject_id,
                session_id=session_id,
                split="calibration",
            ),
            split="calibration",
            config=step5_config,
        )
        step5_online_payload = _validate_saved_score_payload(
            path=score_split_path(
                output_root=config.step5_root,
                subject_id=subject_id,
                session_id=session_id,
                split="online",
            ),
            split="online",
            config=step5_config,
        )
        step5_reference_payload = _validate_saved_reference_payload(
            path=reference_path(output_root=config.step5_root, subject_id=subject_id, session_id=session_id),
            config=step5_config,
        )

        _metadata_match(step4_online_payload, step4_tangent_payload, context=f"{subject_id}/{session_id} Step 4")
        _metadata_match(step4_online_payload, step5_online_payload, context=f"{subject_id}/{session_id} Step 4/5")
        _metadata_match(step4_cal_payload, step5_cal_payload, context=f"{subject_id}/{session_id} Step 4/5 calibration")

        session_ref_mean = np.asarray(step4_online_payload["session_ref_mean"], dtype=np.float64)
        if not np.array_equal(session_ref_mean, np.asarray(step4_cal_payload["session_ref_mean"], dtype=np.float64)):
            raise ValueError(f"{subject_id}/{session_id} session_ref_mean mismatch across Step 4 splits")
        if not np.array_equal(session_ref_mean, np.asarray(step4_tangent_payload["session_ref_mean"], dtype=np.float64)):
            raise ValueError(f"{subject_id}/{session_id} session_ref_mean mismatch across Step 4 outputs")

        mdm_correct_online = np.asarray(step5_online_payload["mdm_correct"], dtype=np.int8)
        sessions.append(
            AuditSessionInputs(
                subject_id=subject_id,
                session_id=session_id,
                session_k=_session_k_from_online_metadata(step4_online_payload),
                step3_online_covs=np.asarray(step3_online_payload["covs"], dtype=np.float64),
                step4_calibration_aligned_covs=np.asarray(step4_cal_payload["covs"], dtype=np.float64),
                step4_online_aligned_covs=np.asarray(step4_online_payload["covs"], dtype=np.float64),
                step4_online_features=np.asarray(step4_tangent_payload["features"], dtype=np.float32),
                session_ref_mean=session_ref_mean,
                score_raw_cal=np.asarray(step5_cal_payload["score_raw"], dtype=np.float32),
                score_raw_online=np.asarray(step5_online_payload["score_raw"], dtype=np.float32),
                d_target_cal=np.asarray(step5_cal_payload["d_target"], dtype=np.float32),
                d_nontarget_cal=np.asarray(step5_cal_payload["d_nontarget"], dtype=np.float32),
                d_target_online=np.asarray(step5_online_payload["d_target"], dtype=np.float32),
                d_nontarget_online=np.asarray(step5_online_payload["d_nontarget"], dtype=np.float32),
                online_mdm_accuracy=float(np.mean(mdm_correct_online)),
                session_cal_mean_raw=float(np.asarray(step5_online_payload["session_cal_mean_raw"], dtype=np.float64)),
                ref_separation_riemann=float(np.asarray(step5_reference_payload["ref_separation_riemann"], dtype=np.float64)),
            )
        )

    return sessions


def raw_log_features_from_covariances(
    covariances: np.ndarray,
    *,
    config: Step6AuditConfig,
) -> np.ndarray:
    """Vectorize raw Step 3 covariances around the identity for pre-alignment comparison."""

    step4_config = step4_validation_config(config)
    covariances = np.asarray(covariances, dtype=np.float64)
    features = np.zeros((covariances.shape[0], config.tangent_dim), dtype=np.float32)
    for index, covariance in enumerate(covariances):
        tangent = matrix_log_spd(covariance, config=step4_config, context=f"raw covariance[{index}]")
        features[index] = vectorize_upper_triangle(tangent, config=step4_config).astype(np.float32)
    return features


def _group_session_arrays(
    sessions: list[AuditSessionInputs],
    *,
    extractor: Callable[[AuditSessionInputs], np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for session in sessions:
        grouped.setdefault(session.subject_id, {})[session.session_id] = np.asarray(extractor(session), dtype=np.float64)
    return grouped


def compute_nested_dispersion(grouped: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    """Compute nested variance shares for 1-D or 2-D session arrays."""

    all_arrays = [
        arr.reshape(arr.shape[0], -1).astype(np.float64)
        for session_map in grouped.values()
        for arr in session_map.values()
    ]
    if not all_arrays:
        raise ValueError("At least one session array is required")

    all_values = np.concatenate(all_arrays, axis=0)
    var_total = float(np.mean(np.var(all_values, axis=0, ddof=0)))

    subject_means = np.stack(
        [
            np.mean(
                np.concatenate(
                    [arr.reshape(arr.shape[0], -1).astype(np.float64) for arr in session_map.values()],
                    axis=0,
                ),
                axis=0,
            )
            for session_map in grouped.values()
        ],
        axis=0,
    )
    var_between_subject = float(np.mean(np.var(subject_means, axis=0, ddof=0)))

    within_subject_session_vars: list[float] = []
    within_session_vars: list[float] = []
    for session_map in grouped.values():
        session_arrays = [arr.reshape(arr.shape[0], -1).astype(np.float64) for arr in session_map.values()]
        session_means = np.stack([np.mean(arr, axis=0) for arr in session_arrays], axis=0)
        within_subject_session_vars.append(float(np.mean(np.var(session_means, axis=0, ddof=0))))
        within_session_vars.extend(float(np.mean(np.var(arr, axis=0, ddof=0))) for arr in session_arrays)

    var_between_session_within_subject = float(np.mean(within_subject_session_vars))
    var_within_session = float(np.mean(within_session_vars))
    var_sum = var_between_subject + var_between_session_within_subject + var_within_session
    if var_sum <= 0.0:
        raise ValueError("Dispersion components must sum to a positive value")

    share_between_subject = var_between_subject / var_sum
    share_between_session = var_between_session_within_subject / var_sum
    share_within_session = var_within_session / var_sum

    return {
        "var_total": var_total,
        "var_between_subject": var_between_subject,
        "var_between_session_within_subject": var_between_session_within_subject,
        "var_within_session": var_within_session,
        "share_between_subject": share_between_subject,
        "share_between_session_within_subject": share_between_session,
        "share_within_session": share_within_session,
        "session_identity_share": share_between_subject + share_between_session,
    }


def build_step4_dispersion_records(
    sessions: list[AuditSessionInputs],
    *,
    config: Step6AuditConfig,
) -> list[dict[str, Any]]:
    """Compare pre-alignment raw-log variability against Step 4 aligned tangent variability."""

    pre_grouped = _group_session_arrays(sessions, extractor=lambda s: raw_log_features_from_covariances(s.step3_online_covs, config=config))
    post_grouped = _group_session_arrays(sessions, extractor=lambda s: s.step4_online_features)
    return [
        {"representation": "pre_alignment_raw_log", **compute_nested_dispersion(pre_grouped)},
        {"representation": "post_alignment_tangent", **compute_nested_dispersion(post_grouped)},
    ]


def _riemann_block_dispersion(covariances: np.ndarray, mean_covariance: np.ndarray) -> float:
    distances = [float(distance_riemann(covariance, mean_covariance)) for covariance in np.asarray(covariances, dtype=np.float64)]
    return float(np.mean(distances))


def build_step4_session_reference_records(
    sessions: list[AuditSessionInputs],
    *,
    config: Step6AuditConfig,
) -> list[dict[str, Any]]:
    """Build one row per session describing how Step 4 references anchor online geometry."""

    identity = np.eye(config.covariance_dim, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for session in sessions:
        raw_online_mean = mean_riemann(np.asarray(session.step3_online_covs, dtype=np.float64))
        raw_online_mean = 0.5 * (raw_online_mean + raw_online_mean.T)
        aligned_online_mean = mean_riemann(np.asarray(session.step4_online_aligned_covs, dtype=np.float64))
        aligned_online_mean = 0.5 * (aligned_online_mean + aligned_online_mean.T)
        aligned_calibration_mean = mean_riemann(np.asarray(session.step4_calibration_aligned_covs, dtype=np.float64))
        aligned_calibration_mean = 0.5 * (aligned_calibration_mean + aligned_calibration_mean.T)

        raw_features = raw_log_features_from_covariances(session.step3_online_covs, config=config)
        records.append(
            {
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "session_k": session.session_k,
                "raw_online_mean_to_session_ref": float(distance_riemann(raw_online_mean, session.session_ref_mean)),
                "aligned_online_mean_to_identity": float(distance_riemann(aligned_online_mean, identity)),
                "raw_online_block_dispersion": _riemann_block_dispersion(session.step3_online_covs, raw_online_mean),
                "aligned_online_block_dispersion": _riemann_block_dispersion(session.step4_online_aligned_covs, aligned_online_mean),
                "raw_online_feature_var": float(np.mean(np.var(np.asarray(raw_features, dtype=np.float64), axis=0, ddof=0))),
                "aligned_online_feature_var": float(
                    np.mean(np.var(np.asarray(session.step4_online_features, dtype=np.float64), axis=0, ddof=0))
                ),
                "aligned_calibration_identity_error": float(np.linalg.norm(aligned_calibration_mean - identity, ord="fro")),
            }
        )
    return records


def build_step5_session_diagnostics(
    sessions: list[AuditSessionInputs],
    *,
    config: Step6AuditConfig,
) -> list[dict[str, Any]]:
    """Build per-session score-construction diagnostics."""

    records: list[dict[str, Any]] = []
    for session in sessions:
        centered_score = np.asarray(session.score_raw_online, dtype=np.float64) - float(np.mean(session.score_raw_online))
        centered_d_target = np.asarray(session.d_target_online, dtype=np.float64) - float(np.mean(session.d_target_online))
        centered_d_nontarget = np.asarray(session.d_nontarget_online, dtype=np.float64) - float(np.mean(session.d_nontarget_online))
        records.append(
            {
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "session_k": session.session_k,
                "ref_separation_riemann": session.ref_separation_riemann,
                "session_cal_mean_raw": session.session_cal_mean_raw,
                "cal_mean_score_raw": float(np.mean(np.asarray(session.score_raw_cal, dtype=np.float64))),
                "online_mean_score_raw": float(np.mean(np.asarray(session.score_raw_online, dtype=np.float64))),
                "cal_std_score_raw": float(np.std(np.asarray(session.score_raw_cal, dtype=np.float64), ddof=0)),
                "online_std_score_raw": float(np.std(np.asarray(session.score_raw_online, dtype=np.float64), ddof=0)),
                "cal_mean_d_target": float(np.mean(np.asarray(session.d_target_cal, dtype=np.float64))),
                "online_mean_d_target": float(np.mean(np.asarray(session.d_target_online, dtype=np.float64))),
                "cal_mean_d_nontarget": float(np.mean(np.asarray(session.d_nontarget_cal, dtype=np.float64))),
                "online_mean_d_nontarget": float(np.mean(np.asarray(session.d_nontarget_online, dtype=np.float64))),
                "online_mdm_accuracy": session.online_mdm_accuracy,
                "score_roundtrip_error_max": _max_score_roundtrip_error(
                    session.score_raw_online,
                    session.d_target_online,
                    session.d_nontarget_online,
                ),
                "corr_centered_score_vs_centered_d_target": safe_corr(
                    centered_score,
                    centered_d_target,
                    epsilon=config.corr_epsilon,
                ),
                "corr_centered_score_vs_centered_d_nontarget": safe_corr(
                    centered_score,
                    centered_d_nontarget,
                    epsilon=config.corr_epsilon,
                ),
            }
        )
    return records


def build_step5_score_decomposition_summary(
    session_records: list[dict[str, Any]],
    sessions: list[AuditSessionInputs],
    *,
    config: Step6AuditConfig,
) -> dict[str, Any]:
    """Summarize how Step 5 references and distances shape the saved raw score."""

    def series(column: str) -> np.ndarray:
        return np.asarray([float(record[column]) for record in session_records], dtype=np.float64)

    centered_score = np.concatenate(
        [
            np.asarray(session.score_raw_online, dtype=np.float64) - float(np.mean(session.score_raw_online))
            for session in sessions
        ]
    )
    centered_d_target = np.concatenate(
        [
            np.asarray(session.d_target_online, dtype=np.float64) - float(np.mean(session.d_target_online))
            for session in sessions
        ]
    )
    centered_d_nontarget = np.concatenate(
        [
            np.asarray(session.d_nontarget_online, dtype=np.float64) - float(np.mean(session.d_nontarget_online))
            for session in sessions
        ]
    )

    return {
        "max_score_roundtrip_error": float(max(float(record["score_roundtrip_error_max"]) for record in session_records)),
        "corr_online_mean_score_vs_cal_mean_score": safe_corr(series("online_mean_score_raw"), series("cal_mean_score_raw"), epsilon=config.corr_epsilon),
        "corr_online_mean_score_vs_session_cal_mean_raw": safe_corr(series("online_mean_score_raw"), series("session_cal_mean_raw"), epsilon=config.corr_epsilon),
        "corr_online_mean_score_vs_ref_separation": safe_corr(series("online_mean_score_raw"), series("ref_separation_riemann"), epsilon=config.corr_epsilon),
        "corr_online_mean_score_vs_online_mdm_accuracy": safe_corr(series("online_mean_score_raw"), series("online_mdm_accuracy"), epsilon=config.corr_epsilon),
        "corr_online_mean_score_vs_session_k": safe_corr(series("online_mean_score_raw"), series("session_k"), epsilon=config.corr_epsilon),
        "corr_online_std_score_vs_session_cal_mean_raw": safe_corr(series("online_std_score_raw"), series("session_cal_mean_raw"), epsilon=config.corr_epsilon),
        "corr_online_std_score_vs_ref_separation": safe_corr(series("online_std_score_raw"), series("ref_separation_riemann"), epsilon=config.corr_epsilon),
        "corr_online_std_score_vs_online_mdm_accuracy": safe_corr(series("online_std_score_raw"), series("online_mdm_accuracy"), epsilon=config.corr_epsilon),
        "corr_online_std_score_vs_session_k": safe_corr(series("online_std_score_raw"), series("session_k"), epsilon=config.corr_epsilon),
        "corr_cal_mean_d_target_vs_online_mean_d_target": safe_corr(series("cal_mean_d_target"), series("online_mean_d_target"), epsilon=config.corr_epsilon),
        "corr_cal_mean_d_nontarget_vs_online_mean_d_nontarget": safe_corr(series("cal_mean_d_nontarget"), series("online_mean_d_nontarget"), epsilon=config.corr_epsilon),
        "corr_online_mean_score_vs_online_mean_d_target": safe_corr(series("online_mean_score_raw"), series("online_mean_d_target"), epsilon=config.corr_epsilon),
        "corr_online_mean_score_vs_online_mean_d_nontarget": safe_corr(series("online_mean_score_raw"), series("online_mean_d_nontarget"), epsilon=config.corr_epsilon),
        "corr_centered_score_vs_centered_d_target": safe_corr(centered_score, centered_d_target, epsilon=config.corr_epsilon),
        "corr_centered_score_vs_centered_d_nontarget": safe_corr(centered_score, centered_d_nontarget, epsilon=config.corr_epsilon),
    }


def target_view_series(score_raw: np.ndarray, *, target_view: str) -> np.ndarray:
    """Transform one raw score trajectory into an audit-only target view."""

    score_raw = np.asarray(score_raw, dtype=np.float64)
    if target_view == "raw_level":
        return score_raw
    if target_view == "session_mean_centered":
        return score_raw - float(np.mean(score_raw))
    if target_view == "first_difference":
        return np.diff(score_raw)
    raise KeyError(f"Unknown target view: {target_view}")


def build_step6_target_variance_records(
    sessions: list[AuditSessionInputs],
    *,
    config: Step6AuditConfig,
) -> list[dict[str, Any]]:
    """Compute variance shares and separability for audit target views."""

    records: list[dict[str, Any]] = []
    for target_view in TARGET_VIEWS:
        grouped = _group_session_arrays(
            sessions,
            extractor=lambda session, view=target_view: target_view_series(session.score_raw_online, target_view=view),
        )
        decomposition = compute_nested_dispersion(grouped)
        records.append(
            {
                "target_view": target_view,
                "var_total": decomposition["var_total"],
                "var_between_subject": decomposition["var_between_subject"],
                "var_between_session_within_subject": decomposition["var_between_session_within_subject"],
                "var_within_session": decomposition["var_within_session"],
                "share_between_subject": decomposition["share_between_subject"],
                "share_between_session_within_subject": decomposition["share_between_session_within_subject"],
                "share_within_session": decomposition["share_within_session"],
                "subject_separability": decomposition["share_between_subject"],
                "session_separability": decomposition["session_identity_share"],
            }
        )
    return records


def build_step6_target_lagged_records(
    sessions: list[AuditSessionInputs],
    *,
    config: Step6AuditConfig,
) -> list[dict[str, Any]]:
    """Compute lagged correlation and persistence MAE for audit target views."""

    records: list[dict[str, Any]] = []
    for target_view in TARGET_VIEWS:
        transformed = [target_view_series(session.score_raw_online, target_view=target_view) for session in sessions]
        series_length = transformed[0].shape[0]
        for horizon in config.horizons:
            if horizon >= series_length:
                raise ValueError(f"Horizon {horizon} is invalid for target view {target_view!r} length {series_length}")
            anchors: list[float] = []
            targets: list[float] = []
            for series in transformed:
                for target_idx in range(horizon, series.shape[0]):
                    anchor_idx = target_idx - horizon
                    anchors.append(float(series[anchor_idx]))
                    targets.append(float(series[target_idx]))
            anchor_arr = np.asarray(anchors, dtype=np.float64)
            target_arr = np.asarray(targets, dtype=np.float64)
            records.append(
                {
                    "target_view": target_view,
                    "horizon": horizon,
                    "lagged_corr": safe_corr(anchor_arr, target_arr, epsilon=config.corr_epsilon),
                    "persistence_mae": float(np.mean(np.abs(target_arr - anchor_arr))),
                }
            )
    return records


def _step4_judgment(
    step4_dispersion_records: list[dict[str, Any]],
    *,
    config: Step6AuditConfig,
) -> tuple[str, str, str]:
    pre_row = next(record for record in step4_dispersion_records if record["representation"] == "pre_alignment_raw_log")
    post_row = next(record for record in step4_dispersion_records if record["representation"] == "post_alignment_tangent")
    identity_share_delta = float(post_row["session_identity_share"]) - float(pre_row["session_identity_share"])
    within_share_delta = float(pre_row["share_within_session"]) - float(post_row["share_within_session"])
    judgment = (
        "YES"
        if identity_share_delta >= config.step4_identity_share_increase_threshold
        and within_share_delta >= config.step4_within_share_drop_threshold
        else "NO"
    )
    return (
        judgment,
        f"session_identity_share {pre_row['session_identity_share']:.3f} -> {post_row['session_identity_share']:.3f}",
        f"within_session_share {pre_row['share_within_session']:.3f} -> {post_row['share_within_session']:.3f}",
    )


def _step5_judgment(
    step5_summary: dict[str, Any],
    *,
    config: Step6AuditConfig,
) -> tuple[str, str, str]:
    cal_link = float(step5_summary["corr_online_mean_score_vs_cal_mean_score"])
    ref_link = float(step5_summary["corr_online_mean_score_vs_ref_separation"])
    judgment = (
        "YES"
        if abs(cal_link) >= config.step5_calibration_link_threshold
        or abs(ref_link) >= config.step5_reference_link_threshold
        else "NO"
    )
    return (
        judgment,
        f"corr(online_mean_score, cal_mean_score)={cal_link:.3f}",
        f"corr(online_mean_score, ref_separation)={ref_link:.3f}",
    )


def _step6_judgment(
    step6_variance_records: list[dict[str, Any]],
    step6_lagged_records: list[dict[str, Any]],
    *,
    config: Step6AuditConfig,
) -> tuple[str, str, str]:
    raw_row = next(record for record in step6_variance_records if record["target_view"] == "raw_level")
    raw_corrs = [float(record["lagged_corr"]) for record in step6_lagged_records if record["target_view"] == "raw_level"]
    centered_corrs = [
        abs(float(record["lagged_corr"]))
        for record in step6_lagged_records
        if record["target_view"] == "session_mean_centered"
    ]
    judgment = (
        "YES"
        if float(np.mean(raw_corrs)) >= config.step6_raw_corr_threshold
        and float(np.max(centered_corrs)) <= config.step6_centered_corr_ceiling
        and float(raw_row["session_separability"]) >= config.step6_session_identity_share_threshold
        else "NO"
    )
    return (
        judgment,
        f"mean raw lagged corr={float(np.mean(raw_corrs)):.3f}, max centered lagged corr={float(np.max(centered_corrs)):.3f}",
        f"raw session_separability={float(raw_row['session_separability']):.3f}",
    )


def build_hypothesis_judgments(
    step4_dispersion_records: list[dict[str, Any]],
    step5_summary: dict[str, Any],
    step6_variance_records: list[dict[str, Any]],
    step6_lagged_records: list[dict[str, Any]],
    *,
    config: Step6AuditConfig,
) -> list[dict[str, str]]:
    """Return the final yes/no judgments requested by the audit plan."""

    step4_judgment, step4_primary, step4_secondary = _step4_judgment(step4_dispersion_records, config=config)
    step5_judgment, step5_primary, step5_secondary = _step5_judgment(step5_summary, config=config)
    step6_judgment, step6_primary, step6_secondary = _step6_judgment(
        step6_variance_records,
        step6_lagged_records,
        config=config,
    )
    deeper_review = "YES" if all(value == "NO" for value in (step4_judgment, step5_judgment, step6_judgment)) else "NO"
    deeper_primary = "Escalate to Steps 1-3 only if Steps 4-6 do not explain baseline dominance"
    deeper_secondary = "Triggered because all upstream judgments were NO" if deeper_review == "YES" else "Blocked because at least one upstream stage already explains the result"

    return [
        {
            "hypothesis": "Step 4 alignment is the main source of baseline dominance",
            "judgment": step4_judgment,
            "primary_evidence": step4_primary,
            "secondary_evidence": step4_secondary,
        },
        {
            "hypothesis": "Step 5 score construction is the main source of baseline dominance",
            "judgment": step5_judgment,
            "primary_evidence": step5_primary,
            "secondary_evidence": step5_secondary,
        },
        {
            "hypothesis": "Step 6 target definition is the main source of baseline dominance",
            "judgment": step6_judgment,
            "primary_evidence": step6_primary,
            "secondary_evidence": step6_secondary,
        },
        {
            "hypothesis": "Steps 1-3 need deeper review because Steps 4-6 do not explain the result",
            "judgment": deeper_review,
            "primary_evidence": deeper_primary,
            "secondary_evidence": deeper_secondary,
        },
    ]


def plot_step4_dispersion(
    step4_dispersion_records: list[dict[str, Any]],
    *,
    figure_root: Path,
) -> Path:
    """Save a grouped bar chart of Step 4 pre/post dispersion shares."""

    labels = ["Between-subject", "Between-session", "Within-session"]
    pre_row = next(record for record in step4_dispersion_records if record["representation"] == "pre_alignment_raw_log")
    post_row = next(record for record in step4_dispersion_records if record["representation"] == "post_alignment_tangent")
    pre_values = [
        float(pre_row["share_between_subject"]),
        float(pre_row["share_between_session_within_subject"]),
        float(pre_row["share_within_session"]),
    ]
    post_values = [
        float(post_row["share_between_subject"]),
        float(post_row["share_between_session_within_subject"]),
        float(post_row["share_within_session"]),
    ]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, pre_values, width, label="Step 3 raw-log")
    ax.bar(x + width / 2, post_values, width, label="Step 4 aligned tangent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Step 4 Dispersion Before vs After Alignment")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = figure_root / "step6c_step4_dispersion.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_step5_calibration_link(
    step5_session_records: list[dict[str, Any]],
    *,
    figure_root: Path,
) -> Path:
    """Save a Step 5 scatter plot linking calibration and online score means."""

    cal_means = np.asarray([float(record["cal_mean_score_raw"]) for record in step5_session_records], dtype=np.float64)
    online_means = np.asarray([float(record["online_mean_score_raw"]) for record in step5_session_records], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cal_means, online_means, alpha=0.7)
    lo = float(min(np.min(cal_means), np.min(online_means)))
    hi = float(max(np.max(cal_means), np.max(online_means)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="tab:red", linewidth=1.5)
    ax.set_xlabel("Calibration mean score_raw")
    ax.set_ylabel("Online mean score_raw")
    ax.set_title("Step 5 Calibration-to-Online Mean Link")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = figure_root / "step6c_step5_calibration_vs_online.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_step6_target_semantics(
    step6_lagged_records: list[dict[str, Any]],
    *,
    figure_root: Path,
) -> Path:
    """Save lagged-correlation curves for raw, centered, and differenced targets."""

    view_to_style = {
        "raw_level": ("o-", "tab:blue", "Raw level"),
        "session_mean_centered": ("s--", "tab:orange", "Session-mean centered"),
        "first_difference": ("^:", "tab:green", "First difference"),
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    for target_view in TARGET_VIEWS:
        horizons = [int(record["horizon"]) for record in step6_lagged_records if record["target_view"] == target_view]
        corrs = [float(record["lagged_corr"]) for record in step6_lagged_records if record["target_view"] == target_view]
        style, color, label = view_to_style[target_view]
        ax.plot(horizons, corrs, style, color=color, linewidth=2, markersize=7, label=label)

    ax.set_xlabel("Horizon (blocks ahead)")
    ax.set_ylabel("Lagged correlation")
    ax.set_title("Step 6 Audit: Target View Lagged Correlations")
    ax.set_xticks([int(record["horizon"]) for record in step6_lagged_records if record["target_view"] == "raw_level"])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = figure_root / "step6c_step6_target_semantics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_conclusion_markdown(
    *,
    path: Path,
    step4_dispersion_records: list[dict[str, Any]],
    step5_summary: dict[str, Any],
    step6_variance_records: list[dict[str, Any]],
    step6_lagged_records: list[dict[str, Any]],
    hypothesis_rows: list[dict[str, str]],
) -> None:
    """Write the short audit conclusion requested by the plan."""

    raw_variance = next(record for record in step6_variance_records if record["target_view"] == "raw_level")
    centered_lagged = [float(record["lagged_corr"]) for record in step6_lagged_records if record["target_view"] == "session_mean_centered"]
    raw_lagged = [float(record["lagged_corr"]) for record in step6_lagged_records if record["target_view"] == "raw_level"]
    pre_step4 = next(record for record in step4_dispersion_records if record["representation"] == "pre_alignment_raw_log")
    post_step4 = next(record for record in step4_dispersion_records if record["representation"] == "post_alignment_tangent")

    lines = [
        "# Step 6c Upstream Target Audit",
        "",
        "## Bottom Line",
        "",
        (
            f"The current Step 6 canonical target remains strongly baseline-dominated: "
            f"raw-level session separability is {float(raw_variance['session_separability']):.3f}, "
            f"mean raw lagged correlation is {float(np.mean(raw_lagged)):.3f}, "
            f"and the maximum session-mean-centered lagged correlation is {float(np.max(np.abs(centered_lagged))):.3f}."
        ),
        "",
        "## Stage Judgments",
        "",
    ]
    for row in hypothesis_rows:
        lines.append(
            f"- **{row['judgment']}**: {row['hypothesis']}. "
            f"{row['primary_evidence']}. {row['secondary_evidence']}."
        )

    lines.extend(
        [
            "",
            "## Key Evidence",
            "",
            (
                f"- Step 4 dispersion shift: session-identity share moved from "
                f"{float(pre_step4['session_identity_share']):.3f} to {float(post_step4['session_identity_share']):.3f}, "
                f"while within-session share moved from {float(pre_step4['share_within_session']):.3f} "
                f"to {float(post_step4['share_within_session']):.3f}."
            ),
            (
                f"- Step 5 calibration link: corr(online mean score, calibration mean score) = "
                f"{float(step5_summary['corr_online_mean_score_vs_cal_mean_score']):.3f}, "
                f"corr(online mean score, ref separation) = "
                f"{float(step5_summary['corr_online_mean_score_vs_ref_separation']):.3f}."
            ),
            (
                f"- Step 6 semantics: raw-level mean lagged corr = {float(np.mean(raw_lagged)):.3f}, "
                f"centered max lagged corr = {float(np.max(np.abs(centered_lagged))):.3f}."
            ),
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step6_upstream_target_audit(config: Step6AuditConfig | None = None) -> Step6AuditSummary:
    """Run the full read-only upstream target audit."""

    config = config or Step6AuditConfig()
    validate_config(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.figure_root.mkdir(parents=True, exist_ok=True)

    logger.info("Step 6c: loading Step 3-5 artifacts ...")
    sessions = load_audit_sessions(config)
    logger.info("Loaded %d sessions for upstream audit.", len(sessions))

    logger.info("Step 6c: auditing Step 4 alignment ...")
    step4_dispersion_records = build_step4_dispersion_records(sessions, config=config)
    step4_session_records = build_step4_session_reference_records(sessions, config=config)
    write_csv(
        step4_dispersion_path(output_root=config.output_root),
        columns=STEP4_DISPERSION_COLUMNS,
        records=step4_dispersion_records,
    )
    write_csv(
        step4_session_reference_path(output_root=config.output_root),
        columns=STEP4_SESSION_REFERENCE_COLUMNS,
        records=step4_session_records,
    )
    plot_step4_dispersion(step4_dispersion_records, figure_root=config.figure_root)

    logger.info("Step 6c: auditing Step 5 score construction ...")
    step5_session_records = build_step5_session_diagnostics(sessions, config=config)
    step5_summary = build_step5_score_decomposition_summary(step5_session_records, sessions, config=config)
    write_csv(
        step5_session_diagnostic_path(output_root=config.output_root),
        columns=STEP5_SESSION_DIAGNOSTIC_COLUMNS,
        records=step5_session_records,
    )
    write_csv(
        step5_score_decomposition_path(output_root=config.output_root),
        columns=STEP5_SCORE_DECOMPOSITION_COLUMNS,
        records=[step5_summary],
    )
    plot_step5_calibration_link(step5_session_records, figure_root=config.figure_root)

    logger.info("Step 6c: auditing Step 6 target semantics ...")
    step6_variance_records = build_step6_target_variance_records(sessions, config=config)
    step6_lagged_records = build_step6_target_lagged_records(sessions, config=config)
    write_csv(
        step6_target_variance_path(output_root=config.output_root),
        columns=STEP6_TARGET_VARIANCE_COLUMNS,
        records=step6_variance_records,
    )
    write_csv(
        step6_target_lagged_path(output_root=config.output_root),
        columns=STEP6_TARGET_LAGGED_COLUMNS,
        records=step6_lagged_records,
    )
    plot_step6_target_semantics(step6_lagged_records, figure_root=config.figure_root)

    logger.info("Step 6c: writing final judgments ...")
    hypothesis_rows = build_hypothesis_judgments(
        step4_dispersion_records,
        step5_summary,
        step6_variance_records,
        step6_lagged_records,
        config=config,
    )
    write_csv(
        hypothesis_judgment_path(output_root=config.output_root),
        columns=HYPOTHESIS_COLUMNS,
        records=hypothesis_rows,
    )

    conclusion_path = conclusion_markdown_path(figure_root=config.figure_root)
    write_conclusion_markdown(
        path=conclusion_path,
        step4_dispersion_records=step4_dispersion_records,
        step5_summary=step5_summary,
        step6_variance_records=step6_variance_records,
        step6_lagged_records=step6_lagged_records,
        hypothesis_rows=hypothesis_rows,
    )
    logger.info("Saved %s", conclusion_path)

    return Step6AuditSummary(
        session_count=len(sessions),
        step4_dispersion_rows=len(step4_dispersion_records),
        step5_session_rows=len(step5_session_records),
        step6_target_variance_rows=len(step6_variance_records),
        step6_target_lagged_rows=len(step6_lagged_records),
        hypothesis_rows=len(hypothesis_rows),
        conclusion_path=conclusion_path,
    )


def load_saved_step6_variance_decomposition(config: Step6AuditConfig) -> dict[str, float]:
    """Load the saved Step 6 addendum variance decomposition."""

    _, rows = load_csv_rows(variance_decomposition_path(output_root=config.step6_root))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one saved Step 6 variance row, got {len(rows)}")
    return {key: float(value) for key, value in rows[0].items()}


def load_saved_step6_exploration_summary(config: Step6AuditConfig) -> dict[int, dict[str, float]]:
    """Load the saved Step 6 exploration summary keyed by horizon."""

    _, rows = load_csv_rows(exploration_summary_path(output_root=config.step6_root))
    return {
        int(row["horizon"]): {key: float(value) for key, value in row.items() if key != "horizon"}
        for row in rows
    }
