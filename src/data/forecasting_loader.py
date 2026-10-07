"""Step 7 forecasting artifact loading and fold construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import SUBJECT_IDS, expected_session_ids
from src.features.step4_riemannian import (
    METADATA_FIELDS,
    Step4Config,
    _validate_saved_aligned_payload,
    _validate_saved_tangent_payload,
    aligned_split_path,
    ts_split_path,
)
from src.features.step5_scores import Step5Config, _validate_saved_score_payload, score_split_path
from src.features.step6_forecasting_task import load_csv_rows, manifest_path

TARGET_VIEWS: tuple[str, ...] = ("raw_level", "context_residual")


@dataclass(frozen=True)
class Step7Config:
    """Frozen Step 7 configuration and paths."""

    step4_root: Path = Path("data/features")
    step5_root: Path = Path("data/scores")
    step6_root: Path = Path("data/forecasting")
    output_root: Path = Path("data/forecasting")
    figure_root: Path = Path("outputs/forecasting/step7")
    horizons: tuple[int, ...] = (5, 10, 15, 20, 25)
    rolling_horizons: tuple[int, ...] = (5, 25)
    target_views: tuple[str, ...] = TARGET_VIEWS
    online_blocks: int = 50
    calibration_blocks: int = 20
    calibration_social_start: int = 10
    tangent_dim: int = 300
    covariance_dim: int = 24
    context_length: int = 10
    score_history_window: int = 5
    online_k_min: int = 3
    online_k_max: int = 10
    alpha_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    elastic_net_alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    elastic_net_l1_ratio_grid: tuple[float, ...] = (0.1, 0.5, 0.9)
    pcr_component_grid: tuple[int, ...] = (5, 10, 20, 40, 80, 160)
    pls_component_grid: tuple[int, ...] = (2, 5, 10, 20, 40)
    geometry_k: int = 5
    geometry_weight_epsilon: float = 1e-6
    geometry_batch_size: int = 48
    max_workers: int = 0
    reserve_cores: int = 2
    fallback_ridge_alpha: float = 1.0
    fallback_elastic_net_alpha: float = 1e-2
    fallback_elastic_net_l1_ratio: float = 0.5
    fallback_component_count: int = 5
    sigma_floor: float = 1e-6
    corr_epsilon: float = 1e-12
    permute_seed: int = 20260322
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 20260324


@dataclass(frozen=True)
class LoadedOnlineSessionArtifacts:
    """Validated Step 4/5 online artifacts for one subject-session pair."""

    subject_id: str
    session_id: str
    session_ord: int
    session_k: int
    features: np.ndarray
    aligned_covs: np.ndarray
    score_raw: np.ndarray
    score_drop_rel_session: np.ndarray
    block_ids: np.ndarray
    runs_per_block: np.ndarray


@dataclass(frozen=True)
class ForecastingTable:
    """One horizon-specific Step 7 row table with all predictor views."""

    horizon: int
    subject_id: np.ndarray
    session_id: np.ndarray
    session_ord: np.ndarray
    session_k: np.ndarray
    target_block_id: np.ndarray
    elapsed_runs: np.ndarray
    anchor_idx: np.ndarray
    target_idx: np.ndarray
    anchor_score_raw: np.ndarray
    target_score_raw: np.ndarray
    context_score_mean_value: np.ndarray
    target_score_context_residual: np.ndarray
    score_history_5: np.ndarray
    score_history_l: np.ndarray
    anchor_ts: np.ndarray
    context_ts_flat: np.ndarray
    anchor_aligned_cov: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.target_score_raw.shape[0])


@dataclass(frozen=True)
class PrimaryFoldDefinition:
    """One primary LOSO fold."""

    fold_subject_id: str
    horizon: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class RollingFoldDefinition:
    """One secondary rolling within-subject fold."""

    subject_id: str
    train_through_session: str
    test_session: str
    horizon: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def validate_step7_config(config: Step7Config) -> None:
    """Validate the fixed Step 7 configuration."""

    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.score_history_window <= 0:
        raise ValueError("score_history_window must be positive")
    if config.context_length < config.score_history_window:
        raise ValueError("context_length must be >= score_history_window")
    if not config.horizons:
        raise ValueError("At least one primary horizon is required")
    if len(set(config.horizons)) != len(config.horizons):
        raise ValueError(f"Primary horizons must be unique, got {config.horizons}")
    if len(set(config.rolling_horizons)) != len(config.rolling_horizons):
        raise ValueError(f"Rolling horizons must be unique, got {config.rolling_horizons}")
    if len(set(config.target_views)) != len(config.target_views):
        raise ValueError(f"Target views must be unique, got {config.target_views}")
    if set(config.target_views) != set(TARGET_VIEWS):
        raise ValueError(f"Target views must be exactly {TARGET_VIEWS}, got {config.target_views}")
    for horizon in (*config.horizons, *config.rolling_horizons):
        if horizon <= 0 or horizon >= config.online_blocks:
            raise ValueError(
                f"Horizon {horizon} is invalid for online_blocks={config.online_blocks}; "
                f"expected 1..{config.online_blocks - 1}",
            )
    if not set(config.rolling_horizons).issubset(set(config.horizons)):
        raise ValueError("rolling_horizons must be a subset of horizons")
    if config.geometry_k <= 0:
        raise ValueError("geometry_k must be positive")
    if config.geometry_batch_size <= 0:
        raise ValueError("geometry_batch_size must be positive")
    if config.max_workers < 0:
        raise ValueError("max_workers must be >= 0, where 0 means use all logical CPUs")
    if config.reserve_cores < 0:
        raise ValueError("reserve_cores must be >= 0")
    if config.fallback_ridge_alpha <= 0.0:
        raise ValueError("fallback_ridge_alpha must be positive")
    if config.fallback_elastic_net_alpha <= 0.0:
        raise ValueError("fallback_elastic_net_alpha must be positive")
    if not 0.0 < config.fallback_elastic_net_l1_ratio <= 1.0:
        raise ValueError("fallback_elastic_net_l1_ratio must be in (0, 1]")
    if config.fallback_component_count <= 0:
        raise ValueError("fallback_component_count must be positive")
    if config.sigma_floor <= 0.0:
        raise ValueError("sigma_floor must be positive")
    if config.corr_epsilon <= 0.0:
        raise ValueError("corr_epsilon must be positive")
    if config.bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")


def step4_validation_config(config: Step7Config) -> Step4Config:
    """Build a Step 4 validation config aligned with Step 7 constants."""

    return Step4Config(
        output_root=config.step4_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        tangent_dim=config.tangent_dim,
        covariance_dim=config.covariance_dim,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
    )


def step5_validation_config(config: Step7Config) -> Step5Config:
    """Build a Step 5 validation config aligned with Step 7 constants."""

    return Step5Config(
        output_root=config.step5_root,
        calibration_blocks=config.calibration_blocks,
        online_blocks=config.online_blocks,
        calibration_social_start=config.calibration_social_start,
        online_k_min=config.online_k_min,
        online_k_max=config.online_k_max,
    )


def expected_usable_rows_per_session(*, horizon: int, config: Step7Config) -> int:
    """Return the exact Step 7 usable rows per session for one horizon."""

    usable = config.online_blocks - int(horizon) - config.context_length + 1
    return max(0, usable)


def target_view_values(table: ForecastingTable, *, target_view: str) -> np.ndarray:
    """Return the requested target-view array."""

    if target_view == "raw_level":
        return np.asarray(table.target_score_raw, dtype=np.float64)
    if target_view == "context_residual":
        return np.asarray(table.target_score_context_residual, dtype=np.float64)
    raise KeyError(f"Unknown Step 7 target view: {target_view}")


def _require_matching_metadata(
    *,
    subject_id: str,
    session_id: str,
    ts_payload: dict[str, np.ndarray],
    aligned_payload: dict[str, np.ndarray],
    score_payload: dict[str, np.ndarray],
) -> None:
    for field_name in METADATA_FIELDS:
        ts_value = np.asarray(ts_payload[field_name])
        aligned_value = np.asarray(aligned_payload[field_name])
        score_value = np.asarray(score_payload[field_name])
        if not np.array_equal(ts_value, aligned_value):
            raise ValueError(
                f"{subject_id}/{session_id} Step 4 tangent and aligned metadata mismatch for {field_name!r}",
            )
        if not np.array_equal(ts_value, score_value):
            raise ValueError(
                f"{subject_id}/{session_id} Step 4/5 metadata mismatch for field {field_name!r}",
            )


def load_online_session_artifacts(
    *,
    subject_id: str,
    session_id: str,
    config: Step7Config,
) -> LoadedOnlineSessionArtifacts:
    """Load one validated Step 4/5 online session payload trio."""

    step4_config = step4_validation_config(config)
    step5_config = step5_validation_config(config)

    ts_path = ts_split_path(output_root=config.step4_root, subject_id=subject_id, session_id=session_id, split="online")
    aligned_path = aligned_split_path(
        output_root=config.step4_root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
    )
    score_path = score_split_path(
        output_root=config.step5_root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
    )

    ts_payload = _validate_saved_tangent_payload(path=ts_path, split="online", config=step4_config)
    aligned_payload = _validate_saved_aligned_payload(path=aligned_path, split="online", config=step4_config)
    score_payload = _validate_saved_score_payload(path=score_path, split="online", config=step5_config)
    _require_matching_metadata(
        subject_id=subject_id,
        session_id=session_id,
        ts_payload=ts_payload,
        aligned_payload=aligned_payload,
        score_payload=score_payload,
    )

    features = np.asarray(ts_payload["features"], dtype=np.float32)
    aligned_covs = np.asarray(aligned_payload["covs"], dtype=np.float64)
    score_raw = np.asarray(score_payload["score_raw"], dtype=np.float32)
    score_drop_rel_session = np.asarray(score_payload["score_drop_rel_session"], dtype=np.float32)
    block_ids = np.asarray(ts_payload["block_ids"], dtype=np.int16)
    runs_per_block = np.asarray(ts_payload["runs_per_block"], dtype=np.int16)

    if features.shape != (config.online_blocks, config.tangent_dim):
        raise ValueError(
            f"{subject_id}/{session_id} tangent features shape {features.shape}, "
            f"expected ({config.online_blocks}, {config.tangent_dim})",
        )
    if aligned_covs.shape != (config.online_blocks, config.covariance_dim, config.covariance_dim):
        raise ValueError(
            f"{subject_id}/{session_id} aligned covs shape {aligned_covs.shape}, "
            f"expected ({config.online_blocks}, {config.covariance_dim}, {config.covariance_dim})",
        )
    if score_raw.shape != (config.online_blocks,):
        raise ValueError(
            f"{subject_id}/{session_id} score_raw shape {score_raw.shape}, expected ({config.online_blocks},)",
        )
    if score_drop_rel_session.shape != (config.online_blocks,):
        raise ValueError(
            f"{subject_id}/{session_id} score_drop_rel_session shape {score_drop_rel_session.shape}, "
            f"expected ({config.online_blocks},)",
        )
    if block_ids.shape != (config.online_blocks,):
        raise ValueError(
            f"{subject_id}/{session_id} block_ids shape {block_ids.shape}, expected ({config.online_blocks},)",
        )
    unique_runs = np.unique(runs_per_block)
    if unique_runs.size != 1:
        raise ValueError(f"{subject_id}/{session_id} runs_per_block is not constant within the online split")
    session_k = int(unique_runs[0])
    if session_k < config.online_k_min or session_k > config.online_k_max:
        raise ValueError(
            f"{subject_id}/{session_id} session_k={session_k} outside expected {config.online_k_min}..{config.online_k_max}",
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(f"{subject_id}/{session_id} tangent features contain non-finite values")
    if not np.all(np.isfinite(aligned_covs)):
        raise ValueError(f"{subject_id}/{session_id} aligned covariances contain non-finite values")
    if not np.all(np.isfinite(score_raw)):
        raise ValueError(f"{subject_id}/{session_id} score_raw contains non-finite values")
    if not np.all(np.isfinite(score_drop_rel_session)):
        raise ValueError(f"{subject_id}/{session_id} score_drop_rel_session contains non-finite values")

    return LoadedOnlineSessionArtifacts(
        subject_id=subject_id,
        session_id=session_id,
        session_ord=int(session_id[1:]),
        session_k=session_k,
        features=features,
        aligned_covs=aligned_covs,
        score_raw=score_raw,
        score_drop_rel_session=score_drop_rel_session,
        block_ids=block_ids,
        runs_per_block=runs_per_block,
    )


def _parse_manifest_rows(
    *,
    horizon: int,
    config: Step7Config,
) -> list[dict[str, str]]:
    manifest_file = manifest_path(output_root=config.step6_root, horizon=horizon)
    fieldnames, rows = load_csv_rows(manifest_file)
    required_columns = {
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
    }
    missing = sorted(required_columns.difference(fieldnames))
    if missing:
        raise ValueError(f"{manifest_file.name} is missing required Step 6 columns: {missing}")
    filtered = [row for row in rows if int(row["anchor_idx"]) >= config.context_length - 1]
    if not filtered:
        raise ValueError(f"{manifest_file.name} produced zero Step 7-usable rows after anchor_idx filtering")
    filtered.sort(key=lambda row: (row["subject_id"], row["session_id"], int(row["target_idx"])))
    return filtered


def load_forecasting_table(
    *,
    horizon: int,
    config: Step7Config,
) -> ForecastingTable:
    """Load one horizon-specific Step 7 forecasting table."""

    if horizon not in config.horizons:
        raise ValueError(f"Horizon {horizon} is not part of the primary Step 7 horizon set {config.horizons}")

    rows = _parse_manifest_rows(horizon=horizon, config=config)
    session_cache: dict[tuple[str, str], LoadedOnlineSessionArtifacts] = {}

    subject_id: list[str] = []
    session_id: list[str] = []
    session_ord: list[int] = []
    session_k: list[int] = []
    target_block_id: list[int] = []
    elapsed_runs: list[int] = []
    anchor_idx: list[int] = []
    target_idx: list[int] = []
    anchor_score_raw: list[float] = []
    target_score_raw: list[float] = []
    context_score_mean_value: list[float] = []
    target_score_context_residual: list[float] = []
    score_history_5: list[np.ndarray] = []
    score_history_l: list[np.ndarray] = []
    anchor_ts: list[np.ndarray] = []
    context_ts_flat: list[np.ndarray] = []
    anchor_aligned_cov: list[np.ndarray] = []

    for row in rows:
        row_subject_id = row["subject_id"]
        row_session_id = row["session_id"]
        cache_key = (row_subject_id, row_session_id)
        session = session_cache.get(cache_key)
        if session is None:
            session = load_online_session_artifacts(
                subject_id=row_subject_id,
                session_id=row_session_id,
                config=config,
            )
            session_cache[cache_key] = session

        row_horizon = int(row["horizon"])
        if row_horizon != horizon:
            raise ValueError(f"Manifest row horizon {row_horizon} does not match requested horizon {horizon}")

        anchor = int(row["anchor_idx"])
        target = int(row["target_idx"])
        if anchor < config.context_length - 1:
            raise ValueError(f"{row_subject_id}/{row_session_id} anchor_idx={anchor} violates Step 7 context rule")
        if target - anchor != horizon:
            raise ValueError(
                f"{row_subject_id}/{row_session_id} manifest target/anchor mismatch: "
                f"target={target}, anchor={anchor}, horizon={horizon}",
            )

        manifest_anchor_score = float(row["anchor_score_raw"])
        manifest_target_score = float(row["target_score_raw"])
        session_anchor_score = float(np.asarray(session.score_raw[anchor], dtype=np.float32))
        session_target_score = float(np.asarray(session.score_raw[target], dtype=np.float32))
        if not np.isclose(manifest_anchor_score, session_anchor_score, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{row_subject_id}/{row_session_id} anchor score drift: "
                f"manifest={manifest_anchor_score}, session={session_anchor_score}",
            )
        if not np.isclose(manifest_target_score, session_target_score, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{row_subject_id}/{row_session_id} target score drift: "
                f"manifest={manifest_target_score}, session={session_target_score}",
            )

        row_session_k = int(row["session_k"])
        if row_session_k != session.session_k:
            raise ValueError(
                f"{row_subject_id}/{row_session_id} manifest session_k={row_session_k} "
                f"does not match loaded session_k={session.session_k}",
            )

        history_l = np.asarray(
            session.score_raw[anchor - config.context_length + 1 : anchor + 1],
            dtype=np.float32,
        )
        context_mean = float(np.mean(np.asarray(history_l, dtype=np.float64)))

        subject_id.append(row_subject_id)
        session_id.append(row_session_id)
        session_ord.append(session.session_ord)
        session_k.append(session.session_k)
        target_block_id.append(int(row["target_block_id"]))
        elapsed_runs.append(int(row["elapsed_runs"]))
        anchor_idx.append(anchor)
        target_idx.append(target)
        anchor_score_raw.append(session_anchor_score)
        target_score_raw.append(session_target_score)
        context_score_mean_value.append(context_mean)
        target_score_context_residual.append(session_target_score - context_mean)
        score_history_5.append(np.asarray(session.score_raw[anchor - 4 : anchor + 1], dtype=np.float32))
        score_history_l.append(history_l)
        anchor_ts.append(np.asarray(session.features[anchor], dtype=np.float32))
        context_ts_flat.append(
            np.asarray(
                session.features[anchor - config.context_length + 1 : anchor + 1],
                dtype=np.float32,
            ).reshape(-1),
        )
        anchor_aligned_cov.append(np.asarray(session.aligned_covs[anchor], dtype=np.float64))

    expected_rows = len(expected_session_ids()) * expected_usable_rows_per_session(horizon=horizon, config=config)
    if len(subject_id) != expected_rows:
        raise ValueError(f"Horizon {horizon} loaded {len(subject_id)} usable rows, expected {expected_rows}")

    return ForecastingTable(
        horizon=horizon,
        subject_id=np.asarray(subject_id, dtype=object),
        session_id=np.asarray(session_id, dtype=object),
        session_ord=np.asarray(session_ord, dtype=np.int16),
        session_k=np.asarray(session_k, dtype=np.int16),
        target_block_id=np.asarray(target_block_id, dtype=np.int16),
        elapsed_runs=np.asarray(elapsed_runs, dtype=np.int16),
        anchor_idx=np.asarray(anchor_idx, dtype=np.int16),
        target_idx=np.asarray(target_idx, dtype=np.int16),
        anchor_score_raw=np.asarray(anchor_score_raw, dtype=np.float32),
        target_score_raw=np.asarray(target_score_raw, dtype=np.float32),
        context_score_mean_value=np.asarray(context_score_mean_value, dtype=np.float32),
        target_score_context_residual=np.asarray(target_score_context_residual, dtype=np.float32),
        score_history_5=np.stack(score_history_5, axis=0).astype(np.float32),
        score_history_l=np.stack(score_history_l, axis=0).astype(np.float32),
        anchor_ts=np.stack(anchor_ts, axis=0).astype(np.float32),
        context_ts_flat=np.stack(context_ts_flat, axis=0).astype(np.float32),
        anchor_aligned_cov=np.stack(anchor_aligned_cov, axis=0).astype(np.float64),
    )


def load_primary_tables(config: Step7Config) -> dict[int, ForecastingTable]:
    """Load all primary Step 7 forecasting tables."""

    validate_step7_config(config)
    return {horizon: load_forecasting_table(horizon=horizon, config=config) for horizon in config.horizons}


def build_primary_folds(table: ForecastingTable) -> list[PrimaryFoldDefinition]:
    """Build deterministic LOSO folds for one horizon-specific table."""

    folds: list[PrimaryFoldDefinition] = []
    for subject in SUBJECT_IDS:
        test_idx = np.flatnonzero(table.subject_id == subject)
        train_idx = np.flatnonzero(table.subject_id != subject)
        if test_idx.size == 0:
            raise ValueError(f"Horizon {table.horizon} LOSO fold {subject} has zero test rows")
        if train_idx.size == 0:
            raise ValueError(f"Horizon {table.horizon} LOSO fold {subject} has zero train rows")
        folds.append(
            PrimaryFoldDefinition(
                fold_subject_id=subject,
                horizon=table.horizon,
                train_idx=train_idx.astype(np.int64),
                test_idx=test_idx.astype(np.int64),
            ),
        )
    return folds


def build_rolling_folds(table: ForecastingTable) -> list[RollingFoldDefinition]:
    """Build deterministic rolling within-subject folds for one horizon-specific table."""

    folds: list[RollingFoldDefinition] = []
    for subject in SUBJECT_IDS:
        subject_mask = table.subject_id == subject
        for train_through_session_ord in range(1, 7):
            test_session_ord = train_through_session_ord + 1
            train_idx = np.flatnonzero(subject_mask & (table.session_ord <= train_through_session_ord))
            test_idx = np.flatnonzero(subject_mask & (table.session_ord == test_session_ord))
            if train_idx.size == 0:
                raise ValueError(
                    f"Horizon {table.horizon} rolling fold {subject} through "
                    f"S{train_through_session_ord:02d} has zero train rows",
                )
            if test_idx.size == 0:
                raise ValueError(
                    f"Horizon {table.horizon} rolling fold {subject} into "
                    f"S{test_session_ord:02d} has zero test rows",
                )
            folds.append(
                RollingFoldDefinition(
                    subject_id=subject,
                    train_through_session=f"S{train_through_session_ord:02d}",
                    test_session=f"S{test_session_ord:02d}",
                    horizon=table.horizon,
                    train_idx=train_idx.astype(np.int64),
                    test_idx=test_idx.astype(np.int64),
                ),
            )
    return folds


def table_row_dict(table: ForecastingTable, row_index: int) -> dict[str, Any]:
    """Return one row's metadata as plain Python scalars."""

    return {
        "subject_id": str(table.subject_id[row_index]),
        "session_id": str(table.session_id[row_index]),
        "session_ord": int(table.session_ord[row_index]),
        "session_k": int(table.session_k[row_index]),
        "target_block_id": int(table.target_block_id[row_index]),
        "elapsed_runs": int(table.elapsed_runs[row_index]),
        "anchor_idx": int(table.anchor_idx[row_index]),
        "target_idx": int(table.target_idx[row_index]),
        "anchor_score_raw": float(table.anchor_score_raw[row_index]),
        "target_score_raw": float(table.target_score_raw[row_index]),
        "context_score_mean_value": float(table.context_score_mean_value[row_index]),
        "target_score_context_residual": float(table.target_score_context_residual[row_index]),
    }
