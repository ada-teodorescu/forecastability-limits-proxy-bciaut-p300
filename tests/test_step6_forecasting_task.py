"""Unit tests for Step 6 forecasting-task helpers and pipeline."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import expected_session_ids
from src.features.step4_riemannian import Step4Config, aligned_split_path, tangent_features_from_covariances, ts_split_path
from src.features.step5_scores import (
    Step5Config,
    align_references,
    construct_reference_covariances,
    run_step5,
    score_split_path,
)
from src.features.step6_forecasting_task import (
    MANIFEST_COLUMNS,
    OnlineSessionInputs,
    Step6Config,
    block_position_summary_path,
    build_manifest_records_for_session,
    compute_future_drop20_onset,
    exploration_summary_path,
    load_online_session_inputs,
    manifest_path,
    run_step6,
    session_summary_path,
    task_summary_path,
)


def _pytest_local_dir(name: str) -> Path:
    root = Path(".pytest-local") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_offset(subject_id: str, session_id: str) -> int:
    return (int(subject_id[-2:]) - 1) * 7 + (int(session_id[-2:]) - 1)


def _online_k(subject_id: str, session_id: str) -> int:
    return 3 + (_session_offset(subject_id, session_id) % 8)


def _block_targets(n_blocks: int) -> np.ndarray:
    repeats = (n_blocks + 7) // 8
    return np.tile(np.arange(1, 9, dtype=np.int8), repeats)[:n_blocks]


def _session_prototypes(
    *,
    subject_id: str,
    session_id: str,
    config: Step5Config,
) -> tuple[np.ndarray, np.ndarray]:
    offset = _session_offset(subject_id, session_id)
    time = np.linspace(0.0, 1.0, config.epoch_samples, dtype=np.float64)
    p_target = np.vstack(
        [
            np.sin((channel + 1) * np.pi * time) + 0.05 * channel + 0.001 * offset
            for channel in range(config.n_channels)
        ]
    )
    p_nontarget = np.vstack(
        [
            0.45 * np.cos((channel + 1.5) * np.pi * time + 0.2) - 0.03 * channel - 0.0005 * offset
            for channel in range(config.n_channels)
        ]
    )
    return p_target.astype(np.float64), p_nontarget.astype(np.float64)


def _session_ref_mean(
    *,
    subject_id: str,
    session_id: str,
    config: Step5Config,
) -> np.ndarray:
    offset = _session_offset(subject_id, session_id)
    diagonal = 1.2 + 0.015 * np.arange(config.covariance_dim, dtype=np.float64) + 0.0005 * offset
    return np.diag(diagonal)


def _blended_covariance(
    cov_target_ref: np.ndarray,
    cov_nontarget_ref: np.ndarray,
    *,
    alpha: float,
    global_index: int,
) -> np.ndarray:
    basis = np.arange(1, cov_target_ref.shape[0] + 1, dtype=np.float64)
    vec = np.sin((global_index + 1) * basis / 11.0) + 0.5 * np.cos((global_index + 1) * basis / 17.0)
    perturbation = 5e-4 * np.outer(vec, vec)
    blended = alpha * cov_target_ref + (1.0 - alpha) * cov_nontarget_ref + perturbation
    return 0.5 * (blended + blended.T)


def _write_step3_calibration_covariance_file(
    *,
    root: Path,
    subject_id: str,
    session_id: str,
    config: Step5Config,
    p_target: np.ndarray,
    p_nontarget: np.ndarray,
    cov_target_ref: np.ndarray,
    cov_nontarget_ref: np.ndarray,
) -> None:
    offset = _session_offset(subject_id, session_id)
    path = root / f"sub-{subject_id[-2:]}_ses-{session_id[-2:]}_split-calibration_cov.npz"
    calibration_condition = np.concatenate(
        [
            np.zeros(config.calibration_social_start, dtype=np.int8),
            np.ones(config.calibration_blocks - config.calibration_social_start, dtype=np.int8),
        ]
    )
    calibration_covs = np.stack(
        [
            _blended_covariance(
                cov_target_ref,
                cov_nontarget_ref,
                alpha=0.55 + 0.02 * (block_id / max(config.calibration_blocks - 1, 1)),
                global_index=10_000 + 100 * offset + block_id,
            )
            for block_id in range(config.calibration_blocks)
        ],
        axis=0,
    ).astype(np.float64)

    np.savez_compressed(
        path,
        covs=calibration_covs,
        block_ids=np.arange(config.calibration_blocks, dtype=np.int16),
        condition=calibration_condition,
        block_targets=_block_targets(config.calibration_blocks),
        runs_per_block=np.full(config.calibration_blocks, config.calibration_runs_per_block, dtype=np.int16),
        n_target_epochs=np.full(config.calibration_blocks, config.calibration_runs_per_block, dtype=np.int16),
        n_nontarget_epochs=np.full(
            config.calibration_blocks,
            7 * config.calibration_runs_per_block,
            dtype=np.int16,
        ),
        P_target=p_target.astype(np.float64),
        P_nontarget=p_nontarget.astype(np.float64),
        shrinkage_alphas=np.full(config.calibration_blocks, 0.2 + 0.0005 * offset, dtype=np.float64),
        condition_numbers=np.full(config.calibration_blocks, 2.0 + 0.01 * offset, dtype=np.float64),
        lambda_max_over_median=np.full(config.calibration_blocks, 5.0 + 0.02 * offset, dtype=np.float64),
    )


def _write_step4_split_files(
    *,
    root: Path,
    subject_id: str,
    session_id: str,
    split: str,
    step4_config: Step4Config,
    cov_target_ref_aligned: np.ndarray,
    cov_nontarget_ref_aligned: np.ndarray,
    session_ref_mean: np.ndarray,
) -> None:
    n_blocks = step4_config.calibration_blocks if split == "calibration" else step4_config.online_blocks
    offset = _session_offset(subject_id, session_id)
    if split == "calibration":
        alphas = np.concatenate(
            [
                np.linspace(0.58, 0.72, step4_config.calibration_social_start, dtype=np.float64),
                np.linspace(
                    0.84,
                    0.97,
                    step4_config.calibration_blocks - step4_config.calibration_social_start,
                    dtype=np.float64,
                ),
            ]
        )
        condition = np.concatenate(
            [
                np.zeros(step4_config.calibration_social_start, dtype=np.int8),
                np.ones(step4_config.calibration_blocks - step4_config.calibration_social_start, dtype=np.int8),
            ]
        )
        runs_per_block = np.full(n_blocks, step4_config.calibration_runs_per_block, dtype=np.int16)
    else:
        alphas = np.linspace(0.86, 0.18, step4_config.online_blocks, dtype=np.float64)
        condition = np.ones(n_blocks, dtype=np.int8)
        runs_per_block = np.full(n_blocks, _online_k(subject_id, session_id), dtype=np.int16)

    covs = np.stack(
        [
            _blended_covariance(
                cov_target_ref_aligned,
                cov_nontarget_ref_aligned,
                alpha=float(alphas[block_id]),
                global_index=20_000 + 100 * offset + (1_000 if split == "online" else 0) + block_id,
            )
            for block_id in range(n_blocks)
        ],
        axis=0,
    ).astype(np.float64)
    features = tangent_features_from_covariances(
        covs,
        config=step4_config,
        context=f"{subject_id}/{session_id}/{split}",
    )

    aligned_path = aligned_split_path(output_root=root, subject_id=subject_id, session_id=session_id, split=split)
    np.savez_compressed(
        aligned_path,
        covs=covs,
        block_ids=np.arange(n_blocks, dtype=np.int16),
        condition=condition,
        block_targets=_block_targets(n_blocks),
        runs_per_block=runs_per_block,
        n_target_epochs=runs_per_block.copy(),
        n_nontarget_epochs=(7 * runs_per_block).astype(np.int16),
        session_ref_mean=session_ref_mean.astype(np.float64),
    )
    ts_path = ts_split_path(output_root=root, subject_id=subject_id, session_id=session_id, split=split)
    np.savez_compressed(
        ts_path,
        features=features.astype(np.float32),
        block_ids=np.arange(n_blocks, dtype=np.int16),
        condition=condition,
        block_targets=_block_targets(n_blocks),
        runs_per_block=runs_per_block,
        n_target_epochs=runs_per_block.copy(),
        n_nontarget_epochs=(7 * runs_per_block).astype(np.int16),
        session_ref_mean=session_ref_mean.astype(np.float64),
    )


def _write_valid_step3_and_step4_inputs(
    *,
    root: Path,
    subject_id: str,
    session_id: str,
    step5_config: Step5Config,
    step4_config: Step4Config,
) -> None:
    p_target, p_nontarget = _session_prototypes(subject_id=subject_id, session_id=session_id, config=step5_config)
    cov_target_ref, cov_nontarget_ref = construct_reference_covariances(
        p_target,
        p_nontarget,
        config=step5_config,
        context=f"{subject_id}/{session_id}",
    )
    session_ref_mean = _session_ref_mean(subject_id=subject_id, session_id=session_id, config=step5_config)
    cov_target_ref_aligned, cov_nontarget_ref_aligned, _ = align_references(
        cov_target_ref,
        cov_nontarget_ref,
        session_ref_mean,
        config=step5_config,
        context=f"{subject_id}/{session_id}",
    )
    _write_step3_calibration_covariance_file(
        root=root,
        subject_id=subject_id,
        session_id=session_id,
        config=step5_config,
        p_target=p_target,
        p_nontarget=p_nontarget,
        cov_target_ref=cov_target_ref,
        cov_nontarget_ref=cov_nontarget_ref,
    )
    _write_step4_split_files(
        root=root,
        subject_id=subject_id,
        session_id=session_id,
        split="calibration",
        step4_config=step4_config,
        cov_target_ref_aligned=cov_target_ref_aligned,
        cov_nontarget_ref_aligned=cov_nontarget_ref_aligned,
        session_ref_mean=session_ref_mean,
    )
    _write_step4_split_files(
        root=root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
        step4_config=step4_config,
        cov_target_ref_aligned=cov_target_ref_aligned,
        cov_nontarget_ref_aligned=cov_nontarget_ref_aligned,
        session_ref_mean=session_ref_mean,
    )


def _populate_valid_step3_and_step4_inputs(
    *,
    root: Path,
    step5_config: Step5Config,
    step4_config: Step4Config,
) -> None:
    for subject_id, session_id in expected_session_ids():
        _write_valid_step3_and_step4_inputs(
            root=root,
            subject_id=subject_id,
            session_id=session_id,
            step5_config=step5_config,
            step4_config=step4_config,
        )


def _write_manual_online_ts_file(
    *,
    root: Path,
    subject_id: str,
    session_id: str,
    features: np.ndarray,
    block_targets: np.ndarray,
    runs_per_block: np.ndarray,
) -> None:
    np.savez_compressed(
        ts_split_path(output_root=root, subject_id=subject_id, session_id=session_id, split="online"),
        features=np.asarray(features, dtype=np.float32),
        block_ids=np.arange(features.shape[0], dtype=np.int16),
        condition=np.ones(features.shape[0], dtype=np.int8),
        block_targets=np.asarray(block_targets, dtype=np.int8),
        runs_per_block=np.asarray(runs_per_block, dtype=np.int16),
        n_target_epochs=np.asarray(runs_per_block, dtype=np.int16),
        n_nontarget_epochs=(7 * np.asarray(runs_per_block, dtype=np.int16)).astype(np.int16),
        session_ref_mean=np.diag(1.0 + 0.01 * np.arange(24, dtype=np.float64)),
    )


def _write_manual_online_score_file(
    *,
    root: Path,
    subject_id: str,
    session_id: str,
    score_raw: np.ndarray,
    score_drop_rel_session: np.ndarray,
    block_targets: np.ndarray,
    runs_per_block: np.ndarray,
) -> None:
    score_raw = np.asarray(score_raw, dtype=np.float32)
    score_drop_rel_session = np.asarray(score_drop_rel_session, dtype=np.float32)
    session_cal_mean_raw = np.float64(1.2)
    np.savez_compressed(
        score_split_path(output_root=root, subject_id=subject_id, session_id=session_id, split="online"),
        score_raw=score_raw,
        score_norm=np.linspace(0.2, 0.8, score_raw.shape[0], dtype=np.float32),
        score_threshold_rel_global=(score_raw - np.float32(0.3)).astype(np.float32),
        score_session_robust_z=((score_raw - np.float32(0.5)) / np.float32(0.2)).astype(np.float32),
        score_drop_rel_session=score_drop_rel_session,
        d_target=np.linspace(0.3, 0.7, score_raw.shape[0], dtype=np.float32),
        d_nontarget=np.linspace(0.8, 1.2, score_raw.shape[0], dtype=np.float32),
        mdm_correct=np.where(score_raw > 0.0, np.int8(1), np.int8(0)),
        session_cal_mean_raw=session_cal_mean_raw,
        session_cal_std_raw=np.float64(0.15),
        session_cal_median_raw=np.float64(0.55),
        session_cal_mad_raw=np.float64(0.10),
        session_cal_robust_scale_raw=np.float64(0.14826),
        session_cal_q10_raw=np.float64(0.40),
        session_cal_q25_raw=np.float64(0.50),
        session_cal_q50_raw=np.float64(0.55),
        session_cal_ref_count=np.int16(10),
        scale_degenerate_flag=np.int8(0),
        block_ids=np.arange(score_raw.shape[0], dtype=np.int16),
        condition=np.ones(score_raw.shape[0], dtype=np.int8),
        block_targets=np.asarray(block_targets, dtype=np.int8),
        runs_per_block=np.asarray(runs_per_block, dtype=np.int16),
        n_target_epochs=np.asarray(runs_per_block, dtype=np.int16),
        n_nontarget_epochs=(7 * np.asarray(runs_per_block, dtype=np.int16)).astype(np.int16),
    )


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz_file:
        return {name: np.asarray(npz_file[name]) for name in npz_file.files}


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def test_compute_future_drop20_onset_matches_contract_edge_cases() -> None:
    scores = np.asarray([-0.05, -0.10, -0.18, -0.22, -0.24, -0.19], dtype=np.float32)
    threshold = -0.20

    assert compute_future_drop20_onset(scores, anchor_idx=0, target_idx=2, threshold=threshold) == 0
    assert compute_future_drop20_onset(scores, anchor_idx=1, target_idx=3, threshold=threshold) == 1
    assert compute_future_drop20_onset(scores, anchor_idx=2, target_idx=4, threshold=threshold) == 1
    assert compute_future_drop20_onset(scores, anchor_idx=3, target_idx=5, threshold=threshold) == 0


def test_build_manifest_records_for_session_populates_research_fields() -> None:
    config = Step6Config(online_blocks=6, tangent_dim=3, horizons=(2,))
    session = OnlineSessionInputs(
        subject_id="SBJ01",
        session_id="S01",
        session_k=4,
        features=np.zeros((6, 3), dtype=np.float32),
        score_raw=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        score_drop_rel_session=np.asarray([-0.05, -0.10, -0.18, -0.22, -0.24, -0.19], dtype=np.float32),
        block_ids=np.arange(6, dtype=np.int16),
        condition=np.ones(6, dtype=np.int8),
        block_targets=_block_targets(6),
        runs_per_block=np.full(6, 4, dtype=np.int16),
        n_target_epochs=np.full(6, 4, dtype=np.int16),
        n_nontarget_epochs=np.full(6, 28, dtype=np.int16),
    )

    rows = build_manifest_records_for_session(session, horizon=2, config=config)

    assert len(rows) == 4
    assert rows[0]["anchor_idx"] == 0
    assert rows[0]["target_idx"] == 2
    assert rows[0]["elapsed_runs"] == 8
    assert rows[0]["delta_score_raw"] == pytest.approx(2.0)
    assert rows[0]["target_drop20_state"] == 0
    assert rows[0]["future_drop20_onset"] == 0
    assert rows[1]["anchor_idx"] == 1
    assert rows[1]["target_idx"] == 3
    assert rows[1]["target_drop20_state"] == 1
    assert rows[1]["future_drop20_onset"] == 1
    assert rows[3]["anchor_idx"] == 3
    assert rows[3]["target_idx"] == 5
    assert rows[3]["target_drop20_state"] == 0
    assert rows[3]["future_drop20_onset"] == 0


def test_load_online_session_inputs_rejects_step4_step5_metadata_mismatch() -> None:
    root = _pytest_local_dir("step6_metadata_mismatch")
    feature_root = root / "features"
    score_root = root / "scores"
    feature_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    subject_id = "SBJ01"
    session_id = "S01"
    features = np.arange(50 * 300, dtype=np.float32).reshape(50, 300) / np.float32(1000.0)
    score_raw = np.linspace(0.9, 0.1, 50, dtype=np.float32)
    score_drop_rel_session = (score_raw / np.float32(1.2) - np.float32(1.0)).astype(np.float32)
    runs_per_block = np.full(50, 3, dtype=np.int16)

    _write_manual_online_ts_file(
        root=feature_root,
        subject_id=subject_id,
        session_id=session_id,
        features=features,
        block_targets=_block_targets(50),
        runs_per_block=runs_per_block,
    )
    bad_block_targets = np.roll(_block_targets(50), 1).astype(np.int8)
    _write_manual_online_score_file(
        root=score_root,
        subject_id=subject_id,
        session_id=session_id,
        score_raw=score_raw,
        score_drop_rel_session=score_drop_rel_session,
        block_targets=bad_block_targets,
        runs_per_block=runs_per_block,
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_online_session_inputs(
            subject_id=subject_id,
            session_id=session_id,
            config=Step6Config(step4_root=feature_root, step5_root=score_root),
        )


def test_run_step6_and_smoke_test_outputs_validate_generated_files() -> None:
    root = _pytest_local_dir("step6_run_and_smoke")
    feature_root = root / "features"
    score_root = root / "scores"
    forecasting_root = root / "forecasting"
    figure_root = root / "outputs" / "forecasting" / "step6"
    feature_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    step5_config = Step5Config(step3_root=feature_root, step4_root=feature_root, output_root=score_root)
    step4_config = Step4Config(output_root=feature_root)
    _populate_valid_step3_and_step4_inputs(root=feature_root, step5_config=step5_config, step4_config=step4_config)

    run_step5(step5_config)
    summary = run_step6(
        Step6Config(
            step4_root=feature_root,
            step5_root=score_root,
            output_root=forecasting_root,
            figure_root=figure_root,
        )
    )

    assert summary.session_count == 105
    assert summary.manifest_file_count == 5
    assert summary.manifest_rows_by_horizon == (4725, 4200, 3675, 3150, 2625)
    assert summary.usable_rows_by_horizon == (3780, 3255, 2730, 2205, 1680)
    assert summary.session_summary_rows == 105
    assert summary.block_position_rows == 50
    assert summary.exploration_rows == 5
    assert summary.task_summary_rows == 5
    assert summary.figure_count == 3

    assert session_summary_path(output_root=forecasting_root).is_file()
    assert exploration_summary_path(output_root=forecasting_root).is_file()
    assert block_position_summary_path(output_root=forecasting_root).is_file()
    assert task_summary_path(output_root=forecasting_root).is_file()

    fieldnames, manifest_rows = _read_csv_rows(manifest_path(output_root=forecasting_root, horizon=5))
    assert fieldnames == list(MANIFEST_COLUMNS)
    assert len(manifest_rows) == 4725
    first_row = manifest_rows[0]
    assert first_row["subject_id"] == "SBJ01"
    assert first_row["session_id"] == "S01"
    assert first_row["horizon"] == "5"
    assert first_row["anchor_idx"] == "0"
    assert first_row["target_idx"] == "5"
    assert first_row["session_k"] == str(_online_k("SBJ01", "S01"))
    assert first_row["elapsed_runs"] == str(5 * _online_k("SBJ01", "S01"))

    sbj01_scores = _load_npz_payload(
        score_split_path(output_root=score_root, subject_id="SBJ01", session_id="S01", split="online"),
    )
    sbj01_raw = np.asarray(sbj01_scores["score_raw"], dtype=np.float32)
    sbj01_drop = np.asarray(sbj01_scores["score_drop_rel_session"], dtype=np.float32)
    assert float(first_row["anchor_score_raw"]) == float(sbj01_raw[0])
    assert float(first_row["target_score_raw"]) == float(sbj01_raw[5])
    assert float(first_row["delta_score_raw"]) == float(sbj01_raw[5] - sbj01_raw[0])
    assert float(first_row["target_score_drop_rel_session"]) == float(sbj01_drop[5])
    assert int(first_row["target_drop20_state"]) == int(sbj01_drop[5] < -0.20)
    assert int(first_row["future_drop20_onset"]) == compute_future_drop20_onset(
        sbj01_drop,
        anchor_idx=0,
        target_idx=5,
        threshold=-0.20,
    )

    session_fieldnames, session_rows = _read_csv_rows(session_summary_path(output_root=forecasting_root))
    assert "online_runs_per_block" in session_fieldnames
    assert len(session_rows) == 105
