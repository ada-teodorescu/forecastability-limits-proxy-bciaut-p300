"""Unit tests for Step 5 attention-score helpers and pipeline."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import expected_session_ids
from src.features.step4_riemannian import aligned_split_path
from src.features.step5_scores import (
    SessionCalibrationSummary,
    Step5Config,
    _validate_saved_normalization_payload,
    _validate_saved_score_payload,
    align_references,
    compute_global_threshold_relative_scores,
    compute_raw_scores,
    compute_session_calibration_summary,
    compute_session_relative_drop_scores,
    compute_session_robust_z_scores,
    construct_reference_covariances,
    ensure_spd,
    extract_social_calibration_scores,
    fit_global_normalization,
    load_session_inputs,
    normalization_path,
    process_session,
    qc_summary_path,
    reference_path,
    run_step5,
    score_split_path,
    smoke_test_outputs,
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
            np.sin((channel + 1) * np.pi * time)
            + 0.05 * channel
            + 0.001 * offset
            for channel in range(config.n_channels)
        ]
    )
    p_nontarget = np.vstack(
        [
            0.45 * np.cos((channel + 1.5) * np.pi * time + 0.2)
            - 0.03 * channel
            - 0.0005 * offset
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


def _write_step4_aligned_file(
    *,
    root: Path,
    subject_id: str,
    session_id: str,
    split: str,
    config: Step5Config,
    cov_target_ref_aligned: np.ndarray,
    cov_nontarget_ref_aligned: np.ndarray,
    session_ref_mean: np.ndarray,
) -> None:
    n_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    offset = _session_offset(subject_id, session_id)
    if split == "calibration":
        alphas = np.concatenate(
            [
                np.linspace(0.58, 0.72, config.calibration_social_start, dtype=np.float64),
                np.linspace(0.84, 0.97, config.calibration_blocks - config.calibration_social_start, dtype=np.float64),
            ]
        )
        condition = np.concatenate(
            [
                np.zeros(config.calibration_social_start, dtype=np.int8),
                np.ones(config.calibration_blocks - config.calibration_social_start, dtype=np.int8),
            ]
        )
        runs_per_block = np.full(n_blocks, config.calibration_runs_per_block, dtype=np.int16)
    else:
        alphas = np.linspace(0.86, 0.18, config.online_blocks, dtype=np.float64)
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

    path = aligned_split_path(
        output_root=root,
        subject_id=subject_id,
        session_id=session_id,
        split=split,
    )
    np.savez_compressed(
        path,
        covs=covs,
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
    config: Step5Config,
) -> None:
    p_target, p_nontarget = _session_prototypes(subject_id=subject_id, session_id=session_id, config=config)
    cov_target_ref, cov_nontarget_ref = construct_reference_covariances(
        p_target,
        p_nontarget,
        config=config,
        context=f"{subject_id}/{session_id}",
    )
    session_ref_mean = _session_ref_mean(subject_id=subject_id, session_id=session_id, config=config)
    cov_target_ref_aligned, cov_nontarget_ref_aligned, _ = align_references(
        cov_target_ref,
        cov_nontarget_ref,
        session_ref_mean,
        config=config,
        context=f"{subject_id}/{session_id}",
    )
    _write_step3_calibration_covariance_file(
        root=root,
        subject_id=subject_id,
        session_id=session_id,
        config=config,
        p_target=p_target,
        p_nontarget=p_nontarget,
        cov_target_ref=cov_target_ref,
        cov_nontarget_ref=cov_nontarget_ref,
    )
    _write_step4_aligned_file(
        root=root,
        subject_id=subject_id,
        session_id=session_id,
        split="calibration",
        config=config,
        cov_target_ref_aligned=cov_target_ref_aligned,
        cov_nontarget_ref_aligned=cov_nontarget_ref_aligned,
        session_ref_mean=session_ref_mean,
    )
    _write_step4_aligned_file(
        root=root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
        config=config,
        cov_target_ref_aligned=cov_target_ref_aligned,
        cov_nontarget_ref_aligned=cov_nontarget_ref_aligned,
        session_ref_mean=session_ref_mean,
    )


def _populate_valid_step5_inputs(*, root: Path, config: Step5Config) -> None:
    for subject_id, session_id in expected_session_ids():
        _write_valid_step3_and_step4_inputs(
            root=root,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz_file:
        return {name: np.asarray(npz_file[name]) for name in npz_file.files}


def test_construct_reference_covariances_returns_spd_matrices() -> None:
    config = Step5Config()
    p_target, p_nontarget = _session_prototypes(subject_id="SBJ01", session_id="S01", config=config)

    cov_target_ref, cov_nontarget_ref = construct_reference_covariances(
        p_target,
        p_nontarget,
        config=config,
        context="SBJ01/S01",
    )

    assert cov_target_ref.shape == (config.covariance_dim, config.covariance_dim)
    assert cov_nontarget_ref.shape == (config.covariance_dim, config.covariance_dim)
    ensure_spd(cov_target_ref, config=config, context="cov_target_ref")
    ensure_spd(cov_nontarget_ref, config=config, context="cov_nontarget_ref")
    assert not np.allclose(cov_target_ref, cov_nontarget_ref)


def test_compute_raw_scores_sign_and_mdm_correct_follow_reference_distance() -> None:
    config = Step5Config()
    p_target, p_nontarget = _session_prototypes(subject_id="SBJ01", session_id="S01", config=config)
    cov_target_ref, cov_nontarget_ref = construct_reference_covariances(
        p_target,
        p_nontarget,
        config=config,
        context="SBJ01/S01",
    )
    session_ref_mean = np.eye(config.covariance_dim, dtype=np.float64)
    cov_target_ref_aligned, cov_nontarget_ref_aligned, _ = align_references(
        cov_target_ref,
        cov_nontarget_ref,
        session_ref_mean,
        config=config,
        context="SBJ01/S01",
    )

    score_raw, d_target, d_nontarget, mdm_correct = compute_raw_scores(
        np.stack([cov_target_ref_aligned, cov_nontarget_ref_aligned], axis=0),
        cov_target_ref_aligned,
        cov_nontarget_ref_aligned,
        context="SBJ01/S01 ideal",
    )

    assert score_raw[0] > 0.0
    assert score_raw[1] < 0.0
    assert d_target[0] == pytest.approx(0.0)
    assert d_nontarget[1] == pytest.approx(0.0)
    assert np.array_equal(mdm_correct, np.array([1, 0], dtype=np.int8))


def test_align_references_keeps_outputs_spd_after_recentering() -> None:
    config = Step5Config()
    p_target, p_nontarget = _session_prototypes(subject_id="SBJ02", session_id="S03", config=config)
    cov_target_ref, cov_nontarget_ref = construct_reference_covariances(
        p_target,
        p_nontarget,
        config=config,
        context="SBJ02/S03",
    )
    session_ref_mean = _session_ref_mean(subject_id="SBJ02", session_id="S03", config=config)

    cov_target_ref_aligned, cov_nontarget_ref_aligned, ref_separation = align_references(
        cov_target_ref,
        cov_nontarget_ref,
        session_ref_mean,
        config=config,
        context="SBJ02/S03",
    )

    ensure_spd(cov_target_ref_aligned, config=config, context="cov_target_ref_aligned")
    ensure_spd(cov_nontarget_ref_aligned, config=config, context="cov_nontarget_ref_aligned")
    assert ref_separation > 0.0


def test_compute_global_threshold_relative_scores_uses_saved_precision_semantics() -> None:
    raw_scores = np.array([1.123456789, 2.5, 3.987654321], dtype=np.float64)
    threshold_raw = 2.253776506923494

    actual = compute_global_threshold_relative_scores(
        raw_scores,
        disengagement_threshold_raw=threshold_raw,
    )

    expected = (
        np.asarray(raw_scores, dtype=np.float32).astype(np.float64) - float(threshold_raw)
    ).astype(np.float32)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_compute_session_calibration_summary_uses_social_blocks_10_to_19() -> None:
    config = Step5Config()
    raw_scores = np.arange(config.calibration_blocks, dtype=np.float64) + 0.25
    calibration_metadata = {
        "block_ids": np.arange(config.calibration_blocks, dtype=np.int16),
        "condition": np.concatenate(
            [
                np.zeros(config.calibration_social_start, dtype=np.int8),
                np.ones(config.calibration_blocks - config.calibration_social_start, dtype=np.int8),
            ],
        ),
    }

    actual = compute_session_calibration_summary(
        raw_scores,
        calibration_metadata=calibration_metadata,
        config=config,
        context="SBJ01/S01",
    )

    expected_ref = np.asarray(raw_scores, dtype=np.float32)[config.calibration_social_start : config.calibration_blocks]
    expected_median = float(np.median(expected_ref.astype(np.float64)))
    expected_mad = float(np.median(np.abs(expected_ref.astype(np.float64) - expected_median)))
    assert actual.ref_count == 10
    assert actual.mean_raw == float(np.mean(expected_ref.astype(np.float64)))
    assert actual.std_raw == float(np.std(expected_ref.astype(np.float64), ddof=0))
    assert actual.median_raw == expected_median
    assert actual.mad_raw == expected_mad
    assert actual.robust_scale_raw == float(1.4826 * expected_mad)
    assert actual.q10_raw == float(np.percentile(expected_ref.astype(np.float64), 10.0, method="linear"))
    assert actual.q25_raw == float(np.percentile(expected_ref.astype(np.float64), 25.0, method="linear"))
    assert actual.q50_raw == float(np.percentile(expected_ref.astype(np.float64), 50.0, method="linear"))
    assert actual.scale_degenerate_flag == 0
    assert np.array_equal(
        extract_social_calibration_scores(
            raw_scores,
            calibration_metadata=calibration_metadata,
            config=config,
            context="SBJ01/S01",
        ),
        expected_ref,
    )


def test_compute_session_robust_z_scores_falls_back_to_std_when_mad_is_degenerate() -> None:
    config = Step5Config()
    raw_scores = np.concatenate(
        [
            np.linspace(0.0, 0.9, config.calibration_social_start, dtype=np.float64),
            np.array([1.0] * 9 + [2.0], dtype=np.float64),
        ],
    )
    calibration_metadata = {
        "block_ids": np.arange(config.calibration_blocks, dtype=np.int16),
        "condition": np.concatenate(
            [
                np.zeros(config.calibration_social_start, dtype=np.int8),
                np.ones(config.calibration_blocks - config.calibration_social_start, dtype=np.int8),
            ],
        ),
    }
    session_summary = compute_session_calibration_summary(
        raw_scores,
        calibration_metadata=calibration_metadata,
        config=config,
        context="SBJ01/S01",
    )

    actual = compute_session_robust_z_scores(
        np.array([1.0, 2.0], dtype=np.float64),
        session_summary=session_summary,
        config=config,
    )

    expected = (
        (np.asarray([1.0, 2.0], dtype=np.float32).astype(np.float64) - session_summary.median_raw)
        / session_summary.std_raw
    ).astype(np.float32)
    assert session_summary.robust_scale_raw == 0.0
    assert session_summary.std_raw > 0.0
    assert session_summary.scale_degenerate_flag == 0
    assert np.array_equal(actual, expected)


def test_compute_session_calibration_summary_rejects_non_positive_mean() -> None:
    config = Step5Config()
    raw_scores = np.zeros(config.calibration_blocks, dtype=np.float64)
    calibration_metadata = {
        "block_ids": np.arange(config.calibration_blocks, dtype=np.int16),
        "condition": np.concatenate(
            [
                np.zeros(config.calibration_social_start, dtype=np.int8),
                np.ones(config.calibration_blocks - config.calibration_social_start, dtype=np.int8),
            ],
        ),
    }

    with pytest.raises(ValueError, match="session_cal_mean_raw"):
        compute_session_calibration_summary(
            raw_scores,
            calibration_metadata=calibration_metadata,
            config=config,
            context="SBJ01/S01",
        )


def test_compute_session_robust_z_scores_returns_zeros_when_scale_is_fully_degenerate() -> None:
    config = Step5Config()
    session_summary = SessionCalibrationSummary(
        mean_raw=2.0,
        std_raw=0.0,
        median_raw=2.0,
        mad_raw=0.0,
        robust_scale_raw=0.0,
        q10_raw=2.0,
        q25_raw=2.0,
        q50_raw=2.0,
        ref_count=10,
        scale_degenerate_flag=1,
    )

    actual = compute_session_robust_z_scores(
        np.array([1.5, 2.0, 2.5], dtype=np.float64),
        session_summary=session_summary,
        config=config,
    )

    assert actual.dtype == np.float32
    assert np.array_equal(actual, np.zeros(3, dtype=np.float32))


def test_compute_session_relative_drop_scores_rejects_non_positive_baseline() -> None:
    config = Step5Config()

    with pytest.raises(ValueError, match="session_cal_mean_raw"):
        compute_session_relative_drop_scores(
            np.array([1.0, 2.0], dtype=np.float64),
            session_cal_mean_raw=0.0,
            config=config,
        )


def test_load_session_inputs_rejects_unexpected_step4_extra_fields() -> None:
    root = _pytest_local_dir("step5_step4_contract_drift")
    config = Step5Config(step3_root=root, step4_root=root, output_root=root / "scores")
    _write_valid_step3_and_step4_inputs(root=root, subject_id="SBJ01", session_id="S01", config=config)

    calibration_aligned_path = aligned_split_path(
        output_root=root,
        subject_id="SBJ01",
        session_id="S01",
        split="calibration",
    )
    payload = _load_npz_payload(calibration_aligned_path)
    payload["extra_field"] = np.array([1], dtype=np.int8)
    np.savez_compressed(calibration_aligned_path, **payload)

    with pytest.raises(ValueError, match="unexpected Step 4 fields"):
        load_session_inputs(subject_id="SBJ01", session_id="S01", config=config)


def test_run_step5_writes_outputs_and_roundtrips_normalization_and_metadata() -> None:
    root = _pytest_local_dir("step5_run_and_smoke")
    feature_root = root / "features"
    score_root = root / "scores"
    feature_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)
    config = Step5Config(step3_root=feature_root, step4_root=feature_root, output_root=score_root)
    _populate_valid_step5_inputs(root=feature_root, config=config)

    session_output = process_session(subject_id="SBJ01", session_id="S01", config=config)
    assert np.array_equal(
        session_output.cal_metadata["block_ids"],
        np.arange(config.calibration_blocks, dtype=np.int16),
    )
    normalization_preview = fit_global_normalization([session_output], config=config)
    assert normalization_preview.sigma_social_cal > 0.0

    summary = run_step5(config)

    assert summary.calibration_score_file_count == 105
    assert summary.online_score_file_count == 105
    assert summary.reference_file_count == 105
    assert summary.total_raw_scores == 7350
    assert summary.total_normalized_scores == 7350
    assert summary.mu_social_cal == pytest.approx(float(summary.mu_social_cal))

    smoke_summary = smoke_test_outputs(config)
    assert smoke_summary.total_raw_scores == 7350
    assert smoke_summary.total_normalized_scores == 7350

    cal_score_path = score_split_path(
        output_root=score_root,
        subject_id="SBJ01",
        session_id="S01",
        split="calibration",
    )
    online_score_path = score_split_path(
        output_root=score_root,
        subject_id="SBJ01",
        session_id="S01",
        split="online",
    )
    calibration_score_payload = _validate_saved_score_payload(
        path=cal_score_path,
        split="calibration",
        config=config,
    )
    online_score_payload = _validate_saved_score_payload(
        path=online_score_path,
        split="online",
        config=config,
    )
    calibration_step4_payload = _load_npz_payload(
        aligned_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="calibration"),
    )
    online_step4_payload = _load_npz_payload(
        aligned_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="online"),
    )
    for field_name in (
        "block_ids",
        "condition",
        "block_targets",
        "runs_per_block",
        "n_target_epochs",
        "n_nontarget_epochs",
    ):
        assert np.array_equal(calibration_score_payload[field_name], calibration_step4_payload[field_name])
        assert np.array_equal(online_score_payload[field_name], online_step4_payload[field_name])

    normalization_payload = _validate_saved_normalization_payload(normalization_path(output_root=score_root))
    threshold_raw = float(np.asarray(normalization_payload["disengagement_threshold_raw"]))
    sbj01_session_summary = compute_session_calibration_summary(
        calibration_score_payload["score_raw"],
        calibration_metadata={
            "block_ids": calibration_score_payload["block_ids"],
            "condition": calibration_score_payload["condition"],
        },
        config=config,
        context="SBJ01/S01",
    )
    assert np.array_equal(
        np.asarray(calibration_score_payload["score_threshold_rel_global"], dtype=np.float32),
        compute_global_threshold_relative_scores(
            calibration_score_payload["score_raw"],
            disengagement_threshold_raw=threshold_raw,
        ),
    )
    assert np.array_equal(
        np.asarray(online_score_payload["score_threshold_rel_global"], dtype=np.float32),
        compute_global_threshold_relative_scores(
            online_score_payload["score_raw"],
            disengagement_threshold_raw=threshold_raw,
        ),
    )
    assert np.array_equal(
        np.asarray(calibration_score_payload["score_session_robust_z"], dtype=np.float32),
        compute_session_robust_z_scores(
            calibration_score_payload["score_raw"],
            session_summary=sbj01_session_summary,
            config=config,
        ),
    )
    assert np.array_equal(
        np.asarray(online_score_payload["score_session_robust_z"], dtype=np.float32),
        compute_session_robust_z_scores(
            online_score_payload["score_raw"],
            session_summary=sbj01_session_summary,
            config=config,
        ),
    )
    assert np.array_equal(
        np.asarray(calibration_score_payload["score_drop_rel_session"], dtype=np.float32),
        compute_session_relative_drop_scores(
            calibration_score_payload["score_raw"],
            session_cal_mean_raw=sbj01_session_summary.mean_raw,
            config=config,
        ),
    )
    assert np.array_equal(
        np.asarray(online_score_payload["score_drop_rel_session"], dtype=np.float32),
        compute_session_relative_drop_scores(
            online_score_payload["score_raw"],
            session_cal_mean_raw=sbj01_session_summary.mean_raw,
            config=config,
        ),
    )
    assert float(np.asarray(calibration_score_payload["session_cal_mean_raw"])) == sbj01_session_summary.mean_raw
    assert float(np.asarray(calibration_score_payload["session_cal_std_raw"])) == sbj01_session_summary.std_raw
    assert float(np.asarray(calibration_score_payload["session_cal_median_raw"])) == sbj01_session_summary.median_raw
    assert float(np.asarray(calibration_score_payload["session_cal_mad_raw"])) == sbj01_session_summary.mad_raw
    assert float(np.asarray(calibration_score_payload["session_cal_robust_scale_raw"])) == sbj01_session_summary.robust_scale_raw
    assert float(np.asarray(calibration_score_payload["session_cal_q10_raw"])) == sbj01_session_summary.q10_raw
    assert float(np.asarray(calibration_score_payload["session_cal_q25_raw"])) == sbj01_session_summary.q25_raw
    assert float(np.asarray(calibration_score_payload["session_cal_q50_raw"])) == sbj01_session_summary.q50_raw
    assert int(np.asarray(calibration_score_payload["session_cal_ref_count"])) == sbj01_session_summary.ref_count
    assert int(np.asarray(calibration_score_payload["scale_degenerate_flag"])) == sbj01_session_summary.scale_degenerate_flag
    assert float(np.asarray(online_score_payload["session_cal_mean_raw"])) == sbj01_session_summary.mean_raw
    assert int(np.asarray(online_score_payload["scale_degenerate_flag"])) == sbj01_session_summary.scale_degenerate_flag

    recomputed_social_scores: list[np.ndarray] = []
    for subject_id, session_id in expected_session_ids():
        payload = _validate_saved_score_payload(
            path=score_split_path(
                output_root=score_root,
                subject_id=subject_id,
                session_id=session_id,
                split="calibration",
            ),
            split="calibration",
            config=config,
        )
        recomputed_social_scores.append(
            extract_social_calibration_scores(
                payload["score_raw"],
                calibration_metadata={
                    "block_ids": payload["block_ids"],
                    "condition": payload["condition"],
                },
                config=config,
                context=f"{subject_id}/{session_id}",
            )
        )
    recomputed_social_scores_array = np.concatenate(recomputed_social_scores).astype(np.float64)

    assert np.array_equal(
        normalization_payload["raw_social_calibration_scores"],
        recomputed_social_scores_array,
    )
    assert float(np.asarray(normalization_payload["mu_social_cal"])) == float(np.mean(recomputed_social_scores_array))
    assert float(np.asarray(normalization_payload["sigma_social_cal"])) == float(np.std(recomputed_social_scores_array, ddof=0))

    reference_payload = _load_npz_payload(reference_path(output_root=score_root, subject_id="SBJ01", session_id="S01"))
    assert reference_payload["cov_target_ref_aligned"].shape == (config.covariance_dim, config.covariance_dim)
    assert reference_payload["cov_nontarget_ref_aligned"].shape == (config.covariance_dim, config.covariance_dim)

    with open(qc_summary_path(output_root=score_root), "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 105
    assert "online_global_below_threshold_fraction" in rows[0]
    assert "online_session_z_below0_fraction" in rows[0]
    assert "online_drop_below_10_fraction" in rows[0]
    assert "online_drop_below_20_fraction" in rows[0]
    assert "online_drop_below_30_fraction" in rows[0]
    assert "scale_degenerate_flag" in rows[0]
    row_map = {(row["subject_id"], row["session_id"]): row for row in rows}
    sbj01_row = row_map[("SBJ01", "S01")]
    expected_online_global_below_threshold_fraction = float(
        np.mean(np.asarray(online_score_payload["score_threshold_rel_global"], dtype=np.float32) < 0.0)
    )
    expected_online_session_z_below0_fraction = float(
        np.mean(np.asarray(online_score_payload["score_session_robust_z"], dtype=np.float32) < 0.0)
    )
    expected_online_drop_below_10_fraction = float(
        np.mean(np.asarray(online_score_payload["score_drop_rel_session"], dtype=np.float32) < -0.10)
    )
    expected_online_drop_below_20_fraction = float(
        np.mean(np.asarray(online_score_payload["score_drop_rel_session"], dtype=np.float32) < -0.20)
    )
    expected_online_drop_below_30_fraction = float(
        np.mean(np.asarray(online_score_payload["score_drop_rel_session"], dtype=np.float32) < -0.30)
    )
    assert float(sbj01_row["online_global_below_threshold_fraction"]) == expected_online_global_below_threshold_fraction
    assert float(sbj01_row["online_session_z_below0_fraction"]) == expected_online_session_z_below0_fraction
    assert float(sbj01_row["online_drop_below_10_fraction"]) == expected_online_drop_below_10_fraction
    assert float(sbj01_row["online_drop_below_20_fraction"]) == expected_online_drop_below_20_fraction
    assert float(sbj01_row["online_drop_below_30_fraction"]) == expected_online_drop_below_30_fraction
    assert int(sbj01_row["scale_degenerate_flag"]) == sbj01_session_summary.scale_degenerate_flag
