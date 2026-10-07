"""Unit tests for Step 4 Riemannian alignment and tangent-space helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import expected_session_ids
from src.features.step3_covariance import (
    Step3Config,
    _validate_saved_covariance_payload,
    feature_split_path as step3_feature_split_path,
)
from src.features.step4_riemannian import (
    Step4Config,
    _validate_saved_aligned_payload,
    _validate_saved_tangent_payload,
    aligned_split_path,
    alignment_log_path,
    process_session,
    run_step4,
    smoke_test_outputs,
    ts_split_path,
    validate_alignment_log_contents,
    vectorize_upper_triangle,
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


def _reference_covariance(*, subject_id: str, session_id: str, config: Step4Config) -> np.ndarray:
    offset = _session_offset(subject_id, session_id)
    diagonal = 1.0 + 0.02 * np.arange(config.covariance_dim, dtype=np.float64) + 0.001 * offset
    return np.diag(diagonal)


def _online_spd_perturbation(*, global_index: int, config: Step4Config) -> np.ndarray:
    basis = np.arange(1, config.covariance_dim + 1, dtype=np.float64)
    vec1 = np.sin((global_index + 1) * basis / 11.0)
    vec2 = np.cos((global_index + 1) * basis / 17.0)
    stacked = np.stack([vec1, vec2], axis=1)
    return 0.0025 * (stacked @ stacked.T)


def _make_covariance_stack(
    *,
    split: str,
    subject_id: str,
    session_id: str,
    config: Step4Config,
) -> np.ndarray:
    n_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    reference_cov = _reference_covariance(subject_id=subject_id, session_id=session_id, config=config)
    covariances = np.repeat(reference_cov[None, :, :], n_blocks, axis=0)
    if split == "online":
        for block_id in range(n_blocks):
            covariances[block_id] = reference_cov + _online_spd_perturbation(
                global_index=10_000 + 100 * _session_offset(subject_id, session_id) + block_id,
                config=config,
            )
    return covariances


def _write_valid_step3_covariance_outputs(
    *,
    output_root: Path,
    step3_config: Step3Config,
    step4_config: Step4Config,
    subject_id: str,
    session_id: str,
) -> None:
    calibration_path = step3_feature_split_path(
        output_root=output_root,
        subject_id=subject_id,
        session_id=session_id,
        split="calibration",
    )
    online_path = step3_feature_split_path(
        output_root=output_root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
    )
    calibration_condition = np.concatenate(
        [
            np.zeros(step4_config.calibration_social_start, dtype=np.int8),
            np.ones(step4_config.calibration_blocks - step4_config.calibration_social_start, dtype=np.int8),
        ]
    )
    p_target = np.ones((step3_config.n_channels, step3_config.epoch_samples), dtype=np.float64)
    p_nontarget = np.zeros((step3_config.n_channels, step3_config.epoch_samples), dtype=np.float64)
    session_k = _online_k(subject_id, session_id)

    np.savez_compressed(
        calibration_path,
        covs=_make_covariance_stack(split="calibration", subject_id=subject_id, session_id=session_id, config=step4_config),
        block_ids=np.arange(step4_config.calibration_blocks, dtype=np.int16),
        condition=calibration_condition,
        block_targets=_block_targets(step4_config.calibration_blocks),
        runs_per_block=np.full(step4_config.calibration_blocks, step3_config.calibration_runs_per_block, dtype=np.int16),
        n_target_epochs=np.full(step4_config.calibration_blocks, step3_config.calibration_runs_per_block, dtype=np.int16),
        n_nontarget_epochs=np.full(step4_config.calibration_blocks, 7 * step3_config.calibration_runs_per_block, dtype=np.int16),
        P_target=p_target,
        P_nontarget=p_nontarget,
        shrinkage_alphas=np.full(step4_config.calibration_blocks, 0.20, dtype=np.float64),
        condition_numbers=np.full(step4_config.calibration_blocks, 2.00, dtype=np.float64),
        lambda_max_over_median=np.full(step4_config.calibration_blocks, 5.00, dtype=np.float64),
    )
    np.savez_compressed(
        online_path,
        covs=_make_covariance_stack(split="online", subject_id=subject_id, session_id=session_id, config=step4_config),
        block_ids=np.arange(step4_config.online_blocks, dtype=np.int16),
        condition=np.ones(step4_config.online_blocks, dtype=np.int8),
        block_targets=_block_targets(step4_config.online_blocks),
        runs_per_block=np.full(step4_config.online_blocks, session_k, dtype=np.int16),
        n_target_epochs=np.full(step4_config.online_blocks, session_k, dtype=np.int16),
        n_nontarget_epochs=np.full(step4_config.online_blocks, 7 * session_k, dtype=np.int16),
        P_target=p_target,
        P_nontarget=p_nontarget,
        shrinkage_alphas=np.full(step4_config.online_blocks, 0.35, dtype=np.float64),
        condition_numbers=np.full(step4_config.online_blocks, 2.50, dtype=np.float64),
        lambda_max_over_median=np.full(step4_config.online_blocks, 5.50, dtype=np.float64),
    )


def _populate_valid_step3_inputs(*, feature_root: Path, step3_config: Step3Config, step4_config: Step4Config) -> None:
    for subject_id, session_id in expected_session_ids():
        _write_valid_step3_covariance_outputs(
            output_root=feature_root,
            step3_config=step3_config,
            step4_config=step4_config,
            subject_id=subject_id,
            session_id=session_id,
        )


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz_file:
        return {name: np.asarray(npz_file[name]) for name in npz_file.files}


def test_vectorize_upper_triangle_applies_sqrt2_and_returns_length_300() -> None:
    config = Step4Config()
    tangent = np.zeros((config.covariance_dim, config.covariance_dim), dtype=np.float64)
    tangent[0, 0] = 3.0
    tangent[0, 1] = 5.0
    tangent[1, 0] = 5.0
    tangent[1, 1] = 7.0

    vector = vectorize_upper_triangle(tangent, config=config)

    assert vector.shape == (300,)
    assert vector[0] == pytest.approx(3.0)
    assert vector[1] == pytest.approx(5.0 * np.sqrt(2.0))
    assert vector[24] == pytest.approx(7.0)
    assert np.count_nonzero(vector) == 3


def test_process_session_writes_aligned_and_tangent_outputs_and_preserves_metadata() -> None:
    root = _pytest_local_dir("step4_process_session")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _write_valid_step3_covariance_outputs(
        output_root=feature_root,
        step3_config=step3_config,
        step4_config=step4_config,
        subject_id="SBJ01",
        session_id="S01",
    )

    session_output = process_session(subject_id="SBJ01", session_id="S01", config=step4_config)

    calibration_aligned_payload = _validate_saved_aligned_payload(
        path=aligned_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="calibration"),
        split="calibration",
        config=step4_config,
    )
    online_aligned_payload = _validate_saved_aligned_payload(
        path=aligned_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="online"),
        split="online",
        config=step4_config,
    )
    calibration_ts_payload = _validate_saved_tangent_payload(
        path=ts_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="calibration"),
        split="calibration",
        config=step4_config,
    )
    online_ts_payload = _validate_saved_tangent_payload(
        path=ts_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="online"),
        split="online",
        config=step4_config,
    )
    calibration_step3_payload = _validate_saved_covariance_payload(
        path=step3_feature_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="calibration"),
        split="calibration",
        config=step3_config,
    )
    online_step3_payload = _validate_saved_covariance_payload(
        path=step3_feature_split_path(output_root=feature_root, subject_id="SBJ01", session_id="S01", split="online"),
        split="online",
        config=step3_config,
    )

    assert calibration_aligned_payload["covs"].shape == (20, 24, 24)
    assert online_aligned_payload["covs"].shape == (50, 24, 24)
    assert calibration_ts_payload["features"].shape == (20, 300)
    assert online_ts_payload["features"].shape == (50, 300)
    for field_name in ("block_ids", "condition", "block_targets", "runs_per_block", "n_target_epochs", "n_nontarget_epochs"):
        assert np.array_equal(calibration_aligned_payload[field_name], calibration_step3_payload[field_name])
        assert np.array_equal(online_aligned_payload[field_name], online_step3_payload[field_name])
        assert np.array_equal(calibration_ts_payload[field_name], calibration_step3_payload[field_name])
        assert np.array_equal(online_ts_payload[field_name], online_step3_payload[field_name])
    assert np.array_equal(calibration_aligned_payload["session_ref_mean"], online_aligned_payload["session_ref_mean"])
    assert np.array_equal(calibration_aligned_payload["session_ref_mean"], calibration_ts_payload["session_ref_mean"])
    assert np.array_equal(calibration_aligned_payload["session_ref_mean"], online_ts_payload["session_ref_mean"])
    assert session_output.identity_error <= step4_config.identity_error_tolerance
    assert np.allclose(
        np.asarray(calibration_aligned_payload["covs"], dtype=np.float64),
        np.repeat(np.eye(step4_config.covariance_dim, dtype=np.float64)[None, :, :], 20, axis=0),
        atol=1e-8,
    )

    log_payload = validate_alignment_log_contents(
        alignment_log_path(output_root=feature_root, subject_id="SBJ01", session_id="S01"),
    )
    assert log_payload["subject_id"] == "SBJ01"
    assert log_payload["session_id"] == "S01"
    assert log_payload["reference_source"] == "all calibration blocks 0-19"
    assert log_payload["session_k"] == _online_k("SBJ01", "S01")


def test_run_step4_and_smoke_test_outputs_validate_generated_files() -> None:
    root = _pytest_local_dir("step4_run_and_smoke")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _populate_valid_step3_inputs(feature_root=feature_root, step3_config=step3_config, step4_config=step4_config)

    summary = run_step4(step4_config)

    assert summary.calibration_aligned_file_count == 105
    assert summary.online_aligned_file_count == 105
    assert summary.calibration_tangent_file_count == 105
    assert summary.online_tangent_file_count == 105
    assert summary.alignment_log_count == 105
    assert summary.total_aligned_matrices == 7350
    assert summary.total_tangent_vectors == 7350
    assert summary.identity_error_max <= step4_config.identity_error_tolerance
    assert summary.zero_variance_dimensions == 0
    assert summary.warning_session_count == 0

    smoke_summary = smoke_test_outputs(step4_config)
    assert smoke_summary.total_aligned_matrices == 7350
    assert smoke_summary.total_tangent_vectors == 7350


def test_smoke_test_outputs_rejects_mismatched_session_ref_mean_between_artifacts() -> None:
    root = _pytest_local_dir("step4_ref_mean_mismatch")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _populate_valid_step3_inputs(feature_root=feature_root, step3_config=step3_config, step4_config=step4_config)
    run_step4(step4_config)

    online_aligned_path = aligned_split_path(
        output_root=feature_root,
        subject_id="SBJ01",
        session_id="S01",
        split="online",
    )
    with np.load(online_aligned_path, allow_pickle=False) as npz_file:
        payload = {name: np.asarray(npz_file[name]) for name in npz_file.files}
    payload["session_ref_mean"] = payload["session_ref_mean"].copy()
    payload["session_ref_mean"][0, 0] = np.float64(payload["session_ref_mean"][0, 0] + 0.5)
    np.savez_compressed(online_aligned_path, **payload)

    with pytest.raises(ValueError, match="session_ref_mean arrays do not match"):
        smoke_test_outputs(step4_config)


def test_smoke_test_outputs_rejects_log_session_k_mismatch() -> None:
    root = _pytest_local_dir("step4_log_k_mismatch")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _populate_valid_step3_inputs(feature_root=feature_root, step3_config=step3_config, step4_config=step4_config)
    run_step4(step4_config)

    log_path = alignment_log_path(output_root=feature_root, subject_id="SBJ01", session_id="S01")
    original_text = log_path.read_text(encoding="utf-8")
    mutated_text = original_text.replace("Session K: 3", "Session K: 4", 1)
    log_path.write_text(mutated_text, encoding="utf-8")

    with pytest.raises(ValueError, match="log Session K"):
        smoke_test_outputs(step4_config)


def test_smoke_test_outputs_enforces_raw_identity_error_threshold() -> None:
    root = _pytest_local_dir("step4_identity_error_raw_threshold")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _populate_valid_step3_inputs(feature_root=feature_root, step3_config=step3_config, step4_config=step4_config)
    run_step4(step4_config)

    calibration_aligned_path = aligned_split_path(
        output_root=feature_root,
        subject_id="SBJ01",
        session_id="S01",
        split="calibration",
    )
    aligned_payload = _load_npz_payload(calibration_aligned_path)
    epsilon = 1.4e-6
    biased_covariance = np.eye(step4_config.covariance_dim, dtype=np.float64)
    biased_covariance[0, 0] = np.float64(1.0 + epsilon)
    aligned_payload["covs"] = np.repeat(
        biased_covariance[None, :, :],
        step4_config.calibration_blocks,
        axis=0,
    )
    np.savez_compressed(calibration_aligned_path, **aligned_payload)

    log_path = alignment_log_path(output_root=feature_root, subject_id="SBJ01", session_id="S01")
    mutated_log = log_path.read_text(encoding="utf-8")
    mutated_log = mutated_log.replace(
        "Identity-centering error: 0.000000",
        "Identity-centering error: 0.000001",
        1,
    )
    mutated_log = mutated_log.replace(
        "Aligned calibration value range: 0.000000 to 1.000000",
        "Aligned calibration value range: 0.000000 to 1.000001",
        1,
    )
    log_path.write_text(mutated_log, encoding="utf-8")

    with pytest.raises(ValueError, match="identity error"):
        smoke_test_outputs(step4_config)


def test_validate_saved_aligned_payload_rejects_unexpected_extra_fields() -> None:
    root = _pytest_local_dir("step4_aligned_extra_fields")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _populate_valid_step3_inputs(feature_root=feature_root, step3_config=step3_config, step4_config=step4_config)
    run_step4(step4_config)

    calibration_aligned_path = aligned_split_path(
        output_root=feature_root,
        subject_id="SBJ01",
        session_id="S01",
        split="calibration",
    )
    aligned_payload = _load_npz_payload(calibration_aligned_path)
    aligned_payload["P_target"] = np.zeros((8, 128), dtype=np.float64)
    np.savez_compressed(calibration_aligned_path, **aligned_payload)

    with pytest.raises(ValueError, match="unexpected Step 4 fields"):
        _validate_saved_aligned_payload(
            path=calibration_aligned_path,
            split="calibration",
            config=step4_config,
        )


def test_validate_saved_tangent_payload_rejects_unexpected_extra_fields() -> None:
    root = _pytest_local_dir("step4_tangent_extra_fields")
    feature_root = root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    step3_config = Step3Config(output_root=feature_root)
    step4_config = Step4Config(input_root=feature_root, output_root=feature_root)
    _populate_valid_step3_inputs(feature_root=feature_root, step3_config=step3_config, step4_config=step4_config)
    run_step4(step4_config)

    calibration_ts_path = ts_split_path(
        output_root=feature_root,
        subject_id="SBJ01",
        session_id="S01",
        split="calibration",
    )
    tangent_payload = _load_npz_payload(calibration_ts_path)
    tangent_payload["condition_numbers"] = np.zeros(step4_config.calibration_blocks, dtype=np.float64)
    np.savez_compressed(calibration_ts_path, **tangent_payload)

    with pytest.raises(ValueError, match="unexpected Step 4 fields"):
        _validate_saved_tangent_payload(
            path=calibration_ts_path,
            split="calibration",
            config=step4_config,
        )
