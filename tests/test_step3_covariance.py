"""Unit tests for Step 3 covariance helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.covariance import OAS

from src.data.loader import CHANNEL_NAMES, expected_session_ids
from src.features.step3_covariance import (
    _validate_saved_covariance_payload,
    Step3Config,
    aggregate_blocks,
    build_session_prototypes,
    estimate_oas_covariance,
    feature_split_path,
    load_processed_npz,
    parse_k_values_from_inventory,
    session_log_path,
    smoke_test_outputs,
    validate_session_log_contents,
)


def _pytest_local_dir(name: str) -> Path:
    root = Path(".pytest-local") / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_well_formed_payload(
    *,
    split: str,
    runs_per_block: int,
    config: Step3Config | None = None,
) -> dict[str, np.ndarray]:
    config = config or Step3Config()
    n_blocks = config.calibration_blocks if split == "calibration" else config.online_blocks
    n_epochs = n_blocks * runs_per_block * config.events_per_run

    epochs = np.zeros((n_epochs, config.n_channels, config.epoch_samples), dtype=np.float64)
    labels = np.zeros(n_epochs, dtype=np.int8)
    block_ids = np.zeros(n_epochs, dtype=np.int16)
    run_ids = np.zeros(n_epochs, dtype=np.int16)
    flash_ids = np.zeros(n_epochs, dtype=np.int16)
    condition = np.zeros(n_epochs, dtype=np.int8)
    block_targets = ((np.arange(n_blocks, dtype=np.int16) % config.events_per_run) + 1).astype(np.int16)

    index = 0
    for block_id in range(n_blocks):
        target_flash = int(block_targets[block_id])
        block_condition = 1 if split == "online" or block_id >= config.prototype_block_start else 0
        for run_id in range(runs_per_block):
            for flash_id in range(1, config.events_per_run + 1):
                block_ids[index] = block_id
                run_ids[index] = run_id
                flash_ids[index] = flash_id
                condition[index] = block_condition
                if flash_id == target_flash:
                    labels[index] = 1
                    epochs[index, :, :] = float(block_id + 1)
                index += 1

    return {
        "epochs": epochs,
        "labels": labels,
        "block_ids": block_ids,
        "run_ids": run_ids,
        "flash_ids": flash_ids,
        "block_targets": block_targets,
        "condition": condition,
        "fs": np.int16(config.output_fs),
        "ch_names": np.asarray(CHANNEL_NAMES),
    }


def _write_valid_step3_outputs(*, output_root: Path, config: Step3Config, subject_id: str, session_id: str, k_value: int) -> None:
    identity_cov = np.eye(config.covariance_dim, dtype=np.float64)
    p_target = np.ones((config.n_channels, config.epoch_samples), dtype=np.float64)
    p_nontarget = np.zeros((config.n_channels, config.epoch_samples), dtype=np.float64)
    prototype_separation = float(np.linalg.norm(p_target - p_nontarget, ord="fro"))
    calibration_condition = np.concatenate(
        [
            np.zeros(config.prototype_block_start, dtype=np.int8),
            np.ones(config.calibration_blocks - config.prototype_block_start, dtype=np.int8),
        ]
    )

    calibration_path = feature_split_path(
        output_root=output_root,
        subject_id=subject_id,
        session_id=session_id,
        split="calibration",
    )
    online_path = feature_split_path(
        output_root=output_root,
        subject_id=subject_id,
        session_id=session_id,
        split="online",
    )
    log_path = session_log_path(output_root=output_root, subject_id=subject_id, session_id=session_id)

    np.savez_compressed(
        calibration_path,
        covs=np.repeat(identity_cov[None, :, :], config.calibration_blocks, axis=0),
        block_ids=np.arange(config.calibration_blocks, dtype=np.int16),
        condition=calibration_condition,
        block_targets=np.tile(np.arange(1, 9, dtype=np.int8), 3)[: config.calibration_blocks],
        runs_per_block=np.full(config.calibration_blocks, config.calibration_runs_per_block, dtype=np.int16),
        n_target_epochs=np.full(config.calibration_blocks, config.calibration_runs_per_block, dtype=np.int16),
        n_nontarget_epochs=np.full(config.calibration_blocks, 7 * config.calibration_runs_per_block, dtype=np.int16),
        P_target=p_target,
        P_nontarget=p_nontarget,
        shrinkage_alphas=np.full(config.calibration_blocks, 0.25, dtype=np.float64),
        condition_numbers=np.ones(config.calibration_blocks, dtype=np.float64),
        lambda_max_over_median=np.ones(config.calibration_blocks, dtype=np.float64),
    )
    np.savez_compressed(
        online_path,
        covs=np.repeat(identity_cov[None, :, :], config.online_blocks, axis=0),
        block_ids=np.arange(config.online_blocks, dtype=np.int16),
        condition=np.ones(config.online_blocks, dtype=np.int8),
        block_targets=np.tile(np.arange(1, 9, dtype=np.int8), 7)[: config.online_blocks],
        runs_per_block=np.full(config.online_blocks, k_value, dtype=np.int16),
        n_target_epochs=np.full(config.online_blocks, k_value, dtype=np.int16),
        n_nontarget_epochs=np.full(config.online_blocks, 7 * k_value, dtype=np.int16),
        P_target=p_target,
        P_nontarget=p_nontarget,
        shrinkage_alphas=np.full(config.online_blocks, 0.50, dtype=np.float64),
        condition_numbers=np.ones(config.online_blocks, dtype=np.float64),
        lambda_max_over_median=np.ones(config.online_blocks, dtype=np.float64),
    )
    log_path.write_text(
        (
            f"Session: {subject_id}/{session_id}\n"
            "Prototype source: social calibration blocks 10-19\n"
            "Prototype epoch counts: 100 target, 700 non-target\n"
            f"Prototype separation Frobenius norm: {prototype_separation:.6f}\n"
            f"Session K: {k_value}\n"
            "Calibration shrinkage alpha range: 0.250000 to 0.250000\n"
            "Online shrinkage alpha range: 0.500000 to 0.500000\n"
            "Calibration condition-number range: 1.000000 to 1.000000\n"
            "Online condition-number range: 1.000000 to 1.000000\n"
            "Calibration dominance-ratio range: 1.000000 to 1.000000\n"
            "Online dominance-ratio range: 1.000000 to 1.000000\n"
            "Warnings:\n"
            "- none\n"
        ),
        encoding="utf-8",
    )


def _populate_valid_step3_output_root(*, output_root: Path, config: Step3Config) -> None:
    for subject_id, session_id in expected_session_ids():
        subject_number = int(subject_id[-2:])
        k_value = 3 + ((subject_number - 1) % 8)
        _write_valid_step3_outputs(
            output_root=output_root,
            config=config,
            subject_id=subject_id,
            session_id=session_id,
            k_value=k_value,
        )


def test_parse_k_values_from_inventory_reads_step1_table_format() -> None:
    tmp_dir = _pytest_local_dir("step3_inventory")
    inventory_path = tmp_dir / "dataset_inventory_step3_test.txt"
    lines = [
        "BCIAUT-P300 Dataset Inventory",
        "K values per session:",
        "--------------------------------------------------",
        "Subject  S01  S02  S03  S04  S05  S06  S07",
    ]
    for subject_index in range(1, 16):
        lines.append(f"SBJ{subject_index:02d}      6   7   8   9   5   6   7")
    inventory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    k_values = parse_k_values_from_inventory(inventory_path)

    assert len(k_values) == 105
    assert k_values[("SBJ01", "S01")] == 6
    assert k_values[("SBJ15", "S07")] == 7


def test_build_session_prototypes_uses_social_blocks_only() -> None:
    config = Step3Config()
    epochs = np.zeros((1600, 8, 128), dtype=np.float64)
    labels = np.zeros(1600, dtype=np.int8)
    block_ids = np.repeat(np.arange(20, dtype=np.int16), 80)
    condition = np.repeat(np.concatenate([np.zeros(10, dtype=np.int8), np.ones(10, dtype=np.int8)]), 80)

    for block_id in range(20):
        start = block_id * 80
        labels[start : start + 10] = 1
        target_value = 5.0 if block_id >= 10 else 1.0
        epochs[start : start + 10, :, :] = target_value

    p_target, p_nontarget, separation = build_session_prototypes(
        {
            "epochs": epochs,
            "labels": labels,
            "block_ids": block_ids,
            "condition": condition,
        },
        config,
    )

    assert np.allclose(p_target, 5.0)
    assert np.allclose(p_nontarget, 0.0)
    assert separation > 0.0


def test_aggregate_blocks_preserves_k_metadata_for_online() -> None:
    config = Step3Config()
    k_value = 6
    payload = _make_well_formed_payload(split="online", runs_per_block=k_value, config=config)

    aggregated = aggregate_blocks(
        payload=payload,
        split="online",
        runs_per_block=k_value,
        config=config,
    )

    assert aggregated.target_averages.shape == (50, 8, 128)
    assert np.all(aggregated.runs_per_block == k_value)
    assert np.all(aggregated.n_target_epochs == k_value)
    assert np.all(aggregated.n_nontarget_epochs == 7 * k_value)
    assert np.all(aggregated.condition == 1)


def test_aggregate_blocks_rejects_two_targets_in_one_run_even_if_block_total_matches() -> None:
    config = Step3Config()
    payload = _make_well_formed_payload(
        split="calibration",
        runs_per_block=config.calibration_runs_per_block,
        config=config,
    )

    run0_extra_target = (payload["block_ids"] == 0) & (payload["run_ids"] == 0) & (payload["flash_ids"] == 2)
    run1_missing_target = (payload["block_ids"] == 0) & (payload["run_ids"] == 1) & (payload["flash_ids"] == 1)
    payload["labels"][run0_extra_target] = 1
    payload["labels"][run1_missing_target] = 0

    with pytest.raises(ValueError, match="labels/flash_ids disagree"):
        aggregate_blocks(
            payload=payload,
            split="calibration",
            runs_per_block=config.calibration_runs_per_block,
            config=config,
        )


def test_load_processed_npz_rejects_block_targets_inconsistent_with_flash_ids() -> None:
    config = Step3Config()
    tmp_dir = _pytest_local_dir("step3_bad_block_targets")
    path = tmp_dir / "sub-01_ses-01_split-calibration.npz"
    payload = _make_well_formed_payload(
        split="calibration",
        runs_per_block=config.calibration_runs_per_block,
        config=config,
    )
    payload["block_targets"] = ((payload["block_targets"] % config.events_per_run) + 1).astype(np.int16)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="labels/flash_ids disagree"):
        load_processed_npz(path, split="calibration", config=config)


def test_estimate_oas_covariance_matches_manually_centered_oas() -> None:
    config = Step3Config()
    rng = np.random.default_rng(42)
    x_aug = rng.normal(size=(24, 128)) + np.linspace(0.5, 2.5, 24, dtype=np.float64)[:, None]

    covariance, alpha, _ = estimate_oas_covariance(x_aug, config)
    centered = x_aug - x_aug.mean(axis=1, keepdims=True)
    estimator = OAS(store_precision=False, assume_centered=True)
    estimator.fit(centered.T)
    expected_covariance = 0.5 * (estimator.covariance_ + estimator.covariance_.T)

    assert np.allclose(covariance, expected_covariance)
    assert alpha == np.float64(estimator.shrinkage_)


def test_smoke_test_outputs_validates_generated_files() -> None:
    root = _pytest_local_dir("step3_smoke")
    output_root = root / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    config = Step3Config(output_root=output_root)
    _populate_valid_step3_output_root(output_root=output_root, config=config)

    summary = smoke_test_outputs(config)

    assert summary.calibration_file_count == 105
    assert summary.online_file_count == 105
    assert summary.session_log_count == 105
    assert summary.total_covariance_matrices == 7350
    assert summary.prototype_separation_min > 0.0
    assert summary.spectral_outlier_blocks == 0
    assert summary.near_singular_sessions == 0


def test_smoke_test_outputs_rejects_log_k_mismatch() -> None:
    root = _pytest_local_dir("step3_log_k_mismatch")
    output_root = root / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    config = Step3Config(output_root=output_root)
    _populate_valid_step3_output_root(output_root=output_root, config=config)

    log_path = session_log_path(output_root=output_root, subject_id="SBJ01", session_id="S01")
    original_text = log_path.read_text(encoding="utf-8")
    log_path.write_text(original_text.replace("Session K: 3", "Session K: 4"), encoding="utf-8")

    with pytest.raises(ValueError, match="Session K=4|does not match saved runs_per_block"):
        smoke_test_outputs(config)


def test_smoke_test_outputs_rejects_prototype_mismatch_between_splits() -> None:
    root = _pytest_local_dir("step3_prototype_mismatch")
    output_root = root / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    config = Step3Config(output_root=output_root)
    _populate_valid_step3_output_root(output_root=output_root, config=config)

    online_path = feature_split_path(
        output_root=output_root,
        subject_id="SBJ01",
        session_id="S01",
        split="online",
    )
    with np.load(online_path, allow_pickle=False) as npz:
        payload = {name: np.asarray(npz[name]) for name in npz.files}
    payload["P_target"] = payload["P_target"] + 1.0
    np.savez_compressed(online_path, **payload)

    with pytest.raises(ValueError, match="calibration and online P_target arrays do not match"):
        smoke_test_outputs(config)


def test_validate_saved_covariance_payload_rejects_wrong_dtype_and_block_targets() -> None:
    root = _pytest_local_dir("step3_bad_saved_payload")
    output_root = root / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    config = Step3Config(output_root=output_root)
    _write_valid_step3_outputs(
        output_root=output_root,
        config=config,
        subject_id="SBJ01",
        session_id="S01",
        k_value=6,
    )

    calibration_path = feature_split_path(
        output_root=output_root,
        subject_id="SBJ01",
        session_id="S01",
        split="calibration",
    )
    with np.load(calibration_path, allow_pickle=False) as npz:
        payload = {name: np.asarray(npz[name]) for name in npz.files}
    payload["block_ids"] = payload["block_ids"].astype(np.int32)
    payload["block_targets"][0] = np.int8(99)
    np.savez_compressed(calibration_path, **payload)

    with pytest.raises(ValueError, match="dtype|block_targets"):
        _validate_saved_covariance_payload(path=calibration_path, split="calibration", config=config)


def test_validate_session_log_contents_rejects_stub_log() -> None:
    tmp_dir = _pytest_local_dir("step3_bad_log")
    log_path = tmp_dir / "sub-01_ses-01_cov_log.txt"
    log_path.write_text(
        "Session: SBJ01/S01\nPrototype source: social calibration blocks 10-19\nWarnings:\n- none\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required log line"):
        validate_session_log_contents(log_path)


def test_validate_session_log_contents_rejects_invalid_k_and_ranges() -> None:
    tmp_dir = _pytest_local_dir("step3_bad_log_content")
    log_path = tmp_dir / "sub-01_ses-01_cov_log.txt"
    log_path.write_text(
        (
            "Session: SBJ01/S01\n"
            "Prototype source: social calibration blocks 10-19\n"
            "Prototype epoch counts: 100 target, 700 non-target\n"
            "Prototype separation Frobenius norm: 1.000000\n"
            "Session K: 999\n"
            "Calibration shrinkage alpha range: nonsense to values\n"
            "Online shrinkage alpha range: 0.500000 to 0.500000\n"
            "Calibration condition-number range: 1.000000 to 1.000000\n"
            "Online condition-number range: 1.000000 to 1.000000\n"
            "Calibration dominance-ratio range: 1.000000 to 1.000000\n"
            "Online dominance-ratio range: 1.000000 to 1.000000\n"
            "Warnings:\n"
            "- none\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Session K=999|invalid range line"):
        validate_session_log_contents(log_path)
