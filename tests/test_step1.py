"""Unit tests for Step 1: Data Loading and Preprocessing.

All tests use synthetic numpy arrays — no real data I/O required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.preprocess import (
    CH_NAMES_ARRAY,
    EXPECTED_FLASH_IDS,
    PreprocessingConfig,
    SmokeTestWarning,
    _validate_saved_file,
    apply_winsorizing,
    baseline_correct,
    build_downstream_qc_flags,
    build_npz_payload,
    check_block_sizes,
    check_sequential_flash_structure,
    check_target_flash_match,
    compute_baseline_std_per_channel,
    compute_block_and_run_ids,
    compute_winsor_thresholds,
    downsample_epochs,
    find_noisy_channel_indices,
    format_aborted_structure_log,
    format_downstream_qc_flags_csv,
    format_step1_report,
    hard_clip,
    MANUAL_OVERRIDES,
    process_session,
    run_step1,
    transpose_and_crop,
    validate_split_metadata,
)
from src.data.loader import (
    CALIBRATION_RUNS_PER_BLOCK,
    CHANNEL_NAMES,
    DatasetInventory,
    EVENTS_PER_RUN,
    LoadedSession,
    LoadedSplit,
    MetadataRepairOperation,
    N_CALIBRATION_BLOCKS,
    N_CHANNELS,
    N_ONLINE_BLOCKS,
    SessionInventoryEntry,
    SessionMetadataRepair,
    SessionPaths,
    build_dataset_inventory,
    expected_session_ids,
    load_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_epochs(n_channels: int, n_timepoints: int, n_epochs: int) -> np.ndarray:
    """Create synthetic raw EEG array in [C, T, N] layout."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_channels, n_timepoints, n_epochs)) * 10.0


def _make_calibration_labels(n_blocks: int = 20, runs_per_block: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create valid calibration labels, events, and block_targets.

    Returns:
        (targets, events, block_targets) matching the BCIAUT-P300 structure.
    """
    rng = np.random.default_rng(42)
    n_epochs = n_blocks * EVENTS_PER_RUN * runs_per_block
    block_targets = rng.integers(1, 9, size=n_blocks).astype(np.int8)

    targets = np.zeros(n_epochs, dtype=np.int8)
    events = np.zeros(n_epochs, dtype=np.int8)

    for block in range(n_blocks):
        for run in range(runs_per_block):
            offset = block * (EVENTS_PER_RUN * runs_per_block) + run * EVENTS_PER_RUN
            # Permutation of 1..8
            perm = rng.permutation(np.arange(1, 9)).astype(np.int8)
            events[offset:offset + EVENTS_PER_RUN] = perm
            # Exactly one target matching block_targets[block]
            target_flash = block_targets[block]
            target_idx = np.where(perm == target_flash)[0][0]
            targets[offset + target_idx] = 1

    return targets, events, block_targets


def _make_sbj05_s05_online_corrupted() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic online metadata matching the SBJ05/S05 corruption pattern."""
    labels, events, block_targets = _make_calibration_labels(n_blocks=50, runs_per_block=6)
    block_targets[11] = 7

    corrupted_groups = {
        66: ([1, 3, 2, 4, 5, 7, 6, 8], 5),
        67: ([2, 5, 7, 6, 8, 4, 1, 6], 2),
        68: ([4, 3, 8, 5, 7, 1, 2, 3], 4),
        69: ([1, 7, 4, 2, 6, 8, 5, 6], 1),
        70: ([4, 7, 5, 2, 1, 8, 3, 8], 1),
        71: ([1, 5, 7, 4, 2, 6, 3, 3], 2),
    }

    for group_idx, (group_flash, target_pos) in corrupted_groups.items():
        start = group_idx * EVENTS_PER_RUN
        end = start + EVENTS_PER_RUN
        events[start:end] = np.array(group_flash, dtype=np.int8)
        labels[start:end] = 0
        labels[start + target_pos] = 1

    return labels, events, block_targets


def _make_registered_repair() -> SessionMetadataRepair:
    """Return the deterministic SBJ05/S05 online flash_ids repair object."""
    return SessionMetadataRepair(
        subject_id="SBJ05",
        session_id="S05",
        split="online",
        field="flash_ids",
        operations=(
            MetadataRepairOperation(index=543, old_value=6, new_value=3),
            MetadataRepairOperation(index=551, old_value=3, new_value=6),
            MetadataRepairOperation(index=559, old_value=6, new_value=3),
            MetadataRepairOperation(index=567, old_value=8, new_value=6),
            MetadataRepairOperation(index=575, old_value=3, new_value=8),
        ),
        reason=(
            "Deterministic repair for SBJ05/S05 online testEvents: five runs in "
            "block 11 contain one duplicated non-target flash and one missing "
            "flash, while the target flash remains correct."
        ),
    )


def _load_mock_sbj05_s05_session(monkeypatch: pytest.MonkeyPatch) -> LoadedSession:
    """Load SBJ05/S05 through the real loader path with monkeypatched file I/O."""
    cal_labels, cal_flash, cal_block_targets = _make_calibration_labels(
        n_blocks=20, runs_per_block=10,
    )
    online_labels, online_flash, online_block_targets = _make_sbj05_s05_online_corrupted()
    cal_epochs = np.zeros((8, 350, 1600), dtype=np.float64)
    online_epochs = np.zeros((8, 350, 2400), dtype=np.float64)

    paths = SessionPaths(
        subject_id="SBJ05",
        session_id="S05",
        train_dir=Path("SBJ05/S05/Train"),
        test_dir=Path("SBJ05/S05/Test"),
        train_mat=Path("trainData.mat"),
        test_mat=Path("testData.mat"),
        train_events=Path("trainEvents.txt"),
        train_targets=Path("trainTargets.txt"),
        train_labels=Path("trainLabels.txt"),
        test_events=Path("testEvents.txt"),
        test_targets=Path("testTargets.txt"),
        test_labels=Path("testLabels.txt"),
        runs_per_block=Path("runs_per_block.txt"),
    )
    entry = SessionInventoryEntry(
        subject_id="SBJ05",
        session_id="S05",
        paths=paths,
        k_value=6,
        train_mat_key="train_key",
        test_mat_key="test_key",
        train_format="scipy",
        test_format="scipy",
    )
    inventory = DatasetInventory(entries={("SBJ05", "S05"): entry})

    def _fake_load_mat(path: Path, key: str, fmt: str) -> np.ndarray:
        if path.name == "trainData.mat":
            return cal_epochs
        if path.name == "testData.mat":
            return online_epochs
        raise AssertionError(f"Unexpected MAT path: {path}")

    def _fake_load_text(path: Path, dtype: type = np.int8) -> np.ndarray:
        mapping = {
            "trainTargets.txt": cal_labels,
            "trainEvents.txt": cal_flash,
            "trainLabels.txt": cal_block_targets,
            "testTargets.txt": online_labels,
            "testEvents.txt": online_flash,
            "testLabels.txt": online_block_targets,
        }
        if path.name not in mapping:
            raise AssertionError(f"Unexpected text path: {path}")
        return mapping[path.name].astype(dtype, copy=True)

    monkeypatch.setattr("src.data.loader._load_mat_array", _fake_load_mat)
    monkeypatch.setattr("src.data.loader._load_text_file", _fake_load_text)

    return load_session(inventory=inventory, subject_id="SBJ05", session_id="S05")


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestTransposeAndCrop:
    def test_shape_conversion(self) -> None:
        raw = _make_raw_epochs(8, 350, 100)
        result = transpose_and_crop(raw, crop_samples=250)
        assert result.shape == (100, 8, 250)

    def test_crop_discards_tail(self) -> None:
        raw = _make_raw_epochs(8, 350, 5)
        full = transpose_and_crop(raw, crop_samples=350)
        cropped = transpose_and_crop(raw, crop_samples=250)
        np.testing.assert_array_equal(cropped, full[:, :, :250])


class TestComputeBlockAndRunIds:
    def test_calibration_layout(self) -> None:
        block_ids, run_ids = compute_block_and_run_ids(1600, runs_per_block=10)
        assert block_ids.shape == (1600,)
        assert run_ids.shape == (1600,)
        assert set(block_ids.tolist()) == set(range(20))
        assert set(run_ids.tolist()) == set(range(10))
        # Each block has 80 epochs
        for bid in range(20):
            assert (block_ids == bid).sum() == 80

    def test_online_layout_k6(self) -> None:
        n = 50 * 8 * 6  # 2400
        block_ids, run_ids = compute_block_and_run_ids(n, runs_per_block=6)
        assert set(block_ids.tolist()) == set(range(50))
        assert set(run_ids.tolist()) == set(range(6))
        for bid in range(50):
            assert (block_ids == bid).sum() == 48  # 8*6

    def test_online_layout_k10(self) -> None:
        n = 50 * 8 * 10  # 4000
        block_ids, run_ids = compute_block_and_run_ids(n, runs_per_block=10)
        assert set(block_ids.tolist()) == set(range(50))
        assert set(run_ids.tolist()) == set(range(10))


# ---------------------------------------------------------------------------
# Processing function tests
# ---------------------------------------------------------------------------


class TestHardClip:
    def test_clips_extreme_values(self) -> None:
        epochs = np.array([[[600.0, -700.0, 100.0, -100.0]]])  # [1, 1, 4]
        clipped, counts = hard_clip(epochs, 500.0)
        assert clipped.max() == 500.0
        assert clipped.min() == -500.0
        assert counts.shape == (1,)  # 1 channel
        assert counts[0] == 2  # two values clipped

    def test_no_clip_within_range(self) -> None:
        epochs = np.ones((5, 8, 250)) * 10.0
        clipped, counts = hard_clip(epochs, 500.0)
        np.testing.assert_array_equal(clipped, epochs)
        assert counts.sum() == 0


class TestBaselineCorrect:
    def test_zeroes_baseline_mean(self) -> None:
        rng = np.random.default_rng(0)
        epochs = rng.standard_normal((10, 8, 250)) * 5.0
        # Add a known offset to baseline region
        epochs[:, :, :50] += 100.0
        corrected = baseline_correct(epochs, baseline_samples=50)
        # Baseline mean should now be ~0
        baseline_mean = corrected[:, :, :50].mean(axis=2)
        np.testing.assert_allclose(baseline_mean, 0.0, atol=1e-10)

    def test_shape_preserved(self) -> None:
        epochs = np.ones((20, 8, 250))
        corrected = baseline_correct(epochs, 50)
        assert corrected.shape == (20, 8, 250)


class TestWinsorThresholds:
    def test_per_channel_computation(self) -> None:
        rng = np.random.default_rng(0)
        epochs = rng.standard_normal((1000, 2, 100))
        lower, upper = compute_winsor_thresholds(epochs, 1.0, 99.0)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        # Thresholds should be within data range
        for ch in range(2):
            ch_data = epochs[:, ch, :].ravel()
            assert lower[ch] >= ch_data.min()
            assert upper[ch] <= ch_data.max()
            assert lower[ch] < upper[ch]


class TestApplyWinsorizing:
    def test_clips_to_thresholds(self) -> None:
        epochs = np.array([
            [[10.0, -10.0, 5.0, -5.0]],
            [[20.0, -20.0, 3.0, -3.0]],
        ])  # [2, 1, 4]
        lower = np.array([-8.0])
        upper = np.array([8.0])
        result = apply_winsorizing(epochs, lower, upper)
        assert result.max() == 8.0
        assert result.min() == -8.0
        # Values within range are unchanged
        assert result[0, 0, 2] == 5.0
        assert result[0, 0, 3] == -5.0


class TestDownsampleEpochs:
    def test_output_shape(self) -> None:
        epochs = np.ones((10, 8, 250))
        result = downsample_epochs(epochs, up=64, down=125)
        assert result.shape == (10, 8, 128)

    def test_channel_axis_preserved(self) -> None:
        """Ensure axis=2 is used, not axis=0."""
        epochs = np.ones((10, 8, 250))
        result = downsample_epochs(epochs, up=64, down=125)
        # First two axes should be unchanged
        assert result.shape[0] == 10
        assert result.shape[1] == 8


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestFlashStructure:
    def test_valid_structure(self) -> None:
        # 3 groups of 8, each a valid permutation
        rng = np.random.default_rng(0)
        flash_ids = np.concatenate([
            rng.permutation(np.arange(1, 9))
            for _ in range(3)
        ]).astype(np.int8)
        assert check_sequential_flash_structure(flash_ids) is True

    def test_invalid_structure(self) -> None:
        # All ones — not a permutation
        flash_ids = np.ones(8, dtype=np.int8)
        assert check_sequential_flash_structure(flash_ids) is False


class TestBlockSizes:
    def test_correct_sizes(self) -> None:
        block_ids, _ = compute_block_and_run_ids(1600, 10)
        failures = check_block_sizes(block_ids, 10)
        assert len(failures) == 0

    def test_detects_wrong_size(self) -> None:
        # Remove one epoch from block 0
        block_ids = np.zeros(79, dtype=np.int16)  # should be 80
        failures = check_block_sizes(block_ids, 10)
        assert len(failures) == 1
        assert failures[0][1] == 79  # actual count


class TestTargetFlashMatch:
    def test_valid_match(self) -> None:
        targets, events, block_targets = _make_calibration_labels(n_blocks=5, runs_per_block=3)
        block_ids, run_ids = compute_block_and_run_ids(len(targets), 3)
        mismatches = check_target_flash_match(
            events, targets, block_ids, run_ids, block_targets, 3,
        )
        assert len(mismatches) == 0

    def test_detects_mismatch(self) -> None:
        targets, events, block_targets = _make_calibration_labels(n_blocks=2, runs_per_block=2)
        # Corrupt: flip first target to 0 and a non-target to 1
        first_target = np.where(targets == 1)[0][0]
        targets[first_target] = 0
        targets[first_target + 1] = 1  # wrong flash
        block_ids, run_ids = compute_block_and_run_ids(len(targets), 2)
        mismatches = check_target_flash_match(
            events, targets, block_ids, run_ids, block_targets, 2,
        )
        assert len(mismatches) > 0


class TestValidateSplitMetadata:
    def test_all_checks_pass(self) -> None:
        targets, events, block_targets = _make_calibration_labels(n_blocks=20, runs_per_block=10)
        block_ids, run_ids = compute_block_and_run_ids(1600, 10)
        result = validate_split_metadata(
            name="test_calibration",
            labels=targets,
            flash_ids=events,
            block_ids=block_ids,
            run_ids=run_ids,
            block_targets=block_targets,
            runs_per_block=10,
            expected_targets=200,
            expected_nontargets=1400,
        )
        assert result["flash_ok"] is True
        assert result["block_size_ok"] is True
        assert result["target_flash_ok"] is True
        assert result["target_count_ok"] is True
        assert result["nontarget_count_ok"] is True


# ---------------------------------------------------------------------------
# NPZ payload tests
# ---------------------------------------------------------------------------


class TestBuildNpzPayload:
    def test_dtypes_and_shapes(self) -> None:
        n = 100
        payload = build_npz_payload(
            epochs=np.ones((n, 8, 128)),
            labels=np.ones(n),
            block_ids=np.zeros(n),
            run_ids=np.zeros(n),
            flash_ids=np.ones(n),
            block_targets=np.ones(20),
            condition=np.ones(n),
            fs=128,
        )
        assert payload["epochs"].dtype == np.float32
        assert payload["epochs"].shape == (n, 8, 128)
        assert payload["labels"].dtype == np.int8
        assert payload["block_ids"].dtype == np.int16
        assert payload["run_ids"].dtype == np.int16
        assert payload["flash_ids"].dtype == np.int8
        assert payload["block_targets"].dtype == np.int8
        assert payload["condition"].dtype == np.int8
        assert payload["fs"].dtype == np.int16
        assert int(payload["fs"]) == 128

    def test_ch_names_is_u4(self) -> None:
        payload = build_npz_payload(
            epochs=np.ones((1, 8, 128)),
            labels=np.zeros(1),
            block_ids=np.zeros(1),
            run_ids=np.zeros(1),
            flash_ids=np.ones(1),
            block_targets=np.ones(1),
            condition=np.zeros(1),
            fs=128,
        )
        ch = payload["ch_names"]
        assert ch.dtype == np.dtype("U4")
        assert ch.shape == (8,)
        assert list(ch) == list(CHANNEL_NAMES)

    def test_roundtrip_npz_no_pickle(self) -> None:
        """Saved NPZ can be loaded without allow_pickle."""
        payload = build_npz_payload(
            epochs=np.ones((10, 8, 128), dtype=np.float32),
            labels=np.zeros(10),
            block_ids=np.zeros(10),
            run_ids=np.zeros(10),
            flash_ids=np.ones(10),
            block_targets=np.ones(5),
            condition=np.zeros(10),
            fs=128,
        )
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **payload)
            loaded = np.load(f.name, allow_pickle=False)
            assert set(loaded.keys()) == set(payload.keys())
            np.testing.assert_array_equal(loaded["epochs"], payload["epochs"])
            np.testing.assert_array_equal(loaded["ch_names"], payload["ch_names"])


# ---------------------------------------------------------------------------
# Critical correctness test: calibration-only thresholds
# ---------------------------------------------------------------------------


class TestCalibrationOnlyThresholds:
    def test_thresholds_from_calibration_only(self) -> None:
        """Winsorizing thresholds must come from calibration, not combined data."""
        rng = np.random.default_rng(42)

        # Calibration: tight distribution around 0
        cal = rng.standard_normal((100, 2, 50)) * 1.0

        # Online: wider distribution — if thresholds came from combined,
        # they would be wider
        online = rng.standard_normal((100, 2, 50)) * 10.0

        cal_lower, cal_upper = compute_winsor_thresholds(cal, 1.0, 99.0)

        # Combined thresholds (what the buggy code would compute)
        combined = np.concatenate([cal, online], axis=0)
        combined_lower, combined_upper = compute_winsor_thresholds(combined, 1.0, 99.0)

        # Calibration-only thresholds should be tighter
        for ch in range(2):
            assert cal_upper[ch] < combined_upper[ch], (
                f"Channel {ch}: calibration upper {cal_upper[ch]:.3f} should be "
                f"less than combined upper {combined_upper[ch]:.3f}"
            )
            assert cal_lower[ch] > combined_lower[ch], (
                f"Channel {ch}: calibration lower {cal_lower[ch]:.3f} should be "
                f"greater than combined lower {combined_lower[ch]:.3f}"
            )


# ---------------------------------------------------------------------------
# QC tests
# ---------------------------------------------------------------------------


class TestBaselineStdComputation:
    def test_per_channel(self) -> None:
        rng = np.random.default_rng(0)
        epochs = rng.standard_normal((50, 4, 100))
        std = compute_baseline_std_per_channel(epochs, baseline_samples=26)
        assert std.shape == (4,)
        assert np.all(std > 0)


class TestFindNoisyChannels:
    def test_detects_noisy(self) -> None:
        baseline_std = np.array([2.0, 5.0, 0.5, 20.0, 3.0, 3.0, 3.0, 3.0])
        indices, names = find_noisy_channel_indices(baseline_std, 1.0, 15.0)
        assert 2 in indices  # 0.5 < 1.0
        assert 3 in indices  # 20.0 > 15.0
        assert CHANNEL_NAMES[2] in names
        assert CHANNEL_NAMES[3] in names

    def test_clean_channels(self) -> None:
        baseline_std = np.array([3.0] * 8)
        indices, names = find_noisy_channel_indices(baseline_std, 1.0, 15.0)
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# Downstream QC flags tests
# ---------------------------------------------------------------------------


class TestDownstreamQcFlags:
    def test_manual_overrides_present(self) -> None:
        flags = build_downstream_qc_flags(warnings=[])
        assert len(flags) == len(MANUAL_OVERRIDES)
        assert all(f.flag_type == "manual_override" for f in flags)

    def test_csv_format(self) -> None:
        flags = build_downstream_qc_flags(warnings=[])
        csv_str = format_downstream_qc_flags_csv(flags)
        lines = csv_str.strip().splitlines()
        assert lines[0] == "subject,session,split,flag_type,channel,details"
        assert len(lines) == len(MANUAL_OVERRIDES) + 1  # header + data rows

    def test_csv_has_no_double_crlf(self) -> None:
        flags = build_downstream_qc_flags(warnings=[])
        csv_str = format_downstream_qc_flags_csv(flags)
        assert "\r\r\n" not in csv_str

    def test_baseline_warning_generates_flag(self) -> None:
        warning = SmokeTestWarning(
            file_path=Path("sub-01_ses-01_split-calibration.npz"),
            warning_type="baseline_std_out_of_range",
            channel_indices=(5,),
            channel_names=("Pz",),
            observed_values_uv=(20.0,),
            lower_bound_uv=1.0,
            upper_bound_uv=15.0,
            message="test",
        )
        flags = build_downstream_qc_flags([warning])
        # Should have manual overrides + 1 channel-level warning
        channel_flags = [f for f in flags if f.flag_type == "channel_level_warning"]
        assert len(channel_flags) == 1
        assert channel_flags[0].channel == "Pz"

    def test_applied_repair_generates_manual_override_row(self) -> None:
        repair = _make_registered_repair()
        flags = build_downstream_qc_flags(warnings=[], applied_repairs=[repair])
        repair_rows = [
            f for f in flags
            if f.subject == "SBJ05" and f.session == "S05" and f.split == "online"
        ]
        assert len(repair_rows) == 1
        assert repair_rows[0].flag_type == "manual_override"
        assert repair_rows[0].channel == "ALL"
        assert "543:6->3" in repair_rows[0].details


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestExpectedSessionIds:
    def test_count(self) -> None:
        ids = expected_session_ids()
        assert len(ids) == 105

    def test_first_and_last(self) -> None:
        ids = expected_session_ids()
        assert ids[0] == ("SBJ01", "S01")
        assert ids[-1] == ("SBJ15", "S07")

    def test_all_unique(self) -> None:
        ids = expected_session_ids()
        assert len(set(ids)) == 105


class TestDatasetInventory:
    def test_blocking_issues_drive_completeness(self) -> None:
        inventory = DatasetInventory(
            entries={},
            missing_files=["missing"],
            unexpected_items=["unexpected"],
            invalid_sessions=["invalid"],
            deviations=[],
        )
        assert inventory.is_complete() is False
        assert inventory.blocking_issues == ["missing", "unexpected", "invalid"]

    def test_build_inventory_records_unexpected_subject_and_session_dirs(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        raw_root.mkdir()
        (raw_root / "SBJ99").mkdir()
        (raw_root / "SBJ01").mkdir()
        (raw_root / "SBJ01" / "S99").mkdir(parents=True)

        inventory = build_dataset_inventory(raw_root)

        assert any("Unexpected subject folder" in item for item in inventory.unexpected_items)
        assert any("Unexpected session folder" in item for item in inventory.unexpected_items)


class TestRunStep1InventoryGate:
    def test_incomplete_inventory_stops_before_session_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inventory = DatasetInventory(
            entries={},
            missing_files=["missing file"],
            unexpected_items=[],
            invalid_sessions=[],
            deviations=[],
        )

        monkeypatch.setattr(
            "src.preprocessing.preprocess.build_dataset_inventory",
            lambda **_: inventory,
        )
        monkeypatch.setattr(
            "src.preprocessing.preprocess.format_dataset_inventory",
            lambda _: "inventory",
        )

        def _fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("run_step1 should not enter the per-session loop")

        monkeypatch.setattr("src.preprocessing.preprocess.load_session", _fail)

        config = PreprocessingConfig(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
        )

        with pytest.raises(ValueError, match="Dataset inventory is incomplete"):
            run_step1(config)


class TestLoadSessionMetadataRepair:
    def test_sbj05_s05_flash_ids_are_repaired_exactly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _load_mock_sbj05_s05_session(monkeypatch)
        repair = session.repairs[0]
        expected = {
            543: 3,
            551: 6,
            559: 3,
            567: 6,
            575: 8,
        }
        assert len(session.repairs) == 1
        assert repair.subject_id == "SBJ05"
        assert repair.session_id == "S05"
        assert repair.split == "online"
        assert repair.field == "flash_ids"
        for index, value in expected.items():
            assert int(session.online.flash_ids[index]) == value

    def test_repaired_online_metadata_passes_step13_checks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _load_mock_sbj05_s05_session(monkeypatch)
        block_ids, run_ids = compute_block_and_run_ids(
            len(session.online.flash_ids),
            session.k_value,
        )
        result = validate_split_metadata(
            name="SBJ05/S05/online",
            labels=session.online.labels,
            flash_ids=session.online.flash_ids,
            block_ids=block_ids,
            run_ids=run_ids,
            block_targets=session.online.block_targets,
            runs_per_block=session.k_value,
            expected_targets=50 * session.k_value,
            expected_nontargets=350 * session.k_value,
        )
        assert result["count_ok"] is True
        assert result["flash_ok"] is True
        assert result["block_size_ok"] is True
        assert result["target_flash_ok"] is True


# ---------------------------------------------------------------------------
# NPZ helper for smoke-test validator tests
# ---------------------------------------------------------------------------


def _save_valid_npz(
    path: Path,
    split: str = "calibration",
    n_blocks: int | None = None,
    runs_per_block: int | None = None,
    ch_names: np.ndarray | None = None,
    flash_ids: np.ndarray | None = None,
    block_targets: np.ndarray | None = None,
    block_ids: np.ndarray | None = None,
    run_ids: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Save a valid NPZ file and return (n_epochs, n_blocks, runs_per_block)."""
    if n_blocks is None:
        n_blocks = N_CALIBRATION_BLOCKS if split == "calibration" else N_ONLINE_BLOCKS
    if runs_per_block is None:
        runs_per_block = CALIBRATION_RUNS_PER_BLOCK if split == "calibration" else 6
    n_epochs = n_blocks * EVENTS_PER_RUN * runs_per_block

    rng = np.random.default_rng(42)
    epochs = rng.standard_normal((n_epochs, N_CHANNELS, 128)).astype(np.float32)

    if block_ids is None or run_ids is None:
        _block_ids, _run_ids = compute_block_and_run_ids(n_epochs, runs_per_block)
        if block_ids is None:
            block_ids = _block_ids
        if run_ids is None:
            run_ids = _run_ids

    if flash_ids is None:
        # Valid permutations
        n_runs = n_blocks * runs_per_block
        flash_ids = np.concatenate([
            rng.permutation(np.arange(1, 9)) for _ in range(n_runs)
        ]).astype(np.int8)

    labels = np.zeros(n_epochs, dtype=np.int8)
    # Set 1:7 target ratio
    for i in range(0, n_epochs, EVENTS_PER_RUN):
        labels[i] = 1

    if block_targets is None:
        block_targets = rng.integers(1, 9, size=n_blocks).astype(np.int8)

    condition = np.zeros(n_epochs, dtype=np.int8) if split == "calibration" else np.ones(n_epochs, dtype=np.int8)

    payload = {
        "epochs": epochs,
        "labels": labels,
        "block_ids": block_ids.astype(np.int16),
        "run_ids": run_ids.astype(np.int16),
        "flash_ids": flash_ids.astype(np.int8),
        "block_targets": block_targets.astype(np.int8),
        "condition": condition.astype(np.int8),
        "fs": np.int16(128),
        "ch_names": ch_names if ch_names is not None else CH_NAMES_ARRAY,
    }
    np.savez_compressed(str(path), **payload)
    return n_epochs, n_blocks, runs_per_block


# ---------------------------------------------------------------------------
# Fix 1 tests: validation_flags + per-split rows
# ---------------------------------------------------------------------------


class TestValidationFlagsPropagation:
    def test_per_split_flags_no_both(self) -> None:
        """build_downstream_qc_flags must not emit split='both'."""
        validation_flags = {
            "SBJ01/S01": ["BLOCK_SIZE_FAIL_calibration", "TARGET_FLASH_MISMATCH_online"],
        }
        flags = build_downstream_qc_flags(warnings=[], validation_flags=validation_flags)
        validation_rows = [f for f in flags if f.flag_type == "session_level_warning"]
        assert len(validation_rows) == 2
        splits = {f.split for f in validation_rows}
        assert "both" not in splits
        assert "calibration" in splits
        assert "online" in splits
        # Check details are the base flag name (without _calibration/_online suffix)
        details = {f.details for f in validation_rows}
        assert "BLOCK_SIZE_FAIL" in details
        assert "TARGET_FLASH_MISMATCH" in details

    def test_empty_validation_flags(self) -> None:
        """No validation flags → only manual overrides."""
        flags = build_downstream_qc_flags(warnings=[], validation_flags={})
        assert all(f.flag_type == "manual_override" for f in flags)


# ---------------------------------------------------------------------------
# Fix 2 tests: smoke-test validator additional checks
# ---------------------------------------------------------------------------


class TestSmokeValidatorChNamesContent:
    def test_wrong_ch_names_triggers_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            wrong_names = np.array(["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"], dtype="U4")
            n_epochs, n_blocks, rpb = _save_valid_npz(path, ch_names=wrong_names)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="calibration",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "ch_names_content" in types

    def test_correct_ch_names_no_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            n_epochs, n_blocks, rpb = _save_valid_npz(path)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="calibration",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "ch_names_content" not in types
            assert "ch_names_mismatch" not in types


class TestSmokeValidatorFlashIdsRange:
    def test_flash_ids_with_zero_triggers_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            rng = np.random.default_rng(99)
            n_blocks = N_CALIBRATION_BLOCKS
            rpb = CALIBRATION_RUNS_PER_BLOCK
            n_epochs = n_blocks * EVENTS_PER_RUN * rpb
            # Flash IDs with 0 (out of range)
            bad_flash = rng.integers(0, 8, size=n_epochs).astype(np.int8)  # includes 0
            _save_valid_npz(path, flash_ids=bad_flash)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="calibration",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "flash_ids_range" in types

    def test_flash_ids_with_nine_triggers_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            n_blocks = N_CALIBRATION_BLOCKS
            rpb = CALIBRATION_RUNS_PER_BLOCK
            n_epochs = n_blocks * EVENTS_PER_RUN * rpb
            bad_flash = np.ones(n_epochs, dtype=np.int8) * 9
            _save_valid_npz(path, flash_ids=bad_flash)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="calibration",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "flash_ids_range" in types


class TestSmokeValidatorBlockIdsContiguous:
    def test_noncontiguous_block_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            n_blocks = N_CALIBRATION_BLOCKS
            rpb = CALIBRATION_RUNS_PER_BLOCK
            n_epochs = n_blocks * EVENTS_PER_RUN * rpb
            # Create block_ids that skip block 5
            block_ids, run_ids = compute_block_and_run_ids(n_epochs, rpb)
            block_ids[block_ids == 5] = 6  # duplicate block 6, skip block 5
            _save_valid_npz(path, block_ids=block_ids, run_ids=run_ids)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="calibration",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "block_ids_not_contiguous" in types


class TestSmokeValidatorOnlineEpochBand:
    def test_too_few_online_epochs_warns(self) -> None:
        """Online epochs below 1200 should trigger sanity warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            # K=3 → 50*8*3 = 1200 (boundary), use n_blocks < 50 to get below
            # Actually let's just make a small file with 800 epochs
            n_blocks = 10
            rpb = 3
            n_epochs = n_blocks * EVENTS_PER_RUN * rpb  # 10*8*3 = 240
            _save_valid_npz(path, split="online", n_blocks=n_blocks, runs_per_block=rpb)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="online",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "epoch_count_sanity" in types


class TestSmokeValidatorMetadataShapes:
    def test_wrong_labels_shape_triggers_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            n_epochs, n_blocks, rpb = _save_valid_npz(path)
            with np.load(str(path), allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
            payload["labels"] = payload["labels"][:-1]
            np.savez_compressed(str(path), **payload)
            result = _validate_saved_file(
                file_path=path,
                expected_epochs=n_epochs,
                expected_bt_len=n_blocks,
                k_value=rpb,
                split="calibration",
                config=PreprocessingConfig(),
            )
            types = [w.warning_type for w in result.warnings]
            assert "metadata_shape" in types


# ---------------------------------------------------------------------------
# Fix 3 tests: aborted structure log
# ---------------------------------------------------------------------------


class TestAbortedStructureLog:
    def test_contains_abort_status(self) -> None:
        """Aborted structure log must contain EPOCH_ORDER_FAIL status."""
        from src.data.loader import LoadedSession, LoadedSplit, SessionPaths

        # Create minimal mock session
        rng = np.random.default_rng(0)
        cal_flash = rng.permutation(np.tile(np.arange(1, 9), 200)).astype(np.int8)
        online_flash = np.ones(2400, dtype=np.int8)  # bad — all 1s

        cal_validation = {
            "flash_ok": True, "block_size_ok": True, "target_flash_ok": True,
            "target_count": 200, "nontarget_count": 1400, "count_ok": True,
        }
        online_validation = {
            "flash_ok": False, "block_size_ok": True, "target_flash_ok": True,
            "target_count": 300, "nontarget_count": 2100, "count_ok": True,
        }

        config = PreprocessingConfig()
        cal_block_ids, _ = compute_block_and_run_ids(1600, 10)
        online_block_ids, _ = compute_block_and_run_ids(2400, 6)

        # We need a LoadedSession-like object — use a simple namespace
        class FakeSession:
            subject_id = "SBJ05"
            session_id = "S05"
            sub_label = "sub-05"
            ses_label = "ses-05"
            k_value = 6

        log_text = format_aborted_structure_log(
            session=FakeSession(),
            config=config,
            cal_validation=cal_validation,
            online_validation=online_validation,
            cal_flash_ids=cal_flash,
            online_flash_ids=online_flash,
            cal_block_ids=cal_block_ids,
            online_block_ids=online_block_ids,
            cal_raw_min=-100.0,
            cal_raw_max=100.0,
            online_raw_min=-200.0,
            online_raw_max=200.0,
            abort_reason="EPOCH_ORDER_FAIL",
        )
        assert "ABORTED" in log_text
        assert "EPOCH_ORDER_FAIL" in log_text
        assert "SBJ05/S05" in log_text
        assert "Flash-ID permutation (online): FAIL" in log_text
        assert "no hard-clip" in log_text.lower() or "Processing aborted" in log_text


class TestProcessSessionTargetCountMismatch:
    def test_target_count_mismatch_aborts_before_save(self, tmp_path: Path) -> None:
        cal_labels, cal_flash, cal_block_targets = _make_calibration_labels(n_blocks=1, runs_per_block=10)
        online_labels, online_flash, online_block_targets = _make_calibration_labels(n_blocks=1, runs_per_block=6)
        cal_labels[:] = 0

        session = LoadedSession(
            subject_id="SBJ01",
            session_id="S01",
            sub_label="sub-01",
            ses_label="ses-01",
            calibration=LoadedSplit(
                epochs=np.zeros((8, 350, 80), dtype=np.float32),
                labels=cal_labels,
                flash_ids=cal_flash,
                block_targets=cal_block_targets,
            ),
            online=LoadedSplit(
                epochs=np.zeros((8, 350, 48), dtype=np.float32),
                labels=online_labels,
                flash_ids=online_flash,
                block_targets=online_block_targets,
            ),
            k_value=6,
            train_mat_key="data",
            test_mat_key="data",
            train_format="scipy",
            test_format="scipy",
        )
        config = PreprocessingConfig(processed_root=tmp_path)

        with pytest.raises(ValueError, match="TARGET_COUNT_MISMATCH"):
            process_session(session, config)

        log_path = tmp_path / "sub-01_ses-01_structure_log.txt"
        assert log_path.is_file()
        assert "TARGET_COUNT_MISMATCH" in log_path.read_text(encoding="utf-8")


class TestProcessSessionFlashFailure:
    def test_unregistered_flash_failure_still_aborts(self, tmp_path: Path) -> None:
        cal_labels, cal_flash, cal_block_targets = _make_calibration_labels(
            n_blocks=20, runs_per_block=10,
        )
        online_labels, online_flash, online_block_targets = _make_calibration_labels(
            n_blocks=50, runs_per_block=6,
        )
        online_flash[7] = online_flash[0]

        session = LoadedSession(
            subject_id="SBJ04",
            session_id="S04",
            sub_label="sub-04",
            ses_label="ses-04",
            calibration=LoadedSplit(
                epochs=np.zeros((8, 350, 1600), dtype=np.float32),
                labels=cal_labels,
                flash_ids=cal_flash,
                block_targets=cal_block_targets,
            ),
            online=LoadedSplit(
                epochs=np.zeros((8, 350, 2400), dtype=np.float32),
                labels=online_labels,
                flash_ids=online_flash,
                block_targets=online_block_targets,
            ),
            k_value=6,
            train_mat_key="data",
            test_mat_key="data",
            train_format="scipy",
            test_format="scipy",
        )
        config = PreprocessingConfig(processed_root=tmp_path)

        result = process_session(session, config)

        assert result.aborted is True
        log_path = tmp_path / "sub-04_ses-04_structure_log.txt"
        assert log_path.is_file()
        assert "EPOCH_ORDER_FAIL" in log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix 4 tests: observed values in structure log
# ---------------------------------------------------------------------------


class TestStructureLogObservedValues:
    def test_no_hardcoded_flash_ids(self) -> None:
        """Structure log must show observed flash IDs, not hardcoded {1..8}."""
        from src.preprocessing.preprocess import format_structure_log

        repair = _make_registered_repair()

        class FakeSession:
            subject_id = "SBJ01"
            session_id = "S01"
            sub_label = "sub-01"
            ses_label = "ses-01"
            k_value = 6
            repairs = (repair,)

        config = PreprocessingConfig()

        # Use a subset of flash IDs to distinguish from hardcoded full set
        cal_flash = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
        online_flash = np.array([1, 2, 3], dtype=np.int8)  # only 3 unique

        cal_block_ids, _ = compute_block_and_run_ids(1600, 10)
        online_block_ids, _ = compute_block_and_run_ids(2400, 6)

        cal_validation = {
            "flash_ok": True, "block_size_ok": True, "target_flash_ok": True,
            "target_count": 200, "nontarget_count": 1400, "count_ok": True,
        }
        online_validation = {
            "flash_ok": True, "block_size_ok": True, "target_flash_ok": True,
            "target_count": 300, "nontarget_count": 2100, "count_ok": True,
        }

        cal_qc = type("SplitQcSummary", (), {
            "baseline_std_uv": np.ones(N_CHANNELS) * 3.0,
            "noisy_channel_indices": (),
            "noisy_channel_names": (),
        })()
        session_qc = type("SessionQcSummary", (), {
            "calibration": cal_qc,
            "online": cal_qc,
            "noisy_channel_indices": (),
            "noisy_channel_names": (),
            "qc_status": "clean",
        })()

        log_text = format_structure_log(
            session=FakeSession(),
            config=config,
            cal_validation=cal_validation,
            online_validation=online_validation,
            cal_clip_counts=np.zeros(N_CHANNELS),
            online_clip_counts=np.zeros(N_CHANNELS),
            winsor_lower=np.ones(N_CHANNELS) * -10,
            winsor_upper=np.ones(N_CHANNELS) * 10,
            session_qc=session_qc,
            cal_flash_ids=cal_flash,
            online_flash_ids=online_flash,
            cal_block_ids=cal_block_ids,
            online_block_ids=online_block_ids,
            cal_raw_min=-50.0,
            cal_raw_max=50.0,
            online_raw_min=-80.0,
            online_raw_max=80.0,
        )
        # Online only has {1,2,3} — should appear in log, NOT {1..8}
        assert "Unique flash IDs (online): {1, 2, 3}" in log_text
        # Raw amplitude range should appear
        assert "min=-50.0000" in log_text
        assert "max=80.0000" in log_text
        # Block IDs should appear
        assert "Unique block IDs (calibration):" in log_text
        assert "Applied metadata repairs:" in log_text
        assert "543:6->3" in log_text


class TestStep1Report:
    def test_report_includes_execution_summary(self, tmp_path: Path) -> None:
        config = PreprocessingConfig(processed_root=tmp_path)
        inventory = DatasetInventory(
            entries={},
            missing_files=[],
            unexpected_items=[],
            invalid_sessions=[],
            deviations=["Non-uniform train MAT keys"],
        )

        report = format_step1_report(
            config=config,
            inventory=inventory,
            processed_sessions=["SBJ01/S01"],
            aborted_sessions=["SBJ02/S02"],
            warnings=[],
            flags=[],
            applied_repairs=[_make_registered_repair()],
        )

        assert "Processed sessions: 1" in report
        assert "Aborted sessions: 1" in report
        assert "Saved NPZ files: 2" in report
        assert "Step 1 report:" in report
        assert "Applied metadata repairs: 1" in report
        assert "543:6->3" in report
