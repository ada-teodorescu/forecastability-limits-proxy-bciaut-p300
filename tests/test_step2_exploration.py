"""Unit tests for Step 2 ERP exploration helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.step2_exploration import (
    Step2Config,
    SubjectP300Summary,
    build_class_balance_row,
    compute_peak_metrics,
    compute_within_session_variability,
    parse_k_values_from_inventory,
    select_representative_session,
    select_representative_subjects,
)


def _make_online_payload(*, identical_targets: bool) -> dict[str, np.ndarray]:
    n_blocks = 50
    epochs_per_block = 3  # 2 target + 1 non-target
    n_epochs = n_blocks * epochs_per_block
    epochs = np.zeros((n_epochs, 8, 128), dtype=np.float64)
    labels = np.tile(np.array([1, 1, 0], dtype=np.int8), n_blocks)
    block_ids = np.repeat(np.arange(n_blocks, dtype=np.int16), epochs_per_block)

    for block_id in range(n_blocks):
        start = block_id * epochs_per_block
        first_value = float(block_id + 1)
        second_value = first_value if identical_targets else first_value + 1.0
        epochs[start, 5, 64:91] = first_value
        epochs[start + 1, 5, 64:91] = second_value

    return {
        "epochs": epochs,
        "labels": labels,
        "block_ids": block_ids,
    }


def test_parse_k_values_from_inventory_reads_step1_table_format() -> None:
    tmp_dir = Path(".pytest-local")
    tmp_dir.mkdir(exist_ok=True)
    inventory_path = tmp_dir / "dataset_inventory_step2_test.txt"
    lines = [
        "BCIAUT-P300 Dataset Inventory",
        "K values per session:",
        "--------------------------------------------------",
        "Subject  S01  S02  S03  S04  S05  S06  S07",
    ]
    for subject_index in range(1, 16):
        lines.append(f"SBJ{subject_index:02d}      6   7   8   9   5   6   7")
    inventory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        k_values = parse_k_values_from_inventory(inventory_path)
    finally:
        if inventory_path.exists():
            inventory_path.unlink()

    assert len(k_values) == 105
    assert k_values[("SBJ01", "S01")] == 6
    assert k_values[("SBJ15", "S07")] == 7


def test_compute_peak_metrics_includes_end_sample_of_window() -> None:
    waveform = np.zeros(128, dtype=np.float64)
    waveform[90] = 2.5

    peak = compute_peak_metrics(
        waveform=waveform,
        window=(64, 90),
        fs=128,
        time_offset_ms=-200.0,
    )

    assert peak.amplitude_uV == 2.5
    assert peak.sample_index == 90
    assert peak.latency_ms == (90 * 1000.0 / 128.0) - 200.0


def test_select_representative_subjects_picks_strong_average_weakest() -> None:
    summaries = [
        SubjectP300Summary(
            subject_id=f"SBJ{index:02d}",
            target_waveform_pz=np.zeros(128, dtype=np.float64),
            diff_wave_peak_amplitude_uV=float(index),
            diff_wave_peak_latency_ms=350.0 + index,
            target_peak_amplitude_uV=float(index) + 0.5,
            target_peak_latency_ms=340.0 + index,
        )
        for index in range(1, 16)
    ]

    representatives = select_representative_subjects(summaries)

    assert [row.category for row in representatives] == ["strong", "average", "weakest"]
    assert [row.subject_id for row in representatives] == ["SBJ15", "SBJ08", "SBJ01"]


def test_select_representative_session_prefers_earliest_k_5_or_6() -> None:
    k_values = {
        ("SBJ01", "S01"): 8,
        ("SBJ01", "S02"): 6,
        ("SBJ01", "S03"): 5,
        ("SBJ01", "S04"): 9,
        ("SBJ01", "S05"): 7,
        ("SBJ01", "S06"): 10,
        ("SBJ01", "S07"): 8,
    }

    session_id, k_value = select_representative_session(
        subject_id="SBJ01",
        k_values=k_values,
        preferred_ks=(5, 6),
    )

    assert session_id == "S02"
    assert k_value == 6


def test_compute_within_session_variability_uses_finite_snr() -> None:
    payload = _make_online_payload(identical_targets=False)

    row, snr = compute_within_session_variability(
        payload=payload,
        subject_id="SBJ01",
        session_id="S01",
        config=Step2Config(),
    )

    assert row["subject_id"] == "SBJ01"
    assert row["session"] == "S01"
    assert float(row["noise_floor_uV"]) == 0.5
    assert float(row["snr"]) == pytest.approx(snr)
    assert snr > 1.0


def test_compute_within_session_variability_fails_on_zero_noise_floor() -> None:
    payload = _make_online_payload(identical_targets=True)

    with pytest.raises(ValueError, match="noise_floor_uV == 0"):
        compute_within_session_variability(
            payload=payload,
            subject_id="SBJ01",
            session_id="S01",
            config=Step2Config(),
        )


def test_build_class_balance_row_normalizes_ratio_to_one_to_seven() -> None:
    row = build_class_balance_row(
        subject_id="SBJ01",
        session="S01",
        split="online",
        n_target=300,
        n_nontarget=2100,
    )

    assert row["ratio"] == "1:7"
