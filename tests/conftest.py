"""Identify tests that need artifacts from the complete upstream EEG pipeline."""
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "data/features/sub-01_ses-01_split-online_ts.npz",
        root / "data/forecasting/step6_h10_manifest.csv",
        root / "data/forecasting/step6_h25_manifest.csv",
        root / "data/scores/sub-01_ses-01_split-calibration_scores.npz",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    for item in items:
        needs_data = bool({"table_h05", "table_h10", "table_h25"}.intersection(item.fixturenames))
        needs_data |= item.path.name == "test_step6_upstream_target_audit.py" and item.name not in {
            "test_compute_nested_dispersion_shares_sum_to_one",
            "test_target_view_series_and_lagged_metrics_are_deterministic",
        }
        if needs_data:
            item.add_marker(pytest.mark.requires_full_data)
            if missing:
                item.add_marker(pytest.mark.skip(reason="Run the full EEG pipeline first; missing: " + ", ".join(missing)))
