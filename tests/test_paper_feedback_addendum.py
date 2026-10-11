"""Checks for causal information use in the additional paper baselines."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "pipeline/build_paper_feedback_addendum.py"
spec = importlib.util.spec_from_file_location("paper_feedback_addendum", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("horizon", [1, 5, 25])
def test_future_mutation_cannot_change_anchor_predictions(horizon):
    scores = np.random.default_rng(73).normal(size=50)
    original = module.causal_predictions(scores, 2.0, horizon)
    anchor = 15
    altered = scores.copy()
    altered[anchor + 1:] += 1000
    changed = module.causal_predictions(altered, 2.0, horizon)
    fields = ["context_score_mean", "calibration_mean", "expanding_online_mean"]
    np.testing.assert_array_equal(original.loc[original.anchor_idx == anchor, fields],
                                  changed.loc[changed.anchor_idx == anchor, fields])
    assert original.loc[original.anchor_idx == anchor, "y_true_raw"].item() != \
           changed.loc[changed.anchor_idx == anchor, "y_true_raw"].item()


def test_expanding_mean_includes_earlier_observed_blocks():
    original = np.zeros(50)
    altered = original.copy()
    altered[0] = 20
    a = module.causal_predictions(original, 2.0, 5)
    b = module.causal_predictions(altered, 2.0, 5)
    assert a.loc[a.anchor_idx == 19, "context_score_mean"].item() == \
           b.loc[b.anchor_idx == 19, "context_score_mean"].item()
    assert b.loc[b.anchor_idx == 19, "expanding_online_mean"].item() == 1


def test_empty_correlation_set_remains_undefined():
    assert np.isnan(module.ci(np.array([]))).all()


def test_calibration_fit_excludes_all_held_out_online_outcomes():
    rng = np.random.default_rng(74)
    calibration = rng.uniform(1, 3, 105)
    sessions = pd.DataFrame({
        "subject_id": np.repeat(module.SUBJECTS, 7),
        "avatar_calibration_mean": calibration,
        "online_mean": -.4 + .98 * calibration + rng.normal(0, .03, 105),
    })
    held_out = "SBJ08"
    original = module.fit_calibration_correction(sessions, held_out)
    changed = sessions.copy()
    changed.loc[changed.subject_id == held_out, "online_mean"] += 1000
    assert module.fit_calibration_correction(changed, held_out) == original
    assert held_out not in original["training_subject_ids"].split("|")
    # Positive control: training outcomes must affect the fitted level map.
    changed.loc[changed.subject_id != held_out, "online_mean"] += 1
    training_changed = module.fit_calibration_correction(changed, held_out)
    assert training_changed["intercept"] == pytest.approx(original["intercept"] + 1)
    assert training_changed["slope"] == pytest.approx(original["slope"])
