"""Compare the restored training procedure with the preserved original GRU."""

import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.paper_artifacts import resolve_artifact
from src.experiments import exp41_historical_step8 as original
from src.models import step8_temporal as current
from src.features.step6_forecasting_task import load_csv_rows

path = Path(__file__).resolve().parents[1] / "pipeline/reproduce_paper_gru.py"
spec = importlib.util.spec_from_file_location("paper_reproduction", path)
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


@pytest.mark.parametrize("features", [1, 4])
def test_paper_training_and_refit_match_original_without_clipping(features):
    torch.set_num_threads(1)
    rng = np.random.default_rng(251)
    x = rng.normal(size=(24, 10, features)).astype(np.float32)
    y = (3 + 2 * x.mean(axis=(1, 2))).astype(np.float32)
    settings = dict(input_dim=features, hidden_dim=8, learning_rate=.001,
                    dropout=.2, weight_decay=.0001, sigma_floor=1e-6,
                    batch_size=6, seed=730)
    training = dict(x_train=x[:18], y_train=y[:18], x_val=x[18:], y_val=y[18:],
                    max_epochs=4, patience=2, **settings)
    historical = original._train_with_validation(**training)
    restored = current._train_with_validation(**training, gradient_clip_max_norm=None)
    assert historical[0] == restored[0]
    assert historical[1] == pytest.approx(restored[1], abs=1e-7)
    np.testing.assert_allclose(historical[2], restored[2], rtol=0, atol=1e-7)
    refitting = dict(x_train=x[:18], y_train=y[:18], x_test=x[18:],
                     epochs=historical[0], **settings)
    first = original._refit_and_predict(**refitting)
    second = current._refit_and_predict(**refitting, gradient_clip_max_norm=None)
    for expected, actual in zip(first, second):
        np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-7)


def test_paper_command_restores_protocol_and_preserves_current_defaults():
    normal = current.Step8Config()
    assert (normal.max_epochs, normal.patience, normal.gradient_clip_max_norm) == (100, 10, 1.0)
    for experiment in ["main", "exp30", "exp31"]:
        config = entrypoint.paper_config(experiment)
        assert (config.max_epochs, config.patience, config.gradient_clip_max_norm) == (200, 20, None)
        assert config.primary_inner_protocol == ("last2_subject_holdout" if experiment == "main" else "group_kfold")
        assert config.dropout_grid == ((0., .2) if experiment == "exp31" else (.2,))


def test_saved_csv_resolution_prefers_regenerated_csv(tmp_path):
    csv = tmp_path / "predictions.csv"
    compressed = tmp_path / "predictions.csv.gz"
    with gzip.open(compressed, "wt") as stream:
        stream.write("y_true,y_pred\n1,1\n")
    assert resolve_artifact(csv) == compressed
    assert load_csv_rows(csv) == (["y_true", "y_pred"], [{"y_true": "1", "y_pred": "1"}])
    csv.write_text("y_true,y_pred\n2,2\n")
    assert resolve_artifact(csv) == csv
    assert load_csv_rows(csv) == (["y_true", "y_pred"], [{"y_true": "2", "y_pred": "2"}])
