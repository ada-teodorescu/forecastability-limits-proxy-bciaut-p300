"""Estimator parity and leakage checks for the paper's short-horizon extension."""

from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from src.data.forecasting_loader import Step7Config
from src.experiments.exp41_paper_short_horizons import HORIZONS, VIEWS, configs, fit_pcr_shared
from src.models import step7_benchmarks as s7
from src.experiments import exp41_historical_step8 as historical


def _table():
    rng = np.random.default_rng(417)
    subjects = np.repeat([f"SBJ{i:02d}" for i in range(1, 16)], 8)
    x = rng.normal(size=(120, 18)) * np.linspace(.2, 3, 18)
    raw = 2 + .4 * x[:, 0] + rng.normal(scale=.3, size=120)
    return SimpleNamespace(subject_id=subjects, session_id=np.full(120, "S01"),
        context_ts_flat=x, target_score_raw=raw,
        target_score_context_residual=.1 * x[:, 2] + rng.normal(scale=.4, size=120))


def test_reused_full_svd_matches_original_pcr_both_targets():
    table = _table()
    config = Step7Config(pcr_component_grid=(2, 5, 10))
    arguments = dict(table=table, train_idx=np.arange(112), test_idx=np.arange(112, 120),
                     model_name="pcr_context_ts", target_views=VIEWS, config=config, protocol="loso")
    with threadpool_limits(limits=1):
        original = s7._fit_pcr_both_views(**arguments)
        reused = fit_pcr_shared(**arguments)
    for view in VIEWS:
        assert reused[view].selected_n_components == original[view].selected_n_components
        np.testing.assert_allclose(reused[view].test_pred, original[view].test_pred, rtol=0, atol=1e-10)
        np.testing.assert_allclose(reused[view].train_pred, original[view].train_pred, rtol=0, atol=1e-10)
        np.testing.assert_allclose(reused[view].calibration_residuals,
                                   original[view].calibration_residuals, rtol=0, atol=1e-10)


def test_held_out_targets_do_not_affect_pcr_fit_or_selection():
    table = _table()
    config = Step7Config(pcr_component_grid=(2, 5, 10))
    arguments = dict(table=table, train_idx=np.arange(112), test_idx=np.arange(112, 120),
                     model_name="pcr_context_ts", target_views=VIEWS, config=config, protocol="loso")
    with threadpool_limits(limits=1):
        first = fit_pcr_shared(**arguments)
        table.target_score_raw[112:] += 1000
        table.target_score_context_residual[112:] -= 1000
        changed = fit_pcr_shared(**arguments)
    for view in VIEWS:
        assert first[view].selected_n_components == changed[view].selected_n_components
        np.testing.assert_array_equal(first[view].test_pred, changed[view].test_pred)
        np.testing.assert_array_equal(first[view].calibration_residuals, changed[view].calibration_residuals)


def test_extension_preserves_historical_gru_protocol():
    task, classical, neural = configs()
    assert task.horizons == classical.horizons == neural.horizons == HORIZONS
    assert classical.context_length == neural.context_length == 10
    inner_train, inner_validation = historical._inner_validation_split_loso(_table(), np.arange(112))
    assert set(_table().subject_id[inner_validation]) == {"SBJ13", "SBJ14"}
    assert set(inner_train).isdisjoint(inner_validation)
    assert (neural.max_epochs, neural.patience) == (200, 20)
    assert neural.hidden_dim_grid == (32, 64)
    assert neural.learning_rate_grid == (1e-3, 5e-4)
    assert neural.rolling_horizons == neural.rolling_ablations == ()
