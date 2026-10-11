"""Independent exhaustive signed-rank and multiplicity audit for exp41."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.paper_artifacts import resolve_artifact
ROOT = REPO / "data/forecasting/review_revision/short_horizons"


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1., np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    restored = np.empty(len(values))
    restored[order] = adjusted
    return restored


def main():
    metrics = pd.read_csv(ROOT / 'subject_metrics.csv')
    paired = pd.read_csv(ROOT / 'paired_mae_stats.csv')
    assert len(metrics) == 600 and len(paired) == 32
    assert metrics.groupby(['horizon', 'target_view', 'model_name']).size().eq(15).all()
    prediction_rows = 0
    for kind in ['classical', 'gru']:
        predictions = pd.read_csv(resolve_artifact(ROOT / f'{kind}/predictions.csv'))
        if kind == 'gru':
            predictions['model_name'] = 'gru_score_only'
            predictions = predictions.rename(columns={'y_pred_mu': 'y_pred_mean'})
        names = ['horizon', 'target_view', 'model_name', 'fold_subject_id']
        calculated = predictions.assign(error=np.abs(predictions.y_true - predictions.y_pred_mean)).groupby(names).error.mean()
        expected = metrics.set_index(names).mae.loc[calculated.index]
        np.testing.assert_allclose(calculated, expected, rtol=0, atol=1e-12)
        prediction_rows += len(predictions)
    assert prediction_rows == 161700
    errors = []
    for row in paired.itertuples():
        subset = metrics[(metrics.horizon == row.horizon) &
                         (metrics.target_view == row.target_view)]
        wide = subset.pivot(index='fold_subject_id', columns='model_name', values='mae')
        delta = (wide[row.model_b] - wide[row.model_a]).to_numpy()
        np.testing.assert_allclose(np.mean(delta), row.mean_delta_b_minus_a, rtol=0, atol=1e-14)
        ranks = rankdata(np.abs(delta), method='average')
        nonzero = delta != 0
        ranks_nonzero = ranks[nonzero]
        n = len(ranks_nonzero)
        assignments = ((np.arange(2**n, dtype=np.uint32)[:, None] >>
                        np.arange(n, dtype=np.uint32)) & 1)
        positive = assignments @ ranks_nonzero
        total = ranks_nonzero.sum()
        statistics = np.minimum(positive, total - positive)
        actual_positive = ranks[delta > 0].sum()
        actual_statistic = min(actual_positive, total - actual_positive)
        independent_p = np.mean(statistics <= actual_statistic + 1e-12)
        np.testing.assert_allclose(independent_p, row.p_value, rtol=0, atol=1e-14)
        errors.append(abs(float(independent_p) - row.p_value))
        rng = np.random.default_rng(20261008)
        indices = rng.integers(0, 15, size=(10000, 15))
        interval = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose(interval, [row.bootstrap_ci_lower, row.bootstrap_ci_upper],
                                   rtol=0, atol=1e-14)
    np.testing.assert_allclose(holm(paired.p_value), paired.p_value_holm_all_32,
                               rtol=0, atol=1e-14)
    for _, subset in paired.groupby(['horizon', 'target_view']):
        np.testing.assert_allclose(holm(subset.p_value),
            subset.p_value_holm_within_horizon_view, rtol=0, atol=1e-14)
    result = {'passed': True, 'predictions_verified': prediction_rows, 'subject_maes_verified': 600, 'paired_contrasts': 32,
              'exhaustive_sign_assignments_per_nonzero_15_subject_test': 32768,
              'maximum_p_value_difference': max(errors),
              'all_32_holm_verified': True, 'all_8_local_holm_families_verified': True,
              'all_32_bootstrap_intervals_verified': True}
    (ROOT / 'independent_statistical_audit.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
