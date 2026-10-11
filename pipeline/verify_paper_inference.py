"""Verify the paper's existing paired tests and sensitivity values.

Uses saved subject-fold metrics only. No model fitting, simulated datasets,
new research comparisons, or equivalence analyses are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata, t, wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.minimum(1., np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1)))
    return adjusted[np.argsort(order)]


def exact_pratt(differences):
    """Exact conditional two-sided sign distribution, with Pratt zeros/ties."""
    ranks = rankdata(np.abs(differences), method="average")
    weights = np.rint(2 * ranks[differences != 0]).astype(int)
    if not weights.size:
        return 1.
    counts = np.zeros(int(weights.sum()) + 1, dtype=np.int64)
    counts[0] = 1
    for weight in weights:
        counts[weight:] += counts[:-weight].copy()
    probabilities = counts / (2 ** weights.size)
    observed = int(round(2 * ranks[differences > 0].sum()))
    return min(1., 2 * min(probabilities[:observed + 1].sum(), probabilities[observed:].sum()))


def check_implementations():
    sys.path.insert(0, str(ROOT))
    from src.evaluation.forecasting_metrics import exhaustive_sign_permutation_pvalue, holm_correction

    for sample in [np.arange(1., 16.), np.array([-1., 2., -3., 4., 5., -6.])]:
        np.testing.assert_allclose(exact_pratt(sample), wilcoxon(sample, method="exact", zero_method="pratt").pvalue, rtol=0, atol=1e-15)
    for sample in [np.array([0., 1., -1., 2., -3.]), np.zeros(5)]:
        np.testing.assert_allclose(exact_pratt(sample), exhaustive_sign_permutation_pvalue(sample), rtol=0, atol=1e-15)
    sample_p = np.array([.001, .02, .4, .1, .05, .003, .9, .12, .3, .01])
    np.testing.assert_allclose(holm(sample_p), holm_correction(sample_p), rtol=0, atol=1e-15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/forecasting/review_revision/inference_sensitivity")
    args = parser.parse_args()
    check_implementations()
    lookup, families = {}, {}
    input_paths = []
    for stage in (7, 8):
        fold_path = ROOT / f"data/forecasting/step{stage}_fold_metrics.csv"
        stats_path = ROOT / f"data/forecasting/step{stage}_paired_stats.csv"
        input_paths.extend([fold_path, stats_path])
        for row in read_csv(fold_path):
            model = row["model_name"] if stage == 7 else "gru_" + row["ablation"]
            key = (int(row["horizon"]), row["target_view"], model, row["fold_subject_id"])
            assert key not in lookup, key
            lookup[key] = row
        for row in read_csv(stats_path):
            family = (stage, int(row["horizon"]), row["target_view"], row["metric"])
            families.setdefault(family, []).append(row)
    subjects = sorted({key[3] for key in lookup})
    assert len(subjects) == 15
    results = []
    for (stage, horizon, view, metric), family in families.items():
        p_values, verified = [], []
        for row in family:
            a = np.array([float(lookup[(horizon, view, row["model_a"], s)][metric]) for s in subjects])
            b = np.array([float(lookup[(horizon, view, row["model_b"], s)][metric]) for s in subjects])
            differences = b - a
            sd = float(np.std(differences, ddof=1))
            mde = float((t.ppf(.975, 14) + t.ppf(.80, 14)) * sd / np.sqrt(15))
            p = float(exact_pratt(differences))
            np.testing.assert_allclose(p, float(row["p_value"]), rtol=0, atol=1e-12)
            np.testing.assert_allclose(sd, float(row["std_delta"]), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(mde, float(row["min_detectable_delta_80"]), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(differences.mean(), float(row["mean_delta_b_minus_a"]), rtol=0, atol=1e-12)
            p_values.append(p)
            verified.append(dict(row, stage=stage, recomputed_p=p, recomputed_std_delta=sd, recomputed_min_detectable_delta_80=mde))
        adjusted = holm(np.array(p_values))
        for row, value in zip(verified, adjusted, strict=True):
            np.testing.assert_allclose(value, float(row["p_value_holm"]), rtol=0, atol=1e-12)
            row["recomputed_p_holm"] = float(value)
            results.append(row)
    assert len(results) == 340
    sensitivity = []
    for row in sorted(results, key=lambda r: int(r["horizon"])):
        if row["stage"] == 8 and row["target_view"] == "raw_level" and row["metric"] == "mae" and row["comparison_id"] == "gru_score_only_vs_context_score_mean":
            baseline = float(row["mean_metric_b"])
            sensitivity.append({"horizon": int(row["horizon"]), "n_subjects": 15, "sd_paired_difference": row["recomputed_std_delta"], "minimum_detectable_delta_80": row["recomputed_min_detectable_delta_80"], "baseline_MAE": baseline, "percent_of_baseline_MAE": 100 * row["recomputed_min_detectable_delta_80"] / baseline, "p_value": row["recomputed_p"], "p_value_holm": row["recomputed_p_holm"]})
    assert len(sensitivity) == 5
    report = {"observed_tests_verified": len(results), "holm_families_verified": len(families), "original_sensitivity_values_verified": len(results), "implementation_checks": "passed", "new_experiments": False, "analysis_method": "Existing paired-t sensitivity approximation; existing exact signed-rank and Holm tests", "formula": "(t_quantile(0.975,14) + t_quantile(0.80,14)) * sample_SD(paired_differences) / sqrt(15)", "power_target": .8, "uncorrected_two_sided_alpha": .05, "sensitivity_values": sensitivity, "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "input_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "observed_tests.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()