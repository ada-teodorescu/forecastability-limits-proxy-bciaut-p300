from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STEP7_SUMMARY = ROOT / "data" / "forecasting" / "step7_summary.csv"
STEP8_SUMMARY = ROOT / "data" / "forecasting" / "step8_summary.csv"
OUT_DIR = ROOT / "outputs" / "forecasting"
CORRELATION_SUMMARY = ROOT / "data/forecasting/step6_detrended_exploration.csv"
PAIRED_SUMMARY = ROOT / "data/forecasting/review_revision/supplementary/full_model_summary.csv"


def _read_mae_by_horizon(path: Path, model_field: str, model_name: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["target_view"] != "raw_level":
                continue
            if row[model_field] != model_name:
                continue
            values[int(row["horizon"])] = float(row["mae_mean"])
    if not values:
        raise ValueError(f"No rows found for {model_name!r} in {path}")
    return values


def make_main_delta_mae() -> Path:
    context = _read_mae_by_horizon(STEP7_SUMMARY, "model_name", "context_score_mean")
    eeg_pcr = _read_mae_by_horizon(STEP7_SUMMARY, "model_name", "pcr_context_ts")
    gru_score = _read_mae_by_horizon(STEP8_SUMMARY, "ablation", "score_only")
    gru_hybrid = _read_mae_by_horizon(STEP8_SUMMARY, "ablation", "hybrid")

    horizons = sorted(context)
    series = [
        ("GRU score-only", "#1f77b4", "s", gru_score),
        ("GRU hybrid", "#9467bd", "^", gru_hybrid),
        ("EEG PCR", "#d62728", "D", eeg_pcr),
    ]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)

    ax.axhline(
        0.0,
        color="#222222",
        linewidth=1.1,
        linestyle="--",
        label="Context mean baseline",
        zorder=1,
    )
    for label, color, marker, values in series:
        delta = [values[h] - context[h] for h in horizons]
        ax.plot(
            horizons,
            delta,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.2,
            label=label,
            zorder=2,
        )

    ax.set_xlabel("Forecast horizon (blocks)")
    ax.set_ylabel("Delta MAE vs. context mean")
    ax.set_xticks(horizons)
    ax.set_ylim(-0.012, 0.18)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(loc="center right", frameon=False, handlelength=2.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "main_result_delta_mae_vs_context.png"
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_proxy_structure_and_forecasting() -> Path:
    """Plot saved correlations and held-out GRU paired errors; fit no model.

    Panels retain their distinct original aggregations: pooled within-session
    lag pairs and equal-subject LOSO MAE differences. Full-session centring
    is retrospective. The intervals are read from the existing supplement.
    """
    with CORRELATION_SUMMARY.open(newline="", encoding="utf-8") as handle:
        correlations = sorted(csv.DictReader(handle), key=lambda row: int(row["horizon"]))
    with PAIRED_SUMMARY.open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    horizons = [int(row["horizon"]) for row in correlations]
    if horizons != [5, 10, 15, 20, 25]:
        raise ValueError(f"Unexpected diagnostic horizons: {horizons}")
    records = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.50), sharex=True)
        raw = [float(row["lagged_raw_corr"]) for row in correlations]
        centred = [float(row["detrended_lagged_corr"]) for row in correlations]
        axes[0].plot(horizons, raw, color="#1764a0", marker="o",
                     markersize=3, linewidth=1.1, label="Raw score")
        axes[0].plot(horizons, centred, color="#c36b22", marker="s",
                     markersize=3, linewidth=1.1, linestyle="--",
                     label="Full-session centred")
        axes[0].axhline(0, color="#555555", linestyle=":", linewidth=.7)
        axes[0].set_ylim(-.14, .93)
        axes[0].set_yticks([0, .4, .8])
        axes[0].set_ylabel("Pearson correlation")
        axes[0].set_title("(a) Retrospective correlation diagnostic", loc="left", pad=5)
        axes[0].legend(loc="center left", frameon=False, handlelength=2.4)
        models = [("gru_score_only", "Score-only GRU", "#1764a0", "o", -.30),
                  ("gru_hybrid", "Hybrid GRU", "#9658a8", "s", .30)]
        for key, label, color, marker, offset in models:
            for view, line_style in [("raw_level", "-"), ("context_residual", "--")]:
                rows = sorted(
                    [row for row in paired if row["model_key"] == key and row["target_view"] == view],
                    key=lambda row: int(row["horizon"]),
                )
                if [int(row["horizon"]) for row in rows] != horizons:
                    raise ValueError(f"Missing/duplicate GRU rows: {key}, {view}")
                if any(int(row["n_subjects"]) != 15 for row in rows):
                    raise ValueError("Expected 15 held-out subjects")
                means = [float(row["delta_mae_model_minus_context"]) for row in rows]
                lower = [float(row["delta_ci_95_lower"]) for row in rows]
                upper = [float(row["delta_ci_95_upper"]) for row in rows]
                if any(not lo <= value <= hi for lo, value, hi in zip(lower, means, upper)):
                    raise ValueError("Saved interval does not contain its paired mean")
                view_offset = -.09 if view == "raw_level" else .09
                axes[1].errorbar(
                    [h + offset + view_offset for h in horizons], means,
                    yerr=[[value-lo for value, lo in zip(means, lower)],
                          [hi-value for value, hi in zip(means, upper)]],
                    color=color, marker=marker, markersize=3.2,
                    markerfacecolor=color if view == "raw_level" else "white",
                    linewidth=1.1, linestyle=line_style, elinewidth=.65, capsize=1.6,
                    label=("Score GRU" if key == "gru_score_only" else "Hybrid GRU")
                    + (", raw" if view == "raw_level" else ", residual"),
                )
                records.extend({
                    "model_key": key, "target_view": view, "horizon": h,
                    "mean_delta": mean, "ci_lower": lo, "ci_upper": hi,
                } for h, mean, lo, hi in zip(horizons, means, lower, upper))
        axes[1].axhline(0, color="#333333", linestyle=":", linewidth=.8)
        axes[1].set_ylim(-.009, .033)
        axes[1].set_yticks([0, .01, .02, .03])
        axes[1].set_ylabel("MAE minus context mean")
        axes[1].set_xlabel("Forecast horizon (blocks)")
        axes[1].set_title("(b) Out-of-sample forecasting", loc="left", pad=5)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .005),
                   ncol=2, frameon=False, fontsize=6.6, handlelength=2.4,
                   columnspacing=1.2)
        for ax in axes:
            ax.set_xticks(horizons)
            ax.set_xlim(4, 26)
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
            ax.tick_params(length=3, width=.7)
        fig.subplots_adjust(left=.20, right=.985, bottom=.235, top=.94, hspace=.45)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "proxy_structure_and_forecasting.pdf"
        fig.savefig(out_path, metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out_path.with_suffix(".png"), dpi=350)
        plt.close(fig)
    metadata = {
        "inputs": [{"path": path.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (CORRELATION_SUMMARY, PAIRED_SUMMARY)],
        "correlation_aggregation": "Pearson correlation pooling all within-session lag pairs.",
        "full_session_centring": "Retrospective diagnostic using the full online session mean.",
        "forecast_aggregation": "Equal-subject mean of 15 paired LOSO MAE differences.",
        "intervals": "Existing unadjusted descriptive 95% subject-bootstrap intervals, conditional on fitted predictions.",
        "plotted_forecast_points": records,
        "models_fitted": False,
        "horizon_offsets": "Forecast points are offset horizontally for display only.",
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate manuscript figures from saved results.")
    parser.add_argument("--figure", choices=["all", "delta", "structure"], default="all")
    args = parser.parse_args()
    if args.figure in ("all", "delta"):
        print(make_main_delta_mae())
    if args.figure in ("all", "structure"):
        print(make_proxy_structure_and_forecasting())
