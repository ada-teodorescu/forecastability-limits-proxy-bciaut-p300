"""Run the main or robustness GRUs with the paper's historical training settings."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.step8_temporal import Step8Config, run_step8


def paper_config(experiment: str = "main", **paths) -> Step8Config:
    if experiment not in {"main", "exp30", "exp31"}:
        raise ValueError(f"Unknown paper experiment: {experiment}")
    return Step8Config(
        max_epochs=200,
        patience=20,
        gradient_clip_max_norm=None,
        primary_inner_protocol="last2_subject_holdout" if experiment == "main" else "group_kfold",
        dropout_grid=(0.0, 0.2) if experiment == "exp31" else (0.2,),
        # The paper's comparisons use LOSO, not the additional rolling runs.
        rolling_horizons=(),
        rolling_ablations=(),
        **paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("main", "exp30", "exp31"), default="main")
    parser.add_argument("--step4-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step5-root", type=Path, default=Path("data/scores"))
    parser.add_argument("--step6-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--step7-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--figure-root", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = paper_config(
        args.experiment,
        step4_root=args.step4_root, step5_root=args.step5_root,
        step6_root=args.step6_root, step7_root=args.step7_root,
        output_root=args.output_root or Path("data/forecasting/paper_reproduction") / args.experiment,
        figure_root=args.figure_root or Path("outputs/forecasting/paper_reproduction") / args.experiment,
        max_workers=args.max_workers,
    )
    logging.info("Paper protocol: 200 epochs, patience 20, no gradient clipping; %s", args.experiment)
    run_step8(config, no_resume=args.no_resume)


if __name__ == "__main__":
    main()
