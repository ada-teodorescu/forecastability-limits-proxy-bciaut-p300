"""Pipeline Step 6.2b: Variance decomposition and detrended diagnostics addendum.

Runs independently from the main Step 6 pipeline. Requires Step 6 exploration
summary CSV to already exist (for original correlation values).
"""

import argparse
import logging
from pathlib import Path

from src.features.step6_forecasting_task import Step6Config, run_step6_addendum


def main() -> None:
    """CLI entrypoint for Step 6.2b addendum."""

    parser = argparse.ArgumentParser(
        description="Run Step 6.2b addendum: variance decomposition and detrended diagnostics."
    )
    parser.add_argument("--step4-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step5-root", type=Path, default=Path("data/scores"))
    parser.add_argument("--output-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/forecasting/step6"))
    parser.add_argument("--horizons", type=int, nargs="*", default=list(Step6Config().horizons))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_step6_addendum(
        Step6Config(
            step4_root=args.step4_root,
            step5_root=args.step5_root,
            output_root=args.output_root,
            figure_root=args.figure_root,
            horizons=tuple(int(h) for h in args.horizons),
        )
    )


if __name__ == "__main__":
    main()
