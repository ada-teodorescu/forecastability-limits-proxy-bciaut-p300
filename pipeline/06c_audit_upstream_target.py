"""Pipeline Step 6c: upstream target audit for BCIAUT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.features.step6_upstream_target_audit import Step6AuditConfig, run_step6_upstream_target_audit


def main() -> None:
    """CLI entrypoint for the Step 6c upstream target audit."""

    parser = argparse.ArgumentParser(description="Run Step 6c upstream target audit for BCIAUT.")
    parser.add_argument("--step3-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step4-root", type=Path, default=Path("data/features"))
    parser.add_argument("--step5-root", type=Path, default=Path("data/scores"))
    parser.add_argument("--step6-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--output-root", type=Path, default=Path("data/forecasting"))
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/forecasting/step6c"))
    parser.add_argument("--horizons", type=int, nargs="*", default=list(Step6AuditConfig().horizons))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_step6_upstream_target_audit(
        Step6AuditConfig(
            step3_root=args.step3_root,
            step4_root=args.step4_root,
            step5_root=args.step5_root,
            step6_root=args.step6_root,
            output_root=args.output_root,
            figure_root=args.figure_root,
            horizons=tuple(int(horizon) for horizon in args.horizons),
        )
    )


if __name__ == "__main__":
    main()
