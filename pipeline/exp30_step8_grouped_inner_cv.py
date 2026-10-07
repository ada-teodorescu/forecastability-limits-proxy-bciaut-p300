"""Entry point for exp30: Step 8 with grouped subject-level inner CV."""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.step8_temporal import Step8Config, run_step8


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_step8(
        Step8Config(
            output_root=Path("data/forecasting/exp30_step8_grouped_inner_cv"),
            figure_root=Path("outputs/forecasting/exp30_step8_grouped_inner_cv"),
            primary_inner_protocol="group_kfold",
            max_workers=4,
        ),
    )
    logging.getLogger(__name__).info(
        "Exp30 complete: %d fold rows, %d prediction rows, %d rolling rows",
        summary.fold_metric_rows,
        summary.prediction_rows,
        summary.rolling_fold_rows,
    )


if __name__ == "__main__":
    main()
