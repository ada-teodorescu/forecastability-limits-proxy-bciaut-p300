"""Entry point for exp31: Step 8 with expanded HP grid + grouped inner CV.

Adds dropout to the tuned grid (fixed at 0.2 in exp30).
Uses group_kfold inner CV throughout (same as exp30).

Grid (8 combinations):
  hidden_dim:      32, 64
  learning_rate:   1e-3, 5e-4
  dropout:         0.0, 0.2       <- new axis vs exp30
  weight_decay:    1e-4           <- fixed (secondary concern on small dataset)
  context_length:  10             <- fixed; see note below
  eeg_pca:         20             <- fixed (EEG null result from exp09)

Note on context_length: comparing context_length=5 vs 10 within a single HP grid
is not valid because Step 8 builds one dataset filtered at anchor_idx >= max_context_length-1.
A c=5 candidate would only see the c=10-eligible row subset, not its own larger eligible set.
Proper context-length comparison requires separate experiments with independent row filtering
(exp31_ctx5 with context_length_grid=(5,) vs this exp31 with context_length_grid=(10,)).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.step8_temporal import Step8Config, run_step8


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_step8(
        Step8Config(
            output_root=Path("data/forecasting/exp31_expanded_hp_search"),
            figure_root=Path("outputs/forecasting/exp31_expanded_hp_search"),
            primary_inner_protocol="group_kfold",
            hidden_dim_grid=(32, 64),
            learning_rate_grid=(1e-3, 5e-4),
            dropout_grid=(0.0, 0.2),
            weight_decay_grid=(1e-4,),
            context_length_grid=(10,),
            eeg_pca_components_grid=(20,),
            max_workers=2,
        ),
    )
    logging.getLogger(__name__).info(
        "Exp31 complete: %d fold rows, %d prediction rows, %d rolling rows",
        summary.fold_metric_rows,
        summary.prediction_rows,
        summary.rolling_fold_rows,
    )


if __name__ == "__main__":
    main()
