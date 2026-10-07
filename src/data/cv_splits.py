"""Cross-validation split helpers shared across forecasting modules."""

from __future__ import annotations

import numpy as np


def group_kfold_splits(groups: np.ndarray, *, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic grouped K-fold splits over a 1-D group array."""

    groups = np.asarray(groups, dtype=object)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    if n_splits <= 1:
        raise ValueError("n_splits must be >= 2 for group K-fold")
    if n_splits > len(unique_groups):
        raise ValueError(f"n_splits={n_splits} exceeds unique groups={len(unique_groups)}")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for held_out_groups in np.array_split(np.asarray(unique_groups, dtype=object), n_splits):
        val_mask = np.isin(groups, held_out_groups)
        train_idx = np.flatnonzero(~val_mask).astype(np.int64)
        val_idx = np.flatnonzero(val_mask).astype(np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("Group K-fold generated an empty train/validation split")
        splits.append((train_idx, val_idx))
    return splits
