"""Preprocessing utilities for BCIAUT-P300 Step 1."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.preprocessing.preprocess import PreprocessingConfig, run_step1

__all__ = ["PreprocessingConfig", "run_step1"]


def __getattr__(name: str) -> Any:
    if name == "PreprocessingConfig":
        from src.preprocessing.preprocess import PreprocessingConfig

        return PreprocessingConfig
    if name == "run_step1":
        from src.preprocessing.preprocess import run_step1

        return run_step1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
