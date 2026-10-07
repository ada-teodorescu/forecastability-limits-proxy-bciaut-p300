"""Evaluation modules for thesis-bci."""

from __future__ import annotations

from typing import Any

__all__ = ["Step2Config", "run_step2"]


def __getattr__(name: str) -> Any:
    """Lazily expose Step 2 symbols without importing during package execution."""

    if name in __all__:
        from src.evaluation import step2_exploration

        return getattr(step2_exploration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
