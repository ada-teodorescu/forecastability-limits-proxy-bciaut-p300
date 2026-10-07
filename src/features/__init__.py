"""Feature-extraction modules for thesis-bci."""

from __future__ import annotations

from typing import Any

__all__ = ["Step3Config", "run_step3", "Step4Config", "run_step4"]


def __getattr__(name: str) -> Any:
    """Lazily expose Step 3 and Step 4 symbols without eager imports."""

    if name in {"Step3Config", "run_step3"}:
        from src.features import step3_covariance

        return getattr(step3_covariance, name)
    if name in {"Step4Config", "run_step4"}:
        from src.features import step4_riemannian

        return getattr(step4_riemannian, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
