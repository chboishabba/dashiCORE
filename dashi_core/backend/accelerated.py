from __future__ import annotations

from typing import Optional

from .base import Backend, BackendCapabilities


class AcceleratedBackend(Backend):
    """Accelerated backend placeholder (CPU-vectorized semantics)."""

    name = "accelerated"

    def __init__(
        self,
        capabilities: Optional[BackendCapabilities] = None,
        precision: str = "float64",
    ):
        accel_caps = capabilities or BackendCapabilities(
            supports_float64=True,
            supports_atomic_ops=False,
            deterministic_reductions=True,
            supports_int8_exact=True,
            allows_mixed_precision=False,
        )
        super().__init__(capabilities=accel_caps, precision=precision)

    # Skip per-op bookkeeping to reduce overhead in tight loops.
    def _record(self, _: str) -> None:
        return
