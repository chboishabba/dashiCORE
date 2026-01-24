from __future__ import annotations

from typing import Optional

from .base import Backend, BackendCapabilities


class CPUBackend(Backend):
    """Deterministic CPU backend using NumPy only."""

    name = "cpu"

    def __init__(self, capabilities: Optional[BackendCapabilities] = None, precision: str = "float64"):
        super().__init__(capabilities=capabilities or BackendCapabilities(), precision=precision)
