from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    supports_float64: bool = True
    supports_atomic_ops: bool = False
    deterministic_reductions: bool = True
    supports_int8_exact: bool = True
    allows_mixed_precision: bool = False


class Backend:
    """Minimal backend protocol used across dashiCORE."""

    name: str = "base"

    def __init__(self, capabilities: Optional[BackendCapabilities] = None, precision: str = "float64"):
        self.capabilities = capabilities or BackendCapabilities()
        self.precision = precision
        self.metrics = {"ops": 0}

    # ---- capability helpers -------------------------------------------------
    def require_capability(self, attr: str) -> None:
        if not getattr(self.capabilities, attr):
            raise ValueError(f"Backend missing capability: {attr}")

    # ---- observability helpers ----------------------------------------------
    def _record(self, _: str) -> None:
        self.metrics["ops"] += 1

    def reset_metrics(self) -> None:
        self.metrics = {"ops": 0}

    # ---- array primitives ----------------------------------------------------
    def array(self, data: Any, dtype: Any = None) -> np.ndarray:
        self._record("array")
        return np.asarray(data, dtype=dtype)

    def zeros_like(self, arr: np.ndarray, dtype: Any = None) -> np.ndarray:
        self._record("zeros_like")
        return np.zeros_like(arr, dtype=dtype)

    def ones_like(self, arr: np.ndarray, dtype: Any = None) -> np.ndarray:
        self._record("ones_like")
        return np.ones_like(arr, dtype=dtype)

    def sign(self, arr: np.ndarray) -> np.ndarray:
        self._record("sign")
        return np.sign(arr)

    def where(self, condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._record("where")
        return np.where(condition, x, y)

    def sum(self, arr: np.ndarray) -> np.generic:
        self._record("sum")
        return np.sum(arr)

    def abs(self, arr: np.ndarray) -> np.ndarray:
        self._record("abs")
        return np.abs(arr)

    def expand_dims(self, arr: np.ndarray, axis: int) -> np.ndarray:
        self._record("expand_dims")
        return np.expand_dims(arr, axis=axis)

    def squeeze(self, arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        self._record("squeeze")
        return np.squeeze(arr, axis=axis)

    def astype(self, arr: np.ndarray, dtype: Any) -> np.ndarray:
        self._record("astype")
        return arr.astype(dtype)

    def allclose(self, a: np.ndarray, b: np.ndarray) -> bool:
        self._record("allclose")
        return np.array_equal(a, b)

    # ---- dtypes --------------------------------------------------------------
    def to_int8(self, arr: Iterable[Any]) -> np.ndarray:
        self._record("to_int8")
        return np.asarray(arr, dtype=np.int8)

    def to_bool(self, arr: Iterable[Any]) -> np.ndarray:
        self._record("to_bool")
        return np.asarray(arr, dtype=bool)

    def to_float(self, arr: Iterable[Any]) -> np.ndarray:
        self.require_capability("supports_float64")
        self._record("to_float")
        return np.asarray(arr, dtype=np.float64)
