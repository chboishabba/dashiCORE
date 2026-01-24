"""
Lightweight vkFFT stub to keep call sites importable without a GPU/driver.

Intended use:
- Callers may `from gpu_vkfft_stub import fft2, ifft2, has_vkfft`.
- If vkFFT Python bindings are unavailable, we fall back to NumPy and raise a
  clear warning the first time. This keeps CPU parity tests running while
  making the missing acceleration explicit.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

_warned = False


def has_vkfft() -> bool:
    try:
        import vkfft  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _warn_once() -> None:
    global _warned
    if not _warned:
        warnings.warn("vkFFT not installed; using NumPy fft as a stub.", RuntimeWarning, stacklevel=2)
        _warned = True


def fft2(x: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    if has_vkfft():
        import vkfft  # type: ignore

        plan = vkfft.fft(x, ndim=2, **kwargs)
        return plan  # type: ignore[return-value]
    _warn_once()
    return np.fft.fft2(x, *args, **kwargs)


def ifft2(x: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    if has_vkfft():
        import vkfft  # type: ignore

        plan = vkfft.ifft(x, ndim=2, **kwargs)
        return plan  # type: ignore[return-value]
    _warn_once()
    return np.fft.ifft2(x, *args, **kwargs)
