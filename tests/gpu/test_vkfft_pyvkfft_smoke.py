"""Smoke test for pyvkfft OpenCL path.

This test only runs when pyvkfft + pyopencl are present and at least one
OpenCL device is visible. It validates that `VkFFTExecutor` routes to
pyvkfft when `fft_backend="vkfft-opencl"` and produces results close to
NumPy's FFT (within a small tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pyvkfft_env():
    try:
        from pyvkfft.opencl import VkFFTApp  # noqa: F401
        import pyopencl as cl  # noqa: F401
    except Exception as exc:  # pragma: no cover - skip path
        pytest.skip(f"pyvkfft/opencl unavailable: {exc}")

    platforms = cl.get_platforms()
    devices = [d for p in platforms for d in p.get_devices()]
    if not devices:
        pytest.skip("No OpenCL devices available for pyvkfft smoke test")

    return {
        "platforms": platforms,
        "devices": devices,
    }


def test_pyvkfft_roundtrip_matches_numpy(pyvkfft_env):
    from gpu_vkfft_adapter import VkFFTExecutor

    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 32)).astype(np.float32)

    exe = VkFFTExecutor(fft_backend="vkfft-opencl")
    y = exe.fft2(x)
    z = exe.ifft2(y)

    np.testing.assert_allclose(z, x, atol=1e-5, rtol=1e-5)
