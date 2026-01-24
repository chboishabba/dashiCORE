from __future__ import annotations

import numpy as np
import pytest


def test_vkfft_vulkan_roundtrip_matches_numpy():
    try:
        import vkfft_vulkan_py  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"vkfft_vulkan_py not available: {exc}")

    try:
        import vulkan  # noqa: F401
        from gpu_vkfft_adapter import VkFFTExecutor
        from gpu_vulkan_dispatcher import VulkanDispatchConfig, create_vulkan_handles
    except Exception as exc:  # pragma: no cover - skip when Vulkan stack missing
        pytest.skip(f"Vulkan bindings unavailable: {exc}")

    try:
        handles = create_vulkan_handles(VulkanDispatchConfig(device_index=0))
    except Exception as exc:  # pragma: no cover - skip when no device/ICD
        pytest.skip(f"Unable to create Vulkan handles: {exc}")

    try:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16, 16)).astype(np.complex64)
        exe = VkFFTExecutor(handles=handles, fft_backend="vkfft-vulkan")
        y = exe.fft2(x)
        z = exe.ifft2(y)
        np.testing.assert_allclose(z, x, atol=1e-5, rtol=1e-5)
    finally:
        handles.close()
