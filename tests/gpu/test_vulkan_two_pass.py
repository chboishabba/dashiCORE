import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from dashi_core.carrier import Carrier
from gpu_common_methods import compile_shader, resolve_shader
from gpu_vulkan_backend import register_vulkan_backend, VulkanKernelConfig
from gpu_vulkan_dispatcher import VulkanDispatchConfig, build_vulkan_dispatcher


_HAS_VULKAN = shutil.which("glslc") is not None and os.getenv("VK_ICD_FILENAMES")


@pytest.mark.skipif(not _HAS_VULKAN, reason="Vulkan ICD or glslc not available")
def test_vulkan_two_pass_ordering(tmp_path: Path):
    try:
        import vulkan as vk  # noqa: F401
    except ImportError:
        pytest.skip("python-vulkan not installed")

    shader_flip = resolve_shader("sign_flip")
    shader_clamp = resolve_shader("clamp_nonnegative")
    spv_flip = tmp_path / "sign_flip.spv"
    spv_clamp = tmp_path / "clamp_nonnegative.spv"
    compile_shader(shader_flip, spv_flip)
    compile_shader(shader_clamp, spv_clamp)

    config_flip = VulkanKernelConfig(shader_path=shader_flip, spv_path=spv_flip, compile_on_dispatch=False)
    config_clamp = VulkanKernelConfig(shader_path=shader_clamp, spv_path=spv_clamp, compile_on_dispatch=False)

    dispatch_flip = build_vulkan_dispatcher(config=config_flip, dispatch_config=VulkanDispatchConfig(device_index=0))
    dispatch_clamp = build_vulkan_dispatcher(config=config_clamp, dispatch_config=VulkanDispatchConfig(device_index=0))

    def two_pass_dispatcher(state: Carrier) -> Carrier:
        mid = dispatch_flip(state)
        return dispatch_clamp(mid)

    backend = register_vulkan_backend(
        name="vulkan_two_pass_test",
        config=config_flip,
        dispatcher=two_pass_dispatcher,
        allow_fallback=False,
    )

    carrier = Carrier.from_signed(np.array([-1, 0, 1, -1], dtype=np.int8))

    # CPU reference: flip then clamp negatives to zero (supports shrink).
    ref_sign = -carrier.sign
    ref_sign = np.where(ref_sign < 0, 0, ref_sign)
    ref_support = carrier.support & (ref_sign != 0)
    ref = Carrier(support=ref_support, sign=ref_sign.astype(np.int8))

    out = backend.run_kernel(carrier)

    np.testing.assert_array_equal(out.to_signed(), ref.to_signed())
    np.testing.assert_array_equal(out.support, ref.support)

    # Determinism: run twice and compare.
    out2 = backend.run_kernel(carrier)
    np.testing.assert_array_equal(out.to_signed(), out2.to_signed())
    np.testing.assert_array_equal(out.support, out2.support)
