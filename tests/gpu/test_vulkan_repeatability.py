import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from dashi_core.carrier import Carrier
from gpu_common_methods import compile_shader
from gpu_vulkan_backend import make_vulkan_kernel, register_vulkan_backend, VulkanKernelConfig
from gpu_vulkan_dispatcher import VulkanDispatchConfig


_HAS_VULKAN = shutil.which("glslc") is not None and os.getenv("VK_ICD_FILENAMES")


@pytest.mark.skipif(not _HAS_VULKAN, reason="Vulkan ICD or glslc not available")
def test_vulkan_repeatability_sign_flip(tmp_path: Path):
    try:
        import vulkan as vk  # noqa: F401
    except ImportError:
        pytest.skip("python-vulkan not installed")

    shader_src = Path("gpu_shaders/sign_flip.comp")
    spv_out = tmp_path / "sign_flip.spv"
    compile_shader(shader_src, spv_out)

    config = VulkanKernelConfig(shader_path=shader_src, spv_path=spv_out, compile_on_dispatch=False)
    backend = register_vulkan_backend(
        name="vulkan_repeatability_test",
        config=config,
        dispatch_config=VulkanDispatchConfig(device_index=0),
        allow_fallback=False,
    )
    kernel = make_vulkan_kernel(backend)

    carrier = Carrier.from_signed(np.array([1, -1, 0, 1, -1], dtype=np.int8))
    outs = [kernel(carrier) for _ in range(50)]
    first = outs[0]
    for out in outs[1:]:
        np.testing.assert_array_equal(out.to_signed(), first.to_signed())
        np.testing.assert_array_equal(out.support, first.support)
