import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from dashi_core.carrier import Carrier
from gpu_common_methods import compile_shader, resolve_shader
from gpu_vulkan_backend import make_vulkan_kernel, register_vulkan_backend, VulkanKernelConfig
from gpu_vulkan_dispatcher import VulkanDispatchConfig

_HAS_VULKAN = shutil.which("glslc") is not None and os.getenv("VK_ICD_FILENAMES")


@pytest.mark.skipif(not _HAS_VULKAN, reason="Vulkan ICD or glslc not available")
def test_vulkan_disjoint_partition_equivalence(tmp_path: Path):
    try:
        import vulkan as vk  # noqa: F401
    except ImportError:
        pytest.skip("python-vulkan not installed")

    shader_src = resolve_shader("sign_flip")
    spv_out = tmp_path / "sign_flip.spv"
    compile_shader(shader_src, spv_out)

    config = VulkanKernelConfig(shader_path=shader_src, spv_path=spv_out, compile_on_dispatch=False)
    backend = register_vulkan_backend(
        name="vulkan_partition_test",
        config=config,
        dispatch_config=VulkanDispatchConfig(device_index=0),
        allow_fallback=False,
    )
    kernel = make_vulkan_kernel(backend)

    carrier = Carrier.from_signed(np.array([1, -1, 0, 1, -1, 1, 0, -1], dtype=np.int8))
    full = kernel(carrier)

    # Split into contiguous blocks and recombine after running on GPU.
    length = carrier.sign.size
    block_size = 2
    parts = []
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        sub = Carrier(support=carrier.support[start:end], sign=carrier.sign[start:end])
        parts.append(kernel(sub))

    recombined_sup = np.concatenate([p.support for p in parts])
    recombined_sig = np.concatenate([p.sign for p in parts])
    recombined = Carrier(support=recombined_sup, sign=recombined_sig)

    np.testing.assert_array_equal(recombined.to_signed(), full.to_signed())
    np.testing.assert_array_equal(recombined.support, full.support)
