from pathlib import Path

import numpy as np

from dashi_core.backend import use_backend
from dashi_core.carrier import Carrier
from dashi_core.testing.mock_kernels import OneStepErodeKernel
from gpu_vulkan_backend import make_vulkan_kernel, register_vulkan_backend, VulkanKernelConfig


def _run_core_ops():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    kernel = OneStepErodeKernel()
    projected = kernel(state)
    defect = projected.to_signed() - state.to_signed()
    return state.to_signed(), projected.to_signed(), defect


def test_vulkan_backend_registers_and_matches_cpu_fallback():
    config = VulkanKernelConfig(shader_path=Path("dummy.glsl"), spv_path=Path("dummy.spv"))
    dispatcher = lambda c: c  # CPU fallback to avoid Vulkan dependency in test
    register_vulkan_backend(config=config, allow_fallback=True, dispatcher=dispatcher, name="vulkan")

    with use_backend("cpu"):
        cpu_out = _run_core_ops()
    with use_backend("vulkan"):
        vk_out = _run_core_ops()

    for cpu_val, vk_val in zip(cpu_out, vk_out):
        np.testing.assert_array_equal(cpu_val, vk_val)


def test_vulkan_backend_dispatcher_invoked_via_kernel():
    config = VulkanKernelConfig(shader_path=Path("dummy.glsl"), spv_path=Path("dummy.spv"))

    def dispatcher(state: Carrier) -> Carrier:
        flipped = -state.sign
        return Carrier(support=state.support, sign=flipped)

    backend = register_vulkan_backend(
        name="vulkan_dispatch",
        config=config,
        dispatcher=dispatcher,
        allow_fallback=False,
    )
    kernel = make_vulkan_kernel(backend)
    carrier = Carrier.from_signed(np.array([1, -1], dtype=np.int8))
    out = kernel(carrier)
    np.testing.assert_array_equal(out.sign, -carrier.sign)
