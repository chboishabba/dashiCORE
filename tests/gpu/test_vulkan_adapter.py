import tempfile
from pathlib import Path

import numpy as np
import pytest

from dashi_core.carrier import Carrier
from gpu_vulkan_adapter import VulkanBackendAdapter, VulkanCarrierKernel, VulkanKernelConfig


def test_vulkan_adapter_noop_preserves_state():
    carrier = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    config = VulkanKernelConfig(shader_path=Path("dummy.glsl"), spv_path=Path("dummy.spv"))
    adapter = VulkanBackendAdapter(config=config, allow_fallback=True)
    kernel = VulkanCarrierKernel(adapter)
    out = kernel(carrier)
    np.testing.assert_array_equal(out.to_signed(), carrier.to_signed())
    np.testing.assert_array_equal(out.support, carrier.support)


def test_vulkan_compile_requires_shader_file(tmp_path: Path):
    shader = tmp_path / "kernel.comp"
    spv = tmp_path / "kernel.spv"
    config = VulkanKernelConfig(shader_path=shader, spv_path=spv, compile_on_init=False)
    adapter = VulkanBackendAdapter(config=config)
    with pytest.raises(FileNotFoundError):
        adapter.compile()


def test_vulkan_adapter_rejects_support_creation():
    carrier = Carrier.from_signed(np.array([1, 0], dtype=np.int8))
    config = VulkanKernelConfig(shader_path=Path("dummy.glsl"), spv_path=Path("dummy.spv"))
    adapter = VulkanBackendAdapter(config=config)

    class BadKernel(VulkanCarrierKernel):
        def apply(self, state: Carrier, ctx=None) -> Carrier:
            # Illegally create support (should be caught by validate_kernel_output)
            sign = np.ones_like(state.sign)
            support = np.ones_like(state.support, dtype=bool)
            return Carrier(support=support, sign=sign)

    kernel = BadKernel(adapter)
    with pytest.raises(ValueError):
        kernel(carrier)


def test_vulkan_adapter_dispatcher_used():
    carrier = Carrier.from_signed(np.array([1, -1], dtype=np.int8))
    config = VulkanKernelConfig(shader_path=Path("dummy.glsl"), spv_path=Path("dummy.spv"))

    def dispatcher(state: Carrier) -> Carrier:
        # Flip sign deterministically without changing support
        flipped_sign = -state.sign
        return Carrier(support=state.support, sign=flipped_sign)

    adapter = VulkanBackendAdapter(config=config, dispatcher=dispatcher)
    kernel = VulkanCarrierKernel(adapter)
    out = kernel(carrier)
    np.testing.assert_array_equal(out.sign, -carrier.sign)


def test_vulkan_adapter_compile_on_dispatch(tmp_path: Path):
    shader = tmp_path / "kernel.comp"
    spv = tmp_path / "kernel.spv"
    shader.write_text("// noop shader")
    spv.write_text("noop")

    compiled = {"called": False}

    def fake_compile() -> None:
        compiled["called"] = True
        # Mirror side effect of real compile without invoking glslc
        adapter._compiled = True

    config = VulkanKernelConfig(shader_path=shader, spv_path=spv, compile_on_dispatch=True)
    adapter = VulkanBackendAdapter(config=config, allow_fallback=True)
    adapter.compile = fake_compile  # type: ignore[assignment]

    carrier = Carrier.from_signed(np.array([1], dtype=np.int8))
    adapter.run_kernel(carrier)
    assert compiled["called"]


def test_vulkan_adapter_requires_dispatcher_when_no_fallback():
    carrier = Carrier.from_signed(np.array([1], dtype=np.int8))
    config = VulkanKernelConfig(shader_path=Path("dummy.glsl"), spv_path=Path("dummy.spv"))
    adapter = VulkanBackendAdapter(config=config, allow_fallback=False)
    with pytest.raises(RuntimeError):
        adapter.run_kernel(carrier)
