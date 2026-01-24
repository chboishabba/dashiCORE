from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from dashi_core.backend import Backend, BackendCapabilities
from dashi_core.carrier import Carrier
from dashi_core.kernel import Kernel, validate_kernel_output
from gpu_common_methods import compile_shader


@dataclass(frozen=True)
class VulkanKernelConfig:
    shader_path: Path
    spv_path: Path
    workgroup: Tuple[int, int, int] = (1, 1, 1)
    compile_on_init: bool = False
    compile_on_dispatch: bool = False


# GPU dispatch signature used by the adapter.
DispatchFn = Callable[[Carrier], Carrier]


class VulkanBackendAdapter(Backend):
    """
    Minimal Vulkan-friendly adapter that keeps GPU concerns outside dashiCORE.

    Current implementation is a CPU no-op that enforces Carrier invariants.
    GPU dispatch hooks (SSBO upload/dispatch/readback) can be filled in later.
    """

    name = "vulkan"

    def __init__(
        self,
        config: VulkanKernelConfig,
        allow_fallback: bool = True,
        dispatcher: Optional[DispatchFn] = None,
        capabilities: Optional[BackendCapabilities] = None,
        precision: str = "float32",
    ):
        super().__init__(capabilities=capabilities, precision=precision)
        self.config = config
        self.allow_fallback = allow_fallback
        self.dispatcher = dispatcher
        self._compiled = False
        if self.config.compile_on_init:
            self.compile()

    def compile(self) -> None:
        """Compile GLSL to SPIR-V; raises if shader is missing or glslc fails."""
        compile_shader(self.config.shader_path, self.config.spv_path)
        self._compiled = True

    def _ensure_compiled(self) -> None:
        if self._compiled:
            return
        if self.config.compile_on_init or self.config.compile_on_dispatch:
            self.compile()

    def run_kernel(self, carrier: Carrier) -> Carrier:
        """
        Placeholder dispatch: returns input state after validation.
        Fill in Vulkan buffer upload/dispatch/readback here when GPU path lands.
        """
        self._ensure_compiled()

        if self.dispatcher is None:
            if not self.allow_fallback:
                raise RuntimeError("Vulkan dispatcher missing and fallback disabled")
            post = carrier
        else:
            post = self.dispatcher(carrier)

        validate_kernel_output(carrier, post)
        return post


class VulkanCarrierKernel(Kernel):
    """Kernel wrapper that delegates to a VulkanBackendAdapter."""

    is_idempotent = True

    def __init__(self, adapter: VulkanBackendAdapter):
        self.adapter = adapter

    def apply(self, state: Carrier, ctx: Optional[object] = None) -> Carrier:
        return self.adapter.run_kernel(state)
