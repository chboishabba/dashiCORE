from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

from dashi_core.backend import BackendCapabilities, register_backend
from gpu_vulkan_adapter import DispatchFn, VulkanBackendAdapter, VulkanCarrierKernel, VulkanKernelConfig
from gpu_vulkan_dispatcher import VulkanDispatchConfig, build_vulkan_dispatcher

DEFAULT_VULKAN_CAPABILITIES = BackendCapabilities(
    supports_float64=True,
    supports_atomic_ops=False,
    deterministic_reductions=True,
    supports_int8_exact=True,
    allows_mixed_precision=False,
)


class VulkanBackend(VulkanBackendAdapter):
    """Selectable Vulkan backend wrapper suitable for registry use."""

    name = "vulkan"

    def __init__(
        self,
        config: VulkanKernelConfig,
        *,
        allow_fallback: bool = True,
        dispatcher: Optional[DispatchFn] = None,
        capabilities: Optional[BackendCapabilities] = None,
        precision: str = "float32",
    ):
        caps = capabilities or DEFAULT_VULKAN_CAPABILITIES
        super().__init__(
            config=config,
            allow_fallback=allow_fallback,
            dispatcher=dispatcher,
            capabilities=caps,
            precision=precision,
        )


def register_vulkan_backend(
    *,
    name: str = "vulkan",
    config: VulkanKernelConfig,
    allow_fallback: bool = True,
    dispatcher: Optional[DispatchFn] = None,
    dispatch_config: Optional[VulkanDispatchConfig] = None,
    capabilities: Optional[BackendCapabilities] = None,
    precision: str = "float32",
) -> VulkanBackend:
    """Create and register a Vulkan backend instance."""
    if dispatcher is None:
        dispatcher = build_vulkan_dispatcher(config=config, dispatch_config=dispatch_config)
    backend = VulkanBackend(
        config=config,
        allow_fallback=allow_fallback,
        dispatcher=dispatcher,
        capabilities=capabilities,
        precision=precision,
    )
    register_backend(name, backend)
    return backend


def make_vulkan_kernel(adapter: VulkanBackendAdapter) -> VulkanCarrierKernel:
    """Helper to build a Kernel wrapper from a registered adapter/backend."""
    return VulkanCarrierKernel(adapter)


def register_default_vulkan_backend(
    *,
    name: str = "vulkan",
    shader_path: Optional[Path] = None,
    spv_path: Optional[Path] = None,
    device_index: int = 0,
    workgroup: tuple[int, int, int] = (64, 1, 1),
    memory_mode: str = "host_visible",
    allow_fallback: bool = False,
) -> VulkanBackend:
    """Register a fully wired Vulkan backend backed by the default CORE mask shader.

    Intended for production/benchmark use where a real GPU dispatch must occur
    (no silent CPU passthrough).
    """

    shader = shader_path or Path("gpu_shaders/core_mask.comp")
    spv = spv_path or shader.with_suffix(".spv")

    config = VulkanKernelConfig(
        shader_path=shader,
        spv_path=spv,
        workgroup=workgroup,
        compile_on_dispatch=True,
        compile_on_init=False,
    )

    dispatch_cfg = VulkanDispatchConfig(device_index=device_index, memory_mode=memory_mode)

    return register_vulkan_backend(
        name=name,
        config=config,
        allow_fallback=allow_fallback,
        dispatch_config=dispatch_cfg,
    )


def probe_and_register_vulkan_backend(
    *,
    name: str = "vulkan",
    shader_path: Optional[Path] = None,
    spv_path: Optional[Path] = None,
    device_index: int = 0,
    workgroup: tuple[int, int, int] = (64, 1, 1),
    memory_mode: str = "host_visible",
    icd_candidates: Optional[Iterable[Path]] = None,
) -> Tuple[Optional[VulkanBackend], Optional[Path]]:
    """
    Try to register a Vulkan backend by probing ICD JSONs.

    Returns (backend, icd_path). If no ICD works or Vulkan is unavailable,
    returns (None, None) without raising, leaving the caller to fall back to CPU.
    """
    import os
    import subprocess

    shader = shader_path or Path("gpu_shaders/core_mask_majority.comp")
    spv = spv_path or shader.with_suffix(".spv")

    # Default ICD search paths
    candidates: Iterable[Path]
    if icd_candidates is None:
        candidates = [
            Path(p)
            for p in (
                "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
                "/usr/share/vulkan/icd.d/amd_icd64.json",
                "/usr/share/vulkan/icd.d/nvidia_icd.json",
            )
        ]
        # Extend with any present files in standard dirs
        candidates += list(Path("/usr/share/vulkan/icd.d").glob("*.json"))  # type: ignore
        candidates += list(Path("/etc/vulkan/icd.d").glob("*.json"))  # type: ignore
    else:
        candidates = icd_candidates

    for icd in candidates:
        if not icd.is_file():
            continue
        env = dict(os.environ)
        env["VK_ICD_FILENAMES"] = str(icd)
        try:
            # Light-touch validation: run vulkaninfo --summary if available.
            if subprocess.call(
                ["which", "vulkaninfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ) == 0:
                subprocess.check_call(
                    ["vulkaninfo", "--summary"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
        except Exception:
            continue
        try:
            backend = register_default_vulkan_backend(
                name=name,
                shader_path=shader,
                spv_path=spv,
                device_index=device_index,
                workgroup=workgroup,
                memory_mode=memory_mode,
                allow_fallback=False,
            )
            return backend, icd
        except Exception:
            # Try next ICD
            continue
    return None, None
