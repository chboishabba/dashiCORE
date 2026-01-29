from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

from dashi_core.backend import BackendCapabilities, register_backend
from gpu_common_methods import resolve_shader, resolve_spv
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

    shader = shader_path or resolve_shader("core_mask")
    spv = spv_path or resolve_spv(shader.stem)

    # Binding order depends on the shader.
    # core_mask_majority expects support, sign, support_out, sign_out.
    # legacy core_mask expects sign, support, sign_out, support_out.
    binding_order = ("support_in", "sign_in", "support_out", "sign_out")
    shader_name = (shader.stem if shader else "")
    if shader_name == "core_mask":
        binding_order = ("sign_in", "support_in", "sign_out", "support_out")

    if shader_name == "core_mask_majority" and workgroup == (64, 1, 1):
        workgroup = (256, 1, 1)

    config = VulkanKernelConfig(
        shader_path=shader,
        spv_path=spv,
        workgroup=workgroup,
        compile_on_dispatch=True,
        compile_on_init=False,
        binding_order=binding_order,
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
    import subprocess

    shader = shader_path or resolve_shader("core_mask_majority")
    spv = spv_path or resolve_spv(shader.stem)

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

    debug = os.getenv("DASHI_VULKAN_DEBUG")

    for icd in candidates:
        if not icd.is_file():
            continue
        env = dict(os.environ)
        env["VK_ICD_FILENAMES"] = str(icd)
        # Light-touch validation: run vulkaninfo --summary if available. If it fails, fall through
        # and still attempt registration so we can see the real Vulkan error (permission, missing /dev/dri, etc.).
        try:
            if subprocess.call(
                ["which", "vulkaninfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ) == 0:
                try:
                    subprocess.check_call(
                        ["vulkaninfo", "--summary"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                    )
                except Exception as exc:
                    if debug:
                        print(f"[vk][probe] vulkaninfo failed for {icd} (continuing): {exc}")
            elif debug:
                print(f"[vk][probe] vulkaninfo not installed; skipping check for {icd}")
        except Exception as exc:
            if debug:
                print(f"[vk][probe] vulkaninfo check raised (continuing) for {icd}: {exc}")
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
            if debug:
                print(f"[vk][probe] registered backend via {icd}")
            return backend, icd
        except Exception as exc:
            # Try next ICD
            if debug:
                print(f"[vk][probe] backend registration failed for {icd}: {exc}")
            continue
    return None, None
