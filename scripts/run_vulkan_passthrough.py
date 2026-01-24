"""
Minimal Vulkan backend smoke test for dashiCORE on a RADV/RX 580 setup.

Steps performed:
1) Compiles (or reuses) gpu_shaders/carrier_passthrough.comp → .spv via glslc.
2) Registers the Vulkan backend with a real compute dispatcher (no CPU fallback).
3) Runs the passthrough kernel against a sample Carrier and prints the result.

Prereqs:
- Environment variable VK_ICD_FILENAMES set to your RADV ICD (see README/CORE_TRANSITION notes).
- glslc on PATH (shaderc from Vulkan SDK).
- vulkan Python package installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Allow running directly from repo root without installing the package.
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashi_core.backend import use_backend
from dashi_core.carrier import Carrier
from gpu_vulkan_backend import make_vulkan_kernel, register_vulkan_backend, VulkanKernelConfig
from gpu_vulkan_dispatcher import VulkanDispatchConfig


def main() -> None:
    shader = Path("gpu_shaders/carrier_passthrough.comp")
    spv = Path("gpu_shaders/carrier_passthrough.spv")
    config = VulkanKernelConfig(
        shader_path=shader,
        spv_path=spv,
        compile_on_dispatch=True,
    )
    dispatch_cfg = VulkanDispatchConfig(device_index=0)

    backend = register_vulkan_backend(
        name="vulkan",
        config=config,
        dispatch_config=dispatch_cfg,
        allow_fallback=False,
    )
    kernel = make_vulkan_kernel(backend)

    carrier = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))

    with use_backend("vulkan"):
        out = kernel(carrier)

    print("Input signed:", carrier.to_signed())
    print("Output signed:", out.to_signed())
    print("Support identical:", np.array_equal(out.support, carrier.support))


if __name__ == "__main__":
    main()
