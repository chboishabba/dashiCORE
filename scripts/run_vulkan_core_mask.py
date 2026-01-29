"""
Smoke test for the CORE mask Vulkan shader.

Steps:
1) Compile spv/comp/core_mask.comp -> spv/core_mask.spv (on-demand, fallback to gpu_shaders).
2) Register a Vulkan backend (no CPU fallback).
3) Dispatch the mask shader on a sample Carrier and compare with CPU reference.

Prereqs:
- VK_ICD_FILENAMES pointing at a valid ICD JSON.
- glslc on PATH.
- python-vulkan installed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashi_core.backend import use_backend
from dashi_core.carrier import Carrier
from gpu_common_methods import resolve_shader
from gpu_vulkan_backend import make_vulkan_kernel, register_default_vulkan_backend


def cpu_core_mask(carrier: Carrier) -> Carrier:
    """CPU reference: preserve support, saturate sign to {-1,0,1}."""
    signed = carrier.to_signed()
    saturated = np.clip(signed, -1, 1).astype(np.int8, copy=False)
    support = carrier.support & (saturated != 0)
    return Carrier(support=support, sign=saturated)


def main() -> None:
    backend = register_default_vulkan_backend(
        name="vulkan_core_mask",
        shader_path=resolve_shader("core_mask"),
    )
    kernel = make_vulkan_kernel(backend)

    # Ternary sample (Carrier enforces {-1,0,1}); saturation is a no-op here but parity is verified.
    carrier = Carrier.from_signed(np.array([1, 1, 0, -1, -1], dtype=np.int8))

    cpu_out = cpu_core_mask(carrier)

    with use_backend("vulkan_core_mask"):
        vk_out = kernel(carrier)

    print("Input        :", carrier.to_signed())
    print("CPU masked   :", cpu_out.to_signed())
    print("Vulkan masked:", vk_out.to_signed())
    print("Support match:", np.array_equal(cpu_out.support, vk_out.support))
    print("Sign match   :", np.array_equal(cpu_out.sign, vk_out.sign))


if __name__ == "__main__":
    main()
