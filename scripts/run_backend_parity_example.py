"""
Run a simple backend parity check between CPU and Vulkan backends.

Usage examples:
- CPU vs Vulkan with real dispatcher (requires Vulkan + glslc + ICD):
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \\
  python scripts/run_backend_parity_example.py

- CPU vs Vulkan using CPU dispatcher (no Vulkan dependencies):
  python scripts/run_backend_parity_example.py --cpu-dispatcher
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashi_core.backend import use_backend
from dashi_core.carrier import Carrier
from gpu_common_methods import compile_shader
from gpu_vulkan_backend import make_vulkan_kernel, register_vulkan_backend, VulkanKernelConfig
from gpu_vulkan_dispatcher import VulkanDispatchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU vs Vulkan backend parity demo")
    parser.add_argument(
        "--shader",
        type=Path,
        default=ROOT / "gpu_shaders" / "carrier_passthrough.comp",
        help="Path to GLSL compute shader",
    )
    parser.add_argument(
        "--spv",
        type=Path,
        default=ROOT / "gpu_shaders" / "carrier_passthrough.spv",
        help="Path to SPIR-V output",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Vulkan device index",
    )
    parser.add_argument(
        "--cpu-dispatcher",
        action="store_true",
        help="Use CPU dispatcher instead of Vulkan (no GPU required)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cpu_dispatcher:
        compile_shader(args.shader, args.spv)

    config = VulkanKernelConfig(
        shader_path=args.shader,
        spv_path=args.spv,
        compile_on_dispatch=not args.cpu_dispatcher,
    )

    dispatcher = None
    if args.cpu_dispatcher:
        dispatcher = lambda c: c  # type: ignore[assignment]

    backend = register_vulkan_backend(
        name="vulkan_parity_demo",
        config=config,
        dispatch_config=VulkanDispatchConfig(device_index=args.device_index),
        allow_fallback=bool(args.cpu_dispatcher),
        dispatcher=dispatcher,
    )
    kernel = make_vulkan_kernel(backend)

    carrier = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))

    with use_backend("cpu"):
        cpu_signed = carrier.to_signed()

    with use_backend("vulkan_parity_demo"):
        vk_out = kernel(carrier)
        vk_signed = vk_out.to_signed()

    print("CPU signed: ", cpu_signed)
    print("Vulkan signed:", vk_signed)
    print("Support identical:", np.array_equal(vk_out.support, carrier.support))
    print("Parity:", np.array_equal(cpu_signed, vk_signed))


if __name__ == "__main__":
    main()
