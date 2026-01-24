"""
Minimal FFT smoke test with optional vkFFT backend (OpenCL via pyvkfft, or Vulkan handles if provided).

Usage:
    python scripts/run_fft_smoke.py --fft-backend numpy   # default, CPU only
    python scripts/run_fft_smoke.py --fft-backend vkfft --device-index 0

Notes:
- No Vulkan handles are created unless fft-backend=vkfft is requested and pyvkfft is unavailable.
- If vkFFT bindings are missing or binding fails, the executor prints a one-line
  warning and falls back to NumPy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running the script from repo root or elsewhere by adding project root to sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_vkfft_adapter import VkFFTExecutor
from gpu_vulkan_dispatcher import VulkanDispatchConfig, create_vulkan_handles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FFT smoke test (NumPy or vkFFT).")
    p.add_argument(
        "--fft-backend",
        choices=["numpy", "vkfft", "vkfft-opencl", "vkfft-vulkan"],
        default="numpy",
    )
    p.add_argument("--size", type=int, default=256, help="1D length (creates size x size grid).")
    p.add_argument("--device-index", type=int, default=0, help="Vulkan device index when using vkFFT.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    handles = None
    if args.fft_backend in {"vkfft", "vkfft-vulkan"}:
        try:
            handles = create_vulkan_handles(VulkanDispatchConfig(device_index=args.device_index))
        except Exception as exc:
            print(
                f"[warn] Vulkan handle setup failed ({exc}); will attempt pyvkfft or fall back to NumPy.",
                file=sys.stderr,
            )

    executor = VkFFTExecutor(handles=handles, fft_backend=args.fft_backend)

    x = np.random.randn(args.size, args.size).astype(np.float32)
    y = executor.fft2(x)
    z = executor.ifft2(y)

    max_err = float(np.max(np.abs(z - x)))
    backend = "NumPy"
    if executor._plans:  # type: ignore[attr-defined]
        plan_backend = next(iter(executor._plans.values())).backend  # type: ignore[attr-defined]
        if plan_backend == "pyvkfft":
            backend = "vkFFT/OpenCL (pyvkfft)"
        elif plan_backend == "vulkan":
            backend = "vkFFT/Vulkan"
    elif args.fft_backend.startswith("vkfft"):
        backend = "vkFFT (fallback to NumPy)"
    print(f"backend: {backend}")
    print(f"shape: {x.shape}, dtype: {x.dtype}")
    print(f"max reconstruction error: {max_err:.3e}")

    executor.close()
    if handles is not None:
        handles.close()


if __name__ == "__main__":
    main()
