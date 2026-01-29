"""
Smoke test for the core_mask_majority Vulkan shader.

Creates K channels of ternary data, runs the majority-reduction shader on GPU,
and checks parity with a CPU reference.

Usage:
  VK_ICD_FILENAMES=/path/to/icd.json MPLBACKEND=Agg \\
    python dashiCORE/scripts/run_vulkan_core_mask_majority.py --n 1024 --k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashi_core.backend import use_backend  # noqa: E402
from dashi_core.carrier import Carrier  # noqa: E402
from gpu_common_methods import resolve_shader  # noqa: E402
from gpu_vulkan_backend import make_vulkan_kernel, register_default_vulkan_backend  # noqa: E402


def cpu_majority(support: np.ndarray, sign: np.ndarray) -> Carrier:
    """Channel-major majority vote."""
    k, n = support.shape
    support_out = np.zeros(n, dtype=bool)
    sign_out = np.ones(n, dtype=np.int8)
    for i in range(n):
        m = support[:, i]
        if not np.any(m):
            continue
        votes = np.sign(sign[:, i][m]).astype(np.int32)
        total = votes.sum()
        if total > 0:
            support_out[i] = True
            sign_out[i] = 1
        elif total < 0:
            support_out[i] = True
            sign_out[i] = -1
        else:
            # tie -> project away
            support_out[i] = False
            sign_out[i] = 0
    sign_out = sign_out * support_out.astype(np.int8)
    return Carrier(support=support_out, sign=sign_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024, help="elements per channel")
    ap.add_argument("--k", type=int, default=3, help="number of channels")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    support = rng.random((args.k, args.n)) < 0.4
    sign = rng.integers(low=0, high=2, size=(args.k, args.n), dtype=np.int8) * 2 - 1
    sign = sign * support.astype(np.int8)

    carrier = Carrier(support=support, sign=sign)

    backend = register_default_vulkan_backend(
        name="vulkan_core_mask_majority",
        shader_path=resolve_shader("core_mask_majority"),
        workgroup=(256, 1, 1),
        allow_fallback=False,
    )
    kernel = make_vulkan_kernel(backend)

    cpu_out = cpu_majority(support, sign)

    with use_backend("vulkan_core_mask_majority"):
        vk_out = kernel(carrier)

    print(f"support match: {np.array_equal(cpu_out.support, vk_out.support)}")
    print(f"sign match   : {np.array_equal(cpu_out.sign, vk_out.sign)}")
    if not np.array_equal(cpu_out.to_signed(), vk_out.to_signed()):
        mism = np.flatnonzero(cpu_out.to_signed() != vk_out.to_signed())[:10]
        print("first mismatches at indices:", mism)


if __name__ == "__main__":
    main()
