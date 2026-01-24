"""Reusable workload generators and benchmark axes for dashiCORE benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np

from dashi_core.carrier import Carrier


# Canonical axes
DEFAULT_SIZES: List[int] = [1024, 4096, 16384, 65536, 262144, 1048576]
DEFAULT_SPARSITIES: List[float] = [0.0, 0.5, 0.9]
DEFAULT_PASSES: List[int] = [1, 2, 4, 8, 16]


@dataclass(frozen=True)
class Workload:
    """Describes a workload family for benchmarks."""

    name: str
    generator: Callable[[int, float, int], Carrier]
    description: str


def _random_sparse(n: int, sparsity: float, seed: int) -> Carrier:
    rng = np.random.default_rng(seed)
    probs = [sparsity, (1 - sparsity) / 2, (1 - sparsity) / 2]
    vals = rng.choice([0, -1, 1], size=n, p=probs).astype(np.int8)
    return Carrier.from_signed(vals)


def _clustered_sparse(n: int, sparsity: float, seed: int, run_len: int = 8) -> Carrier:
    rng = np.random.default_rng(seed)
    vals = np.zeros(n, dtype=np.int8)
    idx = 0
    while idx < n:
        run = min(run_len, n - idx)
        sign = rng.choice([-1, 1])
        keep = rng.random(run) > sparsity
        vals[idx : idx + run] = sign * keep.astype(np.int8)
        idx += run
    return Carrier.from_signed(vals)


def _checkerboard(n: int, sparsity: float, seed: int) -> Carrier:
    vals = np.zeros(n, dtype=np.int8)
    # Alternating +/- with injected zeros per sparsity
    base = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
    rng = np.random.default_rng(seed)
    mask = rng.random(n) > sparsity
    vals[mask] = base[mask]
    return Carrier.from_signed(vals)


def _blocky_tiles(n: int, sparsity: float, seed: int, tile_size: int = 32) -> Carrier:
    rng = np.random.default_rng(seed)
    vals = np.zeros(n, dtype=np.int8)
    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        fill = rng.choice([0, -1, 1], p=[sparsity, (1 - sparsity) / 2, (1 - sparsity) / 2])
        vals[start:end] = fill
    return Carrier.from_signed(vals)


def _adversarial_edges(n: int, sparsity: float, seed: int, radius: int = 1) -> Carrier:
    rng = np.random.default_rng(seed)
    vals = _random_sparse(n, sparsity, seed).to_signed()
    # Force boundary flips around partition seams to stress halos
    for i in range(radius, n, max(radius, 32)):
        vals[i - radius : i + radius] = rng.choice([-1, 1])
    return Carrier.from_signed(vals)


# Registry of workload families; extend as new patterns matter
WORKLOADS: Dict[str, Workload] = {
    "random_sparse": Workload(
        name="random_sparse",
        generator=_random_sparse,
        description="IID ternary with configurable sparsity",
    ),
    "clustered_sparse": Workload(
        name="clustered_sparse",
        generator=_clustered_sparse,
        description="Runs of identical sign with sparsity mask",
    ),
    "checkerboard": Workload(
        name="checkerboard",
        generator=_checkerboard,
        description="Alternating +/- with sparsity masking",
    ),
    "blocky_tiles": Workload(
        name="blocky_tiles",
        generator=_blocky_tiles,
        description="Tiles filled with a single sign or zero",
    ),
    "adversarial_edges": Workload(
        name="adversarial_edges",
        generator=_adversarial_edges,
        description="Stress partition seams / halos with forced sign flips",
    ),
    "stencil_dense_iterated": Workload(
        name="stencil_dense_iterated",
        generator=lambda n, sparsity, seed: _blocky_tiles(n, sparsity=0.0, seed=seed, tile_size=1),
        description="Fully dense ±1 field intended for iterated, high-intensity stencil kernels",
    ),
}


def names() -> Iterable[str]:
    """Return workload names."""
    return WORKLOADS.keys()


def make(workload: str, n: int, sparsity: float, seed: int) -> Carrier:
    """Create a Carrier for the requested workload."""
    if workload not in WORKLOADS:
        raise ValueError(f"Unknown workload '{workload}'")
    return WORKLOADS[workload].generator(n, sparsity, seed)
