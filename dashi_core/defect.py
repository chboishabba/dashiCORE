from __future__ import annotations

import numpy as np

from .backend import get_backend
from .carrier import Carrier
from .kernel import Kernel


def local(pre: Carrier, post: Carrier) -> np.ndarray:
    """Local defect = absolute difference between signed carriers."""
    backend = get_backend()
    pre_signed = pre.to_signed()
    post_signed = post.to_signed()
    if pre_signed.shape != post_signed.shape:
        raise ValueError("Defect requires matching shapes")
    return backend.abs(pre_signed - post_signed).astype(np.float64)


def aggregate(local_defect: np.ndarray) -> float:
    backend = get_backend()
    backend.require_capability("deterministic_reductions")
    return float(backend.sum(local_defect))


def is_zero(state: Carrier, kernel: Kernel) -> bool:
    projected = kernel.apply(state)
    return aggregate(local(state, projected)) == 0.0
