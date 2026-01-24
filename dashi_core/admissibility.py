from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .carrier import Carrier
from .defect import aggregate, local
from .kernel import Kernel


TransformFn = Callable[[Carrier], Carrier]


@dataclass(frozen=True)
class AdmissibilityTransform:
    name: str
    fn: TransformFn

    def __call__(self, state: Carrier) -> Carrier:
        return self.fn(state)


def apply(state: Carrier, transform: AdmissibilityTransform) -> Carrier:
    transformed = transform(state)
    transformed.validate()
    if transformed.support.shape != state.support.shape:
        raise ValueError("Admissibility transforms must preserve shape")
    return transformed


def equivalent(state_a: Carrier, state_b: Carrier) -> bool:
    return np.array_equal(state_a.to_signed(), state_b.to_signed())


def invariant_defect(state: Carrier, kernel: Kernel, transform: AdmissibilityTransform) -> bool:
    baseline = aggregate(local(state, kernel.apply(state)))
    moved = aggregate(local(apply(state, transform), kernel.apply(apply(state, transform))))
    return baseline == moved
