from __future__ import annotations

import numpy as np

from ..backend import get_backend
from ..carrier import Carrier
from ..kernel import Kernel


class IdentityKernel(Kernel):
    is_idempotent = True

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        return state


class ZeroKernel(Kernel):
    is_idempotent = True

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        backend = get_backend()
        support = backend.zeros_like(state.support, dtype=bool)
        sign = backend.zeros_like(state.sign, dtype=np.int8)
        return Carrier(support=support, sign=sign)


class ClampKernel(Kernel):
    is_idempotent = True

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        backend = get_backend()
        raw_sign = backend.sign(state.sign)
        corrected = backend.where(raw_sign == 0, backend.ones_like(raw_sign, dtype=np.int8), raw_sign)
        sign = backend.where(state.support, corrected, backend.zeros_like(state.sign, dtype=np.int8))
        return Carrier(support=state.support.copy(), sign=sign)


class OneStepErodeKernel(Kernel):
    is_idempotent = False

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        backend = get_backend()
        support = state.support.copy()
        sign = state.sign.copy()
        coords = np.argwhere(support)
        if len(coords) > 0:
            idx = tuple(coords[0])
            support[idx] = False
            sign[idx] = 0
        return Carrier(support=support, sign=sign)


class AdmissibilityNeutralKernel(Kernel):
    is_idempotent = True

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        return Carrier(support=state.support.copy(), sign=state.sign.copy())
