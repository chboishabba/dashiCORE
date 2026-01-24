from __future__ import annotations

from .backend import get_backend
from .carrier import Carrier


def lift(state: Carrier, levels: int = 1) -> Carrier:
    backend = get_backend()
    support = state.support
    sign = state.sign
    for _ in range(levels):
        support = backend.expand_dims(support, axis=-1)
        sign = backend.expand_dims(sign, axis=-1)
    return Carrier(support=support, sign=sign)


def project(state: Carrier, levels: int = 1) -> Carrier:
    backend = get_backend()
    support = state.support
    sign = state.sign
    for _ in range(levels):
        if support.shape[-1] != 1:
            # Collapse by logical any over last axis to avoid spurious defect creation
            support = support.any(axis=-1, keepdims=False)
            sign = sign[..., 0]
        else:
            support = backend.squeeze(support, axis=-1)
            sign = backend.squeeze(sign, axis=-1)
    return Carrier(support=support, sign=sign)
