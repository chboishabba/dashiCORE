from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..carrier import Carrier


def to_carrier(legacy_signed: np.ndarray, neutral_mask: Optional[np.ndarray] = None) -> Tuple[Carrier, np.ndarray]:
    """
    Map legacy ternary state into a CORE Carrier with constant support.

    Neutral (legacy zeros) are represented via an external mask; Carrier.sign uses +1 there.
    """
    signed_arr = np.asarray(legacy_signed, dtype=np.int8)
    mask = _validate_or_default_mask(signed_arr, neutral_mask)
    support = np.ones_like(signed_arr, dtype=bool)
    signed_sign = np.sign(signed_arr).astype(np.int8)
    sign = np.where(mask, np.ones_like(signed_arr, dtype=np.int8), signed_sign)
    carrier = Carrier(support=support, sign=sign)
    return carrier, mask


def from_carrier(carrier: Carrier, neutral_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Map a Carrier back to legacy signed representation, restoring neutral zeros if mask is provided."""
    signed = carrier.to_signed().copy()
    if neutral_mask is not None:
        mask = np.asarray(neutral_mask, dtype=bool)
        if mask.shape != signed.shape:
            raise ValueError("neutral_mask shape must match carrier shape")
        signed[mask] = 0
    return signed


def _validate_or_default_mask(signed_arr: np.ndarray, neutral_mask: Optional[np.ndarray]) -> np.ndarray:
    if neutral_mask is None:
        return signed_arr == 0
    mask = np.asarray(neutral_mask, dtype=bool)
    if mask.shape != signed_arr.shape:
        raise ValueError("neutral_mask shape must match legacy signed shape")
    return mask
