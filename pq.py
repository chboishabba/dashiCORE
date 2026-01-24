"""Minimal PQ (packed ternary) encoding helpers.

This module packs a `Carrier`'s `(support, sign)` into 2-bit values for storage
or transport, and decodes back to a dense `Carrier`. PQ is optional and must be
observationally invisible to core semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Tuple

import numpy as np

from dashi_core.carrier import Carrier

# Mapping: 2-bit value per element
_PQ_VALUES = {
    0: (False, 0),  # no support
    1: (True, -1),  # support with sign -1
    2: (True, 1),   # support with sign +1
}
_PQ_ENCODE = {(False, 0): 0, (True, -1): 1, (True, 1): 2}


@dataclass(frozen=True)
class PQBuffer:
    shape: Tuple[int, ...]
    data: bytes


def encode_carrier_to_pq(carrier: Carrier) -> PQBuffer:
    """Pack a Carrier into a PQBuffer using 2 bits per element.

    Value mapping: 0=no support; 1=support & sign -1; 2=support & sign +1.
    """

    flat_support = carrier.support.ravel()
    flat_sign = carrier.sign.ravel()
    values = np.fromiter(
        (_PQ_ENCODE[(bool(s), int(sn))] for s, sn in zip(flat_support, flat_sign)),
        dtype=np.uint8,
        count=flat_sign.size,
    )
    bits = flat_sign.size * 2
    byte_len = ceil(bits / 8)
    out = bytearray(byte_len)
    for idx, val in enumerate(values):
        bit_pos = idx * 2
        byte_index = bit_pos // 8
        offset = bit_pos % 8
        out[byte_index] |= int(val) << offset
    return PQBuffer(shape=carrier.support.shape, data=bytes(out))


def decode_pq_to_carrier(buffer: PQBuffer) -> Carrier:
    """Decode a PQBuffer back into a dense Carrier."""

    total = int(np.prod(buffer.shape))
    values = np.empty(total, dtype=np.uint8)
    for idx in range(total):
        bit_pos = idx * 2
        byte_index = bit_pos // 8
        offset = bit_pos % 8
        byte_val = buffer.data[byte_index]
        values[idx] = (byte_val >> offset) & 0b11
    # Map back to support/sign
    support = np.empty(total, dtype=bool)
    sign = np.empty(total, dtype=np.int8)
    for i, val in enumerate(values):
        if val not in _PQ_VALUES:
            raise ValueError(f"Invalid PQ value: {val}")
        sup, sgn = _PQ_VALUES[val]
        support[i] = sup
        sign[i] = sgn
    support = support.reshape(buffer.shape)
    sign = sign.reshape(buffer.shape)
    return Carrier(support=support, sign=sign)


def expected_pq_length(num_elements: int) -> int:
    """Return expected PQ byte length for given element count."""

    return ceil((num_elements * 2) / 8)
