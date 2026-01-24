import itertools
import random

import numpy as np

from dashi_core.carrier import Carrier
from dashi_core.kernel import Kernel


class SignFlipKernel(Kernel):
    def apply(self, state: Carrier, ctx=None) -> Carrier:
        flipped = -state.sign
        return Carrier(support=state.support, sign=flipped)


class LeftNeighborStencil(Kernel):
    """Stencil kernel depending on left neighbor only (halo width = 1)."""

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        sign = state.sign
        left = np.concatenate([np.array([0], dtype=np.int8), sign[:-1]])
        out_sign = np.sign(sign + left).astype(np.int8)
        out_support = state.support & (out_sign != 0)
        out_sign = np.where(out_support, out_sign, 0).astype(np.int8)
        return Carrier(support=out_support, sign=out_sign)


def _split_contiguous(carrier: Carrier, num_blocks: int):
    length = carrier.sign.size
    block_size = (length + num_blocks - 1) // num_blocks
    parts = []
    for i in range(0, length, block_size):
        j = min(i + block_size, length)
        sup = carrier.support[i:j]
        sig = carrier.sign[i:j]
        parts.append((i, Carrier(support=sup, sign=sig)))
    return parts


def _recombine(parts):
    # parts: sequence of (start_index, Carrier)
    parts_sorted = sorted(parts, key=lambda t: t[0])
    support = np.concatenate([p.support for _, p in parts_sorted])
    sign = np.concatenate([p.sign for _, p in parts_sorted])
    return Carrier(support=support, sign=sign)


def test_disjoint_partition_equivalence():
    carrier = Carrier.from_signed(np.array([1, -1, 0, 1, -1, 1, 0, -1], dtype=np.int8))
    kernel = SignFlipKernel()
    full = kernel(carrier)

    parts = _split_contiguous(carrier, num_blocks=4)
    outs = []
    for idx, part in parts:
        outs.append((idx, kernel(part)))
    recombined = _recombine(outs)

    np.testing.assert_array_equal(recombined.to_signed(), full.to_signed())
    np.testing.assert_array_equal(recombined.support, full.support)


def test_scheduling_invariance():
    carrier = Carrier.from_signed(np.array([1, -1, 0, 1, -1, 1, 0, -1], dtype=np.int8))
    kernel = SignFlipKernel()
    parts = _split_contiguous(carrier, num_blocks=4)
    orders = [parts, list(reversed(parts))]
    random.seed(0)
    for _ in range(3):
        shuffled = parts.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)

    outputs = []
    for order in orders:
        outs = []
        for idx, part in order:
            outs.append((idx, kernel(part)))
        outputs.append(_recombine(outs))

    first = outputs[0]
    for out in outputs[1:]:
        np.testing.assert_array_equal(out.to_signed(), first.to_signed())
        np.testing.assert_array_equal(out.support, first.support)


def _split_with_halo(carrier: Carrier, block_size: int, halo: int):
    """Yield (core slice, halo-extended carrier, core indices)."""
    length = carrier.sign.size
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        halo_start = max(0, start - halo)
        halo_end = min(length, end + halo)
        sup = carrier.support[halo_start:halo_end]
        sig = carrier.sign[halo_start:halo_end]
        yield (start, end, halo_start, halo_end, Carrier(support=sup, sign=sig))


def test_halo_partition_equivalence():
    carrier = Carrier.from_signed(np.array([1, 0, -1, 1, -1, 0, 1], dtype=np.int8))
    kernel = LeftNeighborStencil()
    full = kernel(carrier)

    block_size = 3
    halo = 1
    parts = []
    for start, end, h_start, _, sub in _split_with_halo(carrier, block_size=block_size, halo=halo):
        out = kernel(sub)
        core_slice = slice(start - h_start, end - h_start)
        core_sup = out.support[core_slice]
        core_sig = out.sign[core_slice]
        parts.append((start, Carrier(support=core_sup, sign=core_sig)))

    recombined = _recombine(parts)
    np.testing.assert_array_equal(recombined.to_signed(), full.to_signed())
    np.testing.assert_array_equal(recombined.support, full.support)
