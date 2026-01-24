import numpy as np
import pytest

from dashi_core.carrier import Carrier
from dashi_core.defect import aggregate, local
from dashi_core.kernel import Kernel, validate_kernel_output


class SignFlipKernel(Kernel):
    is_idempotent = False

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        flipped = -state.sign
        return Carrier(support=state.support, sign=flipped)


class ClampNonnegativeKernel(Kernel):
    is_idempotent = True

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        clamped_sign = np.where(state.sign < 0, 0, state.sign)
        support = state.support & (clamped_sign != 0)
        return Carrier(support=support, sign=clamped_sign.astype(np.int8))


class LeftNeighborKernel(Kernel):
    """Reads left neighbor without creating new support."""

    is_idempotent = False

    def apply(self, state: Carrier, ctx=None) -> Carrier:
        sign = state.sign
        left = np.concatenate([np.array([0], dtype=np.int8), sign[:-1]])
        out_sign = np.sign(sign + left).astype(np.int8)
        # Do not create new support; only keep where input already had support.
        out_support = state.support & (out_sign != 0)
        out_sign = np.where(out_support, out_sign, 0).astype(np.int8)
        return Carrier(support=out_support, sign=out_sign)


def test_carrier_closure_and_totality():
    carrier = Carrier.from_signed(np.array([1, 0, -1, 1], dtype=np.int8))
    kernel = SignFlipKernel()
    out = kernel(carrier)
    assert set(np.unique(out.sign)).issubset({-1, 0, 1})
    np.testing.assert_array_equal(out.support, carrier.support)


def test_support_monotonicity_neighbor_kernel():
    support = np.array([True, False, True], dtype=bool)
    sign = np.array([1, 0, -1], dtype=np.int8)
    carrier = Carrier(support=support, sign=sign)
    kernel = LeftNeighborKernel()
    out = kernel(carrier)
    assert np.all(out.support <= carrier.support)
    np.testing.assert_array_equal(out.support, np.array([True, False, True], dtype=bool))


def test_involution_sign_flip():
    carrier = Carrier.from_signed(np.array([1, -1, 0, 1], dtype=np.int8))
    kernel = SignFlipKernel()
    once = kernel(carrier)
    twice = kernel(once)
    np.testing.assert_array_equal(twice.to_signed(), carrier.to_signed())
    np.testing.assert_array_equal(twice.support, carrier.support)


def test_idempotent_clamp():
    carrier = Carrier.from_signed(np.array([1, -1, -1, 0], dtype=np.int8))
    kernel = ClampNonnegativeKernel()
    once = kernel(carrier)
    twice = kernel(once)
    np.testing.assert_array_equal(twice.to_signed(), once.to_signed())
    np.testing.assert_array_equal(twice.support, once.support)


def test_purity_repeatability():
    carrier = Carrier.from_signed(np.array([1, -1, 0, 1], dtype=np.int8))
    kernel = ClampNonnegativeKernel()
    outs = [kernel(carrier) for _ in range(5)]
    first = outs[0]
    for out in outs[1:]:
        np.testing.assert_array_equal(out.to_signed(), first.to_signed())
        np.testing.assert_array_equal(out.support, first.support)


def test_defect_monotonicity_under_clamp():
    carrier = Carrier.from_signed(np.array([1, -1, -1, 1], dtype=np.int8))
    clamp = ClampNonnegativeKernel()
    post = clamp(carrier)
    post_twice = clamp(post)
    defect_once = aggregate(local(carrier, post))
    defect_twice = aggregate(local(post, post_twice))
    assert defect_once >= defect_twice


def test_validate_kernel_output_blocks_support_creation():
    class BadKernel(Kernel):
        def apply(self, state: Carrier, ctx=None) -> Carrier:
            sign = np.ones_like(state.sign, dtype=np.int8)
            support = np.ones_like(state.support, dtype=bool)
            return Carrier(support=support, sign=sign)

    carrier = Carrier.from_signed(np.array([1, 0], dtype=np.int8))
    bad = BadKernel()
    with pytest.raises(ValueError):
        validate_kernel_output(carrier, bad(carrier))
