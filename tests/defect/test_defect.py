import numpy as np

from dashi_core.carrier import Carrier
from dashi_core.defect import aggregate, local
from dashi_core.testing.mock_kernels import IdentityKernel, OneStepErodeKernel


def test_defect_non_negative():
    pre = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    post = Carrier.from_signed(np.array([1, 0, 0], dtype=np.int8))
    loc = local(pre, post)
    assert (loc >= 0).all()
    assert aggregate(loc) >= 0


def test_defect_zero_iff_fixed_point():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    kernel = IdentityKernel()
    assert aggregate(local(state, kernel.apply(state))) == 0


def test_defect_shape_alignment():
    pre = Carrier.from_signed(np.array([[1, 0], [0, -1]], dtype=np.int8))
    post = Carrier.from_signed(np.array([[1, 0], [0, 0]], dtype=np.int8))
    loc = local(pre, post)
    assert loc.shape == pre.support.shape


def test_defect_monotonicity_under_kernel():
    signed = np.ones((2, 2), dtype=np.int8)
    state = Carrier.from_signed(signed)
    kernel = OneStepErodeKernel()
    once = kernel.apply(state)
    twice = kernel.apply(once)
    first_defect = aggregate(local(state, once))
    second_defect = aggregate(local(once, twice))
    assert first_defect >= second_defect
