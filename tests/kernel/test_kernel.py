import numpy as np

from dashi_core.carrier import Carrier
from dashi_core.defect import aggregate, is_zero, local
from dashi_core.testing.mock_kernels import (
    AdmissibilityNeutralKernel,
    ClampKernel,
    IdentityKernel,
    OneStepErodeKernel,
    ZeroKernel,
)


def sample_state():
    signed = np.array([[1, 0], [-1, 1]], dtype=np.int8)
    return Carrier.from_signed(signed)


def test_kernel_shape_preserving():
    state = sample_state()
    for kernel in [IdentityKernel(), ZeroKernel(), ClampKernel(), OneStepErodeKernel(), AdmissibilityNeutralKernel()]:
        out = kernel.apply(state)
        assert out.support.shape == state.support.shape


def test_kernel_valid_output():
    state = sample_state()
    for kernel in [ZeroKernel(), ClampKernel(), OneStepErodeKernel()]:
        out = kernel.apply(state)
        out.validate()


def test_kernel_fixed_point_zero_defect():
    state = sample_state()
    kernel = IdentityKernel()
    assert is_zero(state, kernel)


def test_kernel_idempotence_if_declared():
    state = sample_state()
    kernel = ClampKernel()
    first = kernel.apply(state)
    second = kernel.apply(first)
    assert aggregate(local(first, second)) == 0.0
