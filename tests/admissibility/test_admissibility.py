import numpy as np

from dashi_core.admissibility import AdmissibilityTransform, apply, equivalent, invariant_defect
from dashi_core.carrier import Carrier
from dashi_core.testing.mock_kernels import AdmissibilityNeutralKernel


def _flip_transform(state: Carrier) -> Carrier:
    return Carrier(support=np.flip(state.support, axis=0), sign=np.flip(state.sign, axis=0))


def test_admissibility_invariance_of_defect():
    state = Carrier.from_signed(np.array([[1, 0], [-1, 1]], dtype=np.int8))
    transform = AdmissibilityTransform(name="flip", fn=_flip_transform)
    kernel = AdmissibilityNeutralKernel()
    assert invariant_defect(state, kernel, transform)


def test_admissibility_equivalence_relation():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    assert equivalent(state, state)
    transformed = _flip_transform(state)
    assert equivalent(transformed, transformed)
    assert equivalent(state, Carrier.from_signed(state.to_signed()))


def test_admissibility_preserves_shape():
    state = Carrier.from_signed(np.array([[1, 0], [0, -1]], dtype=np.int8))
    transform = AdmissibilityTransform(name="flip", fn=_flip_transform)
    transformed = apply(state, transform)
    assert transformed.support.shape == state.support.shape
