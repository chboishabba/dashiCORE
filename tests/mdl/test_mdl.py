import numpy as np

from dashi_core.admissibility import AdmissibilityTransform, apply
from dashi_core.carrier import Carrier
from dashi_core.mdl import compare, score


def _rotate(state: Carrier) -> Carrier:
    return Carrier(support=np.rot90(state.support), sign=np.rot90(state.sign))


def test_mdl_non_negative():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    assert score(state) >= 0


def test_mdl_invariance_under_admissibility():
    state = Carrier.from_signed(np.array([[1, 0], [0, -1]], dtype=np.int8))
    transform = AdmissibilityTransform(name="rotate", fn=_rotate)
    assert score(state) == score(apply(state, transform))


def test_mdl_comparison_total_order():
    a = Carrier.from_signed(np.array([1, 0], dtype=np.int8))
    b = Carrier.from_signed(np.array([1, 1], dtype=np.int8))
    assert compare(a, b) in {"A", "B", "tie"}
