import numpy as np

from dashi_core.carrier import Carrier
from dashi_core.hierarchy import lift, project


def test_lift_project_roundtrip_no_spurious_defect():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    lifted = lift(state, levels=1)
    projected = project(lifted, levels=1)
    np.testing.assert_array_equal(projected.to_signed(), state.to_signed())


def test_hierarchy_defect_composition():
    state = Carrier.from_signed(np.array([[1, 0], [0, 1]], dtype=np.int8))
    lifted = lift(state, levels=1)
    projected = project(lifted, levels=1)
    assert projected.support.sum() == state.support.sum()
