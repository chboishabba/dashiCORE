import numpy as np
import pytest

from dashi_core.carrier import Carrier
from dashi_core.kernel import Kernel


class ShapeDriftKernel(Kernel):
    def apply(self, state: Carrier, ctx=None) -> Carrier:
        support = np.ones((state.support.shape[0] + 1,), dtype=bool)
        sign = np.ones_like(support, dtype=np.int8)
        return Carrier(support=support, sign=sign)


def test_shape_drift_detected():
    state = Carrier.from_signed(np.array([1, 0], dtype=np.int8))
    kernel = ShapeDriftKernel()
    with pytest.raises(ValueError):
        kernel(state)
