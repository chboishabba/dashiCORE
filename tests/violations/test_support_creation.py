import numpy as np
import pytest

from dashi_core.carrier import Carrier
from dashi_core.kernel import Kernel


class IllegalKernel(Kernel):
    def apply(self, state: Carrier, ctx=None) -> Carrier:
        support = np.ones_like(state.support, dtype=bool)
        sign = np.ones_like(state.sign, dtype=np.int8)
        return Carrier(support=support, sign=sign)


def test_kernel_must_not_create_support():
    empty = Carrier.from_signed(np.zeros((2, 2), dtype=np.int8))
    kernel = IllegalKernel()
    with pytest.raises(ValueError):
        kernel(empty)
