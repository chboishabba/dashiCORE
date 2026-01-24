import numpy as np
import pytest

from dashi_core.carrier import Carrier


def test_illegal_carrier_states_rejected():
    support = np.array([True, True])
    sign = np.array([2, 0], dtype=np.int8)
    with pytest.raises(ValueError):
        Carrier(support=support, sign=sign)
