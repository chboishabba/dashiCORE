import numpy as np
import pytest

from dashi_core.carrier import Carrier


def test_support_sign_collapse_rejected():
    bad = np.array([0.2, -0.7, 0.0])
    with pytest.raises(ValueError):
        Carrier.from_signed(bad)
