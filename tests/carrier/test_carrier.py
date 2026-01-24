import numpy as np
import pytest

from dashi_core.carrier import Carrier


def test_carrier_from_signed_roundtrip():
    signed = np.array([-1, 0, 1, 1, 0, -1], dtype=np.int8)
    carrier = Carrier.from_signed(signed)
    np.testing.assert_array_equal(carrier.to_signed(), signed)


def test_carrier_support_sign_separation():
    signed = np.array([-1, 0, 1], dtype=np.int8)
    carrier = Carrier.from_signed(signed)
    expected_support = signed != 0
    np.testing.assert_array_equal(carrier.support, expected_support)
    assert set(np.unique(carrier.sign[carrier.support])).issubset({-1, 1})
    np.testing.assert_array_equal(carrier.sign[~carrier.support], np.zeros(1, dtype=np.int8))


def test_carrier_rejects_illegal_values():
    signed = np.array([2, -2, 42], dtype=np.int8)
    with pytest.raises(ValueError):
        Carrier.from_signed(signed)


def test_carrier_shape_preservation():
    signed = np.array([[1, 0], [-1, 0]], dtype=np.int8)
    carrier = Carrier.from_signed(signed)
    assert carrier.support.shape == signed.shape
    assert carrier.sign.shape == signed.shape
