import numpy as np
import pytest

from dashi_core.adapters import from_carrier, to_carrier


def test_adapter_roundtrip_with_neutral_mask():
    legacy = np.array([[-1, 0, 1], [0, 1, -1]], dtype=np.int8)
    carrier, mask = to_carrier(legacy)
    restored = from_carrier(carrier, neutral_mask=mask)
    np.testing.assert_array_equal(restored, legacy)


def test_adapter_support_constant_true():
    legacy = np.array([1, 0, -1], dtype=np.int8)
    carrier, mask = to_carrier(legacy)
    assert carrier.support.all()
    # zeros mapped to +1 internally
    expected_sign = np.array([1, 1, -1], dtype=np.int8)
    np.testing.assert_array_equal(carrier.sign, expected_sign)
    # mask marks neutral positions
    np.testing.assert_array_equal(mask, legacy == 0)


def test_neutral_mask_shape_mismatch():
    legacy = np.array([1, 0], dtype=np.int8)
    bad_mask = np.array([[True, False]])
    with pytest.raises(ValueError):
        to_carrier(legacy, neutral_mask=bad_mask)
