import numpy as np

from dashi_core.carrier import Carrier
from pq import decode_pq_to_carrier, encode_carrier_to_pq, expected_pq_length


def test_pq_roundtrip_identity():
    carrier = Carrier.from_signed(np.array([1, 0, -1, 1, -1], dtype=np.int8))
    buf = encode_carrier_to_pq(carrier)
    decoded = decode_pq_to_carrier(buf)
    np.testing.assert_array_equal(decoded.to_signed(), carrier.to_signed())
    np.testing.assert_array_equal(decoded.support, carrier.support)


def test_pq_length():
    carrier = Carrier.from_signed(np.array([1, 0, -1, 0], dtype=np.int8))
    buf = encode_carrier_to_pq(carrier)
    assert len(buf.data) == expected_pq_length(carrier.sign.size)


def test_pq_support_and_sign_mapping():
    support = np.array([True, False, True], dtype=bool)
    sign = np.array([1, 0, -1], dtype=np.int8)
    carrier = Carrier(support=support, sign=sign)
    buf = encode_carrier_to_pq(carrier)
    decoded = decode_pq_to_carrier(buf)
    np.testing.assert_array_equal(decoded.support, support)
    np.testing.assert_array_equal(decoded.sign, sign)
