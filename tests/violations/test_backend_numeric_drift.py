import numpy as np

from dashi_core.backend import use_backend
from dashi_core.carrier import Carrier
from dashi_core.defect import aggregate, local
from dashi_core.testing.mock_kernels import IdentityKernel


def test_backend_numeric_drift_rejected():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    kernel = IdentityKernel()

    with use_backend("cpu"):
        cpu_defect = aggregate(local(state, kernel(state)))
    with use_backend("accelerated"):
        accel_defect = aggregate(local(state, kernel(state)))

    assert cpu_defect == accel_defect
