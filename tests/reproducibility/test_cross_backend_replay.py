import numpy as np

from dashi_core.backend import use_backend
from dashi_core.carrier import Carrier
from dashi_core.testing.mock_kernels import IdentityKernel


def run_pipeline():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    kernel = IdentityKernel()
    projected = kernel(state)
    return projected.to_signed()


def test_cross_backend_replay():
    with use_backend("cpu"):
        cpu_first = run_pipeline()
    with use_backend("accelerated"):
        accel = run_pipeline()
    with use_backend("cpu"):
        cpu_second = run_pipeline()

    np.testing.assert_array_equal(cpu_first, accel)
    np.testing.assert_array_equal(cpu_first, cpu_second)
