import numpy as np
import pytest

from dashi_core import mdl
from dashi_core.backend import (
    Backend,
    BackendCapabilities,
    get_backend,
    register_backend,
    set_backend,
    use_backend,
)
from dashi_core.carrier import Carrier
from dashi_core.testing.mock_kernels import IdentityKernel, OneStepErodeKernel


def _run_core_ops():
    state = Carrier.from_signed(np.array([1, 0, -1], dtype=np.int8))
    kernel = OneStepErodeKernel()
    projected = kernel(state)
    defect = projected.to_signed() - state.to_signed()
    return state.to_signed(), projected.to_signed(), defect


def test_backend_parity_cpu_vs_accelerated():
    with use_backend("cpu"):
        cpu_out = _run_core_ops()
    with use_backend("accelerated"):
        accel_out = _run_core_ops()
    np.testing.assert_array_equal(cpu_out[0], accel_out[0])
    np.testing.assert_array_equal(cpu_out[1], accel_out[1])
    np.testing.assert_array_equal(cpu_out[2], accel_out[2])
    assert get_backend().capabilities.supports_float64


def test_backend_determinism():
    with use_backend("accelerated"):
        first = _run_core_ops()
        second = _run_core_ops()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_backend_selection_unknown():
    with pytest.raises(ValueError):
        set_backend("unknown")


def test_backend_forbidden_ops():
    class RestrictedBackend(Backend):
        name = "restricted"

        def __init__(self):
            super().__init__(capabilities=BackendCapabilities(supports_float64=False))

    restricted = RestrictedBackend()
    register_backend("restricted", restricted)
    with use_backend("restricted"):
        state = Carrier.from_signed(np.array([1, 0], dtype=np.int8))
        with pytest.raises(ValueError):
            mdl.score(state)
    # reset to default
    set_backend("cpu")


def test_observability_hooks_do_not_change_outputs():
    from dashi_core.backend import get_backend

    with use_backend("cpu"):
        backend = get_backend()
        backend.reset_metrics()
        before = backend.metrics["ops"]
        result_a = _run_core_ops()
        mid = backend.metrics["ops"]
        result_b = _run_core_ops()
        after = backend.metrics["ops"]

    np.testing.assert_array_equal(result_a[0], result_b[0])
    assert after > before
    assert after > mid
