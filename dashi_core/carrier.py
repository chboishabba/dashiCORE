from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import get_backend

TERNARY_VALUES = (-1, 0, 1)


@dataclass(frozen=True)
class Carrier:
    """Canonical balanced ternary carrier."""

    support: np.ndarray
    sign: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "support", np.asarray(self.support, dtype=bool))
        object.__setattr__(self, "sign", np.asarray(self.sign, dtype=np.int8))
        self.validate()

    @classmethod
    def from_signed(cls, signed: Any) -> "Carrier":
        backend = get_backend()
        raw = backend.array(signed)
        cls._validate_ternary(raw)
        signed_arr = backend.to_int8(raw)
        support = backend.to_bool(signed_arr != 0)
        sign = backend.where(support, backend.sign(signed_arr), backend.zeros_like(signed_arr, dtype=np.int8))
        return cls(support=support, sign=sign)

    def to_signed(self) -> np.ndarray:
        backend = get_backend()
        return backend.where(self.support, self.sign, backend.zeros_like(self.sign, dtype=np.int8))

    def validate(self) -> None:
        if self.support.shape != self.sign.shape:
            raise ValueError("support and sign must have identical shape")
        if self.support.dtype != bool:
            raise ValueError("support must be boolean")
        if not np.isin(self.sign, TERNARY_VALUES).all():
            raise ValueError("sign values must be in {-1,0,1}")
        if not np.array_equal(self.sign[~self.support], np.zeros_like(self.sign[~self.support])):
            raise ValueError("sign must be zero wherever support is False")
        if not np.isin(self.sign[self.support], (-1, 1)).all():
            raise ValueError("sign must be ±1 wherever support is True")

    @staticmethod
    def _validate_ternary(arr: np.ndarray) -> None:
        if not np.isin(arr, TERNARY_VALUES).all():
            raise ValueError("Carrier values must be ternary.")
