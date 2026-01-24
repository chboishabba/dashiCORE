from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .carrier import Carrier


def validate_kernel_output(pre: Carrier, post: Carrier) -> None:
    if pre.support.shape != post.support.shape:
        raise ValueError("Kernel may not change shape")
    if np.any(post.support & ~pre.support):
        raise ValueError("Kernel created support from nothing. This is forbidden.")
    post.validate()


class Kernel(ABC):
    """Base kernel interface."""

    is_idempotent: bool = False

    @abstractmethod
    def apply(self, state: Carrier, ctx: Any = None) -> Carrier:
        raise NotImplementedError

    def __call__(self, state: Carrier, ctx: Any = None) -> Carrier:
        out = self.apply(state, ctx=ctx)
        validate_kernel_output(state, out)
        return out
