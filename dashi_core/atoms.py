from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VortexAtom2D:
    """Minimal 2D vortex atom carrying location, sign, strength, and radius."""

    x: float
    y: float
    sign: int
    gamma: float
    radius: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.sign, self.gamma, self.radius], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "VortexAtom2D":
        x, y, sign, gamma, radius = arr
        return VortexAtom2D(float(x), float(y), int(np.sign(sign) or 1), float(abs(gamma)), float(radius))
