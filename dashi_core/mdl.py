from __future__ import annotations

from typing import Literal

from .backend import get_backend
from .carrier import Carrier

Comparison = Literal["A", "B", "tie"]


def score(rep: Carrier) -> float:
    backend = get_backend()
    backend.require_capability("supports_float64")
    support_mass = backend.sum(rep.support.astype("int64"))
    sign_complexity = backend.sum((rep.sign != 0).astype("int64"))
    return float(support_mass + sign_complexity * 0.01)


def compare(rep_a: Carrier, rep_b: Carrier) -> Comparison:
    score_a = score(rep_a)
    score_b = score(rep_b)
    if score_a < score_b:
        return "A"
    if score_b < score_a:
        return "B"
    return "tie"
