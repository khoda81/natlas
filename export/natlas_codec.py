"""The natlas listing codec, mirroring src/codec.ts."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from natlas.v1 import natlas_pb2 as pb  # noqa: E402

SATURATED = (1 << 31) - 1
_LOWEST = -(1 << 31)


def encode(labels: Sequence[int], ln_weights: Sequence[float], horizon: int) -> list[pb.Edge]:
    """Edges for children sorted by decreasing weight, given as natural logs, down to 2^-horizon.

    Each fraction is taken against the previous weight as the client decodes
    it, so rounding never accumulates.
    """
    unit = step(horizon)
    floor = -horizon * math.log(2)
    edges = []
    previous = 0.0
    for label, ln_weight in zip(labels, ln_weights, strict=True):
        if not ln_weight >= floor:
            break
        ln_g = min(0.0, ln_weight - previous)
        odds = SATURATED
        if ln_g < 0 and unit > 0:
            odds = max(_LOWEST, min(SATURATED, round((ln_g - math.log(-math.expm1(ln_g))) / unit)))
        edges.append(pb.Edge(label=label, odds=odds))
        previous += ln_sigmoid(math.inf if odds == SATURATED else odds * unit)
    return edges


def step(horizon: int) -> float:
    """Nats per odds step at ``horizon`` bits."""
    return horizon * math.log(2) / 2**31


def ln_sigmoid(value: float) -> float:
    if value == math.inf:
        return 0.0
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))
