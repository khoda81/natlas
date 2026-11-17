"""What recordings and readers share: children from weights, and each branch's bytes."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Sequence
from fractions import Fraction

from natlas import natlas_pb2 as pb

# A recording answers views without a server to refine them: each child's
# top is as precise as a view of the child this many pixel rows tall needs.
ROWS = 4096


def labels_json(labels: Sequence[bytes]) -> str:
    """A labels file: each branch's bytes in base64, as proto3 JSON writes bytes."""
    return json.dumps([base64.b64encode(label).decode() for label in labels])


def children(
    branches: Sequence[int], ln_weights: Sequence[float], ln_rest: float, count: int | None = None
) -> pb.Children:
    """A node's children in order from rank 0, ``ln_rest`` of its mass
    below them, and how many children have weight, where known.

    Each run's top is exact to 1/256 of a row of a view of its lightest
    child ``ROWS`` rows tall, and a run ends wherever the f64 sum of its
    weights from the top would carry an edge past the top's last digit."""
    total = math.fsum(math.exp(ln) for ln in ln_weights) + math.exp(ln_rest)
    ln_p = [ln - math.log(total) for ln in ln_weights]
    p = [math.exp(ln) for ln in ln_p]
    runs = []
    above = Fraction(0)
    start = 0
    while start < len(p):
        bits = _bits(ln_p[start])
        top = math.floor(above * 256 ** (bits // 8)).to_bytes(bits // 8, "big")
        end, total = start, 0.0
        while True:
            total += p[end]
            above += Fraction(p[end])
            end += 1
            # Sums of n weights are precise to n · 2^-53 of the sum.
            if end == len(p) or bits < _bits(ln_p[end]) or 2.0 ** (53 - bits) <= (end - start + 1) * (total + p[end]):
                break
        runs.append(pb.Run(rank=start, branches=branches[start:end], ln_weights=ln_p[start:end], top=top))
        start = end
    return pb.Children(runs=runs, count=count)


def _bits(ln_p: float) -> int:
    """Whole digits, as bits, that tell a point of a child of weight e^ln_p
    to 1/256 of a row of a view of it ``ROWS`` rows tall."""
    return 8 * math.ceil((math.log2(ROWS) - ln_p / math.log(2) + 8) / 8)
