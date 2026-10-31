"""What recordings and readers share: children from weights, and each branch's bytes."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Sequence

from natlas import natlas_pb2 as pb


def labels_json(labels: Sequence[bytes]) -> str:
    """A labels file: each branch's bytes in base64, as proto3 JSON writes bytes."""
    return json.dumps([base64.b64encode(label).decode() for label in labels])


def children(
    branches: Sequence[int], ln_weights: Sequence[float], ln_rest: float, count: int | None = None
) -> pb.Children:
    """A node's heaviest children from rank 0, ``ln_rest`` of its mass below
    them, and how many children have weight, where known: each edge summed
    from the light end, so tiny masses survive."""
    ln_below = [ln_rest]
    for ln_weight in reversed(ln_weights):
        ln_below.append(_ln_add(ln_below[-1], ln_weight))
    # A whole node is mass 1, whatever rounding made of its sum.
    edges = [min(edge, 0.0) for edge in reversed(ln_below)]
    return pb.Children(runs=[pb.Run(rank=0, branches=branches, ln_below=edges)], count=count)


def _ln_add(a: float, b: float) -> float:
    high = max(a, b)
    return high if high == -math.inf else high + math.log1p(math.exp(min(a, b) - high))
