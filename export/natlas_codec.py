"""Protobuf and label-file adapters around the shared Rust distribution core."""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from natlas.v1 import natlas_pb2 as pb  # noqa: E402
from natlas_core import encode as _encode  # noqa: E402


def encode(
    labels: Sequence[int], ln_weights: Sequence[float], horizon: int
) -> list[pb.Edge]:
    """Encode with the same Rust codec used by the browser and live server."""
    return [
        pb.Edge(label=label, odds=odds)
        for label, odds in _encode(list(labels), list(ln_weights), horizon)
    ]


def labels_json(labels: Sequence[bytes]) -> str:
    """A labels file: each label's bytes in base64, as proto3 JSON writes bytes."""
    return json.dumps([base64.b64encode(label).decode() for label in labels])
