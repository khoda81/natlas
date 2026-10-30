"""The label file a reader of text needs: each branch's bytes."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence


def labels_json(labels: Sequence[bytes]) -> str:
    """A labels file: each branch's bytes in base64, as proto3 JSON writes bytes."""
    return json.dumps([base64.b64encode(label).decode() for label in labels])
