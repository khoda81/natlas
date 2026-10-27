"""Write natwalk discovery trees as natlas snapshots."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from natlas.v1 import natlas_pb2 as pb  # noqa: E402
from natwalk import Tree  # noqa: E402
from natwalk.tree import NodeId  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# Weights of one further measure for the given tokens at one node, plus the
# aggregate weight of every other token there.
type Measure = Callable[[NodeId, Sequence[int]], tuple[Sequence[float], float]]


def listed_count(
    tree: Tree,
    node_id: NodeId,
    *,
    min_list: int,
    max_list: int,
    coverage: float,
) -> int:
    """List enough ranks for discovered children, then until ``coverage`` mass."""
    node = tree[node_id]
    distribution = node.distribution
    size = len(distribution)
    count = max((rank + 1 for rank in node.children), default=0)
    mass = distribution.mass(0, count)
    while count < min(size, max_list) and (count < min_list or mass < coverage):
        mass += distribution.probability(count)
        count += 1
    return count


def snapshot(
    tree: Tree,
    *,
    title: str,
    measures: Sequence[str],
    label: Callable[[int], str],
    extra: Sequence[Measure] = (),
    min_list: int = 16,
    max_list: int = 128,
    coverage: float = 0.999,
) -> pb.Snapshot:
    """Return the natlas stream for ``tree``; the first measure is natwalk's own."""
    if len(measures) != 1 + len(extra):
        raise ValueError("measures must name natwalk's measure plus every extra measure")

    messages = [pb.ServerMessage(hello=pb.Hello(title=title, measures=list(measures)))]
    for node_id, node in enumerate(tree.nodes):
        distribution = node.distribution
        count = listed_count(
            tree, node_id, min_list=min_list, max_list=max_list, coverage=coverage
        )
        tokens = [distribution.token(rank) for rank in range(count)]
        weights = [pb.Weights(values=[distribution.probability(rank) for rank in range(count)])]
        tail = [distribution.mass(count, len(distribution))]
        for measure in extra:
            values, rest = measure(node_id, tokens) if len(distribution) else ((), 0.0)
            weights.append(pb.Weights(values=[float(value) for value in values]))
            tail.append(float(rest))

        page = pb.Page(
            start=0,
            labels=[label(token) for token in tokens],
            weights=weights,
            tail=tail,
        )
        messages.append(
            pb.ServerMessage(
                node=pb.Node(
                    id=node_id,
                    parent=node.parent or 0,
                    slot=max(node.rank, 0),
                    size=len(distribution),
                    page=page,
                )
            )
        )
    return pb.Snapshot(messages=messages)


def write(name: str, title: str, data: pb.Snapshot) -> Path:
    """Write ``data`` under public/data and register it in the dataset manifest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.pb"
    path.write_bytes(data.SerializeToString())

    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [entry for entry in manifest if entry["file"] != path.name]
    manifest.append({"file": path.name, "title": title})
    manifest.sort(key=lambda entry: entry["file"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    nodes = sum(1 for message in data.messages if message.HasField("node"))
    print(f"wrote {path} ({nodes} nodes, {path.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)
    return path


def fsum_except(values: Sequence[float], tokens: Sequence[int]) -> float:
    """Exact-ish aggregate weight of every index not in ``tokens``."""
    excluded = set(tokens)
    return math.fsum(value for index, value in enumerate(values) if index not in excluded)
