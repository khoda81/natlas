"""Write natwalk discovery trees as natlas snapshots: one recorded server per measure."""

from __future__ import annotations

import heapq
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from natlas.v1 import natlas_pb2 as pb  # noqa: E402
from natlas_codec import encode, labels_json  # noqa: E402
from natwalk import Tree  # noqa: E402
from natwalk.tree import NodeId  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# A node without children lists nothing at any horizon.
_LEAF_HORIZON = (1 << 32) - 1

# The weight of every label at one node under one further measure, indexed by label.
type Measure = Callable[[NodeId], Sequence[float]]
# The same for a path of labels from the root.
type Ancestry = Callable[[Sequence[int]], Sequence[float]]


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


def snapshots(
    tree: Tree,
    *,
    extra: Sequence[Measure] = (),
    start: Sequence[int] = (),
    ancestry: Sequence[Ancestry] = (),
    min_list: int = 16,
    max_list: int = 128,
    coverage: float = 0.999,
) -> list[pb.Snapshot]:
    """Record every discovered node under natwalk's own measure, then under each extra one.

    A node's listing goes down to the whole bit below its ``listed_count``-th
    child: natwalk's discovered children always fit, and so does every child
    at least that heavy, as Expand requires.

    natwalk's root sits at ``start``, a path of labels from the tree's root;
    ``ancestry`` gives every measure's weights, natwalk's own first, at each
    node on that path, whose listing then reaches at least the next label.
    """
    if start and len(ancestry) != 1 + len(extra):
        raise ValueError("a start path needs the weights along it under every measure")
    # Handles are the recording's own: the nodes on the start path by depth,
    # then natwalk's nodes by id after them, natwalk's root last on the path.
    def along(depth: int) -> pb.Child | None:
        return None if depth == 0 else pb.Child(parent=depth - 1, label=start[depth - 1])

    def handle(node_id: NodeId) -> int:
        return len(start) + node_id

    def named(node_id: NodeId) -> tuple[pb.Child | None, int]:
        parent = tree[node_id].parent
        request = along(len(start)) if parent is None else pb.Child(parent=handle(parent), label=tree.token(node_id))
        return request, handle(node_id)

    def along_start(measure: int) -> list[pb.Expansion]:
        return [
            _expansion_of((along(depth), depth), ancestry[measure](start[:depth]), min_list, max_list, coverage, label)
            for depth, label in enumerate(start)
        ]

    own = pb.Snapshot(expansions=along_start(0) if start else [])
    for node_id, node in enumerate(tree.nodes):
        distribution = node.distribution
        count = listed_count(tree, node_id, min_list=min_list, max_list=max_list, coverage=coverage)
        while count and not math.isfinite(distribution.nats(count - 1)):
            count -= 1
        # natwalk's own nats keep precision where p is close to 1.
        horizon = _horizon(-distribution.nats(count - 1)) if count else _LEAF_HORIZON
        listed = list(range(count))
        while count and len(listed) < len(distribution) and distribution.nats(len(listed)) <= horizon * math.log(2):
            listed.append(len(listed))
        own.expansions.append(
            _expansion(
                named(node_id),
                horizon,
                [distribution.token(rank) for rank in listed],
                [-distribution.nats(rank) for rank in listed],
            )
        )

    recorded = [own]
    for index, measure in enumerate(extra, start=1):
        snapshot = pb.Snapshot(expansions=along_start(index) if start else [])
        for node_id in range(len(tree.nodes)):
            # A sequence natwalk has ended has no continuation under any measure.
            weights = measure(node_id) if len(tree[node_id].distribution) else []
            snapshot.expansions.append(_expansion_of(named(node_id), weights, min_list, max_list, coverage))
        recorded.append(snapshot)
    return recorded


def _expansion_of(
    named: tuple[pb.Child | None, int],
    weights: Sequence[float],
    min_list: int,
    max_list: int,
    coverage: float,
    include: int | None = None,
) -> pb.Expansion:
    """A node's listing from the weight of every label, down to the bit below
    ``coverage``, and at least down to label ``include``."""
    top = heapq.nlargest(max_list, range(len(weights)), key=weights.__getitem__)
    count = _coverage_count([weights[label] for label in top], min_list, max_list, coverage)
    if not count:
        return _expansion(named, _LEAF_HORIZON, [], [])
    horizon = _horizon(_ln(weights[top[count - 1]]))
    if include is not None:
        horizon = max(horizon, _horizon(_ln(weights[include])))
    # Ties within the last bit can reach past the first max_list.
    threshold = math.exp(-horizon * math.log(2))
    heavy = [label for label, weight in enumerate(weights) if weight >= threshold]
    listed = sorted(heavy, key=lambda label: (-weights[label], label))
    return _expansion(named, horizon, listed, [_ln(weights[label]) for label in listed])


def _expansion(
    named: tuple[pb.Child | None, int], horizon: int, labels: Sequence[int], ln_weights: Sequence[float]
) -> pb.Expansion:
    """``named`` is how a request names the node, and the handle the recording gives it."""
    child, node = named
    request = pb.ExpandRequest(node=child, horizon=horizon)
    return pb.Expansion(request=request, node=node, edges=encode(labels, ln_weights, horizon))


def _coverage_count(weights: Sequence[float], min_list: int, max_list: int, coverage: float) -> int:
    """How many of ``weights``, sorted decreasing, reach ``coverage`` mass, within the bounds."""
    count = 0
    mass = 0.0
    limit = min(sum(1 for weight in weights if weight > 0), max_list)
    while count < limit and (count < min_list or mass < coverage):
        mass += weights[count]
        count += 1
    return count


def _horizon(ln_weight: float) -> int:
    """The fewest whole bits whose cutoff admits ``ln_weight``."""
    horizon = max(0, math.ceil(-ln_weight / math.log(2)))
    while -horizon * math.log(2) > ln_weight:
        horizon += 1
    return horizon


def _ln(weight: float) -> float:
    return math.log(weight) if weight > 0 else -math.inf


def write(
    name: str,
    title: str,
    labels: Sequence[bytes],
    measures: Sequence[str],
    recorded: Sequence[pb.Snapshot],
    start: Sequence[int] = (),
) -> list[Path]:
    """Write one snapshot per measure and the label texts, and register them in
    the manifest, with ``start`` as the path the client opens at."""
    if len(measures) != len(recorded):
        raise ValueError("one measure name per snapshot")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label_file = f"{name}.labels.json"
    (DATA_DIR / label_file).write_text(labels_json(labels))
    paths = []
    entries = []
    for index, (measure, snapshot) in enumerate(zip(measures, recorded, strict=True)):
        path = DATA_DIR / f"{name}.{index}.pb"
        path.write_bytes(snapshot.SerializeToString())
        paths.append(path)
        entries.append({"name": measure, "file": path.name})
        size = path.stat().st_size / 1e6
        print(f"wrote {path} ({measure}, {len(snapshot.expansions)} nodes, {size:.2f} MB)", file=sys.stderr)

    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [entry for entry in manifest if "name" in entry and entry["name"] != name]
    entry = {"name": name, "title": title, "labels": label_file, "measures": entries}
    if start:
        entry["start"] = list(start)
    manifest.append(entry)
    manifest.sort(key=lambda entry: entry["name"])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return paths
