"""Write natwalk discovery trees as natlas snapshots: one recorded server per measure."""

from __future__ import annotations

import heapq
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from natlas import natlas_pb2 as pb  # noqa: E402
from natlas_codec import children, labels_json  # noqa: E402
from natwalk import Tree  # noqa: E402
from natwalk.tree import NodeId  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# The weight of every branch at one node under one further measure, indexed by branch.
type Measure = Callable[[NodeId], Sequence[float]]
# The same for a path of branches from the root.
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

    A node's run reaches its ``listed_count``-th child, so natwalk's
    discovered children always fit; natwalk's own nats keep precision where p
    is close to 1.

    natwalk's root sits at ``start``, a structural path of branches from the
    global root; ``ancestry`` gives every measure's weights, natwalk's own
    first, at each node on that path, whose run then reaches at least the
    next branch. Every recorded node holds its complete path from the
    global root.
    """
    if start and len(ancestry) != 1 + len(extra):
        raise ValueError("a start path needs the weights along it under every measure")

    def path(node_id: NodeId) -> tuple[int, ...]:
        return tuple(start) + tuple(tree.path(node_id))

    def along_start(measure: int) -> list[pb.Recording]:
        return [
            _recording_of(start[:depth], ancestry[measure](start[:depth]), min_list, max_list, coverage, branch)
            for depth, branch in enumerate(start)
        ]

    own = pb.Snapshot(recordings=along_start(0) if start else [])
    for node_id, node in enumerate(tree.nodes):
        distribution = node.distribution
        count = listed_count(tree, node_id, min_list=min_list, max_list=max_list, coverage=coverage)
        while count and not math.isfinite(distribution.nats(count - 1)):
            count -= 1
        # Ranks are natwalk's own, so the run needs no ties past them.
        listed = range(count)
        own.recordings.append(
            _recording(
                path(node_id),
                [distribution.token(rank) for rank in listed],
                [-distribution.nats(rank) for rank in listed],
                distribution.mass(len(listed), len(distribution)),
                _weighted(distribution),
            )
        )

    recorded = [own]
    for index, measure in enumerate(extra, start=1):
        snapshot = pb.Snapshot(recordings=along_start(index) if start else [])
        for node_id in range(len(tree.nodes)):
            # A sequence natwalk has ended has no continuation under any measure.
            weights = measure(node_id) if len(tree[node_id].distribution) else []
            snapshot.recordings.append(_recording_of(path(node_id), weights, min_list, max_list, coverage))
        recorded.append(snapshot)
    return recorded


def _recording_of(
    path: Sequence[int],
    weights: Sequence[float],
    min_list: int,
    max_list: int,
    coverage: float,
    include: int | None = None,
) -> pb.Recording:
    """A node's run from the weight of every branch, down to the bit below
    ``coverage``, and at least down to branch ``include``."""
    top = heapq.nlargest(max_list, range(len(weights)), key=weights.__getitem__)
    count = _coverage_count([weights[branch] for branch in top], min_list, max_list, coverage)
    if not count:
        return _recording(path, [], [], 0.0, 0)
    horizon = _horizon(_ln(weights[top[count - 1]]))
    if include is not None:
        horizon = max(horizon, _horizon(_ln(weights[include])))
    # Ties within the last bit can reach past the first max_list.
    threshold = math.exp(-horizon * math.log(2))
    heavy = [branch for branch, weight in enumerate(weights) if weight >= threshold]
    listed = sorted(heavy, key=lambda branch: (-weights[branch], branch))
    rest = math.fsum(weight for weight in weights if weight < threshold)
    count = sum(1 for weight in weights if 0 < weight)
    return _recording(path, listed, [_ln(weights[branch]) for branch in listed], rest, count)


def _recording(
    path: Sequence[int],
    branches: Sequence[int],
    ln_weights: Sequence[float],
    rest: float,
    count: int,
) -> pb.Recording:
    """A recorded node: its heaviest children, what they leave unknown, and
    how many children have weight."""
    return pb.Recording(path=path, children=children(branches, ln_weights, _ln(rest), count))


def _weighted(distribution) -> int:
    """How many of natwalk's ranked tokens have weight: they rank heaviest
    first, so the weightless ones are a suffix."""
    low, high = 0, len(distribution)
    while low < high:
        middle = (low + high) // 2
        if math.isfinite(distribution.nats(middle)):
            low = middle + 1
        else:
            high = middle
    return low


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
) -> list[Path]:
    """Write one snapshot per measure and the label texts, and register them in the manifest."""
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
        print(f"wrote {path} ({measure}, {len(snapshot.recordings)} nodes, {size:.2f} MB)", file=sys.stderr)

    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for entry in manifest:
        entry.pop("start", None)
    manifest = [entry for entry in manifest if "name" in entry and entry["name"] != name]
    entry = {"name": name, "title": title, "labels": label_file, "measures": entries}
    manifest.append(entry)
    manifest.sort(key=lambda entry: entry["name"])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return paths
