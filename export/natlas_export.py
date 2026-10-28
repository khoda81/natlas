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
from natwalk import Tree  # noqa: E402
from natwalk.tree import NodeId  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

ROOT = 0
SATURATED = (1 << 31) - 1
_LOWEST = -(1 << 31)
# A node without children lists nothing at any horizon.
_LEAF_HORIZON = (1 << 32) - 1
_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

# The weight of every label at one node under one further measure, indexed by label.
type Measure = Callable[[NodeId], Sequence[float]]


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
    min_list: int = 16,
    max_list: int = 128,
    coverage: float = 0.999,
) -> list[pb.Snapshot]:
    """Record every discovered node under natwalk's own measure, then under each extra one.

    A node's listing goes down to the whole bit below its ``listed_count``-th
    child: natwalk's discovered children always fit, and so does every child
    at least that heavy, as Expand requires.
    """
    handles: dict[NodeId, int] = {}
    for node_id, node in enumerate(tree.nodes):
        parent = node.parent
        handles[node_id] = ROOT if parent is None else child_handle(handles[parent], tree.token(node_id))

    own = pb.Snapshot()
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
                handles[node_id],
                horizon,
                [distribution.token(rank) for rank in listed],
                [-distribution.nats(rank) for rank in listed],
            )
        )

    recorded = [own]
    for measure in extra:
        snapshot = pb.Snapshot()
        for node_id in range(len(tree.nodes)):
            # A sequence natwalk has ended has no continuation under any measure.
            weights = measure(node_id) if len(tree[node_id].distribution) else []
            top = heapq.nlargest(max_list, range(len(weights)), key=weights.__getitem__)
            count = _coverage_count([weights[label] for label in top], min_list, max_list, coverage)
            horizon = _horizon(_ln(weights[top[count - 1]])) if count else _LEAF_HORIZON
            floor = -horizon * math.log(2)
            # Ties within the last bit can reach past the first max_list.
            threshold = math.exp(floor) if count else math.inf
            heavy = [label for label, weight in enumerate(weights) if weight >= threshold]
            listed = sorted(heavy, key=lambda label: (-weights[label], label))
            snapshot.expansions.append(
                _expansion(handles[node_id], horizon, listed, [_ln(weights[label]) for label in listed])
            )
        recorded.append(snapshot)
    return recorded


def encode(labels: Sequence[int], ln_weights: Sequence[float], horizon: int) -> list[pb.Edge]:
    """Edges for children sorted by decreasing weight, given as natural logs, down to 2^-horizon.

    Mirrors src/codec.ts: each fraction is taken against the previous weight
    as the client decodes it, so rounding never accumulates.
    """
    unit = _step(horizon)
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
        previous += _ln_sigmoid(math.inf if odds == SATURATED else odds * unit)
    return edges


def child_handle(parent: int, label: int) -> int:
    """Handle of the child with ``label`` of the node with handle ``parent``."""
    z = (parent + _GOLDEN * (label + 1)) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _expansion(handle: int, horizon: int, labels: Sequence[int], ln_weights: Sequence[float]) -> pb.Expansion:
    request = pb.ExpandRequest(node=handle, horizon=horizon)
    return pb.Expansion(request=request, edges=encode(labels, ln_weights, horizon))


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


def _step(horizon: int) -> float:
    return horizon * math.log(2) / 2**31


def _ln_sigmoid(value: float) -> float:
    if value == math.inf:
        return 0.0
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _ln(weight: float) -> float:
    return math.log(weight) if weight > 0 else -math.inf


def write(
    name: str,
    title: str,
    labels: Sequence[str],
    measures: Sequence[str],
    recorded: Sequence[pb.Snapshot],
) -> list[Path]:
    """Write one snapshot per measure and the label texts, and register them in the manifest."""
    if len(measures) != len(recorded):
        raise ValueError("one measure name per snapshot")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label_file = f"{name}.labels.json"
    (DATA_DIR / label_file).write_text(json.dumps(list(labels), ensure_ascii=False))
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
    manifest.append({"name": name, "title": title, "labels": label_file, "measures": entries})
    manifest.sort(key=lambda entry: entry["name"])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return paths
