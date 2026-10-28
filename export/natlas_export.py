"""Write natwalk discovery trees as natlas snapshots."""

from __future__ import annotations

import itertools
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

# Log weights travel in units of 1/LOG_SCALE nat.
LOG_SCALE = 256
ROOT = 0
_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

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
    """Return the natlas stream for ``tree``; the first measure is natwalk's own.

    Nodes run in natwalk's discovery order, best first from the root, so the
    snapshot is what a server would stream for a focus on the root.
    """
    if len(measures) != 1 + len(extra):
        raise ValueError("measures must name natwalk's measure plus every extra measure")

    hello = pb.Hello(title=title, measures=list(measures), log_scale=LOG_SCALE)
    messages = [pb.ServerMessage(hello=hello)]
    handles: dict[NodeId, int] = {}
    for node_id, node in enumerate(tree.nodes):
        parent = node.parent
        handles[node_id] = ROOT if parent is None else child_handle(handles[parent], node.rank)
        distribution = node.distribution
        size = len(distribution)
        count = listed_count(
            tree, node_id, min_list=min_list, max_list=max_list, coverage=coverage
        )
        tokens = [distribution.token(rank) for rank in range(count)]
        # natwalk's own nats keep precision where p is close to 1.
        weights = [[-distribution.nats(rank) for rank in range(count)]]
        rests = [_ln(distribution.mass(count, size))]
        for measure in extra:
            values, rest = measure(node_id, tokens) if size else ((), 0.0)
            weights.append([_ln(value) for value in values])
            rests.append(_ln(rest))
        try:
            run = encode_run(
                handles[node_id],
                [label(token) for token in tokens],
                weights,
                rests if count < size else None,
            )
        except ValueError as error:
            raise ValueError(f"node {node_id}: {error}") from None
        messages.append(pb.ServerMessage(run=run))
    return pb.Snapshot(messages=messages)


def encode_run(
    handle: int,
    labels: Sequence[str],
    ln_weights: Sequence[Sequence[float]],
    ln_rest: Sequence[float] | None,
) -> pb.Run:
    """A run listing every child of a node from rank 0, given natural-log weights per measure.

    ``ln_rest`` holds the log weight of the unlisted children per measure, or
    is None when the run lists them all.
    """
    if not labels and ln_rest is not None:
        raise ValueError("a rest needs a listed child to be measured against")
    if any(len(level) != len(labels) for level in ln_weights):
        raise ValueError("every measure needs one weight per label")
    for level in ln_weights:
        if any(not math.isfinite(value) for value in level):
            raise ValueError("listed children need positive finite weights")
    if ln_rest is not None and not all(math.isfinite(value) for value in ln_rest):
        raise ValueError("an unlisted remainder needs positive finite weight in every measure")

    levels = [[_units(level[0] - value) for value in level] for level in ln_weights]
    gaps = [after - before for before, after in itertools.pairwise(levels[0])]
    if any(gap < 0 for gap in gaps):
        raise ValueError("children are not sorted by measure 0")
    rest = None
    if ln_rest is not None:
        rest = pb.Logs(values=[_units(level[0] - value) for level, value in zip(ln_weights, ln_rest)])
    return pb.Run(
        node=handle,
        start=0,
        labels=list(labels),
        gaps=gaps,
        others=[pb.Logs(values=level[1:]) for level in levels[1:]],
        rest=rest,
    )


def child_handle(parent: int, rank: int) -> int:
    """Handle of the child at ``rank`` of the node with handle ``parent``."""
    z = (parent + _GOLDEN * (rank + 1)) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _units(nats: float) -> int:
    return round(nats * LOG_SCALE)


def _ln(weight: float) -> float:
    return math.log(weight) if weight > 0 else -math.inf


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

    nodes = sum(1 for message in data.messages if message.HasField("run"))
    print(f"wrote {path} ({nodes} nodes, {path.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)
    return path


def fsum_except(values: Sequence[float], tokens: Sequence[int]) -> float:
    """Exact-ish aggregate weight of every index not in ``tokens``."""
    excluded = set(tokens)
    return math.fsum(value for index, value in enumerate(values) if index not in excluded)
