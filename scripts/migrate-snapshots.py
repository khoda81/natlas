# /// script
# requires-python = ">=3.12"
# dependencies = ["protobuf>=7.36.1"]
# ///
"""Convert recordings whose runs carry ln-below edges to runs with exact
tops and weights, keeping .snapshot-backups/edges/.

With no arguments, migrate public/data/*.pb. Explicit paths are also accepted.
Recordings already in the current format are left alone. No model or
inference is needed: each recorded run's edges give its children's weights
and the mass below them, which the codec records anew.
"""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import sys
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
from natlas import natlas_pb2 as pb  # noqa: E402
from natlas_codec import children  # noqa: E402


def edge_schema():
    """The edge recording schema, isolated from the current bindings."""
    file = descriptor_pb2.FileDescriptorProto(name="natlas-edges.proto", package="edges", syntax="proto3")
    # name, number, protobuf type, repeated, message type, proto3 optional
    messages = {
        "Run": [
            ("rank", 1, 13, False, "", False),
            ("branches", 2, 13, True, "", False),
            ("ln_below", 3, 1, True, "", False),
        ],
        "Children": [("runs", 1, 11, True, "Run", False), ("count", 2, 13, False, "", True)],
        "Recording": [("path", 1, 13, True, "", False), ("children", 3, 11, False, "Children", False)],
        "Snapshot": [("recordings", 2, 11, True, "Recording", False)],
    }
    for name, fields in messages.items():
        message = file.message_type.add(name=name)
        for field_name, number, kind, repeated, type_name, optional in fields:
            field = message.field.add(name=field_name, number=number, type=kind)
            field.label = 3 if repeated else 1
            if type_name:
                field.type_name = f".edges.{type_name}"
            if optional:
                field.proto3_optional = True
                field.oneof_index = len(message.oneof_decl)
                message.oneof_decl.add(name=f"_{field_name}")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("edges.Snapshot"))


def convert(data: bytes, Snapshot) -> bytes | None:
    """The recording in the current format, or None if it already is: edge
    runs carry one edge more than branches, weighted runs as many weights."""
    try:
        old = Snapshot.FromString(data)
    except DecodeError:
        return None
    runs = [run for recording in old.recordings for run in recording.children.runs]
    if not runs or not all(len(run.ln_below) == len(run.branches) + 1 for run in runs):
        return None
    snapshot = pb.Snapshot()
    for recording in old.recordings:
        (run,) = recording.children.runs or [None]
        if run is None:
            snapshot.recordings.add(path=recording.path, children=pb.Children(count=recording.children.count))
            continue
        if run.rank != 0:
            raise ValueError("a recording's run starts at its node's first child")
        edges = list(run.ln_below)
        ln_weights = [_ln_between(top, bottom) for top, bottom in itertools.pairwise(edges)]
        count = recording.children.count if recording.children.HasField("count") else None
        converted = children(list(run.branches), ln_weights, edges[-1], count)
        snapshot.recordings.add(path=recording.path, children=converted)
    return snapshot.SerializeToString()


def _ln_between(top: float, bottom: float) -> float:
    """ln(e^top − e^bottom), exact however close they are."""
    return top if bottom == -math.inf else top + math.log(-math.expm1(bottom - top))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "public" / "data").glob("*.pb"))
    Snapshot = edge_schema()
    for path in paths:
        converted = convert(path.read_bytes(), Snapshot)
        if converted is None:
            print(f"{path}: already current, left alone")
            continue
        # Beside, not over, backups of older formats under the same names.
        backups = path.parent / ".snapshot-backups" / "edges"
        backups.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backups / path.name)
        path.write_bytes(converted)
        print(f"{path}: converted, original in {backups / path.name}")


if __name__ == "__main__":
    main()
