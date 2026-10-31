# /// script
# requires-python = ">=3.12"
# dependencies = ["protobuf>=7.36.1"]
# ///
"""Convert run recordings to children recordings, keeping .snapshot-backups/runs/.

With no arguments, migrate public/data/*.pb. Explicit paths are also accepted.
Recordings already in the current format are left alone. No model or
inference is needed: each recorded run, and the count it carried, becomes
the node's recorded children.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
from natlas import natlas_pb2 as pb  # noqa: E402


def run_schema():
    """The run recording schema, isolated from the current bindings."""
    file = descriptor_pb2.FileDescriptorProto(name="natlas-runs.proto", package="runs", syntax="proto3")
    # name, number, protobuf type, repeated, message type, proto3 optional
    messages = {
        "Run": [
            ("rank", 1, 13, False, "", False),
            ("branches", 2, 13, True, "", False),
            ("ln_below", 3, 1, True, "", False),
            ("children", 4, 13, False, "", True),
        ],
        "Recording": [("path", 1, 13, True, "", False), ("run", 2, 11, False, "Run", False)],
        "Snapshot": [("recordings", 2, 11, True, "Recording", False)],
    }
    for name, fields in messages.items():
        message = file.message_type.add(name=name)
        for field_name, number, kind, repeated, type_name, optional in fields:
            field = message.field.add(name=field_name, number=number, type=kind)
            field.label = 3 if repeated else 1
            if type_name:
                field.type_name = f".runs.{type_name}"
            if optional:
                field.proto3_optional = True
                field.oneof_index = len(message.oneof_decl)
                message.oneof_decl.add(name=f"_{field_name}")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("runs.Snapshot"))


def convert(data: bytes, Snapshot) -> bytes | None:
    """The recording in the current format, or None if it already is."""
    try:
        old = Snapshot.FromString(data)
    except DecodeError:
        return None
    if not old.recordings or not all(recording.HasField("run") for recording in old.recordings):
        return None
    snapshot = pb.Snapshot()
    for recording in old.recordings:
        run = recording.run
        children = pb.Children(runs=[pb.Run(rank=run.rank, branches=run.branches, ln_below=run.ln_below)])
        if run.HasField("children"):
            children.count = run.children
        snapshot.recordings.add(path=recording.path, children=children)
    return snapshot.SerializeToString()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "public" / "data").glob("*.pb"))
    Snapshot = run_schema()
    for path in paths:
        converted = convert(path.read_bytes(), Snapshot)
        if converted is None:
            print(f"{path}: already current, left alone")
            continue
        # Beside, not over, backups of older formats under the same names.
        backups = path.parent / ".snapshot-backups" / "runs"
        backups.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backups / path.name)
        path.write_bytes(converted)
        print(f"{path}: converted, original in {backups / path.name}")


if __name__ == "__main__":
    main()
