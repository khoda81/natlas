# /// script
# requires-python = ">=3.12"
# dependencies = ["protobuf>=7.36.1"]
# ///
"""Convert handle-based .pb recordings to paths, keeping .snapshot-backups/.

With no arguments, migrate public/data/*.pb. Explicit paths are also accepted.
Already converted recordings are left alone. No model or inference is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
from natlas.v1 import natlas_pb2 as pb  # noqa: E402


def legacy_schema():
    """The previous recording schema, isolated from the current RPC bindings."""
    file = descriptor_pb2.FileDescriptorProto(
        name="natlas-legacy.proto", package="legacy", syntax="proto3"
    )
    # name, number, protobuf type, repeated, message type
    messages = {
        "Child": [("parent", 1, 6, False, ""), ("label", 2, 13, False, "")],
        "Request": [("node", 1, 11, False, "Child"), ("horizon", 2, 13, False, "")],
        "Edge": [("label", 1, 13, False, ""), ("odds", 2, 15, False, "")],
        "Expansion": [
            ("request", 1, 11, False, "Request"),
            ("node", 2, 6, False, ""),
            ("edges", 3, 11, True, "Edge"),
        ],
        "Snapshot": [("expansions", 1, 11, True, "Expansion")],
    }
    for name, fields in messages.items():
        message = file.message_type.add(name=name)
        for field, number, kind, repeated, target in fields:
            descriptor = message.field.add(
                name=field, number=number, type=kind, label=3 if repeated else 1
            )
            if target:
                descriptor.type_name = f".legacy.{target}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("legacy.Snapshot")
    )


def convert(data: bytes) -> bytes | None:
    current = pb.Snapshot.FromString(data)
    if current.format_version == 2:
        return None
    if current.format_version:
        raise ValueError(f"unknown snapshot version {current.format_version}")
    old = legacy_schema().FromString(data)
    by_handle = {expansion.node: expansion for expansion in old.expansions}
    paths: dict[int, list[int]] = {}
    remaining = list(old.expansions)
    while remaining:
        unresolved = []
        for expansion in remaining:
            if not expansion.request.HasField("node"):
                paths[expansion.node] = []
            elif expansion.request.node.parent in paths:
                child = expansion.request.node
                paths[expansion.node] = paths[child.parent] + [child.label]
            else:
                unresolved.append(expansion)
        if len(unresolved) == len(remaining):
            missing = [
                item.request.node.parent
                for item in unresolved
                if item.request.node.parent not in by_handle
            ]
            raise ValueError(
                f"recording has missing or cyclic parent handles: {missing[:4]}"
            )
        remaining = unresolved
    converted = pb.Snapshot(format_version=2)
    for expansion in old.expansions:
        out = converted.expansions.add()
        out.request.path.extend(paths[expansion.node])
        out.request.horizon = expansion.request.horizon
        out.edges.extend(
            pb.Edge(label=edge.label, odds=edge.odds) for edge in expansion.edges
        )
    return converted.SerializeToString()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    args = parser.parse_args()
    for path in args.paths or sorted((ROOT / "public/data").glob("*.pb")):
        original = path.read_bytes()
        converted = convert(original)
        if converted is None:
            print(f"already current: {path}")
            continue
        backup = ROOT / ".snapshot-backups" / (path.name + ".v1")
        backup.parent.mkdir(exist_ok=True)
        if backup.exists() and backup.read_bytes() != original:
            raise ValueError(f"a different recording already has backup {backup}")
        if not backup.exists():
            backup.write_bytes(original)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(converted)
        temporary.replace(path)
        print(f"converted: {path} (backup: {backup})")
    manifest = ROOT / "public/data/manifest.json"
    if not args.paths and manifest.exists():
        entries = json.loads(manifest.read_text())
        for entry in entries:
            entry.pop("start", None)
        manifest.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
