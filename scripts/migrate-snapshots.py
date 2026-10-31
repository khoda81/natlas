# /// script
# requires-python = ">=3.12"
# dependencies = ["protobuf>=7.36.1"]
# ///
"""Convert listing recordings to run recordings, keeping .snapshot-backups/listing/.

With no arguments, migrate public/data/*.pb. Explicit paths are also accepted.
Recordings already in the current format are left alone. No model or
inference is needed: each recorded listing becomes the run from rank 0 it
lists, and what it left over stays unknown.
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
from natlas_codec import run  # noqa: E402


def listing_schema():
    """The listing recording schema, isolated from the current bindings."""
    file = descriptor_pb2.FileDescriptorProto(name="natlas-listing.proto", package="listing", syntax="proto3")
    # name, number, protobuf type, repeated, message type
    messages = {
        "Listing": [("branches", 1, 13, True, ""), ("ln_weights", 2, 1, True, ""), ("ln_leftover", 3, 1, False, "")],
        "Expansion": [("path", 1, 13, True, ""), ("horizon", 2, 13, False, ""), ("listing", 3, 11, False, "Listing")],
        "Snapshot": [("expansions", 1, 11, True, "Expansion")],
    }
    for name, fields in messages.items():
        message = file.message_type.add(name=name)
        for field_name, number, kind, repeated, type_name in fields:
            field = message.field.add(name=field_name, number=number, type=kind)
            field.label = 3 if repeated else 1
            if type_name:
                field.type_name = f".listing.{type_name}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("listing.Snapshot"))


def convert(data: bytes, Snapshot) -> bytes | None:
    """The recording in the current format, or None if it already is."""
    try:
        old = Snapshot.FromString(data)
    except DecodeError:
        return None
    if not old.expansions or not all(expansion.HasField("listing") for expansion in old.expansions):
        return None
    snapshot = pb.Snapshot()
    for expansion in old.expansions:
        listing = expansion.listing
        snapshot.recordings.add(
            path=expansion.path, run=run(listing.branches, listing.ln_weights, listing.ln_leftover)
        )
    return snapshot.SerializeToString()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "public" / "data").glob("*.pb"))
    Snapshot = listing_schema()
    for path in paths:
        converted = convert(path.read_bytes(), Snapshot)
        if converted is None:
            print(f"{path}: already current, left alone")
            continue
        # Beside, not over, backups of older formats under the same names.
        backups = path.parent / ".snapshot-backups" / "listing"
        backups.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backups / path.name)
        path.write_bytes(converted)
        print(f"{path}: converted, original in {backups / path.name}")


if __name__ == "__main__":
    main()
