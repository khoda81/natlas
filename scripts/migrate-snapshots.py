# /// script
# requires-python = ">=3.12"
# dependencies = ["protobuf>=7.36.1"]
# ///
"""Convert odds-coded .pb recordings to exact listings, keeping .snapshot-backups/.

With no arguments, migrate public/data/*.pb. Explicit paths are also accepted.
Recordings already in the current format are left alone. No model or
inference is needed: each recorded listing is decoded once, as the browser
used to decode it.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
from natlas import natlas_pb2 as pb  # noqa: E402

SATURATED = 0x7FFF_FFFF


def odds_schema():
    """The odds-coded recording schema, isolated from the current bindings."""
    file = descriptor_pb2.FileDescriptorProto(
        name="natlas-odds.proto", package="odds", syntax="proto3"
    )
    # name, number, protobuf type, repeated, message type
    messages = {
        "Request": [("path", 1, 13, True, ""), ("horizon", 2, 13, False, "")],
        "Edge": [("label", 1, 13, False, ""), ("odds", 2, 15, False, "")],
        "Expansion": [
            ("request", 1, 11, False, "Request"),
            ("edges", 2, 11, True, "Edge"),
        ],
        "Snapshot": [
            ("expansions", 1, 11, True, "Expansion"),
            ("format_version", 2, 13, False, ""),
        ],
    }
    for name, fields in messages.items():
        message = file.message_type.add(name=name)
        for field_name, number, kind, repeated, type_name in fields:
            field = message.field.add(name=field_name, number=number, type=kind)
            field.label = 3 if repeated else 1
            if type_name:
                field.type_name = f".odds.{type_name}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("odds.Snapshot"))


def ln_sigmoid(x: float) -> float:
    if x == math.inf:
        return 0.0
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def decode(edges, horizon: int) -> pb.Listing:
    """Each edge's weight is a fraction of the previous one, as log-odds in
    steps of horizon·ln 2 / 2^31; what they leave of 1 is the rest."""
    unit = horizon * math.log(2) / 2**31
    branches: list[int] = []
    ln_weights: list[float] = []
    leftover, slack = 1.0, 0.0
    for edge in edges:
        x = math.inf if edge.odds == SATURATED else edge.odds * unit
        ln_weight = (ln_weights[-1] if ln_weights else 0.0) + ln_sigmoid(x)
        weight = math.exp(ln_weight)
        if not ln_weights:
            leftover = math.exp(ln_sigmoid(-x))
        elif weight > leftover + slack:
            break
        else:
            leftover -= weight
            slack += unit * weight
        branches.append(edge.label)
        ln_weights.append(ln_weight)
    # A leftover within rounding goes to the listed branches.
    if leftover <= slack:
        shift = -math.log1p(-leftover)
        ln_weights = [min(0.0, weight + shift) for weight in ln_weights]
        leftover = 0.0
    ln_leftover = math.log(leftover) if leftover > 0 else -math.inf
    return pb.Listing(branches=branches, ln_weights=ln_weights, ln_leftover=ln_leftover)


def convert(data: bytes, Snapshot) -> bytes | None:
    """The recording in the current format, or None if it already is."""
    try:
        old = Snapshot.FromString(data)
    except DecodeError:
        return None
    if old.format_version != 2:
        return None
    snapshot = pb.Snapshot()
    for expansion in old.expansions:
        snapshot.expansions.add(
            path=expansion.request.path,
            horizon=expansion.request.horizon,
            listing=decode(expansion.edges, expansion.request.horizon),
        )
    return snapshot.SerializeToString()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "public" / "data").glob("*.pb"))
    Snapshot = odds_schema()
    for path in paths:
        converted = convert(path.read_bytes(), Snapshot)
        if converted is None:
            print(f"{path}: already current, left alone")
            continue
        backups = path.parent / ".snapshot-backups"
        backups.mkdir(exist_ok=True)
        shutil.copy2(path, backups / path.name)
        path.write_bytes(converted)
        print(f"{path}: converted, original in {backups / path.name}")


if __name__ == "__main__":
    main()
