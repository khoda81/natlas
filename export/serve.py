# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "starlette>=0.40",
#   "torch>=2.2",
#   "transformers>=4.51",
#   "uvicorn>=0.30",
# ]
# ///

"""Serve a language model's tree of texts to natlas, live.

The root is the empty text, the model's start-of-text token alone; every
node below it is a text, and its children are the next tokens. Every measure
is one Natlas service under its own path prefix, /0/ for the model itself and
/1/, /2/, … for the --q measures, speaking the Connect protocol's server
streaming. GET /manifest.json describes them to the client, with the prompt's
tokens as the place to start, and GET /labels.json holds the token bytes.

A node's handle hashes its path, so handles survive a restart, but the
server cannot invert them: it remembers each node it has expanded, one parent
handle and label apiece, and asks the client to name any other again.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import threading
import time
from collections import OrderedDict

import anyio.to_thread
import lm
import numpy as np
import uvicorn
from natlas_codec import encode, labels_json, pb
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from transformers import AutoTokenizer

# Connect envelopes: one flags byte, a big-endian length, the message.
_COMPRESSED = 0x01
_END_STREAM = 0x02
_CONTENT_TYPE = "application/connect+proto"
# Edges per write; a listing can run to the whole vocabulary.
_CHUNK = 1024
# Sorted listings kept per measure, about 12 bytes per vocabulary entry each.
_CACHED = 64

_ROOT = 0
_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _child_handle(parent: int, label: int) -> int:
    """SplitMix64 of the parent's handle and the label: siblings never collide."""
    z = (parent + _GOLDEN * (label + 1)) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class UnknownNode(Exception):
    pass


class BadLabel(Exception):
    pass


class Model:
    """The tree of texts under each measure, computed on demand."""

    def __init__(self, root: tuple[int, ...], eos: set[int], labels: int, measures: list[lm.Measure]):
        # What the model reads before the root's children: a start-of-text token.
        self.root = root
        self.eos = eos
        self.labels = labels
        self.measures = measures
        self.parents: dict[int, tuple[int, int]] = {}
        self.cache: list[OrderedDict[int, tuple[np.ndarray, np.ndarray]]] = [OrderedDict() for _ in measures]
        # Requests run on worker threads; one at a time keeps the maps whole
        # and the device to one forward pass.
        self.lock = threading.Lock()

    def path(self, handle: int) -> tuple[int, ...]:
        labels = []
        while handle != _ROOT:
            entry = self.parents.get(handle)
            if entry is None:
                raise UnknownNode(f"node {handle:016x} is unknown here")
            handle, label = entry
            labels.append(label)
        return tuple(reversed(labels))

    def expand(self, measure: int, child: pb.Child | None, horizon: int) -> tuple[int, list[pb.Edge]]:
        """The node's handle, and its listing down to 2^-horizon."""
        with self.lock:
            started = time.perf_counter()
            if child is None:
                handle, path = _ROOT, ()
            else:
                if child.label >= self.labels:
                    raise BadLabel(f"label {child.label} is past the vocabulary's {self.labels}")
                path = self.path(child.parent) + (child.label,)
                handle = _child_handle(child.parent, child.label)
                self.parents.setdefault(handle, (child.parent, child.label))
            if horizon == 0 or (path and path[-1] in self.eos):
                return handle, []
            cached = handle in self.cache[measure]
            labels, nats = self.listing(measure, handle, path)
            count = int(np.searchsorted(nats, horizon * math.log(2), side="right"))
            edges = encode(labels[:count].tolist(), (-nats[:count]).tolist(), horizon)
            elapsed = (time.perf_counter() - started) * 1000
            source = "cached" if cached else "computed"
            print(
                f"expand m{measure} depth {len(path)} horizon {horizon}: {len(edges)} edges, {source}, {elapsed:.0f} ms",
                file=sys.stderr,
            )
            return handle, edges

    def listing(self, measure: int, handle: int, path: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Every label at a node, heaviest first, with its surprisal in nats."""
        cache = self.cache[measure]
        cached = cache.get(handle)
        if cached is not None:
            cache.move_to_end(handle)
            return cached
        log_probs = self.measures[measure].log_probs(self.root + path)
        values, order = log_probs.sort(descending=True)
        listing = (order.cpu().numpy().astype(np.int32), (-values).cpu().numpy())
        cache[handle] = listing
        if len(cache) > _CACHED:
            cache.popitem(last=False)
        return listing


def _envelope(flags: int, payload: bytes) -> bytes:
    return struct.pack(">BI", flags, len(payload)) + payload


def _end(error: dict | None = None) -> bytes:
    return _envelope(_END_STREAM, json.dumps({} if error is None else {"error": error}).encode())


def app(model: Model, manifest: list[dict], labels: bytes) -> Starlette:
    def service(measure: int):
        async def expand(request: Request) -> Response:
            if request.headers.get("content-type") != _CONTENT_TYPE:
                return Response(status_code=415, headers={"Accept-Post": _CONTENT_TYPE})
            body = await request.body()
            if len(body) < 5:
                return _stream([_end({"code": "invalid_argument", "message": "no request message"})])
            flags, length = struct.unpack_from(">BI", body)
            if flags & _COMPRESSED:
                return _stream([_end({"code": "unimplemented", "message": "compressed requests"})])
            request_message = pb.ExpandRequest.FromString(body[5 : 5 + length])
            child = request_message.node if request_message.HasField("node") else None
            try:
                handle, edges = await anyio.to_thread.run_sync(model.expand, measure, child, request_message.horizon)
            except UnknownNode as error:
                return _stream([_end({"code": "not_found", "message": str(error)})])
            except BadLabel as error:
                return _stream([_end({"code": "invalid_argument", "message": str(error)})])

            def chunks():
                yield _envelope(0, pb.ExpandResponse(node=handle).SerializeToString())
                for start in range(0, len(edges), _CHUNK):
                    yield b"".join(
                        _envelope(0, pb.ExpandResponse(edge=edge).SerializeToString())
                        for edge in edges[start : start + _CHUNK]
                    )
                yield _end()

            return _stream(chunks())

        return expand

    async def manifest_json(_: Request) -> Response:
        return JSONResponse(manifest)

    async def labels_file(_: Request) -> Response:
        return Response(labels, media_type="application/json")

    routes = [Route("/manifest.json", manifest_json), Route("/labels.json", labels_file)]
    routes += [
        Route(f"/{index}/natlas.v1.Natlas/Expand", service(index), methods=["POST"])
        for index in range(len(model.measures))
    ]
    # Any page may read a local research server.
    cors = Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    return Starlette(routes=routes, middleware=[cors])


def _stream(chunks) -> StreamingResponse:
    return StreamingResponse(iter(chunks), media_type=_CONTENT_TYPE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The most interesting consequence of information theory is",
        help="where the client starts; the server serves every text alike",
    )
    parser.add_argument("--p", default="HuggingFaceTB/SmolLM2-360M", help="the model")
    parser.add_argument(
        "--q",
        action="append",
        default=[],
        help="further measure, repeatable: a model id sharing p's tokenizer, context:N, or temperature:T",
    )
    parser.add_argument("--tokenizer", help="default: p's own")
    parser.add_argument("--device", default=lm.default_device())
    parser.add_argument("--dtype", choices=lm.DTYPES, default="float32")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.p)
    root = lm.start_of_text(tokenizer)
    prompt = [int(token) for token in tokenizer.encode(args.prompt, add_special_tokens=False)]
    print(f"loading {args.p} on {args.device} …", file=sys.stderr)
    p_model = lm.load(args.p, args.device, args.dtype)
    measures = lm.measures(args.p, p_model, args.q, args.device, args.dtype)
    vocabulary = lm.vocabulary_size(tokenizer, p_model)
    labels = labels_json(lm.token_bytes(tokenizer, vocabulary))
    eos = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()

    manifest = [
        {
            "name": f"live-{_slug(args.p.split('/')[-1])}",
            "title": f"{args.p} · live",
            "labels": "labels.json",
            "start": prompt,
            "measures": [{"name": measure.name, "url": f"{index}/"} for index, measure in enumerate(measures)],
        }
    ]
    model = Model(root, eos, vocabulary, measures)
    uvicorn.run(app(model, manifest, labels.encode()), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
