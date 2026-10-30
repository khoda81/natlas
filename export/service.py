"""Model-independent integer-sequence cache and Connect RPC transport.

Language and music adapters supply the measures; the Rust core owns their
cached distributions, path identity and wire encoding.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import anyio.to_thread
from errors import TooLong
from google.protobuf.message import DecodeError
from natlas_codec import pb
from natlas_core import Tree
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


class Measure(Protocol):
    name: str

    def log_probs(self, tokens: Sequence[int]) -> Any: ...


# Connect envelopes: one flags byte, a big-endian length, the message.
_COMPRESSED = 0x01
_END_STREAM = 0x02
_CONTENT_TYPE = "application/connect+proto"
_PROTO_CONTENT_TYPE = "application/proto"
# Edges per write; a listing can run to the whole vocabulary.
_CHUNK = 1024
# Sorted listings kept per measure, about 12 bytes per vocabulary entry each.
_CACHED = 64


class BadLabel(Exception):
    pass


class Model:
    """The integer-sequence distribution under each measure, computed on demand."""

    def __init__(
        self,
        root: tuple[int, ...],
        ends: Callable[[tuple[int, ...]], bool],
        labels: int,
        measures: list[Measure],
    ):
        # What the model reads before the root's children: a start-of-text token.
        self.root = root
        # Whether a path is finished, with no children.
        self.ends = ends
        self.labels = labels
        self.measures = measures
        self.tree = Tree(len(measures))
        # LRU policy belongs to the server; cached distributions and path identity
        # belong to the same core the browser uses.
        self.cache: list[OrderedDict[tuple[int, ...], int]] = [
            OrderedDict() for _ in measures
        ]
        # Requests run on worker threads; one at a time keeps the cache whole
        # and the device to one forward pass.
        self.lock = threading.Lock()

    def expand(self, measure: int, path: Sequence[int], horizon: int) -> list[pb.Edge]:
        """The listing at ``path`` down to 2^-horizon."""
        with self.lock:
            started = time.perf_counter()
            path = tuple(path)
            for label in path:
                if not 0 <= label < self.labels:
                    raise BadLabel(
                        f"label {label} is past the vocabulary's {self.labels}"
                    )
            if horizon == 0 or self.ends(path):
                return []
            cached = path in self.cache[measure]
            node = self.listing(measure, path)
            edges = [
                pb.Edge(label=label, odds=odds)
                for label, odds in self.tree.expand(node, measure, horizon)
            ]
            elapsed = (time.perf_counter() - started) * 1000
            source = "cached" if cached else "computed"
            print(
                f"expand m{measure} depth {len(path)} horizon {horizon}: {len(edges)} edges, {source}, {elapsed:.0f} ms",
                file=sys.stderr,
            )
            return edges

    def listing(self, measure: int, path: tuple[int, ...]) -> int:
        """Cache the complete distribution in Rust and return its path's node."""
        cache = self.cache[measure]
        cached = cache.get(path)
        if cached is not None:
            cache.move_to_end(path)
            return cached
        log_probs = self.measures[measure].log_probs(self.root + path)
        values, order = log_probs.sort(descending=True)
        node = self.tree.node(path)
        self.tree.set_distribution(
            node, measure, order.cpu().tolist(), values.cpu().tolist()
        )
        cache[path] = node
        if len(cache) > _CACHED:
            _, evicted = cache.popitem(last=False)
            self.tree.clear_listing(evicted, measure)
        return node


def _envelope(flags: int, payload: bytes) -> bytes:
    return struct.pack(">BI", flags, len(payload)) + payload


def _end(error: dict | None = None) -> bytes:
    return _envelope(
        _END_STREAM, json.dumps({} if error is None else {"error": error}).encode()
    )


def app(
    model: Model,
    manifest: list[dict],
    files: dict[str, tuple[bytes, str]],
    search: Callable[[str], Sequence[int]] | None = None,
) -> Starlette:
    """The measures' services, manifest, files, and optional text-to-label-path search."""

    def service(measure: int):
        async def expand(request: Request) -> Response:
            if (
                request.headers.get("content-type", "").split(";", 1)[0].strip()
                != _CONTENT_TYPE
            ):
                return Response(status_code=415, headers={"Accept-Post": _CONTENT_TYPE})
            body = await request.body()
            if len(body) < 5:
                return _stream(
                    [
                        _end(
                            {
                                "code": "invalid_argument",
                                "message": "no request message",
                            }
                        )
                    ]
                )
            flags, length = struct.unpack_from(">BI", body)
            if flags & _COMPRESSED:
                return _stream(
                    [_end({"code": "unimplemented", "message": "compressed requests"})]
                )
            if flags or length != len(body) - 5:
                return _stream(
                    [
                        _end(
                            {
                                "code": "invalid_argument",
                                "message": "invalid request envelope",
                            }
                        )
                    ]
                )
            try:
                request_message = pb.ExpandRequest.FromString(body[5:])
                edges = await anyio.to_thread.run_sync(
                    model.expand, measure, request_message.path, request_message.horizon
                )
            except (BadLabel, DecodeError) as error:
                return _stream(
                    [_end({"code": "invalid_argument", "message": str(error)})]
                )
            except TooLong as error:
                return _stream([_end({"code": "out_of_range", "message": str(error)})])
            except Exception as error:
                # Answered as Connect, inside CORS, so the page sees why.
                traceback.print_exc()
                return _stream(
                    [
                        _end(
                            {
                                "code": "internal",
                                "message": f"{type(error).__name__}: {error}",
                            }
                        )
                    ]
                )

            def chunks():
                for start in range(0, len(edges), _CHUNK):
                    yield b"".join(
                        _envelope(0, edge.SerializeToString())
                        for edge in edges[start : start + _CHUNK]
                    )
                yield _end()

            return _stream(chunks())

        return expand

    async def search_rpc(request: Request) -> Response:
        if search is None:
            return JSONResponse(
                {"code": "unimplemented", "message": "text search"}, status_code=501
            )
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip()
            != _PROTO_CONTENT_TYPE
        ):
            return Response(
                status_code=415, headers={"Accept-Post": _PROTO_CONTENT_TYPE}
            )
        try:
            request_message = pb.SearchRequest.FromString(await request.body())
            path = await anyio.to_thread.run_sync(search, request_message.text)
            response = pb.SearchResponse(path=path)
        except DecodeError as error:
            return JSONResponse(
                {"code": "invalid_argument", "message": str(error)}, status_code=400
            )
        except Exception as error:
            traceback.print_exc()
            return JSONResponse(
                {"code": "internal", "message": f"{type(error).__name__}: {error}"},
                status_code=500,
            )
        return Response(response.SerializeToString(), media_type=_PROTO_CONTENT_TYPE)

    async def manifest_json(_: Request) -> Response:
        return JSONResponse(manifest)

    def file(content: bytes, media_type: str):
        async def serve(_: Request) -> Response:
            return Response(content, media_type=media_type)

        return serve

    routes = [Route("/manifest.json", manifest_json)]
    routes += [Route(f"/{path}", file(*served)) for path, served in files.items()]
    routes += [
        Route(f"/{index}/natlas.v1.Natlas/Expand", service(index), methods=["POST"])
        for index in range(len(model.measures))
    ]
    routes += [
        Route(f"/{index}/natlas.v1.Natlas/Search", search_rpc, methods=["POST"])
        for index in range(len(model.measures))
    ]
    # Any page may read a local research server.
    cors = Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    return Starlette(routes=routes, middleware=[cors])


def _stream(chunks) -> StreamingResponse:
    return StreamingResponse(iter(chunks), media_type=_CONTENT_TYPE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
