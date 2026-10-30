"""Model-independent integer-sequence cache and Connect RPC transport.

Language and music adapters supply the measures; the Rust core owns their
cached distributions and the hashed node ids shared with every client.
"""

from __future__ import annotations

import json
import math
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
from natlas import natlas_pb2 as pb
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
# Branches per message; a listing can run to the whole vocabulary.
_CHUNK = 4096
# Sorted listings kept per measure, about 12 bytes per vocabulary entry each.
_CACHED = 64


class Model:
    """The integer-sequence distribution under each measure, computed on demand."""

    def __init__(
        self,
        root: tuple[int, ...],
        ends: Callable[[tuple[int, ...]], bool],
        vocabulary: int,
        measures: list[Measure],
    ):
        # What the model reads before the root's children: a start-of-text token.
        self.root = root
        # Whether a path is finished, with no children.
        self.ends = ends
        # Branches the model has; every other branch has weight zero.
        self.vocabulary = vocabulary
        self.measures = measures
        self.tree = Tree(len(measures))
        # LRU policy belongs to the server; cached distributions and node ids
        # belong to the same core the browser uses.
        self.cache: list[OrderedDict[int, None]] = [OrderedDict() for _ in measures]
        # Requests run on worker threads. One forward pass at a time keeps the
        # device busy once and the cache whole; ``names`` guards the tree for
        # the moments it is read or written, so naming a node never waits on
        # inference. It is taken inside ``lock``, never the other way round.
        self.lock = threading.Lock()
        self.names = threading.Lock()

    def learn(self, path: Sequence[int]) -> None:
        """Learn the ids along ``path``, so requests can name them alone."""
        with self.names:
            self.tree.node(0, list(path))

    def expand(
        self, measure: int, node: int, path: Sequence[int], horizon: int
    ) -> list[pb.Listing]:
        """The listing of the node ``path`` below ``node`` down to 2^-horizon.

        Raises KeyError when this server has not learned ``node``.
        """
        started = time.perf_counter()
        with self.names:
            node = self.tree.node(node, list(path))
            path = tuple(self.tree.path(node))
        # A branch past the vocabulary has weight zero: nothing follows it.
        if self.ends(path) or any(branch >= self.vocabulary for branch in path):
            return [pb.Listing(ln_leftover=-math.inf)]
        if horizon == 0:
            return [pb.Listing(ln_leftover=0.0)]
        with self.lock:
            cached = node in self.cache[measure]
            self.listing(measure, node, path)
            with self.names:
                batches = self.tree.batches(node, measure, horizon, _CHUNK)
            listings = [
                pb.Listing(branches=branches, ln_weights=ln_weights, ln_leftover=rest)
                for branches, ln_weights, rest in batches
            ]
            elapsed = (time.perf_counter() - started) * 1000
            source = "cached" if cached else "computed"
            count = sum(len(listing.branches) for listing in listings)
            print(
                f"expand m{measure} depth {len(path)} horizon {horizon}: {count} branches, {source}, {elapsed:.0f} ms",
                file=sys.stderr,
            )
            return listings

    def listing(self, measure: int, node: int, path: tuple[int, ...]) -> None:
        """Cache ``node``'s complete distribution in Rust, heaviest first."""
        cache = self.cache[measure]
        if node in cache:
            cache.move_to_end(node)
            return
        log_probs = self.measures[measure].log_probs(self.root + path)
        # Stable, so equal weights keep increasing branch order.
        values, order = log_probs.sort(descending=True, stable=True)
        # Normalizing in float32 can leave a certain branch a hair above ln 1.
        ln_weights = [min(value, 0.0) for value in values.cpu().tolist()]
        cache[node] = None
        evicted = cache.popitem(last=False)[0] if len(cache) > _CACHED else None
        with self.names:
            self.tree.set_distribution(node, measure, order.cpu().tolist(), ln_weights)
            if evicted is not None:
                self.tree.clear_listing(evicted, measure)


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
    """The measures' services, manifest, files, and optional text-to-path search."""

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
                listings = await anyio.to_thread.run_sync(
                    model.expand,
                    measure,
                    request_message.node,
                    request_message.path,
                    request_message.horizon,
                )
            except KeyError as error:
                return _stream([_end({"code": "not_found", "message": error.args[0]})])
            except DecodeError as error:
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
                for listing in listings:
                    yield _envelope(0, listing.SerializeToString())
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
            await anyio.to_thread.run_sync(model.learn, path)
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
        Route(f"/{index}/natlas.Natlas/Expand", service(index), methods=["POST"])
        for index in range(len(model.measures))
    ]
    routes += [
        Route(f"/{index}/natlas.Natlas/Search", search_rpc, methods=["POST"])
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
