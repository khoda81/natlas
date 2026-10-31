"""Model-independent integer-sequence cache and Connect RPC transport.

Language and music adapters supply the measures; the Rust core owns their
cached distributions and the hashed node ids shared with every client.
"""

from __future__ import annotations

import json
import math
import queue
import re
import struct
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

import anyio.to_thread
import numpy as np
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
    """Next-token log-probabilities as a float64 array over the vocabulary."""

    name: str

    def walk(self, tokens: Sequence[int], start: int) -> Iterator[np.ndarray]:
        """Those after tokens[:k] for every k from ``start`` through len(tokens)."""
        ...


# Connect envelopes: one flags byte, a big-endian length, the message.
_COMPRESSED = 0x01
_END_STREAM = 0x02
_CONTENT_TYPE = "application/connect+proto"
_PROTO_CONTENT_TYPE = "application/proto"
# Distributions kept per measure, 8 bytes per vocabulary entry each. A walk
# leaves the deepest nodes along it, the ones in view.
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
        # Node ids are the core's; each measure's distributions stay here, on
        # the server, as arrays cut per request.
        self.tree = Tree()
        self.cache: list[OrderedDict[int, np.ndarray]] = [
            OrderedDict() for _ in measures
        ]
        # Requests run on worker threads. One forward pass at a time keeps the
        # device busy once and the cache whole; ``names`` guards the tree for
        # the moments it is read or written, so naming a node never waits on
        # inference. It is taken inside ``lock``, never the other way round.
        self.lock = threading.Lock()
        self.names = threading.Lock()

    def learn(self, path: Sequence[int]) -> None:
        """Learn the ids along ``path``, so requests can name them alone."""
        with self.names:
            self.tree.walk(0, list(path))

    def expand(
        self,
        measure: int,
        node: int,
        path: Sequence[int],
        horizon: int,
        canceled: threading.Event,
    ) -> Iterator[pb.Listing]:
        """The listing of the node ``path`` names below ``node``, preceded by
        every node between them, as the protocol's Expand: each down to
        2^-horizon and through the path's next branch.

        Raises KeyError, before yielding, when this server has not learned
        ``node``. Stops early once ``canceled`` is set.
        """
        with self.names:
            nodes = self.tree.walk(node, list(path))
            nodes = nodes[1:] if path else nodes
            full = tuple(self.tree.path(nodes[-1]))
        first = len(full) - len(nodes) + 1
        # Past a branch beyond the vocabulary, every node has weight zero.
        valid = next(
            (i for i, branch in enumerate(full) if branch >= self.vocabulary), len(full)
        )
        started = time.perf_counter()
        computed = 0
        with self.lock:
            walk, walked = None, None
            for index, node_id in enumerate(nodes):
                if canceled.is_set():
                    return
                end = first + index
                prefix = full[:end]
                through = full[end] if end < len(full) else None
                if end > valid or self.ends(prefix):
                    for _ in nodes[index:]:
                        yield pb.Listing(ln_leftover=-math.inf)
                    return
                if horizon == 0 and through is None:
                    yield pb.Listing(ln_leftover=0.0)
                    continue
                cache = self.cache[measure]
                ln_weights = cache.get(node_id)
                if ln_weights is None:
                    if walk is None or walked != end:
                        walk = self.measures[measure].walk(
                            self.root + full, len(self.root) + end
                        )
                        walked = end
                    ln_weights = next(walk)
                    walked += 1
                    computed += 1
                    cache[node_id] = ln_weights
                    if len(cache) > _CACHED:
                        cache.popitem(last=False)
                else:
                    cache.move_to_end(node_id)
                yield cut(ln_weights, horizon, through)
        elapsed = (time.perf_counter() - started) * 1000
        print(
            f"expand m{measure} depth {len(full)} walking {len(nodes)} horizon {horizon}: {computed} computed, {elapsed:.0f} ms",
            file=sys.stderr,
        )


def cut(ln_weights: np.ndarray, horizon: int, through: int | None) -> pb.Listing:
    """The protocol's listing of one distribution: every branch at least
    2^-horizon heavy and at least as heavy as ``through``, heaviest first and
    ties by branch, with the rest summed directly rather than as 1 minus the
    listed, so a tiny rest survives."""
    threshold = -horizon * math.log(2) if horizon else math.inf
    if (
        through is not None
        and through < len(ln_weights)
        and ln_weights[through] > -math.inf
    ):
        threshold = min(threshold, ln_weights[through])
    keep = ln_weights >= threshold
    listed = np.flatnonzero(keep)
    branches = listed[np.argsort(-ln_weights[listed], kind="stable")]
    rest = ln_weights[~keep]
    ln_leftover = float(np.logaddexp.reduce(rest)) if rest.size else -math.inf
    return pb.Listing(
        branches=branches.tolist(),
        ln_weights=np.minimum(ln_weights[branches], 0.0).tolist(),
        ln_leftover=min(ln_leftover, 0.0),
    )


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
            except DecodeError as error:
                return _stream(
                    [_end({"code": "invalid_argument", "message": str(error)})]
                )
            # The walk runs on its own thread, holding the model for its whole
            # length, and hands each listing over as it is cut. A client that
            # hangs up stops it at the next node.
            items: queue.Queue[bytes | None] = queue.Queue()
            canceled = threading.Event()

            def work() -> None:
                try:
                    for listing in model.expand(
                        measure,
                        request_message.node,
                        request_message.path,
                        request_message.horizon,
                        canceled,
                    ):
                        items.put(_envelope(0, listing.SerializeToString()))
                    items.put(_end())
                except KeyError as error:
                    items.put(_end({"code": "not_found", "message": error.args[0]}))
                except TooLong as error:
                    items.put(_end({"code": "out_of_range", "message": str(error)}))
                except Exception as error:
                    # Answered as Connect, inside CORS, so the page sees why.
                    traceback.print_exc()
                    items.put(
                        _end(
                            {
                                "code": "internal",
                                "message": f"{type(error).__name__}: {error}",
                            }
                        )
                    )
                finally:
                    items.put(None)

            threading.Thread(target=work, daemon=True).start()

            async def chunks():
                try:
                    while (
                        item := await anyio.to_thread.run_sync(items.get)
                    ) is not None:
                        yield item
                finally:
                    canceled.set()

            return StreamingResponse(chunks(), media_type=_CONTENT_TYPE)

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
