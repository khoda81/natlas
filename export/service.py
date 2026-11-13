"""Model-independent integer-sequence cache and Connect RPC transport.

Language and music adapters supply the measures; the Rust core owns their
cached distributions and the hashed node ids shared with every client.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import struct
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
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
# Memory for distributions a server keeps by default, across its measures.
# The core keeps each sorted, 16 bytes per vocabulary entry: some 440 nodes of
# a 150k vocabulary, 1360 of a 50k one.
CACHE = 1 << 30


class Turns:
    """A lock taken in the order it was asked for, which says whether anyone
    waits: a holder can hand it on and take it again behind them."""

    def __init__(self):
        self._changed = threading.Condition()
        self._waiting: deque[object] = deque()
        self._held = False

    def acquire(self, canceled: threading.Event | None = None) -> bool:
        """Wait for a turn and take it; False, without one, once ``canceled``
        is set and ``wake`` called."""
        def gone() -> bool:
            return canceled is not None and canceled.is_set()

        with self._changed:
            ticket = object()
            self._waiting.append(ticket)
            self._changed.wait_for(lambda: gone() or (not self._held and self._waiting[0] is ticket))
            self._waiting.remove(ticket)
            if gone():
                # Whoever is next may be first now.
                self._changed.notify_all()
                return False
            self._held = True
            return True

    def wake(self) -> None:
        """Have waiters look whether they were canceled."""
        with self._changed:
            self._changed.notify_all()

    def release(self) -> None:
        with self._changed:
            assert self._held, "released a lock nobody holds"
            self._held = False
            self._changed.notify_all()

    def wanted(self) -> bool:
        with self._changed:
            return bool(self._waiting)

    def locked(self) -> bool:
        with self._changed:
            return self._held

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *_) -> None:
        self.release()


class Model:
    """The integer-sequence distribution under each measure, computed on demand."""

    def __init__(
        self,
        root: tuple[int, ...],
        ends: Callable[[tuple[int, ...]], bool],
        vocabulary: int,
        measures: list[Measure],
        cache: int = CACHE,
    ):
        # What the model reads before the root's children: a start-of-text token.
        self.root = root
        # Whether a path is finished, with no children.
        self.ends = ends
        # Branches the model has; every other branch has weight zero.
        self.vocabulary = vocabulary
        self.measures = measures
        # Node ids, distributions and Expand's walk are the core's; this keeps
        # only which distributions to forget first: the least recently read,
        # once those kept, by (node, measure), hold more than ``cache`` bytes.
        # A walk leaves the deepest nodes along it, the ones in view.
        self.tree = Tree(len(measures))
        self.cache = cache
        self.recent: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.held = 0
        # Requests run on worker threads. One forward pass at a time keeps the
        # device busy once and the cache whole; ``names`` guards the tree and
        # which distributions to forget for the moments they are read or
        # written, so naming a node and reading one already held never wait
        # on inference. It is taken inside ``lock``, never the other way round.
        self.lock = Turns()
        self.names = threading.Lock()

    def learn(self, path: Sequence[int]) -> None:
        """Learn the ids along ``path``, so requests can name them alone."""
        with self.names:
            self.tree.walk(0, list(path))

    def expand(
        self, measure: int, request: pb.ExpandRequest, canceled: threading.Event
    ) -> Iterator[pb.Children]:
        """The protocol's Expand: one message per node, computing each
        distribution the walk needs as it gets there.

        Raises KeyError, before yielding, when this server has not learned the
        request's node. Stops early once ``canceled`` is set.
        """
        node = request.node
        path = list(request.path)
        with self.names:
            match request.WhichOneof("within"):
                case "points":
                    parts = [(part.top, part.ln_mass, part.count) for part in request.points.parts]
                    walk = self.tree.start(node, path, parts=parts)
                case "branches":
                    walk = self.tree.start(node, path, branches=list(request.branches.branches))
                case _:
                    walk = self.tree.start(node, path)
            # A path is known ahead, so the model reads it in one pass.
            ahead = tuple(self.tree.path(node)) + tuple(path)
        started = time.perf_counter()
        computed = 0
        recent = self.recent
        # The model is taken at the walk's first forward pass, so a walk of
        # distributions already held answers at once, however long another's
        # passes run. It is handed on after a pass whenever another walk
        # waits for it: a long path costs others one pass each, not all of it.
        held = False
        try:
            # A model walk along ``along``, yielding the distribution after
            # its first ``at`` tokens next.
            reading, along, at = None, (), 0
            while not canceled.is_set():
                with self.names:
                    step = self.tree.advance(walk, measure)
                    if step is not None and step[0] == "children" and (step[1], measure) in recent:
                        recent.move_to_end((step[1], measure))
                if step is None:
                    break
                if step[0] == "children":
                    _, _, runs, count = step
                    yield pb.Children(
                        runs=[pb.Run(rank=rank, branches=branches, ln_below=ln_below) for rank, branches, ln_below in runs],
                        count=count,
                    )
                    continue
                if not held:
                    if not self.lock.acquire(canceled):
                        break
                    held = True
                    # Another walk may have computed it while this one waited,
                    # and its passes moved the model's caches elsewhere.
                    reading = None
                    continue
                missing = step[1]
                with self.names:
                    tokens = tuple(self.tree.path(missing))
                if self.ends(tokens) or any(branch >= self.vocabulary for branch in tokens):
                    ln_weights = np.empty(0)
                else:
                    if reading is None or along[: len(tokens)] != tokens or at != len(tokens):
                        along = ahead if ahead[: len(tokens)] == tokens else tokens
                        reading = self.measures[measure].walk(self.root + along, len(self.root) + len(tokens))
                        at = len(tokens)
                    ln_weights = next(reading)
                    at += 1
                    computed += 1
                with self.names:
                    size = self.tree.set(missing, measure, np.ascontiguousarray(ln_weights, dtype=np.float64))
                    recent[missing, measure] = size
                    self.held += size
                    # The one just computed stays, however large.
                    while self.cache < self.held and 1 < len(recent):
                        (forgotten, of), size = recent.popitem(last=False)
                        self.tree.forget(forgotten, of)
                        self.held -= size
                # After a pass, never before: two walks handing it over
                # before theirs would pass it back and forth forever. And
                # after storing it, so a walk waiting for the same node
                # finds it rather than computing it again.
                if self.lock.wanted():
                    self.lock.release()
                    held = False
        finally:
            if held:
                self.lock.release()
        elapsed = (time.perf_counter() - started) * 1000
        print(
            f"expand m{measure} depth {len(ahead)} {request.WhichOneof('within') or 'path'}: {computed} computed, {elapsed:.0f} ms",
            file=sys.stderr,
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
            if any(math.isnan(part.ln_mass) for part in request_message.points.parts):
                return _stream(
                    [_end({"code": "invalid_argument", "message": "a part's ln mass is not a number"})]
                )
            # The walk runs on its own thread and hands each node's children to
            # the event loop as it places them, so a stream holds no thread
            # while it waits. A client that hangs up stops the walk at its
            # next node, or before its turn at the model.
            loop = asyncio.get_running_loop()
            items: asyncio.Queue[bytes | None] = asyncio.Queue()
            canceled = threading.Event()

            def put(item: bytes | None) -> None:
                loop.call_soon_threadsafe(items.put_nowait, item)

            def work() -> None:
                try:
                    for children in model.expand(measure, request_message, canceled):
                        put(_envelope(0, children.SerializeToString()))
                    put(_end())
                except KeyError as error:
                    put(_end({"code": "not_found", "message": error.args[0]}))
                except TooLong as error:
                    put(_end({"code": "out_of_range", "message": str(error)}))
                except Exception as error:
                    # Answered as Connect, inside CORS, so the page sees why.
                    traceback.print_exc()
                    put(
                        _end(
                            {
                                "code": "internal",
                                "message": f"{type(error).__name__}: {error}",
                            }
                        )
                    )
                finally:
                    put(None)

            threading.Thread(target=work, daemon=True).start()

            async def chunks():
                try:
                    while (item := await items.get()) is not None:
                        yield item
                finally:
                    canceled.set()
                    model.lock.wake()

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
