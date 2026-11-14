"""Model-independent integer-sequence cache, scheduler and session transport.

Language and music adapters supply the measures; the Rust core owns their
cached distributions, the hashed node ids shared with every client, and what
each client's view lacks.
"""

from __future__ import annotations

import asyncio
import queue
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio.to_thread
import numpy as np
from errors import TooLong
from google.protobuf.message import DecodeError
from natlas import natlas_pb2 as pb
from natlas_core import Session, Tree
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect


class Measure(Protocol):
    """Next-token log-probabilities as a float64 array over the vocabulary."""

    name: str
    # How many positions of a path one forward pass gives distributions for.
    positions: int

    def walk(self, tokens: Sequence[int], start: int) -> Iterator[np.ndarray]:
        """Those after tokens[:k] for every k from ``start`` through len(tokens)."""
        ...


_PROTO_CONTENT_TYPE = "application/proto"
# Memory for distributions a server keeps by default, across its measures.
# The core keeps each sorted and packed, about 4 bytes per vocabulary entry:
# some 1700 nodes of a 150k vocabulary, 5000 of a 50k one.
CACHE = 1 << 30


@dataclass(eq=False)
class Client:
    """One client's session, and how to hand it an Updates frame from any
    thread."""

    send: Callable[[bytes], None]
    session: Session = field(default_factory=Session)


class Model:
    """The integer-sequence distribution under each measure, computed for the
    clients looking at it.

    One scheduler thread owns the core tree, the distributions kept and the
    model; everything else posts work to its inbox. It steps every client's
    session, sends what is new, and computes the distribution the next client
    in turn waits on most, choosing again after every pass. Clients take
    turns pass by pass; within a client, worth alone decides.
    """

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
        self.tree = Tree(len(measures))
        # Which distributions to forget first: the oldest, once those kept,
        # by (node, measure), hold more than ``cache`` bytes.
        self.cache = cache
        self.kept: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.held = 0
        self.clients: list[Client] = []
        # The client whose turn is next.
        self.turn = 0
        self.inbox: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="scheduler").start()

    def call[T](self, work: Callable[[], T]) -> Future[T]:
        """Run ``work`` on the scheduler, between passes."""
        future: Future[T] = Future()
        self.inbox.put((work, future))
        return future

    def open(self, send: Callable[[bytes], None]) -> Future[Client]:
        def opened() -> Client:
            client = Client(send)
            self.clients.append(client)
            return client

        return self.call(opened)

    def close(self, client: Client) -> Future[None]:
        return self.call(lambda: self.clients.remove(client))

    def demand(self, client: Client, demand: pb.Demand) -> Future[None]:
        """What ``client`` looks at now. The future fails with ValueError for
        a demand the tree cannot take."""
        view = None
        if demand.HasField("view"):
            v = demand.view
            view = (v.node, v.top, v.ln_height, v.rows, v.right)
        path = (demand.path.node, list(demand.path.branches)) if demand.HasField("path") else None
        return self.call(lambda: client.session.demand(self.tree, demand.p, demand.q, view=view, path=path))

    def learn(self, path: Sequence[int]) -> Future[list[int]]:
        """Learn the ids along ``path``, so demands can name them alone."""
        return self.call(lambda: self.tree.walk(0, list(path)))

    def _run(self) -> None:
        while True:
            self._take(wait=False)
            for client in self.clients:
                self._step(client)
            chosen = self._chosen()
            if chosen is None:
                self._take(wait=True)
                continue
            self._compute(*chosen)

    def _take(self, wait: bool) -> None:
        """Run what was posted; with ``wait``, wait for something first."""
        try:
            item = self.inbox.get() if wait else self.inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            work, future = item
            try:
                future.set_result(work())
            except Exception as error:
                future.set_exception(error)
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return

    def _step(self, client: Client) -> None:
        updates = client.session.step(self.tree)
        if updates:
            client.send(_encode(updates))

    def _chosen(self) -> tuple[int, int, float, list[int]] | None:
        """The next client in turn that waits on something, and the most
        valuable distribution it waits on."""
        count = len(self.clients)
        for offset in range(count):
            client = self.clients[(self.turn + offset) % count]
            blocked = client.session.blocked()
            if blocked:
                self.turn = (self.turn + offset + 1) % count
                return blocked[0]
        return None

    def _compute(self, measure: int, node: int, worth: float, ahead: list[int]) -> None:
        """Compute a distribution, and where a path goes on below it, the
        next nodes along it, as many as the measure reads in one pass."""
        started = time.perf_counter()
        tokens = tuple(self.tree.path(node))
        # Down the path as far as one pass reads and the sequence goes on.
        along = tokens
        for branch in ahead[: self.measures[measure].positions - 1]:
            if self._empty(along) or self._empty(along + (branch,)):
                break
            along += (branch,)
        try:
            if self._empty(tokens):
                computed = [np.empty(0)]
            else:
                walk = self.measures[measure].walk(self.root + along, len(self.root) + len(tokens))
                computed = [ln_weights for ln_weights, _ in zip(walk, range(len(along) - len(tokens) + 1), strict=False)]
        except Exception as error:
            if not isinstance(error, TooLong):
                traceback.print_exc()
            message = str(error) if isinstance(error, TooLong) else f"{type(error).__name__}: {error}"
            for client in self.clients:
                if any(blocked[:2] == (measure, node) for blocked in client.session.blocked()):
                    client.session.failed(measure, node, message)
            return
        nodes = self.tree.walk(node, list(along[len(tokens) :]))
        for at, ln_weights in zip(nodes, computed, strict=False):
            self._keep(measure, at, ln_weights)
        elapsed = (time.perf_counter() - started) * 1000
        print(
            f"m{measure} depth {len(tokens)} worth {worth:.1f}: {len(computed)} computed, {elapsed:.0f} ms",
            file=sys.stderr,
        )

    def _empty(self, tokens: tuple[int, ...]) -> bool:
        """Whether a sequence has no children: finished, or past the vocabulary."""
        return self.ends(tokens) or any(branch >= self.vocabulary for branch in tokens)

    def _keep(self, measure: int, node: int, ln_weights: np.ndarray) -> None:
        size = self.tree.set(node, measure, np.ascontiguousarray(ln_weights, dtype=np.float64))
        self.held += size - self.kept.pop((node, measure), 0)
        self.kept[node, measure] = size
        # The one just computed stays, however large.
        while self.cache < self.held and 1 < len(self.kept):
            (forgotten, of), size = self.kept.popitem(last=False)
            self.tree.forget(forgotten, of)
            self.held -= size


def _encode(updates: list[tuple]) -> bytes:
    """A session's updates as one Updates frame."""
    frame = pb.Updates()
    for update in updates:
        match update:
            case ("answer", measure, node, parent, branch, runs, count):
                answer = frame.updates.add().answer
                answer.measure, answer.node = measure, node
                if parent is not None:
                    answer.parent, answer.branch = parent, branch
                answer.children.runs.extend(
                    pb.Run(rank=rank, branches=branches, ln_below=ln_below) for rank, branches, ln_below in runs
                )
                if count is not None:
                    answer.children.count = count
            case ("unknown", node):
                frame.updates.add().unknown = node
            case ("placed", node, branches):
                frame.updates.add().placed.CopyFrom(pb.Path(node=node, branches=branches))
            case ("lacks", measure, node):
                frame.updates.add().lacks.CopyFrom(pb.Missing(measure=measure, node=node))
            case ("failed", measure, node, message):
                frame.updates.add().failed.CopyFrom(pb.Failed(measure=measure, node=node, message=message))
    return frame.SerializeToString()


def app(
    model: Model,
    manifest: list[dict],
    files: dict[str, tuple[bytes, str]],
    search: Callable[[str], Sequence[int]] | None = None,
) -> Starlette:
    """The session endpoint, manifest, files, and optional text-to-path search."""

    async def session(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[bytes] = asyncio.Queue()
        client = await asyncio.wrap_future(model.open(lambda frame: loop.call_soon_threadsafe(outbox.put_nowait, frame)))

        async def send() -> None:
            while True:
                await websocket.send_bytes(await outbox.get())

        sending = asyncio.create_task(send())
        try:
            while True:
                try:
                    demand = pb.Demand.FromString(await websocket.receive_bytes())
                    await asyncio.wrap_future(model.demand(client, demand))
                except (DecodeError, ValueError) as error:
                    # 1007: the message is not one a demand can be.
                    await websocket.close(1007, str(error)[:120])
                    return
        except WebSocketDisconnect:
            pass
        finally:
            sending.cancel()
            await asyncio.wrap_future(model.close(client))

    async def search_rpc(request: Request) -> Response:
        if search is None:
            return JSONResponse({"code": "unimplemented", "message": "text search"}, status_code=501)
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != _PROTO_CONTENT_TYPE:
            return Response(status_code=415, headers={"Accept-Post": _PROTO_CONTENT_TYPE})
        try:
            request_message = pb.SearchRequest.FromString(await request.body())
            path = await anyio.to_thread.run_sync(search, request_message.text)
            await asyncio.wrap_future(model.learn(path))
            response = pb.SearchResponse(path=path)
        except DecodeError as error:
            return JSONResponse({"code": "invalid_argument", "message": str(error)}, status_code=400)
        except Exception as error:
            traceback.print_exc()
            return JSONResponse({"code": "internal", "message": f"{type(error).__name__}: {error}"}, status_code=500)
        return Response(response.SerializeToString(), media_type=_PROTO_CONTENT_TYPE)

    async def manifest_json(_: Request) -> Response:
        return JSONResponse(manifest)

    def file(content: bytes, media_type: str):
        async def serve(_: Request) -> Response:
            return Response(content, media_type=media_type)

        return serve

    routes: list = [Route("/manifest.json", manifest_json), WebSocketRoute("/session", session)]
    routes += [Route(f"/{path}", file(*served)) for path, served in files.items()]
    routes += [Route(f"/{index}/natlas.Natlas/Search", search_rpc, methods=["POST"]) for index in range(len(model.measures))]
    # Any page may read a local research server.
    cors = Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    return Starlette(routes=routes, middleware=[cors])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
