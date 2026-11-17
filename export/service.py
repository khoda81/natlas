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
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Protocol

import anyio.to_thread
import numpy as np
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
    """Next-token ln weights over the vocabulary, up to a constant, as
    float32 rows."""

    name: str
    # How many distributions one forward pass computes, and of how many
    # paths at most.
    width: int
    paths: int
    # The most tokens a path it reads may hold, or None where nothing says.
    longest: int | None

    def walk(self, walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
        """For each walk (tokens, start), the rows after tokens[:k] for every
        k from ``start`` through len(tokens)."""
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


@dataclass(frozen=True)
class Pass:
    """One forward pass under ``measure``: for each walk, the distributions
    of its nodes, each the child of the one before; and the clients waiting
    on them, whose new demands withdraw it."""

    measure: int
    walks: tuple[tuple[int, ...], ...]
    clients: frozenset[Client]


# Forward passes submitted and not yet finished: the one running and the one
# queued behind it, so the model starts the next the moment one ends while
# the scheduler packs the last one's result. A queued pass can still be
# withdrawn; a running one cannot.
_SUBMITTED = 2


class Model:
    """The integer-sequence distribution under each measure, computed for the
    clients looking at it.

    One scheduler thread owns the core tree, the distributions kept and the
    sessions; everything else posts work to its inbox. It takes everything
    posted at once, so a burst of demands costs one step, steps every
    client's session, sends what is new, and keeps the model busy with what
    the clients wait on most. The model runs on a thread of its own; only
    the scheduler submits passes to it. A pass computes as many
    distributions as the measure computes at once, taking a walk from each
    client in turn, from the next client in turn on; within a client, worth
    alone decides. A new demand withdraws the queued pass of the client, so
    the next one is chosen from it.
    """

    def __init__(
        self,
        root: tuple[int, ...],
        ends: Callable[[tuple[int, ...]], bool],
        vocabulary: int,
        measures: list[Measure],
        cache: int = CACHE,
        order: str = "heaviest",
    ):
        # What the model reads before the root's children: a start-of-text token.
        self.root = root
        # Whether a path is finished, with no children.
        self.ends = ends
        # Branches the model has; every other branch has weight zero.
        self.vocabulary = vocabulary
        self.measures = measures
        # The order each node's children rank in: "heaviest" first, or by
        # "branch", where branches already follow the reading's own order.
        self.order = order
        self.tree = Tree(len(measures))
        # Which distributions to forget first: the oldest, once those kept,
        # by (node, measure), hold more than ``cache`` bytes.
        self.cache = cache
        self.kept: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.held = 0
        self.clients: list[Client] = []
        # The client whose turn is next.
        self.turn = 0
        # Passes whose results the scheduler has not taken yet.
        self.passes: dict[Future[list[np.ndarray]], Pass] = {}
        self.inbox: queue.SimpleQueue[tuple[Callable[[], Any], Future]] = queue.SimpleQueue()
        self.inference = ThreadPoolExecutor(1, thread_name_prefix="inference")
        threading.Thread(target=self._run, daemon=True, name="scheduler").start()

    def call[T](self, work: Callable[[], T]) -> Future[T]:
        """Run ``work`` on the scheduler."""
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
        def closed() -> None:
            self.clients.remove(client)
            self._withdraw(client)

        return self.call(closed)

    def demand(self, client: Client, demand: pb.Demand) -> Future[None]:
        """What ``client`` looks at now. The future fails with ValueError for
        a demand the tree cannot take."""
        view = None
        if demand.HasField("view"):
            v = demand.view
            view = (v.node, v.top, v.ln_height, v.rows, v.right)
        path = (demand.path.node, list(demand.path.branches)) if demand.HasField("path") else None

        def demanded() -> None:
            client.session.demand(self.tree, demand.p, demand.q, view=view, path=path)
            self._withdraw(client)

        return self.call(demanded)

    def learn(self, path: Sequence[int]) -> Future[list[int]]:
        """Learn the ids along ``path``, so demands can name them alone."""
        return self.call(lambda: self.tree.walk(0, list(path)))

    def _run(self) -> None:
        while True:
            item = self.inbox.get()
            while True:
                work, future = item
                try:
                    future.set_result(work())
                except Exception as error:
                    future.set_exception(error)
                try:
                    item = self.inbox.get_nowait()
                except queue.Empty:
                    break
            # What needs no pass is settled at once, and read on from.
            while True:
                for client in self.clients:
                    self._step(client)
                if not self._submit():
                    break

    def _step(self, client: Client) -> None:
        updates = client.session.step(self.tree)
        if updates:
            client.send(_encode(updates))

    def _withdraw(self, client: Client) -> None:
        """Withdraw the client's queued pass; its result is taken, and the
        pass forgotten, when the future's callback reaches the inbox."""
        for future, job in self.passes.items():
            if client in job.clients:
                future.cancel()

    def _submit(self) -> bool:
        """Keep the model busy: submit what the clients wait on most, and is
        not on its way already. Says whether it settled distributions that
        need no pass."""
        reserved = {(job.measure, node) for job in self.passes.values() for walk in job.walks for node in walk}
        settled = False
        while sum(not future.done() for future in self.passes) < _SUBMITTED:
            job, settles = self._chosen(reserved)
            settled |= settles
            if job is None:
                return settled
            future = self.inference.submit(self._work(job))
            self.passes[future] = job
            future.add_done_callback(lambda future: self.call(lambda: self._finish(future)))
        return settled

    def _chosen(self, reserved: set[tuple[int, int]]) -> tuple[Pass | None, bool]:
        """A pass under the measure that the next client in turn waiting on
        something not reserved waits on most: a walk from each client in
        turn, from that one on, while the measure has room, each reserved as
        it is taken. Says too whether choosing it settled distributions that
        need no pass."""
        count = len(self.clients)
        order = [self.clients[(self.turn + offset) % count] for offset in range(count)]
        waits = {client: [blocked for blocked in client.session.blocked() if blocked[:2] not in reserved] for client in order}
        first = next((index for index, client in enumerate(order) if waits[client]), None)
        if first is None:
            return None, False
        self.turn = (self.turn + first + 1) % count
        order = order[first:] + order[:first]
        measure = waits[order[0]][0][0]
        turns = zip_longest(*([(client, blocked) for blocked in waits[client] if blocked[0] == measure] for client in order))
        width, paths = self.measures[measure].width, self.measures[measure].paths
        walks: list[tuple[int, ...]] = []
        clients: set[Client] = set()
        settled = False
        for client, (_, node, _, ahead) in (taken for turn in turns for taken in turn if taken is not None):
            room = width - sum(map(len, walks))
            if room == 0 or len(walks) == paths:
                break
            # Another client's wait taken already.
            if (measure, node) in reserved:
                continue
            nodes = self._walk(measure, node, ahead[: room - 1], reserved)
            if nodes is None:
                reserved.add((measure, node))
                settled = True
                continue
            reserved.update((measure, node) for node in nodes)
            walks.append(nodes)
            clients.add(client)
        if not walks:
            return None, settled
        return Pass(measure, tuple(walks), frozenset(clients)), settled

    def _walk(self, measure: int, node: int, ahead: list[int], reserved: set[tuple[int, int]]) -> tuple[int, ...] | None:
        """The nodes a pass computes from ``node`` on: it, and where a path
        goes on below it, the next nodes along ``ahead`` that are neither
        held nor on their way. None where it needs no pass, as a node with no
        children or one deeper than the measure reads, settled at once."""
        tokens = tuple(self.tree.path(node))
        if self._empty(tokens):
            self._keep(measure, [node], [np.empty(0, np.float32)])
            return None
        longest = self.measures[measure].longest
        if not self._reads(measure, tokens):
            self._fail(measure, node, f"{len(self.root) + len(tokens)} tokens, but the context holds {longest}")
            return None
        nodes, along = [node], tokens
        for branch in ahead:
            along += (branch,)
            if self._empty(along) or not self._reads(measure, along):
                break
            (_, child) = self.tree.walk(nodes[-1], [branch])
            if (measure, child) in reserved or self.tree.has(child, measure):
                break
            nodes.append(child)
        return tuple(nodes)

    def _work(self, job: Pass) -> Callable[[], list[np.ndarray]]:
        """What the model runs for a pass."""
        walk = self.measures[job.measure].walk
        reads = []
        for nodes in job.walks:
            path = self.root + tuple(self.tree.path(nodes[-1]))
            reads.append((path, len(path) - len(nodes) + 1))

        def work() -> list[np.ndarray]:
            started = time.perf_counter()
            computed = walk(reads)
            elapsed = (time.perf_counter() - started) * 1000
            depths = sorted({start - len(self.root) for _, start in reads})
            print(
                f"m{job.measure} {len(reads)} paths, depth {depths[0]}–{depths[-1]}: "
                f"{sum(map(len, job.walks))} computed, {elapsed:.0f} ms",
                file=sys.stderr,
            )
            return computed

        return work

    def _finish(self, future: Future[list[np.ndarray]]) -> None:
        """Keep what a pass computed, or fail what waits on it."""
        job = self.passes.pop(future)
        if future.cancelled():
            return
        error = future.exception()
        if error is None:
            computed = future.result()
            if [len(rows) for rows in computed] == list(map(len, job.walks)):
                self._keep(job.measure, [node for nodes in job.walks for node in nodes], [row for rows in computed for row in rows])
                return
            error = RuntimeError(f"walks of {list(map(len, job.walks))} nodes gave {[len(rows) for rows in computed]} distributions")
        traceback.print_exception(error)
        for nodes in job.walks:
            self._fail(job.measure, nodes[0], f"{type(error).__name__}: {error}")

    def _fail(self, measure: int, node: int, message: str) -> None:
        """Fail the distribution for every client waiting on it."""
        for client in self.clients:
            if any(blocked[:2] == (measure, node) for blocked in client.session.blocked()):
                client.session.failed(measure, node, message)

    def _reads(self, measure: int, tokens: tuple[int, ...]) -> bool:
        """Whether the measure reads a path this long."""
        longest = self.measures[measure].longest
        return longest is None or len(self.root) + len(tokens) <= longest

    def _empty(self, tokens: tuple[int, ...]) -> bool:
        """Whether a sequence has no children: finished, or past the vocabulary."""
        return self.ends(tokens) or any(self.vocabulary <= branch for branch in tokens)

    def _keep(self, measure: int, nodes: list[int], rows: list[np.ndarray]) -> None:
        rows = [np.ascontiguousarray(row, dtype=np.float32) for row in rows]
        sizes = self.tree.set(nodes, measure, rows, self.order)
        for node, size in zip(nodes, sizes, strict=True):
            self.held += size - self.kept.pop((node, measure), 0)
            self.kept[node, measure] = size
        # Those just computed stay, however large.
        while self.cache < self.held and len(nodes) < len(self.kept):
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
                    pb.Run(rank=rank, branches=branches, ln_weights=ln_weights, top=bytes(top))
                    for rank, branches, ln_weights, top in runs
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
        # Frames to send, or why the session ends.
        outbox: asyncio.Queue[bytes | Exception] = asyncio.Queue()

        def post(item: bytes | Exception) -> None:
            loop.call_soon_threadsafe(outbox.put_nowait, item)

        def refused(future: Future[None]) -> None:
            if (error := future.exception()) is not None:
                post(error)

        client = await asyncio.wrap_future(model.open(post))

        async def receive() -> Exception | None:
            # Demands are posted without waiting for the scheduler to take
            # them: it takes all posted at once, the latest last.
            try:
                while True:
                    demand = pb.Demand.FromString(await websocket.receive_bytes())
                    model.demand(client, demand).add_done_callback(refused)
            except WebSocketDisconnect:
                return None
            except DecodeError as error:
                return error

        async def send() -> Exception:
            while isinstance(item := await outbox.get(), bytes):
                await websocket.send_bytes(item)
            return item

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if (error := done.pop().result()) is not None:
                # 1007: the message is not one a demand can be.
                await websocket.close(1007, str(error)[:120])
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()
            model.close(client)

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
