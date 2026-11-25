"""Model-independent integer-sequence cache, scheduler and session transport.

Language and music adapters supply a catalog: what their specs may say, and
how to open the measure one names. Measures open the first time a client
asks for one; the Rust core owns their cached distributions, the hashed node
ids shared with every client, and what each client's view lacks.
"""

from __future__ import annotations

import asyncio
import gc
import queue
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

import anyio.to_thread
import numpy as np
from google.protobuf.message import DecodeError
from natlas_core import Session, Tree
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from catalog import Catalog, Measure
from natlas import natlas_pb2 as pb

_PROTO_CONTENT_TYPE = "application/proto"
# Memory for distributions a server keeps by default, across its measures.
# The core keeps each packed, about 4 bytes per vocabulary entry:
# some 1700 nodes of a 150k vocabulary, 5000 of a 50k one.
CACHE = 1 << 30


@dataclass(eq=False)
class Client:
    """One client's session, and how to hand it an Updates frame from any
    thread."""

    send: Callable[[bytes], None]
    session: Session = field(default_factory=Session)
    # What it looks at, as it said it: taken again when a model it names loads.
    demand: pb.Demand | None = None


class _Worker:
    """One daemon thread doing what it is given in turn. A pass or a model
    loading never keeps the process from ending: ^C ends it at once."""

    def __init__(self, name: str):
        self.queue: queue.SimpleQueue[tuple[Future, Callable[[], Any]]] = queue.SimpleQueue()
        threading.Thread(target=self._run, daemon=True, name=name).start()

    def submit[T](self, work: Callable[[], T]) -> Future[T]:
        future: Future[T] = Future()
        self.queue.put((future, work))
        return future

    def _run(self) -> None:
        while True:
            future, work = self.queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(work())
            except BaseException as error:  # noqa: BLE001 (the future holds it)
                future.set_exception(error)
            # Holding on to what it did, a model let go would stay in memory.
            del future, work


@dataclass(frozen=True)
class Pass:
    """One forward pass under ``measure``, by id: for each walk, the
    distributions of its nodes, each the child of the one before; and the
    clients waiting on them, whose new demands withdraw it."""

    measure: int
    walks: tuple[tuple[int, ...], ...]
    clients: frozenset[Client]


@dataclass(frozen=True)
class _Ready:
    measure: Measure


@dataclass(frozen=True)
class _Failed:
    message: str


@dataclass
class _MeasureRecord:
    spec: str
    model: str | None
    # None waits for the model. A future means opening has been submitted.
    state: Future[Measure] | _Ready | _Failed | None = None


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

    Models load when asked to, on a thread of their own, and unload when
    asked to. Measures are numbered as clients first name them, and open
    only of a model loaded: one opens on the model's thread, as a pass does,
    so it never runs beside one. What waits on a measure whose model is
    loading waits; what waits on one that cannot be opened fails, saying
    why, and a client that named it is answered once its model loads.
    """

    def __init__(self, vocabulary: int, catalog: Catalog, cache: int = CACHE, order: str = "heaviest"):
        # Branches the models have; every other branch has weight zero.
        self.vocabulary = vocabulary
        self.catalog = catalog
        # Each model asked for, by spec, as it loads and once it has.
        self.models: dict[str, Future[Any]] = {}
        # Stable ids name records in sessions and stored distributions.
        # Failed records retain their ids while clients may still name them.
        self._measure_records: list[_MeasureRecord] = []
        self._measure_ids: dict[str, int] = {}
        # The order each node's children rank in: "heaviest" first, or by
        # "branch", where branches already follow the reading's own order.
        self.order = order
        self.tree = Tree()
        # Which distributions to forget first: the oldest, once those kept,
        # by (node, measure), hold more than ``cache`` bytes.
        self.cache = cache
        self.kept: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.kept_bytes = 0
        self.clients: list[Client] = []
        # The client whose turn is next.
        self.turn = 0
        # Passes whose results the scheduler has not taken yet.
        self.passes: dict[Future[list[np.ndarray]], Pass] = {}
        self.inbox: queue.SimpleQueue[tuple[Callable[[], Any], Future]] = queue.SimpleQueue()
        self.inference = _Worker("inference")
        self.loader = _Worker("loader")
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
        return self.call(lambda: self._demand(client, demand))

    def _demand(self, client: Client, demand: pb.Demand) -> None:
        view = None
        if demand.HasField("view"):
            v = demand.view
            view = (v.node, v.top, v.ln_height, v.rows, v.right)
        path = (demand.path.node, list(demand.path.branches)) if demand.HasField("path") else None

        p, q = self._measure(demand.p), self._measure(demand.q)
        client.session.demand(self.tree, p, q, view=view, path=path)
        client.demand = demand
        self._withdraw(client)

    def held(self) -> Future[dict[str, Any]]:
        """The models asked for, each loading, loaded or failed, and the
        measures open."""

        def held() -> dict[str, Any]:
            def state(model: str, loading: Future[Any]) -> dict[str, str]:
                if not loading.done():
                    return {"spec": model, "state": "loading"}
                if (error := loading.exception()) is not None:
                    return {"spec": model, "state": "failed", "message": f"{type(error).__name__}: {error}"}
                return {"spec": model, "state": "loaded"}

            return {
                "models": [state(model, loading) for model, loading in self.models.items()],
                "measures": [
                    spec for spec, measure in self._measure_ids.items()
                    if isinstance(self._measure_records[measure].state, _Ready)
                ],
            }

        return self.call(held)

    def load(self, model: str) -> Future[None]:
        """Load the model ``model`` names, unless it is loaded or loading.
        The future fails with ValueError where it names none."""

        def load() -> None:
            self.catalog.model(model)
            loading = self.models.get(model)
            if loading is not None and not (loading.done() and loading.exception() is not None):
                return

            def loaded() -> Any:
                print(f"loading {model} …", file=sys.stderr)
                return self.catalog.load(model)

            future = self.models[model] = self.loader.submit(loaded)
            future.add_done_callback(lambda future: self.call(lambda: self._loaded(model, future)))

        return self.call(load)

    def unload(self, model: str) -> Future[None]:
        """Let the model go, closing its measures, and give back its memory."""

        def unload() -> None:
            if self.models.pop(model, None) is None:
                return
            for measure in tuple(self._measure_ids.values()):
                record = self._measure_records[measure]
                if record.model == model:
                    self._close(record, f"{model} was unloaded")
            # Once no pass holds it.
            self.inference.submit(self._release)

        return self.call(unload)

    def open_measure(self, spec: str) -> Future[None]:
        """Open the measure ``spec`` names. The future fails with ValueError
        where it names none, or none of a model loaded or loading."""

        def open() -> None:
            model, _ = self.catalog.parse(spec)
            if model not in self.models:
                raise ValueError(f"load {model} first")
            self._measure(spec)

        return self.call(open)

    def _release(self) -> None:
        gc.collect()
        self.catalog.released()

    def _loaded(self, model: str, future: Future[Any]) -> None:
        if self.models.get(model) is not future:
            return
        if (error := future.exception()) is not None:
            traceback.print_exception(error)
        retry = set()
        for record in self._measure_records:
            if record.model != model:
                continue
            if record.state is None:
                self._open(record, future)
            elif isinstance(record.state, _Failed) and record.spec not in self._measure_ids:
                retry.add(record.spec)
        if error is not None:
            return
        # Clients waiting on a measure of it, closed while it was not loaded.
        for client in self.clients:
            demand = client.demand
            if demand is not None and (demand.p in retry or demand.q in retry):
                self._demand(client, demand)

    def _measure(self, spec: str) -> int:
        """The id of the measure ``spec`` names, opening it the first time.
        One that closed opens again when named again."""
        known = self._measure_ids.get(spec)
        if known is not None:
            return known
        measure = len(self._measure_records)
        try:
            model, _ = self.catalog.parse(spec)
        except ValueError as error:
            self._measure_records.append(_MeasureRecord(spec, None, _Failed(f"{spec}: {error}")))
            return measure
        record = _MeasureRecord(spec, model)
        self._measure_records.append(record)
        self._measure_ids[spec] = measure
        loading = self.models.get(model)
        if loading is None:
            self._close(record, f"load {model} to read it")
        elif loading.done():
            self._open(record, loading)
        return measure

    def _open(self, record: _MeasureRecord, loading: Future[Any]) -> None:
        """Open a measure of a model that has loaded, or could not."""
        name = record.model
        assert name is not None and record.state is None
        if (error := loading.exception()) is not None:
            self._close(record, f"{name} could not be loaded: {type(error).__name__}: {error}")
            return
        model, spec = loading.result(), record.spec

        def open() -> Measure:
            print(f"opening {spec} …", file=sys.stderr)
            return self.catalog.open(model, spec)

        future = self.inference.submit(open)
        record.state = future
        future.add_done_callback(lambda future: self.call(lambda: self._opened(record, name, loading, future)))

    def _opened(self, record: _MeasureRecord, model: str, loading: Future[Any], future: Future[Measure]) -> None:
        if record.state is not future or self.models.get(model) is not loading:
            return
        if (error := future.exception()) is None:
            record.state = _Ready(future.result())
            return
        traceback.print_exception(error)
        self._close(record, f"{type(error).__name__}: {error}")

    def _close(self, record: _MeasureRecord, why: str) -> None:
        """What waits on the measure fails, saying ``why``; named again, it opens anew."""
        state = record.state
        record.state = _Failed(f"{record.spec}: {why}")
        del self._measure_ids[record.spec]
        if isinstance(state, Future):
            state.cancel()

    def _ready_measure(self, measure: int) -> Measure:
        state = self._measure_records[measure].state
        assert isinstance(state, _Ready), f"measure {measure} must be open"
        return state.measure

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
            client.send(_encode(updates, lambda measure: self._measure_records[measure].spec))

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
        # What waits on a measure that could not be opened fails; on one
        # still opening it waits.
        settled = False
        waits = {client: [] for client in order}
        for client in order:
            for blocked in client.session.blocked():
                measure, node, *_ = blocked
                state = self._measure_records[measure].state
                if isinstance(state, _Failed):
                    client.session.failed(measure, node, state.message)
                    settled = True
                elif isinstance(state, _Ready) and (measure, node) not in reserved:
                    waits[client].append(blocked)
        first = next((index for index, client in enumerate(order) if waits[client]), None)
        if first is None:
            return None, settled
        self.turn = (self.turn + first + 1) % count
        order = order[first:] + order[:first]
        measure = waits[order[0]][0][0]
        turns = zip_longest(*([(client, blocked) for blocked in waits[client] if blocked[0] == measure] for client in order))
        source = self._ready_measure(measure)
        width, paths = source.width, source.paths
        walks: list[tuple[int, ...]] = []
        clients: set[Client] = set()
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
        source = self._ready_measure(measure)
        tokens = tuple(self.tree.path(node))
        if self._empty(source, tokens):
            self._keep(measure, [node], [np.empty(0, np.float32)])
            return None
        if not self._reads(source, tokens):
            self._fail(measure, node, f"{len(source.start) + len(tokens)} tokens, but the context holds {source.longest}")
            return None
        nodes, along = [node], tokens
        for branch in ahead:
            along += (branch,)
            if self._empty(source, along) or not self._reads(source, along):
                break
            (_, child) = self.tree.walk(nodes[-1], [branch])
            if (measure, child) in reserved or self.tree.has(child, measure):
                break
            nodes.append(child)
        return tuple(nodes)

    def _work(self, job: Pass) -> Callable[[], list[np.ndarray]]:
        """What the model runs for a pass."""
        source = self._ready_measure(job.measure)
        walk, start = source.walk, source.start
        spec = self._measure_records[job.measure].spec
        reads = []
        for nodes in job.walks:
            path = start + tuple(self.tree.path(nodes[-1]))
            reads.append((path, len(path) - len(nodes) + 1))

        def work() -> list[np.ndarray]:
            started = time.perf_counter()
            computed = walk(reads)
            elapsed = (time.perf_counter() - started) * 1000
            depths = sorted({first - len(start) for _, first in reads})
            print(
                f"{spec} {len(reads)} paths, depth {depths[0]}–{depths[-1]}: "
                f"{sum(map(len, job.walks))} computed, {elapsed:.0f} ms",
                file=sys.stderr,
            )
            return computed

        return work

    def _finish(self, future: Future[list[np.ndarray]]) -> None:
        """Keep what a pass computed, or fail what waits on it."""
        job = self.passes.pop(future)
        # Withdrawn, or of a measure since closed.
        if future.cancelled() or not isinstance(self._measure_records[job.measure].state, _Ready):
            return
        error = future.exception()
        if error is None:
            computed = future.result()
            if [len(rows) for rows in computed] == list(map(len, job.walks)):
                try:
                    self._keep(job.measure, [node for nodes in job.walks for node in nodes], [row for rows in computed for row in rows])
                    return
                except Exception as exc:
                    error = exc
            else:
                error = RuntimeError(f"walks of {list(map(len, job.walks))} nodes gave {[len(rows) for rows in computed]} distributions")
        traceback.print_exception(error)
        for nodes in job.walks:
            self._fail(job.measure, nodes[0], f"{type(error).__name__}: {error}")

    def _fail(self, measure: int, node: int, message: str) -> None:
        """Fail the distribution for every client waiting on it."""
        for client in self.clients:
            if any(blocked[:2] == (measure, node) for blocked in client.session.blocked()):
                client.session.failed(measure, node, message)

    def _reads(self, measure: Measure, tokens: tuple[int, ...]) -> bool:
        """Whether the measure reads a path this long."""
        longest = measure.longest
        return longest is None or len(measure.start) + len(tokens) <= longest

    def _empty(self, measure: Measure, tokens: tuple[int, ...]) -> bool:
        """Whether a sequence has no children: finished, or past the vocabulary."""
        return measure.ends(tokens) or any(self.vocabulary <= branch for branch in tokens)

    def _keep(self, measure: int, nodes: list[int], rows: list[np.ndarray]) -> None:
        widths = self._ready_measure(measure).widths
        rows = [np.asarray(row, dtype=np.float32) for row in rows]
        sizes = self.tree.set_rows(nodes, measure, rows, heaviest=self.order == "heaviest", normalize=not widths)
        for node, size in zip(nodes, sizes, strict=True):
            self.kept_bytes += size - self.kept.pop((node, measure), 0)
            self.kept[node, measure] = size
        # Those just computed stay, however large.
        while self.cache < self.kept_bytes and len(nodes) < len(self.kept):
            (forgotten, of), size = self.kept.popitem(last=False)
            self.tree.forget(forgotten, of)
            for client in self.clients:
                client.session.forget(of, forgotten)
            self.kept_bytes -= size


def _encode(updates: list[tuple], spec: Callable[[int], str]) -> bytes:
    """A session's updates as one Updates frame, each measure by its spec."""
    frame = pb.Updates()
    for update in updates:
        match update:
            case ("answer", measure, node, parent, branch, runs, count):
                answer = frame.updates.add().answer
                answer.measure, answer.node = spec(measure), node
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
                frame.updates.add().lacks.CopyFrom(pb.Missing(measure=spec(measure), node=node))
            case ("failed", measure, node, message):
                frame.updates.add().failed.CopyFrom(pb.Failed(measure=spec(measure), node=node, message=message))
    return frame.SerializeToString()


def app(
    model: Model,
    manifest: list[dict],
    files: dict[str, tuple[bytes, str]],
    search: Callable[[str], Sequence[int]] | None = None,
    routes: Sequence[Route] = (),
) -> Starlette:
    """The session endpoint, manifest, files, optional text-to-path search,
    and any routes of the server's own."""

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

    async def held(_: Request) -> Response:
        return JSONResponse(await asyncio.wrap_future(model.held()))

    def changing(change: Callable[[str], Future[None]]):
        """A request changing what the model holds, by the spec in its path."""

        async def changed(request: Request) -> Response:
            try:
                await asyncio.wrap_future(change(request.path_params["spec"]))
            except ValueError as error:
                return JSONResponse({"message": str(error)}, status_code=400)
            return JSONResponse(await asyncio.wrap_future(model.held()))

        return changed

    served: list = [
        Route("/manifest.json", manifest_json),
        WebSocketRoute("/session", session),
        # What the model holds, and loading, unloading and opening by spec.
        Route("/held", held),
        Route("/models/{spec:path}", changing(model.load), methods=["POST"]),
        Route("/models/{spec:path}", changing(model.unload), methods=["DELETE"]),
        Route("/measures/{spec:path}", changing(model.open_measure), methods=["POST"]),
    ]
    served += [Route(f"/{path}", file(*content)) for path, content in files.items()]
    served += [Route("/natlas.Natlas/Search", search_rpc, methods=["POST"]), *routes]
    # Any page may read a local research server.
    cors = Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
    return Starlette(routes=served, middleware=[cors])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
