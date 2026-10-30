"""Bounded asynchronous inference, independent of HTTP and tree encodings.

One worker belongs to one fixed set of measures. Backends receive batches of
measure/path jobs and return results keyed by those jobs; a remote or batched
backend can implement the same interface. The local backend owns one thread,
keeping stateful model/KV-cache access serialized and off the request loop.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Job:
    measure: int
    path: tuple[int, ...]


type LogWeights = tuple[float, ...]
type Results = Mapping[Job, LogWeights | Exception]


class Backend(Protocol):
    async def evaluate(self, jobs: tuple[Job, ...]) -> Results:
        """Exactly one result or error per job, for this backend's fixed measures."""
        ...

    async def aclose(self) -> None: ...


class LocalBackend:
    """The current models evaluate one path at a time on a dedicated thread."""

    def __init__(self, evaluate: Callable[[Job], Iterable[float]]):
        self.evaluate_one = evaluate
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="natlas-inference")

    def _batch(self, jobs: tuple[Job, ...]) -> Results:
        results: dict[Job, LogWeights | Exception] = {}
        for job in jobs:
            try:
                # Copy before another evaluation can overwrite a model's logits.
                results[job] = tuple(self.evaluate_one(job))
            except Exception as error:  # noqa: BLE001 - deliver each job's error to its callers
                results[job] = error
        return results

    async def evaluate(self, jobs: tuple[Job, ...]) -> Results:
        return await asyncio.get_running_loop().run_in_executor(self.executor, self._batch, jobs)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)


class Busy(Exception):
    """The bounded number of distinct outstanding jobs has been reached."""


class Closed(Exception):
    """The worker is not accepting work."""


@dataclass
class _Pending:
    result: asyncio.Future[LogWeights]
    waiters: int = 0
    started: bool = False


def _observe(result: asyncio.Future) -> None:
    # Running inference can outlive its final caller. Retrieve its error even
    # then, without changing what any remaining caller receives.
    if not result.cancelled():
        result.exception()


class Worker:
    """Share duplicate jobs, bound admission, and dispatch batches to a backend.

    All bookkeeping belongs to the request event loop. Cancelling one waiter
    never cancels another. A queued job with no waiters is removed; a running
    job finishes before the backend is closed. Capacity includes running jobs.
    """

    def __init__(self, backend: Backend, *, capacity: int = 64, batch_size: int = 1):
        if capacity < 1 or batch_size < 1:
            raise ValueError("capacity and batch size must be positive")
        self.backend = backend
        self.capacity = capacity
        self.batch_size = batch_size
        self.pending: dict[Job, _Pending] = {}
        self.queue: OrderedDict[Job, _Pending] = OrderedDict()
        self.available = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.closing: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.task is not None or self.closing is not None:
            raise Closed("inference worker has already been started")
        self.task = asyncio.create_task(self._run(), name="natlas-inference")

    async def infer(self, job: Job) -> LogWeights:
        if self.task is None or self.task.done() or self.closing is not None:
            raise Closed("inference worker is not accepting work")
        pending = self.pending.get(job)
        if pending is None:
            if len(self.pending) >= self.capacity:
                raise Busy(f"inference capacity of {self.capacity} jobs reached")
            result = asyncio.get_running_loop().create_future()
            result.add_done_callback(_observe)
            pending = self.pending[job] = self.queue[job] = _Pending(result)
            self.available.set()
        pending.waiters += 1
        try:
            return await asyncio.shield(pending.result)
        finally:
            pending.waiters -= 1
            if pending.waiters == 0 and not pending.started:
                self.queue.pop(job, None)
                self.pending.pop(job, None)
                pending.result.cancel()

    async def _run(self) -> None:
        try:
            while True:
                await self.available.wait()
                if self.closing is not None:
                    return
                batch = []
                while self.queue and len(batch) < self.batch_size:
                    job, pending = self.queue.popitem(last=False)
                    pending.started = True
                    batch.append((job, pending))
                if not self.queue:
                    self.available.clear()
                if not batch:
                    continue
                jobs = tuple(job for job, _ in batch)
                try:
                    results = await self.backend.evaluate(jobs)
                    if set(results) != set(jobs):
                        raise RuntimeError("inference backend returned a different set of jobs")
                except Exception as error:  # noqa: BLE001 - a backend failure settles the whole batch
                    results = dict.fromkeys(jobs, error)
                for job, pending in batch:
                    self.pending.pop(job, None)
                    value = results[job]
                    if isinstance(value, Exception):
                        pending.result.set_exception(value)
                    else:
                        pending.result.set_result(value)
        finally:
            # Also settle callers if the event loop or backend cancels the pump.
            for pending in self.pending.values():
                if not pending.result.done():
                    pending.result.set_exception(Closed("inference worker stopped"))
            self.pending.clear()
            self.queue.clear()

    async def aclose(self) -> None:
        if self.closing is None:
            self.closing = asyncio.create_task(self._close(), name="natlas-inference-close")
        await asyncio.shield(self.closing)

    async def _close(self) -> None:
        for job, pending in self.queue.items():
            pending.result.set_exception(Closed("inference worker is shutting down"))
            self.pending.pop(job, None)
        self.queue.clear()
        self.available.set()
        try:
            if self.task is not None:
                await self.task
        finally:
            await self.backend.aclose()
