"""Byte-prefix inverse CDF service shared by the text and music servers."""

from __future__ import annotations

import asyncio
import hashlib
import math
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Sequence

from google.protobuf.message import DecodeError
from natlas.v2 import natlas_pb2 as pb

MAX_PREFIX = 1024
MAX_WEIGHT_BITS = 16384
CACHED_CDFS = 8
RPC = "/natlas.v2.Natlas/Quantile"


def request(data: bytes) -> tuple[tuple[int, int] | None, bytes]:
    try:
        message = pb.QuantileRequest.FromString(data)
    except DecodeError as error:
        raise ValueError("malformed protobuf request") from error
    if len(message.prefix) > MAX_PREFIX:
        raise OverflowError(f"prefix exceeds {MAX_PREFIX} bytes")
    node = (
        (message.node.parent, message.node.branch_id)
        if message.HasField("node")
        else None
    )
    return node, message.prefix


def response(
    handle: int, first: int, deltas: Sequence[int], terminal: bool = False
) -> bytes:
    return pb.QuantileResponse(
        node=handle, first_branch_id=first, delta=deltas, terminal=terminal
    ).SerializeToString()


class CDF:
    """Exact cumulative integer masses for a fixed finite distribution.

    Input weights need not already sum to one. Zero weights retain their ID
    positions but never appear as inverse-CDF answers. No rounded cumulative
    floating-point coordinates are stored.
    """

    def __init__(self, weights: Iterable[float]):
        ratios = []
        for weight in weights:
            if (
                weight < 0
                or (hasattr(weight, "is_finite") and not weight.is_finite())
                or (isinstance(weight, float) and not math.isfinite(weight))
            ):
                raise ValueError("invalid weight")
            ratios.append(weight.as_integer_ratio())
        denominator = math.lcm(*(d for _, d in ratios))
        self.ends: list[int] = []
        total = 0
        for numerator, divisor in ratios:
            total += numerator * (denominator // divisor)
            self.ends.append(total)
        if ratios and total == 0:
            raise ValueError("zero total mass")
        self.total = total

    @classmethod
    def from_log_weights(cls, logs: Iterable[float]) -> CDF:
        """Binary64 significands with arbitrary integer exponents.

        This preserves tiny tails even when exp(logp) would underflow. The
        significands approximate the input logits to ordinary float accuracy;
        normalization and all later coordinate comparisons are exact.
        """
        logs = list(map(float, logs))
        if any(math.isnan(value) or value == math.inf for value in logs):
            raise ValueError("invalid log weight")
        if not logs:
            return cls([])
        maximum = max(logs)
        if maximum == -math.inf:
            raise ValueError("zero total mass")
        ratios = []
        for value in logs:
            if value == -math.inf:
                ratios.append((0, 0))
                continue
            value -= maximum
            if not math.isfinite(value) or value < -MAX_WEIGHT_BITS * math.log(2):
                raise OverflowError(f"log weight range exceeds {MAX_WEIGHT_BITS} bits")
            exponent = math.floor(value / math.log(2))
            numerator, denominator = math.exp(
                value - exponent * math.log(2)
            ).as_integer_ratio()
            ratios.append((numerator, exponent - (denominator.bit_length() - 1)))
        minimum = min(exponent for numerator, exponent in ratios if numerator)
        result = cls.__new__(cls)
        result.ends = []
        total = 0
        for numerator, exponent in ratios:
            if numerator:
                total += numerator << (exponent - minimum)
            result.ends.append(total)
        result.total = total
        return result

    def at(self, numerator: int, denominator: int) -> int:
        if numerator == denominator:
            return len(self.ends)  # Vocabulary-end sentinel, never a token ID.
        threshold = numerator * self.total // denominator
        return bisect_right(self.ends, threshold)

    def page(self, prefix: bytes) -> tuple[int, list[int], bool]:
        if not self.ends:
            return 0, [], True
        a = int.from_bytes(prefix, "big") * 256
        denominator = 1 << (8 * (len(prefix) + 1))
        ids = [self.at(a + j, denominator) for j in range(257)]
        return ids[0], [b - a for a, b in zip(ids, ids[1:])], False


class Service:
    """Event-loop-owned handles and CDFs; inference is an asynchronous provider."""

    def __init__(
        self,
        distribution: Callable[[tuple[int, ...]], Awaitable[Iterable[float]]],
        *,
        log_weights: bool = False,
    ):
        self.distribution = distribution
        self.log_weights = log_weights
        self.paths: dict[int, tuple[int, ...]] = {0: ()}
        self.handles: dict[tuple[int, ...], int] = {(): 0}
        self.cache: OrderedDict[tuple[int, ...], CDF] = OrderedDict()
        self.vocabularies: dict[tuple[int, ...], int] = {}
        self.pending: dict[tuple[int, ...], asyncio.Task[CDF]] = {}

    async def query(self, data: bytes) -> bytes:
        child, prefix = request(data)
        if child is None:
            path: tuple[int, ...] = ()
        else:
            parent, label = child
            if parent not in self.paths:
                raise KeyError("unknown parent handle")
            base = self.paths[parent]
            if base not in self.vocabularies:
                await self.cdf(base)
            if label >= self.vocabularies[base]:
                raise ValueError("invalid branch")
            path = base + (label,)
        if path not in self.handles:
            encoded = b"".join(label.to_bytes(4, "big") for label in path)
            handle = int.from_bytes(
                hashlib.blake2b(encoded, digest_size=8, person=b"natlas-v2").digest(),
                "big",
            )
            if handle in self.paths and self.paths[handle] != path:
                raise RuntimeError("node handle collision")
            self.handles[path] = handle
            self.paths[handle] = path
        cdf = await self.cdf(path)
        # The local reference remains valid if another request evicts this CDF.
        page = await asyncio.to_thread(cdf.page, prefix)
        return response(self.handles[path], *page)

    async def cdf(self, path: tuple[int, ...]) -> CDF:
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached
        task = self.pending.get(path)
        if task is None:
            task = asyncio.create_task(self._load(path), name="natlas-cdf")
            self.pending[path] = task
            task.add_done_callback(lambda done: self._finished(path, done))
        # Accepted work warms the cache even if its HTTP caller disconnects;
        # other callers waiting for this same CDF keep their result.
        return await asyncio.shield(task)

    def _finished(self, path: tuple[int, ...], task: asyncio.Task[CDF]) -> None:
        self.pending.pop(path, None)
        if not task.cancelled():
            task.exception()

    async def _load(self, path: tuple[int, ...]) -> CDF:
        weights = await self.distribution(path)
        make = CDF.from_log_weights if self.log_weights else CDF
        cdf = await asyncio.to_thread(make, weights)
        self.vocabularies[path] = len(cdf.ends)
        self.cache[path] = cdf
        if len(self.cache) > CACHED_CDFS:
            self.cache.popitem(last=False)
        return cdf

    async def aclose(self) -> None:
        await asyncio.gather(*self.pending.values(), return_exceptions=True)
