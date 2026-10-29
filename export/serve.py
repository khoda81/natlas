# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "llama-cpp-python>=0.3.16",
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "starlette>=0.40",
#   "torch>=2.2",
#   "transformers>=4.51",
#   "uvicorn>=0.30",
# ]
#
# # llama.cpp for the GPU, built once; CMAKE_CUDA_ARCHITECTURES=75 is the GTX 1660 Ti.
# [tool.uv]
# config-settings-package = { llama-cpp-python = { "cmake.define.GGML_CUDA" = "ON", "cmake.define.CMAKE_CUDA_ARCHITECTURES" = "75" } }
# ///

"""Serve a language model's tree of texts to natlas, live.

The root is the empty text, the model's start-of-text token alone; every
node below it is a text, and its children are the next tokens. Every measure
is one Natlas service under its own path prefix, /0/ for the model itself and
/1/, /2/, … for the --q measures, speaking Connect's unary quantile protocol
(and retaining the legacy streaming Expand service). GET /manifest.json describes them to the client, with the prompt's
tokens as the place to start, and GET /labels.json holds the token bytes.

A node's handle hashes its path, so handles survive a restart, but the
server cannot invert them: it remembers each node it has expanded, one parent
handle and label apiece, and asks the client to name any other again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import struct
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import lm
from inference import Backend, Busy, Closed, Job, LocalBackend, LogWeights, Worker
from natlas_codec import encode, labels_json, pb

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

    def __init__(
        self,
        root: tuple[int, ...],
        ends: Callable[[tuple[int, ...]], bool],
        labels: int,
        measures: list[lm.Measure],
    ):
        # What the model reads before the root's children: a start-of-text token.
        self.root = root
        # Whether a path is finished, with no children.
        self.ends = ends
        self.labels = labels
        self.measures = measures
        self.parents: dict[int, tuple[int, int]] = {}
        self.cache: list[OrderedDict[int, tuple[np.ndarray, np.ndarray]]] = [OrderedDict() for _ in measures]
        self.pending: dict[tuple[int, int], asyncio.Task[tuple[np.ndarray, np.ndarray]]] = {}
        # Handles and caches belong to the request event loop. Only infer()
        # touches model/KV state, through the shared inference worker.

    def infer(self, job: Job) -> LogWeights:
        return tuple(self.measures[job.measure].log_probs(self.root + job.path).tolist())

    def path(self, handle: int) -> tuple[int, ...]:
        labels = []
        while handle != _ROOT:
            entry = self.parents.get(handle)
            if entry is None:
                raise UnknownNode(f"node {handle:016x} is unknown here")
            handle, label = entry
            labels.append(label)
        return tuple(reversed(labels))

    async def expand(
        self, measure: int, child: pb.Child | None, horizon: int, worker: Worker
    ) -> tuple[int, list[pb.Edge]]:
        """The node's handle, and its listing down to 2^-horizon."""
        started = time.perf_counter()
        if child is None:
            handle, path = _ROOT, ()
        else:
            if child.label >= self.labels:
                raise BadLabel(f"label {child.label} is past the vocabulary's {self.labels}")
            path = self.path(child.parent) + (child.label,)
            handle = _child_handle(child.parent, child.label)
            self.parents.setdefault(handle, (child.parent, child.label))
        if horizon == 0 or self.ends(path):
            return handle, []
        cached = handle in self.cache[measure]
        listing = await self.listing(measure, handle, path, worker)

        def encoded():
            labels, nats = listing
            count = int(np.searchsorted(nats, horizon * math.log(2), side="right"))
            return encode(labels[:count].tolist(), (-nats[:count]).tolist(), horizon)

        edges = await asyncio.to_thread(encoded)
        elapsed = (time.perf_counter() - started) * 1000
        source = "cached" if cached else "computed"
        print(
            f"expand m{measure} depth {len(path)} horizon {horizon}: {len(edges)} edges, {source}, {elapsed:.0f} ms",
            file=sys.stderr,
        )
        return handle, edges

    async def listing(
        self, measure: int, handle: int, path: tuple[int, ...], worker: Worker
    ) -> tuple[np.ndarray, np.ndarray]:
        """Every label at a node, heaviest first, with its surprisal in nats."""
        cache = self.cache[measure]
        cached = cache.get(handle)
        if cached is not None:
            cache.move_to_end(handle)
            return cached
        key = measure, handle
        task = self.pending.get(key)
        if task is None:
            task = asyncio.create_task(self._listing(measure, handle, path, worker), name="natlas-listing")
            self.pending[key] = task
            task.add_done_callback(lambda done: self._finished(key, done))
        return await asyncio.shield(task)

    def _finished(self, key: tuple[int, int], task: asyncio.Task) -> None:
        self.pending.pop(key, None)
        if not task.cancelled():
            task.exception()

    async def _listing(
        self, measure: int, handle: int, path: tuple[int, ...], worker: Worker
    ) -> tuple[np.ndarray, np.ndarray]:
        logs = await worker.infer(Job(measure, path))

        def sorted_listing():
            values = np.asarray(logs, dtype=np.float64)
            order = np.argsort(-values, kind="stable").astype(np.int32)
            return order, -values[order]

        listing = await asyncio.to_thread(sorted_listing)
        cache = self.cache[measure]
        cache[handle] = listing
        if len(cache) > _CACHED:
            cache.popitem(last=False)
        return listing

    async def aclose(self) -> None:
        await asyncio.gather(*self.pending.values(), return_exceptions=True)


def _envelope(flags: int, payload: bytes) -> bytes:
    return struct.pack(">BI", flags, len(payload)) + payload


def _end(error: dict | None = None) -> bytes:
    return _envelope(_END_STREAM, json.dumps({} if error is None else {"error": error}).encode())


def app(
    model: Model,
    manifest: list[dict],
    files: dict[str, tuple[bytes, str]],
    *,
    backend: Backend | None = None,
    capacity: int = 64,
    batch_size: int = 1,
) -> Starlette:
    """The measures' services, the manifest, and ``files``: each published path's bytes and media type."""
    worker = Worker(backend if backend is not None else LocalBackend(model.infer), capacity=capacity, batch_size=batch_size)

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
                handle, edges = await model.expand(measure, child, request_message.horizon, worker)
            except UnknownNode as error:
                return _stream([_end({"code": "not_found", "message": str(error)})])
            except BadLabel as error:
                return _stream([_end({"code": "invalid_argument", "message": str(error)})])
            except lm.TooLong as error:
                return _stream([_end({"code": "out_of_range", "message": str(error)})])
            except Busy as error:
                return _stream([_end({"code": "resource_exhausted", "message": str(error)})])
            except Closed as error:
                return _stream([_end({"code": "unavailable", "message": str(error)})])
            except Exception as error:
                # Answered as Connect, inside CORS, so the page sees why.
                traceback.print_exc()
                return _stream([_end({"code": "internal", "message": f"{type(error).__name__}: {error}"})])

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

    def file(content: bytes, media_type: str):
        async def serve(_: Request) -> Response:
            return Response(content, media_type=media_type)

        return serve

    # v2 uses the same model/path semantics, with independent opaque handles.
    from quantile import Service as QuantileService, RPC as QUANTILE_RPC

    async def distribution(measure, path):
        if model.ends(path):
            return ()
        return await worker.infer(Job(measure, path))

    quantile_services = [
        QuantileService(lambda path, m=m: distribution(m, path), log_weights=True) for m in range(len(model.measures))
    ]

    def quantile_endpoint(index):
        async def query(request):
            if request.headers.get("content-type", "").split(";")[0] != "application/proto":
                return Response(status_code=415)
            body = await request.body()
            if len(body) > 4096:
                return JSONResponse({"code": "resource_exhausted", "message": "request too large"}, status_code=429)
            try:
                result = await quantile_services[index].query(body)
                return Response(result, media_type="application/proto")
            except lm.TooLong as error:
                return JSONResponse({"code": "out_of_range", "message": str(error)}, status_code=400)
            except KeyError:
                return JSONResponse({"code": "not_found", "message": "unknown parent handle"}, status_code=404)
            except Busy as error:
                return JSONResponse({"code": "resource_exhausted", "message": str(error)}, status_code=429)
            except Closed as error:
                return JSONResponse({"code": "unavailable", "message": str(error)}, status_code=503)
            except (ValueError, OverflowError) as error:
                exhausted = isinstance(error, OverflowError)
                return JSONResponse(
                    {"code": "resource_exhausted" if exhausted else "invalid_argument", "message": str(error)},
                    status_code=429 if exhausted else 400,
                )
            except Exception as error:
                traceback.print_exc()
                return JSONResponse(
                    {"code": "internal", "message": f"{type(error).__name__}: {error}"}, status_code=500
                )

        return query

    routes = [Route("/manifest.json", manifest_json)]
    routes += [Route(f"/{path}", file(*served)) for path, served in files.items()]
    routes += [
        Route(f"/{index}/natlas.v1.Natlas/Expand", service(index), methods=["POST"])
        for index in range(len(model.measures))
    ]
    routes += [
        Route(f"/{index}{QUANTILE_RPC}", quantile_endpoint(index), methods=["POST"])
        for index in range(len(model.measures))
    ]
    # Any page may read a local research server.
    cors = Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @asynccontextmanager
    async def lifespan(_: Starlette):
        worker.start()
        try:
            yield
        finally:
            # Reject queued inference, finish the active batch, then settle any
            # detached cache builders before releasing the application's loop.
            await worker.aclose()
            await asyncio.gather(model.aclose(), *(service.aclose() for service in quantile_services))

    application = Starlette(routes=routes, middleware=[cors], lifespan=lifespan)
    application.state.inference = worker
    return application


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
    parser.add_argument(
        "--p", default="HuggingFaceTB/SmolLM2-360M", help="the model: a Hugging Face id or a .gguf file"
    )
    parser.add_argument(
        "--q",
        action="append",
        default=[],
        help="further measure, repeatable: a model id sharing p's tokenizer, context:N, or temperature:T",
    )
    parser.add_argument("--tokenizer", help="default: p's own")
    parser.add_argument("--device", default=lm.default_device())
    parser.add_argument("--dtype", choices=lm.DTYPES, default="float32")
    parser.add_argument("--n-ctx", type=int, default=8192, help="llama.cpp: tokens of context, the deepest node")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="llama.cpp: layers on the GPU, -1 for all")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()

    print(f"loading {args.p} …", file=sys.stderr)

    def open_language(spec: str) -> lm.Language:
        llama = {"context": args.n_ctx, "gpu_layers": args.gpu_layers}
        return lm.open_language(spec, args.device, args.dtype, args.tokenizer if spec == args.p else None, **llama)

    language = open_language(args.p)
    prompt = language.encode(args.prompt)
    measures = lm.measures(language, args.q, open_language)
    labels = labels_json(language.token_bytes())

    manifest = [
        {
            "name": f"live-{_slug(language.name.split('/')[-1])}",
            "title": f"{language.name} · live",
            "labels": "labels.json",
            "start": prompt,
            "measures": [
                {"name": measure.name, "url": f"{index}/", "protocol": "quantile-v2"}
                for index, measure in enumerate(measures)
            ],
        }
    ]
    model = Model(language.start, lambda path: bool(path) and path[-1] in language.eos, language.vocabulary, measures)
    files = {"labels.json": (labels.encode(), "application/json")}
    uvicorn.run(app(model, manifest, files), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
