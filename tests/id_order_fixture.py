"""Deterministic UI fixture for ID-ordered quantiles; no model or GPU needed.

python tests/id_order_fixture.py --static /tmp/natlas-id-order-build
Optionally run a second instance and use --peer http://127.0.0.1:8424/.
"""

import argparse
import asyncio
from pathlib import Path
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
sys.modules["lm"] = types.SimpleNamespace(TooLong=type("TooLong", (Exception,), {}))
from serve import Model, app
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
import uvicorn


class Measure:
    def __init__(self, variant):
        self.variant = variant

    def log_probs(self, path):
        weights = np.ones(1024)
        if self.variant < 2:
            peaks = [0, 11, 79, 301, 781]
            amounts = [400, 900, 200, 500, 300]
            if self.variant:
                amounts.reverse()
            for token, amount in zip(peaks, amounts):
                weights[(token + sum(path) * 7) % len(weights)] += amount
        return np.log(weights / weights.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--port", type=int, default=8423)
    parser.add_argument("--peer", help="Independent server base URL for Q")
    parser.add_argument("--delay", type=float, default=0)
    args = parser.parse_args()
    model = Model((), lambda path: len(path) >= 3, 1024, [Measure(i) for i in range(3)])
    p = {"name": "P peaks", "url": "0/", "protocol": "quantile-v2"}
    q = {"name": "Q reversed peaks", "url": f"{args.peer.rstrip('/')}/1/" if args.peer else "1/", "protocol": "quantile-v2"}
    uniform = {"name": "uniform 1024", "url": "2/", "protocol": "quantile-v2"}
    manifest = [
        {"name": "id-order", "title": "ID order · deterministic fixture", "measures": [p, q]},
        {"name": "uniform", "title": "Unresolved uniform 1024", "measures": [uniform]},
        {"name": "prompt", "title": "ID order · tiny prompt token", "measures": [p, q], "start": [777]},
    ]
    application = app(model, manifest, {})
    counts = {}

    async def track(request, call_next):
        if request.url.path.endswith("/Quantile"):
            counts[request.url.path] = counts.get(request.url.path, 0) + 1
            if args.delay:
                await asyncio.sleep(args.delay)
        return await call_next(request)

    async def state(request):
        return JSONResponse(counts)

    application.add_middleware(BaseHTTPMiddleware, dispatch=track)
    application.routes.append(Route("/_test/state", state))
    application.routes.append(Mount("/", StaticFiles(directory=args.static, html=True)))
    uvicorn.run(application, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
