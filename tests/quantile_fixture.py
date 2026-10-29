"""Serve the real Natlas app with recorded Qwen inference and a built frontend.

python tests/quantile_fixture.py --samples ZIP --static DIR [--port 8423]
"""

import argparse
import io
from pathlib import Path
import sys
import types
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
sys.modules["lm"] = types.SimpleNamespace(TooLong=type("TooLong", (Exception,), {}))
from serve import Model, app
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse
import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--port", type=int, default=8423)
    args = parser.parse_args()
    records = {}
    with zipfile.ZipFile(args.samples) as archive:
        for name in archive.namelist():
            if name.endswith(".npz"):
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                    records[tuple(map(int, data["tokens"]))] = data["logp"].tolist()
    origin = min(records, key=len)

    class Measure:
        def __init__(self, temperature):
            self.temperature = temperature

        def log_probs(self, tokens):
            logs = [value / self.temperature for value in records[tokens]]
            return types.SimpleNamespace(tolist=lambda: logs)

    model = Model(origin, lambda path: origin + path not in records, len(records[origin]), [Measure(1), Measure(1.5)])
    measures = [
        {"name": name, "url": f"{i}/", "protocol": "quantile-v2"}
        for i, name in enumerate(("Qwen recording", "temperature 1.5"))
    ]
    manifest = [
        {"name": "qwen-v2", "title": "Qwen3-4B-Q6_K · recorded", "start": [], "measures": measures},
        {"name": "qwen-prompt", "title": "Qwen prompt path", "start": [785, 1429, 7040], "measures": measures},
    ]
    application = app(model, manifest, {})
    application.state.calls = 0

    async def track(request, call_next):
        if request.url.path.endswith("/Quantile"):
            application.state.calls += 1
        return await call_next(request)

    async def state(request):
        return JSONResponse({"calls": application.state.calls})

    from starlette.middleware.base import BaseHTTPMiddleware

    application.add_middleware(BaseHTTPMiddleware, dispatch=track)
    application.routes.append(Route("/_test/state", state))
    application.routes.append(Mount("/", StaticFiles(directory=args.static, html=True)))
    uvicorn.run(application, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
