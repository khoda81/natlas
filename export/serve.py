# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "natlas-core",
#   "gguf>=0.17",
#   "llama-cpp-python>=0.3.36",
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "starlette>=0.40",
#   "torch>=2.2",
#   "transformers>=4.51",
#   "uvicorn>=0.30",
#   "websockets>=13",
# ]
#
# # llama.cpp for the GPU, built once; CMAKE_CUDA_ARCHITECTURES=75 is the GTX 1660 Ti.
# [tool.uv.sources]
# natlas-core = { path = "../core" }
#
# [tool.uv]
# config-settings-package = { llama-cpp-python = { "cmake.define.GGML_CUDA" = "ON", "cmake.define.CMAKE_CUDA_ARCHITECTURES" = "75" } }
# ///

"""Serve language models' integer-sequence distributions to natlas, live.

The root is the empty sequence; a model reads its start-of-text token
internally before that path. Every other node is a complete token path from
the root, and its children are the next tokens. The config file (serve.toml)
names the sources offered, which all share one tokenizer, the control
vectors, and models to offer as they are. Nothing loads until a page asks:
it loads a model, a source on a device at a precision, and reads measures of
it by spec (see lm.py). The browser sends demands and receives updates over
the /session WebSocket. Search uses Connect unary. GET /labels.json holds
the token bytes.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import lm
import uvicorn
from natlas_codec import labels_json
from service import CACHE, Model, app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("serve.toml"),
        help="the sources, control vectors and models to offer (default serve.toml beside this script)",
    )
    parser.add_argument("--cache", type=float, default=CACHE / 2**30, help=f"GiB of memory for the distributions it keeps, default {CACHE / 2**30:g}")
    parser.add_argument(
        "--order",
        choices=["heaviest", "branch"],
        default="heaviest",
        help="how each node's children rank: heaviest first, or by token id",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()
    try:
        config = tomllib.loads(args.config.read_text())
        sources: dict[str, str] = config["sources"]
        if not sources:
            raise ValueError("no sources")
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.error(f"{args.config}: {error!r}")
    here = args.config.parent
    sources = {name: str((here / Path(source).expanduser()).resolve()) if source.endswith(".gguf") else source for name, source in sources.items()}
    controls = {name: (here / Path(path).expanduser()).resolve() for name, path in config.get("controls", {}).items()}
    tokenizer = config.get("tokenizer")
    vocabulary = lm.Vocabulary(tokenizer or next(iter(sources.values())))
    try:
        catalog = lm.catalog(
            sources,
            vocabulary,
            controls,
            config.get("presets", []),
            tokenizer,
            config.get("n_ctx", 8192),
            config.get("gpu_layers", -1),
        )
    except ValueError as error:
        parser.error(f"{args.config}: {error}")
    manifest = [
        {
            "name": f"live-{args.config.stem}",
            "title": f"{', '.join(sources)} · live",
            "labels": "labels.json",
            "catalog": catalog.describe(),
        }
    ]
    model = Model(vocabulary.size, catalog, round(args.cache * 2**30), args.order)
    files = {"labels.json": (labels_json(vocabulary.token_bytes()).encode(), "application/json")}
    uvicorn.run(app(model, manifest, files, search=vocabulary.encode), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
