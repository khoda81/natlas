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
the root, and its children are the next tokens. A measure is named by a spec
(see lm.py) the page builds from the catalog in GET /manifest.json: any
--model, which all share the first one's tokenizer, at any temperature or
context, steered by any --control vector. The browser sends demands and
receives updates over the /session WebSocket; a measure opens the first time
one names it. Search uses Connect unary. GET /labels.json holds the token
bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lm
import uvicorn
from natlas_codec import labels_json
from service import CACHE, Model, _slug, app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        action="append",
        help="a model offered, repeatable: a Hugging Face id or a .gguf file; the first is opened at once "
        "and its tokenizer is every one's (default HuggingFaceTB/SmolLM2-360M)",
    )
    parser.add_argument("--control", action="append", default=[], type=Path, help="a llama.cpp control vector offered, repeatable")
    parser.add_argument("--tokenizer", help="the first model's tokenizer, where not its own")
    parser.add_argument("--device", default=lm.default_device())
    parser.add_argument("--dtype", choices=lm.DTYPES, default="float32")
    parser.add_argument("--n-ctx", type=int, default=8192, help="llama.cpp: tokens of context, the deepest node")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="llama.cpp: layers on the GPU, -1 for all")
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
    models = args.model or ["HuggingFaceTB/SmolLM2-360M"]

    def open_language(model: str) -> lm.Language:
        print(f"loading {model} …", file=sys.stderr)
        llama = {"context": args.n_ctx, "gpu_layers": args.gpu_layers}
        language = lm.open_language(model, args.device, args.dtype, args.tokenizer if model == models[0] else None, **llama)
        print(f"{model}: {language.width} distributions a pass, of {language.paths} paths at most", file=sys.stderr)
        return language

    try:
        languages = lm.Languages(models, open_language)
    except ValueError as error:
        parser.error(str(error))
    catalog = lm.catalog(languages, args.control)
    language = languages.first
    manifest = [
        {
            "name": f"live-{_slug(lm.model_name(models[0]))}",
            "title": f"{', '.join(languages.models)} · live",
            "labels": "labels.json",
            "catalog": catalog.describe(),
        }
    ]
    model = Model(language.start, language.vocabulary, catalog.open, round(args.cache * 2**30), args.order)
    files = {"labels.json": (labels_json(language.token_bytes()).encode(), "application/json")}
    uvicorn.run(app(model, manifest, files, search=language.encode), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
