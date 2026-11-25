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

Each server instance serves one fixed vocabulary (known at startup from
``tokenizer``, either a Hugging Face repo id or a local .gguf file) and one
fixed branch ordering. Models and measures are loaded and configured live from
the browser through the catalog schema — no TOML config file required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lm
import uvicorn
from natlas_codec import labels_json
from service import CACHE, Model, app


def _discover_ggufs(tokenizer: str, dirs: list[Path]) -> list[str]:
    suggestions: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        if item not in seen:
            seen.add(item)
            suggestions.append(item)

    if lm.is_gguf(tokenizer):
        resolved = Path(tokenizer).expanduser().resolve()
        add(str(resolved))
        dirs = [resolved.parent, *dirs]
    else:
        add(tokenizer)
    for directory in dirs:
        root = directory.expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.gguf")):
            add(str(path))
    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "tokenizer",
        help="HF model/tokenizer id or path to a .gguf file defining this server's fixed vocabulary",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        action="append",
        default=[],
        help="optional directory to scan for .gguf autocomplete suggestions (repeatable)",
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

    is_local_gguf = lm.is_gguf(args.tokenizer)
    tokenizer_spec = str(Path(args.tokenizer).expanduser()) if is_local_gguf else args.tokenizer
    vocabulary = lm.Vocabulary(tokenizer_spec)
    suggestions = _discover_ggufs(tokenizer_spec, args.models_dir)
    hf_tokenizer = None if is_local_gguf else tokenizer_spec
    catalog = lm.catalog(vocabulary, suggestions=suggestions, tokenizer=hf_tokenizer)

    label = Path(tokenizer_spec).stem if tokenizer_spec.endswith(".gguf") else tokenizer_spec.rsplit("/", 1)[-1]
    manifest = [
        {
            "name": f"live-{label}",
            "title": f"{label} · live",
            "labels": "labels.json",
            "catalog": catalog.describe(),
        }
    ]
    model = Model(vocabulary.size, catalog, round(args.cache * 2**30), args.order)
    files = {"labels.json": (labels_json(vocabulary.token_bytes()).encode(), "application/json")}
    uvicorn.run(app(model, manifest, files, search=vocabulary.encode), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
