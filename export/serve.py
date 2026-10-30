# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "natlas-core",
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
# [tool.uv.sources]
# natlas-core = { path = "../core" }
#
# [tool.uv]
# config-settings-package = { llama-cpp-python = { "cmake.define.GGML_CUDA" = "ON", "cmake.define.CMAKE_CUDA_ARCHITECTURES" = "75" } }
# ///

"""Serve a language model's integer-sequence distribution to natlas, live.

The root is the empty label path; the model reads its start-of-text token
internally before that path. Every other node is a complete token path from
the root, and its children are the next tokens. Every measure is one Natlas
service under its own URL prefix, /0/ for the model itself and /1/, /2/, …
for the --q measures. Expand uses Connect server streaming and Search uses
Connect unary. GET /manifest.json describes the measures and GET /labels.json
holds the token bytes.
"""

from __future__ import annotations

import argparse
import sys

import lm
import uvicorn
from natlas_codec import labels_json
from service import Model, _slug, app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", default="HuggingFaceTB/SmolLM2-360M", help="the model: a Hugging Face id or a .gguf file")
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
    measures = lm.measures(language, args.q, open_language)
    labels = labels_json(language.token_bytes())

    manifest = [
        {
            "name": f"live-{_slug(language.name.split('/')[-1])}",
            "title": f"{language.name} · live",
            "labels": "labels.json",
            "measures": [{"name": measure.name, "url": f"{index}/"} for index, measure in enumerate(measures)],
        }
    ]
    model = Model(language.start, lambda path: bool(path) and path[-1] in language.eos, language.vocabulary, measures)
    files = {"labels.json": (labels.encode(), "application/json")}
    uvicorn.run(
        app(model, manifest, files, search=language.encode), host=args.host, port=args.port, access_log=False
    )


if __name__ == "__main__":
    main()
