# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "natlas-core",
#   "natwalk",
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "torch>=2.2",
#   "transformers>=4.51",
# ]
#
# [tool.uv.sources]
# natlas-core = { path = "../core" }
# natwalk = { path = "../../natwalk", editable = true }
# ///

"""Export a language-model tree: natwalk discovers it with the first model, the q measures follow.

A q measure is a spec (see lm.py) of any --model, all sharing the first
one's tokenizer: ``Qwen3-0.6B`` or ``Qwen3-1.7B/temperature=1.6``. Every
discovered node retains a complete vocabulary distribution in memory, so keep
--nodes moderate.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import lm
import numpy as np
from natlas_export import snapshots, write
from natwalk import Session

_HF_APP = Path(__file__).resolve().parents[2] / "natwalk" / "examples" / "hf_app.py"


def _hf_app():
    spec = importlib.util.spec_from_file_location("hf_app", _HF_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The most interesting consequence of information theory is",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Hugging Face model, repeatable; natwalk searches with the first (default Qwen/Qwen3-1.7B and Qwen/Qwen3-0.6B)",
    )
    parser.add_argument("--q", action="append", help="q measure's spec, repeatable (default each other model)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="tokenizer shared by all models")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--device", default=lm.default_device())
    parser.add_argument("--dtype", choices=lm.DTYPES, default="float32")
    parser.add_argument("--name", help="dataset file stem; default derives from the measures")
    args = parser.parse_args()
    models = args.model or ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"]
    specs = args.q or [lm.model_name(model) for model in models[1:]]

    hf_app = _hf_app()
    device = args.device
    languages = lm.Languages(models, lambda model: lm.Transformers(model, device, args.dtype, tokenizer=args.tokenizer))
    language = languages.first
    catalog = lm.catalog(languages)
    # The tree's root is the empty text; natwalk searches from the prompt down.
    root = language.start
    prompt = tuple(language.encode(args.prompt))
    eos = tuple(language.eos)

    print(f"discovering {args.nodes} nodes with {models[0]} …", file=sys.stderr)
    cursor = hf_app.HFCursor(
        language.model,
        root + prompt,
        eos_token_ids=eos,
        terminal=False,
        device=device,
        context_length=None,
    )
    session = Session(cursor)
    while len(session.tree.nodes) < args.nodes and session.search.frontier:
        session.search.discover()
    tree = session.tree

    def weights(measure: lm.Measure):
        def at(node_id: int) -> list[float]:
            return np.exp(measure.log_probs(root + prompt + tree.path(node_id))).tolist()

        return at

    def along(measure: lm.Measure):
        return lambda path: np.exp(measure.log_probs(root + tuple(path))).tolist()

    print(f"evaluating {', '.join(specs)} on the discovered tree …", file=sys.stderr)
    names = [catalog.start, *specs]
    measures = [catalog.open(spec) for spec in names]
    recorded = snapshots(
        tree,
        extra=tuple(weights(measure) for measure in measures[1:]),
        start=prompt,
        ancestry=tuple(along(measure) for measure in measures),
    )
    write(
        args.name or "-".join(_slug(name) for name in names),
        f"{args.prompt} · {lm.model_name(models[0])}",
        language.token_bytes(),
        names,
        recorded,
    )


if __name__ == "__main__":
    main()
