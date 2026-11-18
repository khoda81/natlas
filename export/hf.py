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

"""Export a language-model tree: natwalk discovers it with p, the q measures follow.

p and each q are measure specs (see lm.py) of the sources a config names, as
serve.py reads it: ``Qwen3-1.7B/cuda/float32`` or
``Qwen3-0.6B/cuda/float32/temperature=1.6``. p's source is a Hugging Face
model, which natwalk searches with. Every discovered node retains a complete
vocabulary distribution in memory, so keep --nodes moderate.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import tomllib
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
    parser.add_argument("--config", type=Path, required=True, help="the sources, as serve.py reads them")
    parser.add_argument("--p", required=True, help="the measure natwalk searches with")
    parser.add_argument("--q", action="append", default=[], help="q measure's spec, repeatable")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--name", help="dataset file stem; default derives from the measures")
    args = parser.parse_args()

    hf_app = _hf_app()
    config = tomllib.loads(args.config.read_text())
    sources = config["sources"]
    tokenizer = config.get("tokenizer")
    vocabulary = lm.Vocabulary(tokenizer or next(iter(sources.values())))
    catalog = lm.catalog(sources, vocabulary, {}, tokenizer=tokenizer)
    names = [args.p, *args.q]
    loaded = {}
    for spec in names:
        model, _ = catalog.parse(spec)
        if model not in loaded:
            loaded[model] = catalog.load(model)
    measures = [catalog.open(loaded[catalog.parse(spec)[0]], spec) for spec in names]
    language = loaded[catalog.parse(args.p)[0]]
    if not isinstance(language, lm.Transformers):
        parser.error("natwalk searches with a Hugging Face model")
    device = str(language.device)
    # The tree's root is the empty text; natwalk searches from the prompt down.
    root = language.start
    prompt = tuple(language.encode(args.prompt))
    eos = tuple(language.eos)

    print(f"discovering {args.nodes} nodes with {args.p} …", file=sys.stderr)
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

    print(f"evaluating {', '.join(args.q)} on the discovered tree …", file=sys.stderr)
    recorded = snapshots(
        tree,
        extra=tuple(weights(measure) for measure in measures[1:]),
        start=prompt,
        ancestry=tuple(along(measure) for measure in measures),
    )
    write(
        args.name or "-".join(_slug(name) for name in names),
        f"{args.prompt} · {args.p}",
        language.token_bytes(),
        names,
        recorded,
    )


if __name__ == "__main__":
    main()
