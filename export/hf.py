# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "natwalk",
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "torch>=2.2",
#   "transformers>=4.51",
# ]
#
# [tool.uv.sources]
# natwalk = { path = "../../natwalk", editable = true }
# ///

"""Export a language-model tree: natwalk discovers it with model p, the q measures follow.

A q measure is another model sharing p's tokenizer, ``context:N`` for p seeing
only the last N tokens, or ``temperature:T`` for p at temperature T. Every
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
from natlas_export import snapshots, write
from natwalk import Session
from transformers import AutoTokenizer

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
    parser.add_argument("--p", default="Qwen/Qwen3-1.7B", help="model that natwalk searches with")
    parser.add_argument(
        "--q",
        action="append",
        help="q measure, repeatable: a model id, context:N, or temperature:T "
        "(default Qwen/Qwen3-0.6B)",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="tokenizer shared by all models")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--device", default=lm.default_device())
    parser.add_argument("--dtype", choices=lm.DTYPES, default="float32")
    parser.add_argument("--name", help="dataset file stem; default derives from the measures")
    args = parser.parse_args()
    specs = args.q or ["Qwen/Qwen3-0.6B"]

    hf_app = _hf_app()
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # The tree's root is the empty text; natwalk searches from the prompt down.
    root = lm.start_of_text(tokenizer)
    prompt = tuple(int(token) for token in tokenizer.encode(args.prompt, add_special_tokens=False))
    eos = (int(tokenizer.eos_token_id),)

    print(f"discovering {args.nodes} nodes with {args.p} …", file=sys.stderr)
    p_model = lm.load(args.p, device, args.dtype)
    cursor = hf_app.HFCursor(
        p_model,
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
            return measure.log_probs(root + prompt + tree.path(node_id)).exp().tolist()

        return at

    def along(measure: lm.Measure):
        return lambda path: measure.log_probs(root + tuple(path)).exp().tolist()

    print(f"evaluating {', '.join(specs)} on the discovered tree …", file=sys.stderr)
    measures = lm.measures(args.p, p_model, specs, device, args.dtype)
    recorded = snapshots(
        tree,
        extra=tuple(weights(measure) for measure in measures[1:]),
        start=prompt,
        ancestry=tuple(along(measure) for measure in measures),
    )
    stem = "-".join(_slug(part.split("/")[-1]) for part in (args.p, *specs))
    write(
        args.name or stem,
        f"{args.prompt} · {args.p}",
        lm.token_bytes(tokenizer, lm.vocabulary_size(tokenizer, p_model)),
        [measure.name for measure in measures],
        recorded,
        start=prompt,
    )


if __name__ == "__main__":
    main()
