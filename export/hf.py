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
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from natlas_export import snapshots, write
from natwalk import Session
from transformers import AutoModelForCausalLM, AutoTokenizer

_HF_APP = Path(__file__).resolve().parents[2] / "natwalk" / "examples" / "hf_app.py"


def _hf_app():
    spec = importlib.util.spec_from_file_location("hf_app", _HF_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _load(model_id: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device).eval()


def _probabilities(model, tokens: Sequence[int], device: str, temperature: float = 1.0) -> np.ndarray:
    input_ids = torch.tensor([list(tokens)], device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].to(torch.float64)
    return torch.softmax(logits / temperature, dim=-1).cpu().numpy()


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", help="dataset file stem; default derives from the measures")
    args = parser.parse_args()
    specs = args.q or ["Qwen/Qwen3-0.6B"]

    hf_app = _hf_app()
    device = args.device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt = tuple(int(token) for token in tokenizer.encode(args.prompt))
    eos = (int(tokenizer.eos_token_id),)

    print(f"discovering {args.nodes} nodes with {args.p} …", file=sys.stderr)
    p_model = _load(args.p, device, dtype)
    cursor = hf_app.HFCursor(
        p_model,
        prompt,
        eos_token_ids=eos,
        terminal=False,
        device=device,
        context_length=None,
    )
    session = Session(cursor)
    while len(session.tree.nodes) < args.nodes and session.search.frontier:
        session.search.discover()
    tree = session.tree

    def measure(spec: str) -> tuple[str, Callable[[Sequence[int]], np.ndarray]]:
        kind, _, value = spec.partition(":")
        if kind == "context":
            n = int(value)
            return f"{args.p} · last {n} tokens", lambda tokens: _probabilities(
                p_model, tokens[-n:], device
            )
        if kind == "temperature":
            t = float(value)
            return f"{args.p} · T={t:g}", lambda tokens: _probabilities(p_model, tokens, device, t)
        model = _load(spec, device, dtype)
        return spec, lambda tokens: _probabilities(model, tokens, device)

    def weights(probabilities: Callable[[Sequence[int]], np.ndarray]):
        def at(node_id: int) -> list[float]:
            return probabilities(prompt + tree.path(node_id)).tolist()

        return at

    print(f"evaluating {', '.join(specs)} on the discovered tree …", file=sys.stderr)
    measures = [measure(spec) for spec in specs]
    recorded = snapshots(tree, extra=tuple(weights(probabilities) for _, probabilities in measures))
    display = hf_app.TokenDisplay(tokenizer)
    vocabulary = max(len(tokenizer), int(p_model.config.vocab_size))
    stem = "-".join(_slug(part.split("/")[-1]) for part in (args.p, *specs))
    write(
        args.name or stem,
        f"{args.prompt} · {args.p}",
        [display(token) for token in range(vocabulary)],
        (args.p, *(name for name, _ in measures)),
        recorded,
    )


if __name__ == "__main__":
    main()
