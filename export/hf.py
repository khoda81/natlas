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

"""Export a language-model tree: natwalk discovers it with model p, model q is evaluated on it.

Both models must share one tokenizer. Every discovered node retains a complete
vocabulary distribution in memory, so keep --nodes moderate.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from natlas_export import snapshot, write
from natwalk import Session
from transformers import AutoModelForCausalLM, AutoTokenizer

_HF_APP = Path(__file__).resolve().parents[2] / "natwalk" / "examples" / "hf_app.py"


def _hf_app():
    spec = importlib.util.spec_from_file_location("hf_app", _HF_APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(model_id: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device).eval()


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
    parser.add_argument("--q", default="Qwen/Qwen3-0.6B", help="model evaluated on the same tree")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", help="dataset file stem; default derives from the models")
    args = parser.parse_args()

    hf_app = _hf_app()
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.q)
    prompt = tuple(int(token) for token in tokenizer.encode(args.prompt))
    eos = (int(tokenizer.eos_token_id),)

    print(f"discovering {args.nodes} nodes with {args.p} …", file=sys.stderr)
    p_model = _load(args.p, args.device, dtype)
    cursor = hf_app.HFCursor(
        p_model,
        prompt,
        eos_token_ids=eos,
        terminal=False,
        device=args.device,
        context_length=None,
    )
    session = Session(cursor)
    while len(session.tree.nodes) < args.nodes and session.search.frontier:
        session.search.discover()
    tree = session.tree
    del session, cursor, p_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"evaluating {args.q} on the discovered tree …", file=sys.stderr)
    q_model = _load(args.q, args.device, dtype)

    def q_weights(node_id: int, tokens: Sequence[int]) -> tuple[np.ndarray, float]:
        input_ids = torch.tensor([prompt + tree.path(node_id)], device=args.device)
        with torch.inference_mode():
            logits = q_model(input_ids=input_ids, use_cache=False).logits[0, -1]
        probabilities = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        rest = np.ones(len(probabilities), dtype=bool)
        rest[list(tokens)] = False
        return probabilities[list(tokens)], float(np.sum(probabilities[rest]))

    display = hf_app.TokenDisplay(tokenizer)
    title = f"{args.prompt} · {args.p} vs {args.q}"
    data = snapshot(
        tree,
        title=title,
        measures=(args.p, args.q),
        label=display,
        extra=(q_weights,),
    )
    name = args.name or f"{_slug(args.p.split('/')[-1])}-vs-{_slug(args.q.split('/')[-1])}"
    write(name, title, data)


if __name__ == "__main__":
    main()
