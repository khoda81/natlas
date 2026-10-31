# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gguf>=0.17",
#   "numpy>=1.26",
#   "safetensors>=0.6",
#   "torch>=2.2",
#   "transformers>=5.2",
# ]
# ///

"""Write a llama.cpp control vector that steers a model's MLP neurons.

A neuron LxNn with dose d adds d × median(‖residual leaving layer x‖, tokens
1:) / ‖W_down[:, n]‖ to neuron n's input to layer x's down_proj, at every
position (lyra's recipe, x.com/_lyraaaa_/status/2113833330429083662). That
adds d × median‖resid‖ along W_down[:, n] to the residual stream, which is
what a control vector does at the end of layer x: the MLP is the layer's last
addition. Layers count from 0, as in Hugging Face's ``layers.x``.

The medians are measured on --text with the Hugging Face model on the CPU,
skipping token 0, whose residual norm is an outlier. Serve the result with
``serve.py --q control:<file>``, or give it to llama.cpp's --control-vector.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

NEURON = re.compile(r"L(\d+)N(\d+)([+-]\d+(?:\.\d+)?)")
TEXT = (
    "The history of the city begins with a small fishing village on the river. Over the centuries, merchants "
    "built warehouses along the docks, and the town grew into a busy port. Today it is known for its markets, "
    "its old stone bridges, and a university that attracts students from around the world."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face id or snapshot directory with safetensors")
    parser.add_argument("neurons", nargs="+", help="LxNn±dose, e.g. L14N8949+1 L13N7956-1")
    parser.add_argument("--text", default=TEXT, help="what the residual norms are measured on")
    parser.add_argument("--out", type=Path, required=True, help="the control vector .gguf to write")
    args = parser.parse_args()

    neurons = []
    for spec in args.neurons:
        match = NEURON.fullmatch(spec)
        if match is None:
            parser.error(f"{spec!r} is not LxNn±dose")
        neurons.append((int(match[1]), int(match[2]), float(match[3])))

    root = Path(args.model) if Path(args.model).is_dir() else Path(snapshot_download(args.model))
    config = json.loads((root / "config.json").read_text())
    text_config = config.get("text_config", config)
    layers, width = text_config["num_hidden_layers"], text_config["hidden_size"]
    if any(layer == 0 or layers <= layer for layer, _, _ in neurons):
        parser.error(f"layers must be 1 through {layers - 1}: llama.cpp steers no layer 0")

    medians = residual_medians(root, args.text, sorted({layer for layer, _, _ in neurons}))
    directions = np.zeros((layers, width), np.float32)
    for layer, neuron, dose in neurons:
        column = down_proj(root, layer)[:, neuron].astype(np.float64)
        directions[layer] += dose * medians[layer] * column / np.linalg.norm(column)
        print(f"L{layer}N{neuron}{dose:+g}: median ‖resid‖ {medians[layer]:.2f}", file=sys.stderr)

    writer = gguf.GGUFWriter(args.out, "controlvector")
    writer.add_string("controlvector.model_hint", text_config["model_type"])
    writer.add_int32("controlvector.layer_count", layers - 1)
    for layer in range(1, layers):
        writer.add_tensor(f"direction.{layer}", directions[layer])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out}", file=sys.stderr)


def down_proj(root: Path, layer: int) -> np.ndarray:
    weights = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    (name,) = (name for name in weights if name.endswith(f"layers.{layer}.mlp.down_proj.weight"))
    with safe_open(root / weights[name], "pt") as tensors:
        return tensors.get_tensor(name).float().numpy()


def residual_medians(root: Path, text: str, layers: list[int]) -> dict[int, float]:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(root)
    model = AutoModelForImageTextToText.from_pretrained(root, dtype=torch.bfloat16).eval()
    tokens = tokenizer(text, return_tensors="pt").input_ids
    with torch.inference_mode():
        # hidden_states[x + 1] leaves layer x.
        states = model.model.language_model(input_ids=tokens, output_hidden_states=True).hidden_states
    return {layer: states[layer + 1][0, 1:].float().norm(dim=-1).median().item() for layer in layers}


if __name__ == "__main__":
    main()
