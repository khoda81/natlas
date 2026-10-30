"""Capture two real next-token distributions for the categorical playground.

Run with the same Python environment as export/serve.py. The second GGUF is
re-quantized from the first; this is explicitly recorded, not represented as
an independent quantization of the original unquantized checkpoint.
"""

import argparse
import ctypes
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from llama_cpp import Llama, llama_cpp


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requantized", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("public/playground/qwen3-4b"))
    parser.add_argument("--prompt", default="A person that is hired to handle onboarding and hiring is")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.requantized.parent.mkdir(parents=True, exist_ok=True)
    if not args.requantized.exists():
        print("Re-quantizing cached Q6_K weights to Q4_K_M (CPU)", flush=True)
        params = llama_cpp.llama_model_quantize_default_params()
        params.nthread = args.threads
        params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
        params.allow_requantize = True
        params.max_buf_size = 256 * 1024 * 1024
        result = llama_cpp.llama_model_quantize(bytes(args.model), bytes(args.requantized), ctypes.byref(params))
        if result:
            raise RuntimeError(f"quantization failed: {result}")
    measures = []
    token_ids = None
    vocab_digest = None
    for index, (name, path) in enumerate((("Q6_K", args.model), ("Q4_K_M · from Q6_K", args.requantized))):
        started = time.monotonic()
        print(f"Capture {index + 1}/2: {name}; one prompt evaluation, no generation", flush=True)
        model = Llama(str(path), n_ctx=128, n_batch=64, n_threads=args.threads,
                      n_threads_batch=args.threads, n_gpu_layers=0, offload_kqv=False,
                      logits_all=False, verbose=False)
        ids = model.tokenize(args.prompt.encode(), add_bos=False, special=False)
        start = model.token_bos() if model.token_bos() >= 0 else model.token_eos()
        ids = [start, *ids]
        if token_ids is not None and ids != token_ids:
            raise RuntimeError("tokenizers disagree on the prompt")
        token_ids = ids
        model.eval(ids)
        logits = np.ctypeslib.as_array(model._ctx.get_logits(), shape=(model.n_vocab(),)).astype(np.float64)
        shifted = logits - logits.max()
        logp = shifted - np.log(np.exp(shifted).sum())
        filename = f"{index}.f64.gz"
        (args.output / filename).write_bytes(gzip.compress(logp.astype("<f8").tobytes(), mtime=0))
        labels = [model.detokenize([i], special=True).decode("utf-8", errors="replace") for i in range(model.n_vocab())]
        encoded = json.dumps(labels, ensure_ascii=True, separators=(",", ":")).encode()
        current_digest = hashlib.sha256(encoded).hexdigest()
        if vocab_digest is not None and current_digest != vocab_digest:
            raise RuntimeError("vocabularies differ")
        vocab_digest = current_digest
        if index == 0:
            (args.output / "labels.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
        measures.append({"name": f"Qwen3-4B {name}", "file": filename,
                         "sha256": digest(args.output / filename), "model_sha256": digest(path),
                         "capture_seconds": round(time.monotonic() - started, 3)})
        model.close()
        del model
        gc.collect()
        print(f"Captured {len(logp):,} probabilities in {measures[-1]['capture_seconds']} s", flush=True)
    metadata = {"title": "Qwen3-4B · Q6_K / Q4_K_M", "prompt": args.prompt,
                "tokens": token_ids, "vocabulary": len(logp), "vocabulary_sha256": vocab_digest,
                "labels": "labels.json.gz", "measures": measures,
                "source": "Qwen/Qwen3-4B-GGUF", "source_revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
                "note": "Q4_K_M was re-quantized from the cached Q6_K GGUF, not from FP16. Both captures use the same prompt, tokenizer, CPU backend, batch, and temperature 1. These are model-weight quantizations, not rounded output probabilities.",
                "format": "gzip-compressed little-endian float64 normalized log probabilities in token-ID order"}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
