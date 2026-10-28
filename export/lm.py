"""Language-model measures, shared by the exporter and the server.

A measure is a model's next-token log-probabilities over the whole
vocabulary, given the tokens so far. Besides the model itself, a measure spec
names another model sharing its tokenizer, ``context:N`` for the model seeing
only the last N tokens, or ``temperature:T`` for the model at temperature T.

A model is a Hugging Face id, run by transformers, or a .gguf file, run by
llama.cpp.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

type LogProbs = Callable[[Sequence[int]], torch.Tensor]


@dataclass(frozen=True)
class Measure:
    name: str
    log_probs: LogProbs


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Half precision saves memory on large models but can be slower: a GTX 1660 Ti
# runs SmolLM2-360M about 5x faster in float32.
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load(model_id: str, device: str, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES[dtype])
    return model.to(device).eval()


def log_probs(model, tokens: Sequence[int], device: str, temperature: float = 1.0) -> torch.Tensor:
    """float64 log-probabilities of the next token, on ``device``.

    Normalizing in float64 keeps -ln p precise where p is close to 1.
    """
    input_ids = torch.tensor([list(tokens)], device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].to(torch.float64)
    return torch.log_softmax(logits / temperature, dim=-1)


class TooLong(Exception):
    """More tokens than the model's context holds."""


class Language(Protocol):
    """A model and the tokenizer it reads."""

    name: str
    # Labels the model can emit.
    vocabulary: int
    # What the model reads before a text's first token.
    start: tuple[int, ...]
    # Tokens that end a text.
    eos: frozenset[int]

    def log_probs(self, tokens: Sequence[int], temperature: float = 1.0) -> torch.Tensor: ...
    def encode(self, text: str) -> list[int]: ...
    def token_bytes(self) -> list[bytes]: ...


class Transformers:
    def __init__(self, model_id: str, device: str, dtype: str = "float32", tokenizer: str | None = None):
        from transformers import AutoTokenizer

        self.name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or model_id)
        self.model = load(model_id, device, dtype)
        self.device = device
        self.vocabulary = vocabulary_size(self.tokenizer, self.model)
        self.start = start_of_text(self.tokenizer)
        eos = self.tokenizer.eos_token_id
        self.eos = frozenset() if eos is None else frozenset({int(eos)})

    def log_probs(self, tokens: Sequence[int], temperature: float = 1.0) -> torch.Tensor:
        return log_probs(self.model, tokens, self.device, temperature)

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)]

    def token_bytes(self) -> list[bytes]:
        return token_bytes(self.tokenizer, self.vocabulary)


class LlamaCpp:
    """A GGUF model run by llama.cpp, its layers on the GPU as far as ``gpu_layers``
    says (-1 for all). Requests that share a prefix with the last one reuse
    its KV cache, so walking down a path costs a token per step. Every token
    is evaluated alone, so a node's log-probabilities depend on its path only."""

    def __init__(self, path: Path, context: int = 8192, gpu_layers: int = -1):
        from llama_cpp import Llama

        self.name = path.stem
        try:
            self.llm = Llama(
                str(path),
                n_ctx=context,
                n_gpu_layers=gpu_layers,
                # With no layers on the GPU, keep the KV cache off it too.
                offload_kqv=gpu_layers != 0,
                logits_all=False,
                verbose=False,
            )
        except ValueError as error:
            # llama.cpp says why only in its own log, and a full GPU is the usual reason.
            raise RuntimeError(
                f"llama.cpp could not load {path}: is the GPU's memory taken (nvidia-smi)? "
                "Fewer --gpu-layers or a smaller --n-ctx need less of it."
            ) from error
        self.vocabulary = self.llm.n_vocab()
        bos, eos = self.llm.token_bos(), self.llm.token_eos()
        if bos < 0 and eos < 0:
            raise ValueError(f"{path} has no start- or end-of-text token to begin a text with")
        self.start = (bos if bos >= 0 else eos,)
        self.eos = frozenset({eos} if eos >= 0 else ())

    def log_probs(self, tokens: Sequence[int], temperature: float = 1.0) -> torch.Tensor:
        tokens = [int(token) for token in tokens]
        if len(tokens) > self.llm.n_ctx():
            raise TooLong(f"{len(tokens)} tokens, but the context holds {self.llm.n_ctx()}")
        # input_ids is the whole context buffer: past n_tokens it holds tokens
        # already dropped from the KV cache.
        cached = self.llm.input_ids[: self.llm.n_tokens]
        shared = len(tokens)
        for index, (have, want) in enumerate(zip(cached, tokens)):
            if have != want:
                shared = index
                break
        else:
            shared = min(len(cached), len(tokens))
        # The logits on hand are the last evaluated position's.
        if not shared == len(tokens) == len(cached):
            self.llm.n_tokens = min(shared, len(tokens) - 1)
            try:
                # One token at a time: llama.cpp's kernels round a batch
                # differently, so a node's numbers would hang on what was
                # asked before it.
                for token in tokens[self.llm.n_tokens :]:
                    self.llm.eval([token])
            except Exception:
                # What a failed batch left in the cache is unknown.
                self.llm.reset()
                raise
        logits = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.vocabulary,))  # noqa: SLF001
        return torch.log_softmax(torch.from_numpy(logits.astype(np.float64)) / temperature, dim=-1)

    def encode(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=False, special=False)

    def token_bytes(self) -> list[bytes]:
        return [self.llm.detokenize([token], special=True) for token in range(self.vocabulary)]


def open_language(spec: str, device: str, dtype: str = "float32", tokenizer: str | None = None, **llama) -> Language:
    """A .gguf file through llama.cpp, taking ``llama``'s options, else a Hugging Face model."""
    if spec.endswith(".gguf"):
        return LlamaCpp(Path(spec).expanduser(), **llama)
    return Transformers(spec, device, dtype, tokenizer)


def measures(own: Language, specs: Sequence[str], other: Callable[[str], Language]) -> list[Measure]:
    """The model itself, then one measure per spec; ``other`` opens a model a spec names."""

    def measure(spec: str) -> Measure:
        kind, _, value = spec.partition(":")
        if kind == "context":
            n = int(value)
            return Measure(f"{own.name} · last {n} tokens", lambda tokens: own.log_probs(tokens[-n:]))
        if kind == "temperature":
            t = float(value)
            return Measure(f"{own.name} · T={t:g}", lambda tokens: own.log_probs(tokens, t))
        language = other(spec)
        return Measure(language.name, language.log_probs)

    return [Measure(own.name, own.log_probs), *(measure(spec) for spec in specs)]


def start_of_text(tokenizer) -> tuple[int, ...]:
    """What a model reads before a text's first token: its beginning-of-text
    token, or its end-of-text token where it separates documents with that."""
    token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if token is None:
        raise ValueError("the tokenizer has no start- or end-of-text token to begin a text with")
    return (int(token),)


def vocabulary_size(tokenizer, model) -> int:
    """Labels a model can emit: its logits can outnumber its tokenizer's tokens."""
    return max(len(tokenizer), int(model.config.vocab_size))


def _byte_level_alphabet() -> dict[str, int]:
    """GPT-2's byte-level BPE writes each byte as one printable character."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    shifted = [byte for byte in range(256) if byte not in printable]
    return {
        **{chr(byte): byte for byte in printable},
        **{chr(256 + index): byte for index, byte in enumerate(shifted)},
    }


def _decoders(tokenizer) -> set[str]:
    decoder = json.loads(tokenizer.backend_tokenizer.to_str()).get("decoder") or {}
    return {decoder.get("type"), *(step.get("type") for step in decoder.get("decoders", []))}


def token_bytes(tokenizer, size: int) -> list[bytes]:
    """The bytes of every token id below ``size``.

    A token can hold part of a UTF-8 character, which decoding it alone turns
    into U+FFFD, so this reads the bytes from the token's piece: through the
    byte-level alphabet, or SentencePiece's ▁ and <0xNN> byte fallback.
    Special tokens are their text; anything else is decoded alone.
    """
    kinds = _decoders(tokenizer)
    alphabet = _byte_level_alphabet() if "ByteLevel" in kinds else None
    special = {int(token) for token in tokenizer.all_special_ids} | {
        int(token) for token in tokenizer.added_tokens_decoder
    }
    fallback = re.compile(r"<0x([0-9A-Fa-f]{2})>")
    result = []
    for token in range(size):
        piece = tokenizer.convert_ids_to_tokens(token) if token < len(tokenizer) else None
        if piece is None:
            result.append(f"<token:{token}>".encode())
        elif token in special:
            result.append(piece.encode())
        elif alphabet is not None and all(char in alphabet for char in piece):
            result.append(bytes(alphabet[char] for char in piece))
        elif "ByteFallback" in kinds and (byte := fallback.fullmatch(piece)):
            result.append(bytes([int(byte.group(1), 16)]))
        elif "Metaspace" in kinds or "Replace" in kinds:
            result.append(piece.replace("▁", " ").encode())
        else:
            result.append(tokenizer.decode([token], clean_up_tokenization_spaces=False).encode())
    return result

