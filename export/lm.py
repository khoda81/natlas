"""Language-model measures, shared by the exporter and the server.

A measure is a model's next-token log-probabilities over the whole
vocabulary, given the tokens so far, as a float64 numpy array; walking a path
gives them after every prefix in turn. Besides the model itself, a measure spec
names another model sharing its tokenizer, ``context:N`` for the model seeing
only the last N tokens, or ``temperature:T`` for the model at temperature T.

A model is a Hugging Face id, run by transformers, or a .gguf file, run by
llama.cpp.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from errors import TooLong

type LogProbs = Callable[[Sequence[int]], np.ndarray]
# The log-probabilities after tokens[:k] for every k from start through
# len(tokens), in order.
type Walk = Callable[[Sequence[int], int], Iterator[np.ndarray]]


@dataclass(frozen=True)
class Measure:
    name: str
    log_probs: LogProbs
    walk: Walk


def stepwise[T](log_probs: Callable[..., T], tokens: Sequence[int], start: int, *args) -> Iterator[T]:
    """Walk a path one prefix at a time: right for a model that reuses the
    cache of the prefix it was last asked, where each step is one token."""
    for end in range(start, len(tokens) + 1):
        yield log_probs(tokens[:end], *args)


def _numpy(log_probs: torch.Tensor) -> np.ndarray:
    return log_probs.cpu().numpy()


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Half precision saves memory on large models but can be slower: a GTX 1660 Ti
# runs SmolLM2-360M about 5x faster in float32.
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load(model_id: str, device: str, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES[dtype])
    return model.to(device).eval()


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
    def walk(self, tokens: Sequence[int], start: int, temperature: float = 1.0) -> Iterator[torch.Tensor]: ...
    def encode(self, text: str) -> list[int]: ...
    def token_bytes(self) -> list[bytes]: ...


class Transformers:
    """A Hugging Face model. It keeps the KV cache of the last path it read, so
    a request that shares a prefix with it feeds only the rest, and it walks a
    path in chunks, reading every position's logits off one pass."""

    # Positions per forward pass while walking: logits for this many fit easily.
    CHUNK = 256

    def __init__(self, model_id: str, device: str, dtype: str = "float32", tokenizer: str | None = None):
        from transformers import AutoTokenizer, DynamicCache

        self.name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or model_id)
        self.model = load(model_id, device, dtype)
        self.device = device
        self.vocabulary = vocabulary_size(self.tokenizer, self.model)
        self.start = start_of_text(self.tokenizer)
        eos = self.tokenizer.eos_token_id
        self.eos = frozenset() if eos is None else frozenset({int(eos)})
        self.new_cache = DynamicCache
        self.cache = DynamicCache()
        # The tokens whose keys and values the cache holds.
        self.fed: list[int] = []

    def log_probs(self, tokens: Sequence[int], temperature: float = 1.0) -> torch.Tensor:
        return next(self.walk(tokens, len(tokens), temperature))

    def walk(self, tokens: Sequence[int], start: int, temperature: float = 1.0) -> Iterator[torch.Tensor]:
        """float64 log-probabilities after tokens[:k] for k from ``start`` on.

        Normalizing in float64 keeps -ln p precise where p is close to 1.
        Positions are computed a chunk at a time, so their last bits can
        differ from one computed alone.
        """
        tokens = [int(token) for token in tokens]
        # The distribution after tokens[:k] is the output at position k - 1,
        # so feeding starts at ``start - 1`` at the latest.
        shared = 0
        for have, want in zip(self.fed, tokens[: start - 1]):
            if have != want:
                break
            shared += 1
        if shared == 0:
            self.cache = self.new_cache()
        elif shared < len(self.fed):
            # A negative crop removes that many tokens, in transformers 4 and 5.
            self.cache.crop(shared - len(self.fed))
        self.fed = tokens[:shared]
        with torch.inference_mode():
            for begin in range(shared, len(tokens), self.CHUNK):
                chunk = tokens[begin : begin + self.CHUNK]
                try:
                    logits = self.model(
                        input_ids=torch.tensor([chunk], device=self.device),
                        past_key_values=self.cache,
                        use_cache=True,
                    ).logits[0]
                except Exception:
                    # What a failed pass left in the cache is unknown.
                    self.cache, self.fed = self.new_cache(), []
                    raise
                self.fed += chunk
                for offset in range(max(0, start - 1 - begin), len(chunk)):
                    yield torch.log_softmax(logits[offset].to(torch.float64) / temperature, dim=-1)

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

    def walk(self, tokens: Sequence[int], start: int, temperature: float = 1.0) -> Iterator[torch.Tensor]:
        return stepwise(self.log_probs, tokens, start, temperature)

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

    def of(language: Language, name: str, temperature: float = 1.0) -> Measure:
        return Measure(
            name,
            lambda tokens: _numpy(language.log_probs(tokens, temperature)),
            lambda tokens, start: map(_numpy, language.walk(tokens, start, temperature)),
        )

    def measure(spec: str) -> Measure:
        kind, _, value = spec.partition(":")
        if kind == "context":
            n = int(value)

            def window(tokens: Sequence[int]) -> np.ndarray:
                return _numpy(own.log_probs(tokens[-n:]))

            return Measure(f"{own.name} · last {n} tokens", window, lambda tokens, start: stepwise(window, tokens, start))
        if kind == "temperature":
            t = float(value)
            return of(own, f"{own.name} · T={t:g}", t)
        language = other(spec)
        return of(language, language.name)

    return [of(own, own.name), *(measure(spec) for spec in specs)]


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

