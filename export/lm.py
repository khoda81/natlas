"""Language-model measures, shared by the exporter and the server.

A measure is a model's next-token log-probabilities over the whole
vocabulary, given the tokens so far, as a float64 numpy array; walking a path
gives them after every prefix in turn. Besides the model itself, a measure spec
names another model sharing its tokenizer, ``context:N`` for the model seeing
only the last N tokens, ``temperature:T`` for the model at temperature T, or
``control:FILE`` for a .gguf model steered by a llama.cpp control vector (see
steer.py).

A model is a Hugging Face id, run by transformers, or a .gguf file, run by
llama.cpp.
"""

from __future__ import annotations

import copy
import ctypes
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
    says (-1 for all). Requests that share a prefix with one read before reuse
    its cache, so walking down a path costs a token per step, and walks down
    different branches take turns without reading each other's tokens again.
    Every token is evaluated alone."""

    def __init__(self, path: Path, context: int = 8192, gpu_layers: int = -1):
        from llama_cpp import Llama

        self.name = path.stem
        self.context = context
        try:
            # The model. Its own context holds nothing: readers have theirs.
            self.llm = Llama(
                str(path),
                n_ctx=_UNUSED_CONTEXT,
                n_gpu_layers=gpu_layers,
                # With no layers on the GPU, keep the KV cache off it too.
                offload_kqv=gpu_layers != 0,
                logits_all=False,
                verbose=False,
            )
            context = self.new_context()
        except ValueError as error:
            # llama.cpp says why only in its own log, and a full GPU is the usual reason.
            raise RuntimeError(
                f"llama.cpp could not load {path}: is the GPU's memory taken (nvidia-smi)? "
                "Fewer --gpu-layers or a smaller --n-ctx need less of it."
            ) from error
        self.vocabulary = self.llm.n_vocab()
        # llama.cpp names a BPE vocabulary's token 11 its BOS even where none
        # is added, as in Qwen's: there the model reads EOS before a text.
        bos = self.llm.token_bos() if self.llm._model.add_bos_token() else -1  # noqa: SLF001
        eos = self.llm.token_eos()
        if bos < 0 and eos < 0:
            raise ValueError(f"{path} has no start- or end-of-text token to begin a text with")
        self.start = (bos if bos >= 0 else eos,)
        self.eos = frozenset({eos} if eos >= 0 else ())
        self.reader = _Reader(context, self.vocabulary)

    def new_context(self):
        """A context of ``context`` tokens shared by as many sequences as
        llama.cpp holds: they share the cells of their common prefixes."""
        import llama_cpp
        from llama_cpp._internals import LlamaContext

        params = type(self.llm.context_params).from_buffer_copy(self.llm.context_params)
        params.n_ctx = self.context
        params.n_seq_max = llama_cpp.llama_max_parallel_sequences()
        params.kv_unified = True
        return LlamaContext(model=self.llm._model, params=params, verbose=False)  # noqa: SLF001

    def controlled(self, vector: Path) -> LlamaCpp:
        """The same weights read through a context of its own, with the control
        vector in ``vector`` added to the residual stream after its layers."""
        import llama_cpp

        llm = self.llm
        layers, width = llama_cpp.llama_model_n_layer(llm._model.model), llm._model.n_embd()  # noqa: SLF001
        directions = control_vector(vector, layers, width)
        context = self.new_context()
        data = directions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if llama_cpp.llama_set_adapter_cvec(context.ctx, data, directions.size, width, 1, layers - 1):
            raise RuntimeError(f"llama.cpp refused the control vector in {vector}")
        steered = copy.copy(self)
        steered.name = f"{self.name} · {vector.stem}"
        steered.reader = _Reader(context, self.vocabulary)
        return steered

    def log_probs(self, tokens: Sequence[int], temperature: float = 1.0) -> torch.Tensor:
        logits = self.reader.logits([int(token) for token in tokens])
        return torch.log_softmax(torch.from_numpy(logits.astype(np.float64)) / temperature, dim=-1)

    def walk(self, tokens: Sequence[int], start: int, temperature: float = 1.0) -> Iterator[torch.Tensor]:
        return stepwise(self.log_probs, tokens, start, temperature)

    def encode(self, text: str) -> list[int]:
        # Special tokens in the text are themselves, as transformers reads them.
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def token_bytes(self) -> list[bytes]:
        return [self.llm.detokenize([token], special=True) for token in range(self.vocabulary)]


# Tokens of context llama-cpp-python's own Llama keeps, which nothing here
# reads from: as few as llama.cpp allows.
_UNUSED_CONTEXT = 256


class _Reader:
    """A llama.cpp context and the tokens each of its sequences holds.

    Sequences share the cells of their common prefix. A request reading on
    from a sequence's tokens extends it; one that branches off copies the
    shared prefix into another sequence, at no cost, and reads only its own
    tokens there, so the walk it branched from keeps its place. The least
    recently read sequences give way when the context runs out of cells.
    """

    def __init__(self, context, vocabulary: int):
        from llama_cpp import llama_max_parallel_sequences
        from llama_cpp._internals import LlamaBatch

        self.context = context
        self.batch = LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
        self.vocabulary = vocabulary
        self.sequences = llama_max_parallel_sequences()
        # Tokens by sequence, the least recently read first.
        self.fed: dict[int, list[int]] = {}
        # The sequence and length the logits on hand are the last position of.
        self.last: tuple[int, int] | None = None

    def logits(self, tokens: list[int]) -> np.ndarray:
        """The logits after ``tokens``."""
        import llama_cpp

        if self.context.n_ctx() < len(tokens):
            raise TooLong(f"{len(tokens)} tokens, but the context holds {self.context.n_ctx()}")
        memory = self.context.memory
        best, shared = None, 0
        for sequence, fed in self.fed.items():
            common = _common(fed, tokens)
            if best is None or shared < common:
                best, shared = sequence, common
        # The logits wanted are those of the last token read.
        keep = min(shared, len(tokens) - 1)
        # Read on from the sequence, or its logits on hand are the ones.
        on_hand = self.last == (best, len(tokens)) and shared == len(tokens)
        if best is not None and (keep == len(self.fed[best]) or on_hand):
            sequence = best
        else:
            free = next((sequence for sequence in range(self.sequences) if sequence not in self.fed), None)
            sequence = free if free is not None else next(iter(self.fed))
            if sequence == best:
                # The one sequence there is: cut it back.
                if not llama_cpp.llama_memory_seq_rm(memory, sequence, keep, -1):
                    # A recurrent layer's state cannot be cut back to a prefix.
                    self.clear()
                    keep = 0
            else:
                llama_cpp.llama_memory_seq_rm(memory, sequence, -1, -1)
                if best is not None and 0 < keep:
                    llama_cpp.llama_memory_seq_cp(memory, best, sequence, 0, keep)
                else:
                    keep = 0
            self.fed[sequence] = tokens[:keep]
        fed = self.fed.pop(sequence)
        self.fed[sequence] = fed
        try:
            # One token at a time: llama.cpp's kernels round a batch
            # differently, so a node's numbers would hang on what was
            # asked before it.
            for token in tokens[len(fed) :]:
                self.decode(sequence, token, len(fed))
                fed.append(token)
                self.last = (sequence, len(fed))
        except Exception:
            # What a failed batch left in the memory is unknown.
            self.clear()
            raise
        return np.ctypeslib.as_array(self.context.get_logits(), shape=(self.vocabulary,))

    def decode(self, sequence: int, token: int, position: int) -> None:
        """Read ``token`` at ``position`` of ``sequence``, freeing the least
        recently read other sequences while the context has no room."""
        import llama_cpp

        batch = self.batch.batch
        batch.n_tokens = 1
        batch.token[0], batch.pos[0], batch.n_seq_id[0], batch.logits[0] = token, position, 1, True
        batch.seq_id[0][0] = sequence
        while (code := llama_cpp.llama_decode(self.context.ctx, batch)) == 1:
            # 1: no room for the token's cells.
            other = next((other for other in self.fed if other != sequence), None)
            if other is None:
                break
            llama_cpp.llama_memory_seq_rm(self.context.memory, other, -1, -1)
            del self.fed[other]
        if code != 0:
            raise RuntimeError(f"llama_decode returned {code}")

    def clear(self) -> None:
        self.context.kv_cache_clear()
        self.fed.clear()
        self.last = None


def _common(have: Sequence[int], want: Sequence[int]) -> int:
    """How many tokens two token lists share from their start."""
    shared = 0
    for a, b in zip(have, want):
        if a != b:
            break
        shared += 1
    return shared


def control_vector(path: Path, layers: int, width: int) -> np.ndarray:
    """A llama.cpp control vector file's directions for layers 1 through
    ``layers`` - 1, the rows llama.cpp reads; layers it leaves out are zero."""
    import gguf

    directions = np.zeros((layers - 1, width), np.float32)
    for tensor in gguf.GGUFReader(path).tensors:
        layer = int(tensor.name.removeprefix("direction."))
        if not 1 <= layer < layers or tensor.data.shape != (width,):
            raise ValueError(f"{path}: {tensor.name} of shape {tensor.data.shape} fits no layer of {layers} × {width}")
        directions[layer - 1] = tensor.data
    return directions


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
        if kind == "control":
            if not isinstance(own, LlamaCpp):
                raise ValueError("control vectors need a .gguf model")
            steered = own.controlled(Path(value).expanduser())
            return of(steered, steered.name)
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

