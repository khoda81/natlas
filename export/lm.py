"""Language-model measures, shared by the exporter and the server.

A measure is a model's next-token ln weights over the whole vocabulary, up
to a constant, given the tokens so far, as a float32 numpy row; walking a path
gives them after every prefix in turn, and one call walks several paths.
A model is one of the sources offered, all sharing one tokenizer, loaded on
a device at a precision: ``SmolLM2-360M/cuda/float32``. A measure's spec
names its model, then optionally ``temperature=T``, ``context=N`` for the
model seeing only the last N tokens, and ``control=NAME`` for a .gguf model
steered by one of the llama.cpp control vectors offered (see steer.py):
``Qwen3-4B-Q6_K/cuda/auto/temperature=1.6``.

A source is a Hugging Face id, run by transformers, or a .gguf file, run by
llama.cpp, whose weights are as the file stores them: its precision is
``auto``, as a Hugging Face model's is the one its checkpoint was saved in.
"""

from __future__ import annotations

import copy
import ctypes
import json
import math
import re
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from catalog import Catalog, Choice, Number

# For each walk (tokens, start), the ln weights after tokens[:k] for every k
# from start through len(tokens), a float32 row each, in order.
type Walk = Callable[[Sequence[tuple[Sequence[int], int]]], list[np.ndarray]]


@dataclass(frozen=True)
class Measure:
    # What the model reads before the root's children: a start-of-text token.
    start: tuple[int, ...]
    walk: Walk
    # How many distributions one forward pass computes, and of how many
    # paths at most.
    width: int
    paths: int
    # The most tokens a path it reads may hold, or None where nothing says.
    longest: int | None
    # Whether a path is finished, with no children.
    ends: Callable[[tuple[int, ...]], bool]
    # Whether its rows are each child's own ln weight, -ln of the width it
    # is drawn at, rather than ln weights up to a constant.
    widths: bool = False

    def log_probs(self, tokens: Sequence[int]) -> np.ndarray:
        """The float64 log-probabilities after ``tokens``: normalizing in
        float64 keeps -ln p precise where p is close to 1."""
        (rows,) = self.walk([(tokens, len(tokens))])
        row = rows[-1].astype(np.float64)
        top = row.max()
        return row - top - np.log(np.exp(row - top).sum())


def stepwise(ln_weights: Callable[[Sequence[int]], np.ndarray], walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
    """Walks read one prefix at a time: right for a model that reuses the
    cache of the prefix it was last asked, where each step is one token."""
    return [np.stack([ln_weights(tokens[:end]) for end in range(start, len(tokens) + 1)]) for tokens, start in walks]


def devices() -> tuple[str, ...]:
    """Where a model can load: each GPU, CUDA's or Apple's (MPS), then the CPU."""
    count = torch.cuda.device_count()
    gpus = ("cuda",) if count == 1 else tuple(f"cuda:{index}" for index in range(count))
    if torch.backends.mps.is_available():
        gpus += ("mps",)
    return (*gpus, "cpu")


def released() -> None:
    """Give the memory of models let go back to the GPU's other users."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Half precision saves memory on large models but can be slower: a GTX 1660 Ti
# runs SmolLM2-360M about 5x faster in float32.
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load(model_id: str, device: str, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES.get(dtype, "auto"))
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
    # How many distributions one forward pass of ``walk`` computes, and of
    # how many paths at most.
    width: int
    paths: int
    # The most tokens a path it reads may hold, or None where nothing says.
    longest: int | None

    def walk(self, walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]: ...
    def encode(self, text: str) -> list[int]: ...
    def token_bytes(self) -> list[bytes]: ...


class Transformers:
    """A Hugging Face model. It keeps the KV cache of the last path it read, so
    a request that shares a prefix with it feeds only the rest, and it walks a
    path in chunks, reading every position's logits off one pass. Walks take
    turns: one cache holds one path."""

    # Positions per forward pass while walking: logits for this many fit easily.
    width = 256
    paths = 1
    longest = None

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

    def walk(self, walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
        return [self._walk([int(token) for token in tokens], start) for tokens, start in walks]

    def _walk(self, tokens: list[int], start: int) -> np.ndarray:
        """The logits after tokens[:k] for k from ``start`` on.

        Positions are computed a chunk at a time, so their last bits can
        differ from one computed alone.
        """
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
        rows = []
        with torch.inference_mode():
            for begin in range(shared, len(tokens), self.width):
                chunk = tokens[begin : begin + self.width]
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
                rows.append(logits[max(0, start - 1 - begin) :].float())
        return torch.cat(rows).cpu().numpy()

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)]

    def token_bytes(self) -> list[bytes]:
        return token_bytes(self.tokenizer, self.vocabulary)


class LlamaCpp:
    """A GGUF model run by llama.cpp, its layers on the GPU as far as ``gpu_layers``
    says (-1 for all). Requests that share a prefix with one read before reuse
    its cache, so walking down a path costs a token per step, and walks down
    different branches do not read each other's tokens again. Paths are read
    together, a batch of tokens per forward pass."""

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
        self.start, self.eos = _llama_ends(self.llm, path)
        self.reader = _Reader(context, self.vocabulary)
        self.width = self.paths = self.reader.width()
        self.longest = self.context

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

    def walk(self, walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
        return self.reader.read([([int(token) for token in tokens], start) for tokens, start in walks])

    def encode(self, text: str) -> list[int]:
        # Special tokens in the text are themselves, as transformers reads them.
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def token_bytes(self) -> list[bytes]:
        return [self.llm.detokenize([token], special=True) for token in range(self.vocabulary)]


def _llama_ends(llm, path: Path) -> tuple[tuple[int, ...], frozenset[int]]:
    """What a llama.cpp model reads before a text, and the tokens that end one.
    llama.cpp names a BPE vocabulary's token 11 its BOS even where none is
    added, as in Qwen's: there the model reads EOS before a text."""
    bos = llm.token_bos() if llm._model.add_bos_token() else -1  # noqa: SLF001
    eos = llm.token_eos()
    if bos < 0 and eos < 0:
        raise ValueError(f"{path} has no start- or end-of-text token to begin a text with")
    return (bos if bos >= 0 else eos,), frozenset({eos} if eos >= 0 else ())


class Vocabulary:
    """A source's tokenizer, read without its weights: what a server's
    branches mean, every model it offers sharing it."""

    def __init__(self, source: str):
        if source.endswith(".gguf"):
            from llama_cpp import Llama

            path = Path(source).expanduser()
            llm = Llama(str(path), vocab_only=True, verbose=False)
            self.size = llm.n_vocab()
            self.start, self.eos = _llama_ends(llm, path)
            self.encode = lambda text: llm.tokenize(text.encode(), add_bos=False, special=True)
            self.token_bytes = lambda: [llm.detokenize([token], special=True) for token in range(self.size)]
        else:
            from transformers import AutoConfig, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(source)
            self.size = max(len(tokenizer), int(AutoConfig.from_pretrained(source).vocab_size))
            self.start = start_of_text(tokenizer)
            self.eos = frozenset() if tokenizer.eos_token_id is None else frozenset({int(tokenizer.eos_token_id)})
            self.encode = lambda text: [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
            self.token_bytes = lambda: token_bytes(tokenizer, self.size)


# Tokens of context llama-cpp-python's own Llama keeps, which nothing here
# reads from: as few as llama.cpp allows.
_UNUSED_CONTEXT = 256


class _Reader:
    """A llama.cpp context and the tokens each of its sequences holds.

    Sequences share the cells of their common prefix. A path reading on from
    a sequence's tokens extends it; one that branches off copies the shared
    prefix into another sequence, at no cost, and reads only its own tokens
    there, so the walk it branched from keeps its place. The least recently
    read sequences give way when the context runs out of cells.

    The paths of a read go through the model together, as one batch of
    tokens. llama.cpp's kernels round a batch differently from a token read
    alone, so a node's numbers hang on how it was batched: on Qwen3-4B Q6_K
    by a KL of about 1e-3 nats, where the CPU's differ from the GPU's by
    4e-2. They hang on what else the cache holds too, batched or not: a
    token read alone moves by up to 0.3 in its logits once 1200 cells hold
    other paths. A recomputed node that differs replaces what a client knew.
    """

    def __init__(self, context, vocabulary: int):
        import llama_cpp
        from llama_cpp._internals import LlamaBatch

        self.context = context
        # Tokens one llama_decode reads at most: one micro-batch, so a
        # context out of cells refuses it whole.
        self.tokens = llama_cpp.llama_n_ubatch(context.ctx)
        self.batch = LlamaBatch(n_tokens=self.tokens, embd=0, n_seq_max=1, verbose=False)
        self.vocabulary = vocabulary
        self.sequences = llama_cpp.llama_max_parallel_sequences()
        # Tokens by sequence, the least recently read first.
        self.fed: dict[int, list[int]] = {}
        # The logits after each path the last read computed.
        self.last: dict[tuple[int, ...], np.ndarray] = {}

    def width(self) -> int:
        """How many paths to read in one pass: of batches of 1, 2, 4, … paths
        branching off one token, the one whose throughput over latency,
        k / t(k)², is greatest, the knee of the cost curve. Past it a batch
        takes longer than what it gains, which on a GTX 1660 Ti is 4."""
        limit = min(self.sequences - 1, self.tokens)
        best, power, k = 1, 0.0, 1
        # A batch takes no less time than a smaller one, so once even the
        # largest could not beat the best, none can.
        while k <= limit:
            seconds = math.inf
            for _ in range(3):
                self.clear()
                self.read([([0], 1)])
                started = time.perf_counter()
                self.read([([0, branch], 2) for branch in range(k)])
                seconds = min(seconds, time.perf_counter() - started)
            if power < k / seconds**2:
                best, power = k, k / seconds**2
            if limit / seconds**2 <= power:
                break
            k *= 2
        self.clear()
        return best

    def read(self, walks: list[tuple[list[int], int]]) -> list[np.ndarray]:
        """For each walk (tokens, start), the logits after tokens[:k] for k
        from ``start`` through len(tokens), a row each."""
        import llama_cpp

        if self.sequences < len(walks):
            raise ValueError(f"{len(walks)} paths, but the context holds {self.sequences} sequences")
        # Which sequence each walk reads in, chosen before any is read.
        reading: list[int | np.ndarray] = []
        for tokens, start in walks:
            if self.context.n_ctx() < len(tokens):
                raise ValueError(f"{len(tokens)} tokens, but the context holds {self.context.n_ctx()}")
            on_hand = [self.last.get(tuple(tokens[:end])) for end in range(start, len(tokens) + 1)]
            if all(row is not None for row in on_hand):
                reading.append(np.stack(on_hand))
            else:
                reading.append(self._sequence(tokens, start, {r for r in reading if isinstance(r, int)}))
        # Every token a walk lacks, in order, and whether its logits are wanted.
        entries = [
            (sequence, tokens[position], position, start - 1 <= position)
            for sequence, (tokens, start) in zip(reading, walks, strict=True)
            if isinstance(sequence, int)
            for position in range(len(self.fed[sequence]), len(tokens))
        ]
        rows = []
        try:
            for begin in range(0, len(entries), self.tokens):
                chunk = entries[begin : begin + self.tokens]
                self._decode(chunk, {sequence for sequence in reading if isinstance(sequence, int)})
                wanted = sum(want for *_, want in chunk)
                logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits(self.context.ctx), shape=(wanted, self.vocabulary))
                rows.extend(logits.copy())
                for sequence, token, _, _ in chunk:
                    self.fed[sequence].append(token)
        except Exception:
            # What a failed batch left in the memory is unknown.
            self.clear()
            raise
        computed = iter(rows)
        results = []
        self.last = {}
        for sequence, (tokens, start) in zip(reading, walks, strict=True):
            if isinstance(sequence, int):
                sequence = np.stack([next(computed) for _ in range(start, len(tokens) + 1)])
                self.last.update((tuple(tokens[:end]), row) for end, row in zip(range(start, len(tokens) + 1), sequence))
            results.append(sequence)
        return results

    def _sequence(self, tokens: list[int], start: int, taken: set[int]) -> int:
        """The sequence to read ``tokens`` in, holding as much of them as it
        can short of the logits wanted, the most recently read: the one
        sharing most with them where reading on from it is enough, else one
        not ``taken`` by this read that the shared prefix is copied into."""
        import llama_cpp

        memory = self.context.memory
        best, shared = None, 0
        for sequence, fed in self.fed.items():
            common = _common(fed, tokens)
            if best is None or shared < common:
                best, shared = sequence, common
        # The logits wanted are those after tokens[:start], the last read first.
        keep = min(shared, start - 1)
        if best is not None and best not in taken and keep == len(self.fed[best]):
            sequence = best
        else:
            free = next((sequence for sequence in range(self.sequences) if sequence not in self.fed and sequence not in taken), None)
            sequence = free if free is not None else next(sequence for sequence in self.fed if sequence not in taken)
            if sequence == best:
                # Cut it back.
                if not llama_cpp.llama_memory_seq_rm(memory, sequence, keep, -1):
                    # A recurrent layer's state cannot be cut back to a prefix.
                    llama_cpp.llama_memory_seq_rm(memory, sequence, -1, -1)
                    keep = 0
            else:
                llama_cpp.llama_memory_seq_rm(memory, sequence, -1, -1)
                if best is not None and 0 < keep:
                    llama_cpp.llama_memory_seq_cp(memory, best, sequence, 0, keep)
                else:
                    keep = 0
            self.fed[sequence] = tokens[:keep]
        self.fed[sequence] = self.fed.pop(sequence)
        return sequence

    def _decode(self, entries: list[tuple[int, int, int, bool]], reading: set[int]) -> None:
        """Read each (sequence, token, position, wanted) entry in one pass,
        freeing the least recently read sequences not ``reading`` while the
        context has no room."""
        import llama_cpp

        batch = self.batch.batch
        batch.n_tokens = len(entries)
        for i, (sequence, token, position, wanted) in enumerate(entries):
            batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.logits[i] = token, position, 1, wanted
            batch.seq_id[i][0] = sequence
        while (code := llama_cpp.llama_decode(self.context.ctx, batch)) == 1:
            # 1: no room for the tokens' cells.
            other = next((other for other in self.fed if other not in reading), None)
            if other is None:
                break
            llama_cpp.llama_memory_seq_rm(self.context.memory, other, -1, -1)
            del self.fed[other]
        if code != 0:
            raise RuntimeError(f"llama_decode returned {code}")

    def clear(self) -> None:
        self.context.kv_cache_clear()
        self.fed.clear()
        self.last = {}


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


def tempered(walk: Walk, temperature: float) -> Walk:
    """``walk`` at another temperature: its ln weights divided by it."""
    if not 0 < temperature:
        raise ValueError(f"a temperature is positive, not {temperature:g}")
    if temperature == 1:
        return walk
    return lambda walks: [rows / np.float32(temperature) for rows in walk(walks)]


def measure(language: Language, temperature: float = 1.0, context: float | None = None) -> Measure:
    """The model at ``temperature``, seeing only the last ``context`` tokens
    where that is given."""
    walk, width, longest = language.walk, language.width, language.longest
    if context is not None:
        n = int(context)
        if n != context or n < 1:
            raise ValueError(f"a context is a count of tokens, not {context:g}")

        def walk(walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
            # Each position reads its own window, a path of its own.
            windows = [tokens[:end][-n:] for tokens, start in walks for end in range(start, len(tokens) + 1)]
            rows = iter([rows[0] for rows in language.walk([(tokens, len(tokens)) for tokens in windows])])
            return [np.stack([next(rows) for _ in range(start, len(tokens) + 1)]) for tokens, start in walks]

        width = language.paths
        # A window longer than the model reads fails on paths that long.
        longest = None if language.longest is None or n <= language.longest else language.longest
    eos = language.eos
    return Measure(language.start, tempered(walk, temperature), width, language.paths, longest, lambda tokens: bool(tokens) and tokens[-1] in eos)


def catalog(
    sources: dict[str, str],
    vocabulary: Vocabulary,
    controls: dict[str, Path],
    presets: Sequence[str] = (),
    tokenizer: str | None = None,
    context: int = 8192,
    gpu_layers: int = -1,
) -> Catalog:
    """Every source offered, loaded on any device at any precision, read at
    any temperature and context, and steered by any control vector offered
    where it is a .gguf file. A .gguf model reads ``context`` tokens at most,
    with ``gpu_layers`` of its layers on a GPU (-1 for all)."""
    # Each loaded .gguf model's steered copies, by control: let go with it.
    steered: weakref.WeakKeyDictionary[LlamaCpp, dict[str, LlamaCpp]] = weakref.WeakKeyDictionary()

    def load(values: dict) -> Language:
        source, device, dtype = sources[values["model"]], values["device"], values["dtype"]
        if source.endswith(".gguf"):
            if dtype != "auto":
                raise ValueError("a .gguf file's weights are as it stores them: its dtype is auto")
            language: Language = LlamaCpp(Path(source).expanduser(), context, 0 if device == "cpu" else gpu_layers)
        else:
            language = Transformers(source, device, dtype, tokenizer)
        if language.vocabulary != vocabulary.size:
            raise ValueError(f"{source} has {language.vocabulary} tokens, where the server's tokenizer has {vocabulary.size}")
        return language

    def open(language: Language, values: dict) -> Measure:
        control = values.get("control", "none")
        if control != "none":
            if not isinstance(language, LlamaCpp):
                raise ValueError("control vectors need a .gguf model")
            copies = steered.setdefault(language, {})
            if control not in copies:
                copies[control] = language.controlled(controls[control])
            language = copies[control]
        return measure(language, values["temperature"], values["context"])

    names = tuple(sources)
    models = [Choice("model", names, names[0]), Choice("device", devices(), devices()[0]), Choice("dtype", ("auto", *DTYPES), "auto")]
    parameters = [Number("temperature", 1.0), Number("context", None)]
    if controls:
        parameters.append(Choice("control", ("none", *controls), "none"))
    return Catalog(models, parameters, load, open, presets, released)


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

