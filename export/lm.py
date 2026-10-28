"""Language-model measures, shared by the exporter and the server.

A measure is a model's next-token log-probabilities over the whole
vocabulary, given the tokens so far. Besides the model itself, a measure spec
names another model sharing its tokenizer, ``context:N`` for the model seeing
only the last N tokens, or ``temperature:T`` for the model at temperature T.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

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


def measures(model_id: str, model, specs: Sequence[str], device: str, dtype: str = "float32") -> list[Measure]:
    """The model itself, then one measure per spec."""

    def measure(spec: str) -> Measure:
        kind, _, value = spec.partition(":")
        if kind == "context":
            n = int(value)
            return Measure(f"{model_id} · last {n} tokens", lambda tokens: log_probs(model, tokens[-n:], device))
        if kind == "temperature":
            t = float(value)
            return Measure(f"{model_id} · T={t:g}", lambda tokens: log_probs(model, tokens, device, t))
        other = load(spec, device, dtype)
        return Measure(spec, lambda tokens: log_probs(other, tokens, device))

    own = Measure(model_id, lambda tokens: log_probs(model, tokens, device))
    return [own, *(measure(spec) for spec in specs)]


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


def token_texts(tokenizer, size: int) -> list[str]:
    """The text of every token id below ``size``, as decoded; the client makes whitespace visible."""
    pieces = tokenizer.batch_decode(
        [[token] for token in range(size)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [piece or f"<token:{token}>" for token, piece in enumerate(pieces)]
