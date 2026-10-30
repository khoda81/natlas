# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "natlas-core",
#   "natwalk",
#   "protobuf>=7.36.1",
# ]
#
# [tool.uv.sources]
# natlas-core = { path = "../core" }
# natwalk = { path = "../../natwalk", editable = true }
# ///

"""Export a model-free test tree: character n-gram models of the Zen of Python.

natwalk discovers the tree with the trigram model; the bigram and unigram
models are evaluated on the same paths as alternative measures.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import io
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from natlas_export import snapshots, write
from natwalk import Session

_START = "^"
_END = "$"


def corpus() -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()):
        import this
    text = codecs.decode(this.s, "rot13").lower()
    return [line for line in text.splitlines()[2:] if line]


class NGram:
    """Add-k smoothed character model conditioned on the previous ``order`` symbols."""

    def __init__(self, lines: Sequence[str], alphabet: Sequence[str], order: int, k: float):
        self.alphabet = alphabet
        self.order = order
        self.k = k
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        for line in lines:
            padded = _START * order + line + _END
            for index in range(order, len(padded)):
                self.counts[padded[index - order : index]][padded[index]] += 1

    def distribution(self, history: str) -> list[float]:
        context = (_START * self.order + history)[-self.order :] if self.order else ""
        counts = self.counts.get(context, Counter())
        total = sum(counts.values()) + self.k * len(self.alphabet)
        return [(counts[symbol] + self.k) / total for symbol in self.alphabet]


@dataclass
class NGramCursor:
    model: NGram
    alphabet: Sequence[str]
    history: str = ""

    def predict(self) -> Sequence[float]:
        if self.history.endswith(_END):
            return ()
        return self.model.distribution(self.history)

    def observe(self, token: int) -> None:
        self.history += self.alphabet[token]

    def checkpoint(self) -> object:
        return self.history

    def restore(self, checkpoint: object) -> None:
        assert isinstance(checkpoint, str)
        self.history = checkpoint


def label(symbol: str) -> str:
    return "∎" if symbol == _END else symbol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=4000)
    args = parser.parse_args()

    lines = corpus()
    alphabet = sorted({symbol for line in lines for symbol in line}) + [_END]
    trigram = NGram(lines, alphabet, order=2, k=0.02)
    bigram = NGram(lines, alphabet, order=1, k=0.05)
    unigram = NGram(lines, alphabet, order=0, k=0.5)

    session = Session(NGramCursor(trigram, alphabet))
    while len(session.tree.nodes) < args.nodes and session.search.frontier:
        session.search.discover()

    def measure(model: NGram):
        def weights(node_id: int) -> list[float]:
            return model.distribution("".join(alphabet[token] for token in session.tree.path(node_id)))

        return weights

    recorded = snapshots(
        session.tree,
        extra=(measure(bigram), measure(unigram)),
        min_list=len(alphabet),
    )
    write(
        "zen-ngram",
        "Zen of Python · character n-grams",
        [label(symbol).encode() for symbol in alphabet],
        ("trigram", "bigram", "unigram"),
        recorded,
    )


if __name__ == "__main__":
    main()
