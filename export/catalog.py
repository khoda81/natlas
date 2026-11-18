"""Specs: the names a server gives its models and measures, and what they may say.

A server's branches have one meaning; its measures are what weighs them.
A model is weights loaded on a device at a precision, by hand, and a
measure is a model read one way: conditioned on a recording, at a
temperature. Every measure of a server can be compared with every other,
whatever model each is of. A catalog lists the parameters a spec sets, for
the page to choose them by, loads the model one names and opens a measure
of it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import unquote

import numpy as np


class Measure(Protocol):
    """Next-token ln weights over the vocabulary, as float32 rows: up to a
    constant, or each child's own where ``widths``."""

    # What the model reads before the root's children: a start-of-text token.
    start: tuple[int, ...]
    # How many distributions one forward pass computes, and of how many
    # paths at most.
    width: int
    paths: int
    # The most tokens a path it reads may hold, or None where nothing says.
    longest: int | None
    # Whether its rows are each child's own ln weight, -ln of the width it
    # is drawn at, rather than ln weights up to a constant.
    widths: bool

    def ends(self, tokens: tuple[int, ...]) -> bool:
        """Whether a path is finished, with no children."""
        ...

    def walk(self, walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
        """For each walk (tokens, start), the rows after tokens[:k] for every
        k from ``start`` through len(tokens)."""
        ...


@dataclass(frozen=True)
class Choice:
    """One of a few values."""

    name: str
    options: tuple[str, ...]
    default: str


@dataclass(frozen=True)
class Number:
    """A finite number, or none where ``default`` is None."""

    name: str
    default: float | None


@dataclass(frozen=True)
class Text:
    """Any text, empty by default."""

    name: str


@dataclass(frozen=True)
class Recording:
    """A recording a client uploaded, by id, or none by default."""

    name: str


type Parameter = Choice | Number | Text | Recording


class Catalog:
    """What a server's specs may say: how to load the model one names, and
    open a measure of it.

    A model's spec is each of its parameters' values in turn, as path
    segments: ``large/cuda/bfloat16``. A measure's spec is its model's, then
    ``name=value`` for each of its own parameters not at its default, the
    value percent-encoded: ``large/cuda/bfloat16/temperature=3``. Two specs
    naming the same measure in other words are two measures that compute
    alike.
    """

    def __init__(
        self,
        models: Sequence[Choice],
        parameters: Sequence[Parameter],
        load: Callable[[dict[str, str]], Any],
        open: Callable[[Any, dict[str, Any]], Measure],
        presets: Sequence[str] = (),
        released: Callable[[], None] = lambda: None,
    ):
        self.models = models
        # Told once a model let go is no longer held, to give back its memory.
        self.released = released
        self.parameters = {parameter.name: parameter for parameter in parameters}
        self._load = load
        self._open = open
        # Models to offer as they are.
        self.presets = list(presets)
        for preset in self.presets:
            self.model(preset)

    def model(self, spec: str) -> dict[str, str]:
        """Each of the model's parameters under its spec; ValueError where
        it names none."""
        segments = [unquote(segment) for segment in spec.split("/")]
        if len(segments) != len(self.models):
            raise ValueError(f"{spec!r} is no model: a model is {'/'.join(choice.name for choice in self.models)}")
        return {choice.name: _value(choice, segment) for choice, segment in zip(self.models, segments, strict=True)}

    def parse(self, spec: str) -> tuple[str, dict[str, Any]]:
        """The spec of the model a measure's spec names, and each of the
        measure's own parameters; ValueError where it says what none may."""
        segments = spec.split("/")
        model = "/".join(segments[: len(self.models)])
        self.model(model)
        values: dict[str, Any] = {}
        for segment in segments[len(self.models) :]:
            name, equals, value = segment.partition("=")
            if not equals or name not in self.parameters or name in values:
                raise ValueError(f"{spec!r}: {segment!r} is no parameter, given once, of {', '.join(self.parameters) or 'none'}")
            values[name] = _value(self.parameters[name], unquote(value))
        for parameter in self.parameters.values():
            values.setdefault(parameter.name, _default(parameter))
        return model, values

    def load(self, model: str) -> Any:
        """The model ``model`` names, loaded: slow."""
        return self._load(self.model(model))

    def open(self, model: Any, spec: str) -> Measure:
        """The measure ``spec`` names, of ``model``, loaded."""
        return self._open(model, self.parse(spec)[1])

    def describe(self) -> dict[str, Any]:
        """For the page: the model's parameters and the measure's, in order,
        and the models offered as they are."""
        return {
            "models": [_described(choice) for choice in self.models],
            "parameters": [_described(parameter) for parameter in self.parameters.values()],
            "presets": self.presets,
        }


def _described(parameter: Parameter) -> dict[str, Any]:
    match parameter:
        case Choice(name, options, default):
            return {"name": name, "kind": "choice", "options": list(options), "default": default}
        case Number(name, default):
            return {"name": name, "kind": "number", "default": default}
        case Text(name):
            return {"name": name, "kind": "text"}
        case Recording(name):
            return {"name": name, "kind": "recording"}


def _value(parameter: Parameter, text: str) -> Any:
    match parameter:
        case Choice(name, options, _):
            if text not in options:
                raise ValueError(f"{name} is one of {', '.join(options)}, not {text!r}")
            return text
        case Number(name, _):
            try:
                number = float(text)
            except ValueError:
                number = math.nan
            if not math.isfinite(number):
                raise ValueError(f"{name} is a finite number, not {text!r}")
            return number
        case Text() | Recording():
            return text


def _default(parameter: Parameter) -> Any:
    match parameter:
        case Choice(_, _, default):
            return default
        case Number(_, default):
            return default
        case Text():
            return ""
        case Recording():
            return None
