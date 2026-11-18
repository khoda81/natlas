"""Specs: the names a server gives its measures, and what they may say.

A server's branches have one meaning; its measures are what weighs them,
each named by a spec. A catalog lists the parameters a spec sets, for the
page to choose them by, and opens the measure a spec names.
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
    """What a server's specs may say, and how to open the measure one names.

    A spec is path segments: the first is the source, the first parameter's
    value, and each other is ``name=value`` for one of the rest, its value
    percent-encoded; a parameter left out has its default. Two specs naming
    the same measure in other words are two measures that compute alike.
    """

    def __init__(self, parameters: Sequence[Parameter], open: Callable[[dict[str, Any]], Measure], start: str):
        if not isinstance(parameters[0], Choice):
            raise ValueError("a catalog's first parameter, the source, is a choice")
        self.parameters = {parameter.name: parameter for parameter in parameters}
        self._open = open
        # The spec a client starts from.
        self.start = start
        self.parse(start)

    def parse(self, spec: str) -> dict[str, Any]:
        """Each parameter's value under ``spec``; ValueError where it says
        what none may."""
        source, *segments = spec.split("/")
        first, *rest = self.parameters.values()
        values: dict[str, Any] = {first.name: _value(first, unquote(source))}
        for segment in segments:
            name, equals, value = segment.partition("=")
            parameter = self.parameters.get(name)
            if not equals or parameter is None or parameter is first or name in values:
                raise ValueError(f"{spec!r}: {segment!r} is no parameter, given once, of {', '.join(p.name for p in rest)}")
            values[name] = _value(parameter, unquote(value))
        for parameter in rest:
            values.setdefault(parameter.name, _default(parameter))
        return values

    def open(self, spec: str) -> Measure:
        """The measure ``spec`` names: slow, loading what it needs."""
        return self._open(self.parse(spec))

    def describe(self) -> dict[str, Any]:
        """For the page: the parameters, in order, and the spec to start from."""

        def described(parameter: Parameter) -> dict[str, Any]:
            match parameter:
                case Choice(name, options, default):
                    return {"name": name, "kind": "choice", "options": list(options), "default": default}
                case Number(name, default):
                    return {"name": name, "kind": "number", "default": default}
                case Text(name):
                    return {"name": name, "kind": "text"}
                case Recording(name):
                    return {"name": name, "kind": "recording"}

        return {"parameters": [described(parameter) for parameter in self.parameters.values()], "start": self.start}


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
