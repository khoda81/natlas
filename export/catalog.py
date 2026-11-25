"""Specs: the names a server gives its models and measures, and what they may say.

A server's branches have one fixed vocabulary; its measures are probability
providers over that vocabulary. A model is weights loaded on a device (e.g.
source, device, dtype, n_ctx, gpu_layers), and a measure is a model read one
way (e.g. temperature, control, audio).

A model spec is canonical ``name=value`` pairs joined by ``&`` (omitting
parameters at their default, except keeping the first parameter when all are at
default so a spec is never empty). A measure spec is ``<model_spec>`` when all
measure parameters are at default, or ``<model_spec>|<name=value&...>``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode

import numpy as np


class Measure(Protocol):
    """Next-token ln weights over the vocabulary, as float32 rows: up to a
    constant, or each child's own where ``widths``."""

    start: tuple[int, ...]
    width: int
    paths: int
    longest: int | None
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
    """One of a few closed values."""

    name: str
    options: tuple[str, ...]
    default: str


@dataclass(frozen=True)
class Number:
    """A finite number, or None where ``default`` is None."""

    name: str
    default: float | None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    integer: bool = False


@dataclass(frozen=True)
class Text:
    """Free-form text, with optional default, placeholder, and autocomplete suggestions."""

    name: str
    default: str = ""
    placeholder: str = ""
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Boolean:
    """A boolean flag."""

    name: str
    default: bool = False


@dataclass(frozen=True)
class Recording:
    """A recording a client uploaded, by id, or none by default."""

    name: str


Parameter = Choice | Number | Text | Boolean | Recording


class Catalog:
    """What a server's specs may say: how to load the model one names, and
    open a measure of it."""

    def __init__(
        self,
        models: Sequence[Parameter],
        parameters: Sequence[Parameter],
        load: Callable[[dict[str, Any]], Any],
        open: Callable[[Any, dict[str, Any]], Measure],
        released: Callable[[], None] = lambda: None,
    ):
        self.models = {parameter.name: parameter for parameter in models}
        self.parameters = {parameter.name: parameter for parameter in parameters}
        self._load = load
        self._open = open
        self.released = released

    def model(self, spec: str) -> dict[str, Any]:
        """Each of the model's parameters under its spec; ValueError where invalid."""
        if not spec or "|" in spec:
            raise ValueError(f"{spec!r} is no model spec")
        return _parse_section(spec, self.models, "model")

    def parse(self, spec: str) -> tuple[str, dict[str, Any]]:
        """The model spec and each of the measure's own parameters; ValueError where invalid."""
        model_spec, sep, measure_part = spec.partition("|")
        if "|" in measure_part:
            raise ValueError(f"{spec!r}: expected at most one '|' separating model and measure")
        self.model(model_spec)
        if sep and not measure_part:
            raise ValueError(f"{spec!r}: empty measure section after '|'")
        values = _parse_section(measure_part, self.parameters, "measure") if measure_part else {
            p.name: _default(p) for p in self.parameters.values()
        }
        return model_spec, values

    def load(self, model: str) -> Any:
        """Load the model ``model`` names."""
        return self._load(self.model(model))

    def open(self, model: Any, spec: str) -> Measure:
        """Open the measure ``spec`` of loaded ``model``."""
        return self._open(model, self.parse(spec)[1])

    def describe(self) -> dict[str, Any]:
        """For the page: the model's parameters and the measure's, in order."""
        return {
            "models": [_described(parameter) for parameter in self.models.values()],
            "parameters": [_described(parameter) for parameter in self.parameters.values()],
        }


def format_model(parameters: Sequence[Parameter], values: dict[str, Any]) -> str:
    """Format a canonical model spec from parameter values."""
    pairs: list[tuple[str, str]] = []
    for p in parameters:
        raw = _format_value(p, values.get(p.name, _default(p)))
        if raw != _format_value(p, _default(p)):
            pairs.append((p.name, raw))
    if not pairs and parameters:
        first = parameters[0]
        pairs.append((first.name, _format_value(first, values.get(first.name, _default(first)))))
    return urlencode(pairs)


def _parse_section(section: str, schema: dict[str, Parameter], label: str) -> dict[str, Any]:
    if not section:
        raise ValueError(f"empty {label} spec")
    try:
        pairs = parse_qsl(section, keep_blank_values=True, strict_parsing=True)
    except ValueError as error:
        raise ValueError(f"{section!r}: invalid {label} parameters: {error}") from error
    values: dict[str, Any] = {}
    for name, raw in pairs:
        if name not in schema or name in values:
            allowed = ", ".join(schema) or "none"
            raise ValueError(f"{section!r}: {name!r} is no {label} parameter, given once, of {allowed}")
        values[name] = _value(schema[name], raw)
    for parameter in schema.values():
        values.setdefault(parameter.name, _default(parameter))
    return values


def _described(parameter: Parameter) -> dict[str, Any]:
    match parameter:
        case Choice(name, options, default):
            return {"name": name, "kind": "choice", "options": list(options), "default": default}
        case Number(name, default, lo, hi, step, integer):
            out: dict[str, Any] = {"name": name, "kind": "number", "default": default}
            if lo is not None:
                out["min"] = lo
            if hi is not None:
                out["max"] = hi
            if step is not None:
                out["step"] = step
            if integer:
                out["integer"] = True
            return out
        case Text(name, default, placeholder, suggestions):
            out = {"name": name, "kind": "text"}
            if default:
                out["default"] = default
            if placeholder:
                out["placeholder"] = placeholder
            if suggestions:
                out["suggestions"] = list(suggestions)
            return out
        case Boolean(name, default):
            return {"name": name, "kind": "boolean", "default": default}
        case Recording(name):
            return {"name": name, "kind": "recording"}


def _value(parameter: Parameter, text: str) -> Any:
    match parameter:
        case Choice(name, options, _):
            if text not in options:
                raise ValueError(f"{name} is one of {', '.join(options)}, not {text!r}")
            return text
        case Number(name, default, lo, hi, _, integer):
            if text.strip() == "":
                if default is None:
                    return None
                raise ValueError(f"{name} requires a number")
            try:
                number = float(text)
            except ValueError:
                number = math.nan
            if not math.isfinite(number):
                raise ValueError(f"{name} is a finite number, not {text!r}")
            if integer:
                if int(number) != number:
                    raise ValueError(f"{name} must be an integer, not {text!r}")
                number = int(number)
            if lo is not None and number < lo:
                raise ValueError(f"{name} must be >= {lo:g}, not {text!r}")
            if hi is not None and number > hi:
                raise ValueError(f"{name} must be <= {hi:g}, not {text!r}")
            return number
        case Boolean(name, _):
            if text in ("true", "1"):
                return True
            if text in ("false", "0"):
                return False
            raise ValueError(f"{name} must be true or false, not {text!r}")
        case Text():
            return text
        case Recording():
            return text or None


def _default(parameter: Parameter) -> Any:
    match parameter:
        case Choice(_, _, default):
            return default
        case Number(default=default):
            return default
        case Text(default=default):
            return default
        case Boolean(_, default):
            return default
        case Recording():
            return None


def _format_value(parameter: Parameter, value: Any) -> str:
    match parameter:
        case Boolean():
            return "true" if value else "false"
        case Number():
            return "" if value is None else str(value)
        case _:
            return "" if value is None else str(value)
