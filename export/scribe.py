# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "natlas-core",
#   "muscriptor==0.3.0",
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "soundfile",
#   "starlette>=0.40",
#   "uvicorn>=0.30",
#   "websockets>=13",
# ]
#
# [tool.uv.sources]
# natlas-core = { path = "../core" }
# ///

"""Serve MuScriptor's tree of transcriptions of recordings to natlas, live.

MuScriptor transcribes music 5 seconds at a time, writing each window as
MIDI-like events: a tie prologue naming the notes sustained into the window,
then time shifts, instruments, note-ons and note-offs, pitches and drum hits,
then EOS. A path here is one transcription of the whole recording from a
time on, scrolling through the windows as ``TranscriptionModel.transcribe``
does with its prelude forcing:

* within a window, an event has MuScriptor's probability given that window's
  audio and the window's events so far, among the events that keep the path
  a valid encoding (below);
* EOS is the model's own decision; it ends the window and moves to the next;
* a later window's tie prologue is forced from the notes the path leaves
  sounding, so those events have probability one; so is the first window's,
  an empty one, when it starts at 0 and nothing sounds before the recording;
* a window that reaches MuScriptor's generation limit gets a forced EOS;
* the last window's EOS ends the path.

An event is valid where MuScriptor's decoder acts on it. Probability is
taken from events it would drop or that would make the window malformed:
pitches before an instrument and an on/off switch, note-offs of notes not
sounding, switches to the instrument or on/off already set, or set and never
used before the next switch or EOS, time that does not move forward or
leaves the window, and a tie, velocity, drum, time shift or EOS inside a tie
prologue, which names its instruments and pitches once each in ascending
order, as MuScriptor's encoder writes it. So every path is a transcription
MuScriptor decodes in full, and its probability is the model's, given that
it is one, as in natwalk's MuScriptor demo.

Every measure gives the events this one meaning, so any two compare. Nothing
loads until a page asks: a model is a size loaded on a device at a precision,
``large/cuda/float16``, and a measure's spec names its model, then
optionally ``audio=ID`` for a recording a page uploaded, the model hearing
silence without one (so p against it shows what the audio told the model),
``at=SECONDS`` where its first window starts, ``instruments=NAMES`` for the
model told to expect those instruments, ``temperature=T``, and
``shape=time-area`` for its time shifts drawn as one, keeping their area:
``large/cuda/float16/audio=3f2a9c/at=30``. Under silence a path goes on
window after window; under a recording, its last window's EOS ends it. The
config file (scribe.toml) lists models to offer as they are; every model
loaded since is offered too.

POST /recordings takes an audio file (including M4A via FFmpeg) and says its id;
GET /recordings lists them, and GET /recordings/ID is one as the model
hears it. GET /labels.json describes the events.

The data directory (--data) keeps what outlives a restart: each recording
uploaded, as recordings/ID/NAME, and the models loaded, one spec per line
in models.txt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import tomllib
import wave
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lm
import numpy as np
import torch
import torch.nn.functional as F
import anyio.to_thread
import uvicorn
from catalog import Catalog, Choice, Number, Recording, Text
from muscriptor import TranscriptionModel
from muscriptor.events import ChunkBoundary, OpenNoteTracker
from muscriptor.models.lm import TorchAutocast
from muscriptor.modules.streaming import increment_steps, init_states
from muscriptor.tokenizer.mt3 import (
    MT3Tokenizer,
    instrument_group_from_names,
    resolve_instrument_names,
)
from muscriptor.transcription_model import _build_instrument_for_program
from muscriptor.utils.audio import load_audio
from service import CACHE, Model, app
from soundfile import LibsndfileError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE
SIZES = ("small", "medium", "large")
# TranscriptionModel.transcribe writes at most this many events per window,
# counting the forced prologue and EOS.
MAX_EVENTS = 2000
# Windows entered, remembered with the prologue each forces.
_WINDOW_MEMO = 256

# General MIDI's percussion key map: the drum each pitch plays.
DRUMS = {
    35: "acoustic kick", 36: "kick", 37: "side stick", 38: "snare", 39: "clap", 40: "electric snare",
    41: "low floor tom", 42: "closed hi-hat", 43: "high floor tom", 44: "pedal hi-hat", 45: "low tom",
    46: "open hi-hat", 47: "low-mid tom", 48: "high-mid tom", 49: "crash", 50: "high tom", 51: "ride",
    52: "china", 53: "ride bell", 54: "tambourine", 55: "splash", 56: "cowbell", 57: "crash 2",
    58: "vibraslap", 59: "ride 2", 60: "high bongo", 61: "low bongo", 62: "mute conga", 63: "open conga",
    64: "low conga", 65: "high timbale", 66: "low timbale", 67: "high agogo", 68: "low agogo",
    69: "cabasa", 70: "maracas", 71: "short whistle", 72: "long whistle", 73: "short guiro",
    74: "long guiro", 75: "claves", 76: "high wood block", 77: "low wood block", 78: "mute cuica",
    79: "open cuica", 80: "mute triangle", 81: "open triangle",
}


@dataclass(frozen=True)
class Window:
    """One window as a path enters it: its index, and the prologue forced on it."""

    index: int
    prologue: tuple[int, ...]


class Windows:
    """How paths run through a recording's windows from ``at`` seconds on:
    ``count`` of them, or endless where it is None, as under silence."""

    def __init__(self, tokenizer: MT3Tokenizer, at: float, count: int | None, max_events: int):
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.eos_id
        self.at = at
        self.count = count
        self.max_events = max_events
        vocab = self.tokenizer._vocab  # noqa: SLF001
        self.kinds = np.array([event.type for event in vocab])
        self.values = np.array([event.value for event in vocab])
        # The decoder's state on entering a window, by the path up to that
        # window's start, which ends in the EOS before it.
        self.entered: OrderedDict[tuple[int, ...], tuple[Window, OpenNoteTracker]] = OrderedDict()

    def ends(self, path: tuple[int, ...]) -> bool:
        return self.count is not None and self.count <= path.count(self.eos)

    def locate(self, path: tuple[int, ...]) -> tuple[Window, tuple[int, ...]]:
        """The window ``path`` ends in, and the events written in it so far, its prologue first."""
        start = len(path) - path[::-1].index(self.eos) if self.eos in path else 0
        return self._state(path[:start])[0], path[start:]

    def valid(self, path: tuple[int, ...]) -> np.ndarray:
        """Which events may follow ``path`` and keep it a valid encoding:
        the ones MuScriptor's decoder acts on, by its own state there."""
        window, written = self.locate(path)
        before = path[: len(path) - len(written)]
        vocab = self.tokenizer._vocab  # noqa: SLF001
        tracker = copy.deepcopy(self._state(before)[1], {id(vocab): vocab})
        # Instrument and on/off switches not yet used by a pitch.
        unused: set[str] = set()
        for token in written:
            tracker.feed(token)
            kind = vocab[token].type
            if kind in ("program", "velocity"):
                unused.add(kind)
            elif kind == "pitch":
                unused.clear()
        # MuScriptor's decoder state, as OpenNoteTracker keeps it.
        program, velocity = tracker._program, tracker._velocity  # noqa: SLF001
        kinds, values = self.kinds, self.values
        if tracker._in_prologue:  # noqa: SLF001
            # Instruments ascending, each followed by its pitches ascending.
            ok = (kinds == "program") & (True if program is None else program < values) & ("program" not in unused)
            if program is not None:
                named = [pitch for instrument, pitch in tracker._tie_set if instrument == program]  # noqa: SLF001
                ok |= (kinds == "pitch") & (max(named, default=-1) < values)
            if "program" not in unused:
                ok |= kinds == "tie"
            return ok
        # Time moves forward, and stays in the window unless it is the last.
        seek = tracker._next_seek_time  # noqa: SLF001
        start, rate = tracker._start_tick, tracker._frame_rate  # noqa: SLF001
        shifted = (start + values) / rate
        ok = (kinds == "shift") & (tracker._tick_state - start < values) & (True if seek is None else shifted < seek)  # noqa: SLF001
        ok |= (kinds == "program") & (values != (-1 if program is None else program)) & ("program" not in unused)
        ok |= (kinds == "velocity") & (values != (-1 if velocity is None else velocity)) & ("velocity" not in unused)
        if not unused:
            ok |= kinds == "EOS"
        if seek is None or tracker._tick_state / rate < seek:  # noqa: SLF001
            ok |= kinds == "drum"
            if program is not None and velocity is not None:
                if 0 < velocity:
                    ok |= kinds == "pitch"
                else:
                    sounding = [pitch for instrument, pitch in tracker._open if instrument == program]  # noqa: SLF001
                    ok |= (kinds == "pitch") & np.isin(values, sounding)
        return ok

    def forced(self, window: Window, written: tuple[int, ...]) -> int | None:
        if len(written) < len(window.prologue):
            return window.prologue[len(written)]
        if len(written) >= self.max_events:
            return self.eos
        return None

    def _boundary(self, index: int) -> ChunkBoundary:
        following = self.at + (index + 1) * WINDOW_SECONDS if self.count is None or index + 1 < self.count else None
        return ChunkBoundary(self.at + index * WINDOW_SECONDS, following)

    def _state(self, before: tuple[int, ...]) -> tuple[Window, OpenNoteTracker]:
        known = self.entered.get(before)
        if known is not None:
            self.entered.move_to_end(before)
            return known
        vocab = self.tokenizer._vocab  # noqa: SLF001
        if not before:
            tracker = OpenNoteTracker(vocab, self.tokenizer.frame_rate)
            tracker.feed(self._boundary(0))
            # At the recording's start nothing sounds: its prologue is empty.
            # Later in a recording what sounds before it is unknown, and the
            # model names it.
            prologue = tuple(self.tokenizer.tie_section_token_ids([])) if self.at == 0 else ()
            entered = (Window(0, prologue), tracker)
        else:
            # The window before this one starts after its own preceding EOS.
            last = before[:-1]
            start = len(last) - last[::-1].index(self.eos) if self.eos in last else 0
            previous, tracker = self._state(last[:start])
            tracker = copy.deepcopy(tracker, {id(vocab): vocab})
            for token in last[start:]:
                tracker.feed(token)
            index = previous.index + 1
            tracker.feed(self._boundary(index))
            prologue = tuple(self.tokenizer.tie_section_token_ids(tracker.open_keys()))
            entered = (Window(index, prologue), tracker)
        self.entered[before] = entered
        if len(self.entered) > _WINDOW_MEMO:
            self.entered.popitem(last=False)
        return entered


@dataclass
class _Slot:
    """One KV cache and what it holds: a window's start token and events."""

    state: Any
    window: Window | None = None
    fed: list[int] = field(default_factory=list)
    logits: torch.Tensor | None = None
    used: int = 0


class Scribe:
    """MuScriptor transcribing what it hears: each event's log-probability given the path before it.

    A few KV caches, one per window in reach, each stepped back to the prefix
    a request shares with what it holds, so walking down a path costs one
    step per event.
    """

    width = paths = 1
    longest = None

    def __init__(
        self,
        model: TranscriptionModel,
        windows: Windows,
        hear: Callable[[int], torch.Tensor],
        expect: str | None = None,
        slots: int = 2,
    ):
        self.model = model
        self.lm = model._model  # noqa: SLF001
        self.device = model._device  # noqa: SLF001
        self.windows = windows
        self.hear = hear
        self.expect = expect
        self.vocabulary = windows.tokenizer.num_tokens
        self.start = (self.lm.initial_token_id,)
        self.eos = frozenset({windows.eos})
        self.conditions: dict[int, Any] = {}
        self.prepend = sum(condition.shape[1] for condition, _ in self._conditions(0).values())
        self.slots: list[_Slot] = []
        self.capacity = slots
        self.clock = 0

    def ln_weights(self, tokens: Sequence[int]) -> np.ndarray:
        """Each event's ln weight after ``tokens``, up to a constant."""
        if tuple(tokens[:1]) != self.start:
            raise ValueError("a transcription begins with the start token")
        window, written = self.windows.locate(tuple(int(token) for token in tokens[1:]))
        forced = self.windows.forced(window, written)
        if forced is not None:
            certain = np.full(self.vocabulary, -np.inf, dtype=np.float32)
            certain[forced] = 0.0
            return certain
        ln_weights = self._evaluate(window, list(written))[: self.vocabulary].float().cpu().numpy()
        ln_weights[~self.windows.valid(tuple(int(token) for token in tokens[1:]))] = -np.inf
        return ln_weights

    def walk(self, walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
        return lm.stepwise(self.ln_weights, walks)

    def _conditions(self, index: int) -> Any:
        conditions = self.conditions.get(index)
        if conditions is None:
            with torch.inference_mode():
                attributes = self.model._build_conditions(self.hear(index), self.expect)  # noqa: SLF001
                provider = self.lm.condition_provider
                conditions = provider(provider.tokenize(attributes))
            self.conditions[index] = conditions
        return conditions

    def _slot(self, window: Window, written: list[int]) -> tuple[_Slot, int | None]:
        """The cache to read ``written`` in, and how many of its events it
        holds already, or None to start over.

        A cache holding a prefix of ``written`` goes on in place. Otherwise
        a new cache while there is room, else the least recently used, takes
        what the cache sharing the most with it holds, up to what they share.
        Rewinding the shared cache whatever its use would lose the walk it
        holds: walks that take turns, as passes do while one runs and the
        next waits, would each read their whole path again on every pass.
        """
        shared: dict[int, int] = {}
        for index, slot in enumerate(self.slots):
            if slot.window != window:
                continue
            kept = 0
            for have, want in zip(slot.fed, written):
                if have != want:
                    break
                kept += 1
            # Keep the prologue whole: it is written in one pass.
            if len(window.prologue) <= kept:
                shared[index] = kept
        source = max(shared, key=shared.__getitem__, default=None)
        self.clock += 1
        if source is not None and shared[source] == len(self.slots[source].fed):
            slot = self.slots[source]
        else:
            if len(self.slots) < self.capacity:
                sequence_length = self.prepend + 1 + self.windows.max_events
                self.slots.append(_Slot(init_states(self.lm, batch_size=1, sequence_length=sequence_length)))
                target = len(self.slots) - 1
            else:
                target = min(range(len(self.slots)), key=lambda index: self.slots[index].used)
            slot = self.slots[target]
            if source is not None and target != source:
                self._copy(self.slots[source], slot, shared[source])
        slot.used = self.clock
        return slot, None if source is None else shared[source]

    def _copy(self, source: _Slot, target: _Slot, kept: int) -> None:
        """Make ``target`` hold the first ``kept`` events ``source`` holds."""
        end = self.prepend + 1 + kept
        for name, state in source.state.items():
            into = target.state[name]
            if "cache" in state:
                into["cache"][:, :, :end].copy_(state["cache"][:, :, :end])
        target.window, target.fed, target.logits = source.window, source.fed[:kept], None

    @torch.inference_mode()
    def _evaluate(self, window: Window, written: list[int]) -> torch.Tensor:
        slot, kept = self._slot(window, written)
        if kept == len(written) == len(slot.fed) and slot.logits is not None:
            return slot.logits
        conditions = self._conditions(window.index)
        try:
            if kept is not None and kept == len(written):
                # The last event's logits are wanted, so it is written again.
                kept -= 1
            if kept is None or kept < len(window.prologue):
                slot.window = None
                self._seek(slot, 0)
                # The start token and the prologue in one pass under autocast,
                # as transcribe writes a forced prompt.
                with self.lm.autocast:
                    logits = self._step(slot, [self.lm.initial_token_id, *window.prologue], conditions, first=True)
                slot.window, slot.fed = window, list(window.prologue)
            else:
                self._seek(slot, self.prepend + 1 + kept)
                del slot.fed[kept:]
                logits = slot.logits
            # Single steps run at the weights' precision: transcribe's
            # generator drops its autocast after the first pass.
            for token in written[len(slot.fed) :]:
                logits = self._step(slot, [token], conditions)
                slot.fed.append(token)
        except Exception:
            slot.window = None
            raise
        slot.logits = logits
        return logits

    def _step(self, slot: _Slot, tokens: list[int], conditions: Any, first: bool = False) -> torch.Tensor:
        sequence = torch.tensor([tokens], device=self.device, dtype=torch.long)
        logits = self.lm._compute_logits(sequence, conditions, slot.state, first_step=first, cfg_coef=1.0)  # noqa: SLF001
        increment_steps(self.lm.transformer, slot.state, increment=len(tokens) + (self.prepend if first else 0))
        return logits[0]

    @staticmethod
    def _seek(slot: _Slot, position: int) -> None:
        """Rewind the cache: what lies past ``position`` is overwritten before it is read."""
        for state in slot.state.values():
            if "offset" in state:
                state["offset"] = position
            if "offsets" in state:
                state["offsets"].fill_(position)


def time_area(rows: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Widths that draw each row's time shifts as one: every shift the row
    gives weight is as wide as their mean -ln p, weighted by p, so together
    they keep their area; every other event keeps its own -ln p. Rows of ln
    weights up to a constant in, each child's own ln weight out."""
    rows = rows.astype(np.float64)
    top = rows.max(axis=1, keepdims=True)
    finite = np.isfinite(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_p = rows - (top + np.log(np.exp(rows - top).sum(axis=1, keepdims=True)))
        p = np.where(finite, np.exp(ln_p), 0.0)
        shifts = time[None, :] & finite & (0 < p)
        mass = np.where(shifts, p, 0.0).sum(axis=1, keepdims=True)
        area = np.where(shifts, -p * ln_p, 0.0).sum(axis=1, keepdims=True)
        width = area / mass
    return np.where(shifts, -width, ln_p).astype(np.float32)


# The tokenizer every MuScriptor size reads with, as ``TranscriptionModel.load_model``
# builds it. Its instrument groups name the programs: another vocabulary has the
# same events but names the groups' first programs wrong.
TOKENIZER = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


def labels_file() -> bytes:
    name = _build_instrument_for_program(TOKENIZER)
    instruments = {program: name(program).replace("_", " ") for program in range(130)}
    return json.dumps(
        {
            "kind": "notes",
            "frameRate": TOKENIZER.frame_rate,
            "window": WINDOW_SECONDS,
            "events": [[event.type, event.value] for event in TOKENIZER._vocab],  # noqa: SLF001
            "instruments": instruments,
            "drums": DRUMS,
        }
    ).encode()


@dataclass(frozen=True)
class Recorded:
    """An uploaded recording, as the model hears it: mono at 16 kHz."""

    name: str
    samples: torch.Tensor

    @property
    def seconds(self) -> float:
        return self.samples.shape[-1] / SAMPLE_RATE


def _recording_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:12]


def _decode_m4a(name: str, path: Path) -> torch.Tensor:
    try:
        decoded = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-xerror",
                # Decode the uploaded container without opening external tracks or URLs.
                "-protocol_whitelist", "file", "-f", "mov", "-enable_drefs", "0", "-i", str(path.resolve()),
                "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise ValueError(f"{name}: install FFmpeg on the scribe server to read M4A") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"{name}: FFmpeg could not decode audio: {detail}") from error
    return torch.from_numpy(np.frombuffer(decoded.stdout, dtype="<f4").astype(np.float32)).unsqueeze(0)


def _decoded(name: str, path: Path) -> Recorded:
    """The recording in ``path``; ValueError where it is no audio."""
    try:
        samples = load_audio(path, target_sr=SAMPLE_RATE)
    except (LibsndfileError, OSError, ValueError) as error:
        if not isinstance(error, LibsndfileError) or not name.lower().endswith(".m4a"):
            raise ValueError(f"{name}: could not read audio: {error}") from error
        samples = _decode_m4a(name, path)
    if samples.shape[-1] == 0:
        raise ValueError(f"{name} holds no sound")
    return Recorded(name, samples)


class Recordings:
    """The recordings held, by an id hashed from their bytes, so the same
    file is the same recording however often it comes. Those uploaded are
    kept in ``kept``, as ID/NAME, and held again from there on start."""

    def __init__(self, kept: Path):
        self.kept = kept
        self.held: dict[str, Recorded] = {}
        kept.mkdir(parents=True, exist_ok=True)
        for folder in sorted(kept.iterdir()):
            if folder.is_dir():
                for path in folder.iterdir():
                    self.add(path)

    def add(self, path: Path) -> str:
        """Hold the recording in ``path``; ValueError where it is no audio."""
        id = _recording_id(path.read_bytes())
        if id not in self.held:
            self.held[id] = _decoded(path.name, path)
        return id

    def upload(self, name: str, content: bytes) -> str:
        """Hold and keep an uploaded file; ValueError where it is no audio,
        or ``name`` no file name."""
        id = _recording_id(content)
        if id in self.held:
            return id
        # The name a client gave becomes a path: only its last part.
        name = Path(name).name
        if name in ("", ".."):
            raise ValueError(f"{name!r} is no file name")
        # Decoded before it is moved into place, so only whole recordings
        # are kept, and two uploads of one file never read each other's.
        file, temporary = tempfile.mkstemp(dir=self.kept, prefix=".upload-")
        try:
            with os.fdopen(file, "wb") as out:
                out.write(content)
            recorded = _decoded(name, Path(temporary))
            (self.kept / id).mkdir(exist_ok=True)
            os.replace(temporary, self.kept / id / name)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self.held[id] = recorded
        return id

    def __getitem__(self, id: str) -> Recorded:
        if id not in self.held:
            raise ValueError(f"no recording {id} on the server: upload it again")
        return self.held[id]

    def listed(self, id: str) -> dict:
        recorded = self.held[id]
        return {"id": id, "name": recorded.name, "seconds": recorded.seconds}

    def routes(self) -> list[Route]:
        async def upload(request: Request) -> Response:
            name = request.query_params.get("name", "recording")
            try:
                id = await anyio.to_thread.run_sync(self.upload, name, await request.body())
            except ValueError as error:
                return JSONResponse({"message": str(error)}, status_code=400)
            return JSONResponse(self.listed(id))

        async def listing(_: Request) -> Response:
            return JSONResponse([self.listed(id) for id in self.held])

        async def audio(request: Request) -> Response:
            recorded = self.held.get(request.path_params["id"])
            if recorded is None:
                return Response(status_code=404)
            return Response(await anyio.to_thread.run_sync(wav_file, recorded.samples), media_type="audio/wav")

        return [
            Route("/recordings", upload, methods=["POST"]),
            Route("/recordings", listing),
            Route("/recordings/{id}", audio),
        ]


def wav_file(samples: torch.Tensor) -> bytes:
    pcm = (samples[0].clamp(-1, 1) * 32767).round().to(torch.int16).cpu().numpy()
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm.tobytes())
    return out.getvalue()


def _load_model(size: str, device: torch.device, dtype: torch.dtype | None) -> TranscriptionModel:
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return TranscriptionModel.load_model(size, device=device, dtype=dtype)

    # MuScriptor 0.3.0 allocates FP32 weights on the requested device before casting.
    model = TranscriptionModel.load_model(size, device="cpu", dtype=dtype)
    decoder = model._model.to(device)  # noqa: SLF001
    # Moving the module leaves MuScriptor's device attributes at their original values.
    model._device = device  # noqa: SLF001
    provider = decoder.condition_provider
    provider.device = device
    for conditioner in provider.conditioners.values():
        conditioner.device = device
    decoder.autocast = TorchAutocast(enabled=True, dtype=torch.float16)
    return model


@dataclass
class Loaded:
    """A model size loaded on a device at a precision, and its transcribers
    by what each hears and expects: the measures that only reshape what one
    computes, at another temperature or as time-area, share it and its
    caches."""

    model: TranscriptionModel
    transcribers: dict[tuple, tuple[Scribe, Windows]] = field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("scribe.toml"),
        help="the models to offer as they are and recordings to offer from the start (default scribe.toml beside this script)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "natlas/scribe",
        help="where the recordings uploaded and the models loaded are kept (default $XDG_DATA_HOME/natlas/scribe)",
    )
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS, help="per window, counting its prologue and EOS")
    parser.add_argument("--slots", type=int, default=2, help="KV caches per transcriber, each holding one window")
    parser.add_argument("--cache", type=float, default=CACHE / 2**30, help=f"GiB of memory for the distributions it keeps, default {CACHE / 2**30:g}")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()
    try:
        config = tomllib.loads(args.config.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        parser.error(f"{args.config}: {error}")

    try:
        recordings = Recordings(args.data / "recordings")
    except (OSError, ValueError) as error:
        parser.error(f"{args.data}: {error}")
    for audio in config.get("recordings", []):
        try:
            recordings.add((args.config.parent / Path(audio).expanduser()).resolve())
        except (OSError, ValueError) as error:
            parser.error(f"{args.config}: {error}")
    time = Windows(TOKENIZER, 0.0, None, args.max_events).kinds == "shift"

    def load(values: dict) -> Loaded:
        device = torch.device(values["device"])
        dtype = None if values["dtype"] == "auto" else lm.DTYPES[values["dtype"]]
        return Loaded(_load_model(values["size"], device, dtype))

    def transcriber(loaded: Loaded, audio: str | None, at: float, instruments: str) -> tuple[Scribe, Windows]:
        key = (audio, at, instruments)
        if key not in loaded.transcribers:
            device = loaded.model._device  # noqa: SLF001
            expect = None
            if instruments:
                expect = instrument_group_from_names(resolve_instrument_names(instruments.split(",")))
            if audio is None:
                windows = Windows(TOKENIZER, at, None, args.max_events)
                silence = torch.zeros(1, WINDOW_SAMPLES, device=device)

                def hear(_: int) -> torch.Tensor:
                    return silence
            else:
                samples = recordings[audio].samples
                start = round(at * SAMPLE_RATE)
                if not 0 <= start < samples.shape[-1]:
                    raise ValueError(f"at {at:g} s is outside the recording's {samples.shape[-1] / SAMPLE_RATE:.1f} s")
                heard = samples[:, start:].to(device)
                windows = Windows(TOKENIZER, at, math.ceil(heard.shape[-1] / WINDOW_SAMPLES), args.max_events)

                def hear(index: int) -> torch.Tensor:
                    chunk = heard[:, index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES]
                    return F.pad(chunk, (0, WINDOW_SAMPLES - chunk.shape[-1]))

            loaded.transcribers[key] = (Scribe(loaded.model, windows, hear, expect, args.slots), windows)
        return loaded.transcribers[key]

    def open(loaded: Loaded, values: dict) -> lm.Measure:
        scribe, windows = transcriber(loaded, values["audio"], values["at"], values["instruments"])
        walk = lm.tempered(scribe.walk, values["temperature"])
        widths = values["shape"] == "time-area"
        if widths:
            plain = walk

            def walk(walks: Sequence[tuple[Sequence[int], int]]) -> list[np.ndarray]:
                return [time_area(rows, time) for rows in plain(walks)]

        # The start token carries the audio: paths are read whole.
        return lm.Measure(scribe.start, walk, scribe.width, scribe.paths, None, windows.ends, widths)

    catalog = Catalog(
        [Choice("size", SIZES, "medium"), Choice("device", lm.devices(), lm.devices()[0]), Choice("dtype", ("auto", *lm.DTYPES), "auto")],
        [
            Recording("audio"),
            Number("at", 0.0, min=0.0, step=0.1),
            Text("instruments", placeholder="e.g. piano,drums"),
            Number("temperature", 1.0, min=0.0, step=0.1),
            Choice("shape", ("none", "time-area"), "none"),
        ],
        load,
        open,
        released=lm.released,
    )
    manifest = [{"name": "scribe", "title": "MuScriptor · live", "labels": "labels.json", "catalog": catalog.describe()}]
    # MuScriptor's vocabulary is musical already: time shifts and pitches
    # ascending, so children keep its order.
    model = Model(TOKENIZER.num_tokens, catalog, round(args.cache * 2**30), "branch")
    files = {"labels.json": (labels_file(), "application/json")}
    uvicorn.run(app(model, manifest, files, routes=recordings.routes()), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
