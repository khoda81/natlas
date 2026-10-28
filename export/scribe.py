# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "muscriptor==0.3.0",
#   "numpy>=1.26",
#   "protobuf>=7.36.1",
#   "starlette>=0.40",
#   "uvicorn>=0.30",
# ]
# ///

"""Serve MuScriptor's tree of transcriptions of a recording to natlas, live.

MuScriptor transcribes music 5 seconds at a time, writing each window as
MIDI-like events: a tie prologue naming the notes sustained into the window,
then time shifts, instruments, note-ons and note-offs, pitches and drum hits,
then EOS. A path here is one transcription of the whole recording from --at
on, scrolling through the windows as ``TranscriptionModel.transcribe`` does
with its prelude forcing:

* within a window, an event has MuScriptor's probability given that window's
  audio and the window's events so far;
* EOS is the model's own decision; it ends the window and moves to the next;
* a later window's tie prologue is forced from the notes the path leaves
  sounding, so those events have probability one;
* a window that reaches MuScriptor's generation limit gets a forced EOS;
* the last window's EOS ends the path.

So a path's probability is the probability MuScriptor's chunked decoder
gives that transcription, as in natwalk's MuScriptor demo.

The measures are the model itself, then each --q: another model size hearing
the same recording, ``silence`` for the model hearing nothing (so p against
it shows what the audio told the model), ``instruments:NAMES`` for the model
told to expect those instruments, or ``temperature:T``. GET /labels.json
describes the events, and GET /audio.wav is the recording from --at on.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import sys
import wave
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lm
import torch
import torch.nn.functional as F
import uvicorn
from muscriptor import TranscriptionModel
from muscriptor.events import ChunkBoundary, OpenNoteTracker
from muscriptor.modules.streaming import increment_steps, init_states
from muscriptor.tokenizer.mt3 import instrument_group_from_names, resolve_instrument_names
from serve import Model, _slug, app

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
    """How paths run through the recording's windows; the same for every measure."""

    def __init__(self, model: TranscriptionModel, at: float, count: int, max_events: int):
        self.tokenizer = model._tokenizer  # noqa: SLF001
        self.eos = self.tokenizer.eos_id
        self.at = at
        self.count = count
        self.max_events = max_events
        # The decoder's state on entering a window, by the path up to that
        # window's start, which ends in the EOS before it.
        self.entered: OrderedDict[tuple[int, ...], tuple[Window, OpenNoteTracker]] = OrderedDict()

    def ends(self, path: tuple[int, ...]) -> bool:
        return path.count(self.eos) >= self.count

    def locate(self, path: tuple[int, ...]) -> tuple[Window, tuple[int, ...]]:
        """The window ``path`` ends in, and the events written in it so far, its prologue first."""
        start = len(path) - path[::-1].index(self.eos) if self.eos in path else 0
        return self._state(path[:start])[0], path[start:]

    def forced(self, window: Window, written: tuple[int, ...]) -> int | None:
        if len(written) < len(window.prologue):
            return window.prologue[len(written)]
        if len(written) >= self.max_events:
            return self.eos
        return None

    def _boundary(self, index: int) -> ChunkBoundary:
        following = self.at + (index + 1) * WINDOW_SECONDS if index + 1 < self.count else None
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
            # Nothing came before the first window to force its prologue from.
            entered = (Window(0, ()), tracker)
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
    """MuScriptor transcribing a recording: each event's log-probability given the path before it.

    A few KV caches, one per window in reach, each stepped back to the prefix
    a request shares with what it holds, so walking down a path costs one
    step per event.
    """

    def __init__(
        self,
        model: TranscriptionModel,
        name: str,
        windows: Windows,
        hear: Callable[[int], torch.Tensor],
        expect: str | None = None,
        slots: int = 2,
    ):
        self.name = name
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

    def log_probs(self, tokens: Sequence[int], temperature: float = 1.0) -> torch.Tensor:
        if tuple(tokens[:1]) != self.start:
            raise ValueError("a transcription begins with the start token")
        window, written = self.windows.locate(tuple(int(token) for token in tokens[1:]))
        forced = self.windows.forced(window, written)
        if forced is not None:
            certain = torch.full((self.vocabulary,), -math.inf, dtype=torch.float64)
            certain[forced] = 0.0
            return certain
        logits = self._evaluate(window, list(written))
        return torch.log_softmax(logits[: self.vocabulary].double() / temperature, dim=-1)

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
        """The cache that saves the most steps, and how many of its events to keep, or None to start over."""
        best: tuple[tuple[int, int], _Slot, int | None] | None = None
        for slot in self.slots:
            kept = None
            if slot.window == window:
                shared = 0
                for have, want in zip(slot.fed, written):
                    if have != want:
                        break
                    shared += 1
                # Keep the prologue whole: it is written in one pass.
                if shared >= len(window.prologue):
                    kept = shared
            score = (-1 if kept is None else kept, -slot.used)
            if best is None or score > best[0]:
                best = (score, slot, kept)
        if best is None or (best[2] is None and len(self.slots) < self.capacity):
            sequence_length = self.prepend + 1 + self.windows.max_events
            slot = _Slot(init_states(self.lm, batch_size=1, sequence_length=sequence_length))
            self.slots.append(slot)
            best = ((-1, 0), slot, None)
        _, slot, kept = best
        self.clock += 1
        slot.used = self.clock
        return slot, kept

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


def labels_file(model: TranscriptionModel, windows: Windows) -> bytes:
    tokenizer = model._tokenizer  # noqa: SLF001
    instruments = {program: model._instrument_for_program(program).replace("_", " ") for program in range(130)}  # noqa: SLF001
    return json.dumps(
        {
            "kind": "notes",
            "frameRate": tokenizer.frame_rate,
            "window": WINDOW_SECONDS,
            "windows": windows.count,
            "at": windows.at,
            "audio": "audio.wav",
            "events": [[event.type, event.value] for event in tokenizer._vocab],  # noqa: SLF001
            "instruments": instruments,
            "drums": DRUMS,
        }
    ).encode()


def wav_file(samples: torch.Tensor) -> bytes:
    pcm = (samples[0].clamp(-1, 1) * 32767).round().to(torch.int16).cpu().numpy()
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm.tobytes())
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="any file soundfile reads: wav, mp3, flac, …")
    parser.add_argument("--at", type=float, default=0.0, help="where the first 5-second window starts, in seconds")
    parser.add_argument("--p", default="medium", choices=SIZES, help="the model size")
    parser.add_argument(
        "--q",
        action="append",
        default=[],
        help="further measure, repeatable: a model size, silence, instruments:NAMES or temperature:T",
    )
    parser.add_argument("--device", default=lm.default_device())
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS, help="per window, counting its prologue and EOS")
    parser.add_argument("--slots", type=int, default=2, help="KV caches per measure, each holding one window")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()
    if any(spec.startswith("context:") for spec in args.q):
        parser.error("context:N would cut the start token, which carries the audio")

    print(f"loading muscriptor-{args.p} …", file=sys.stderr)
    models = {args.p: TranscriptionModel.load_model(args.p, device=args.device)}
    wav = models[args.p]._load_wav(args.audio, None)  # noqa: SLF001
    start = round(args.at * SAMPLE_RATE)
    if start >= wav.shape[-1]:
        parser.error(f"--at {args.at:g} is past the audio's end at {wav.shape[-1] / SAMPLE_RATE:.1f} s")
    recording = wav[:, start:]
    windows = Windows(models[args.p], args.at, math.ceil(recording.shape[-1] / WINDOW_SAMPLES), args.max_events)

    def heard(index: int) -> torch.Tensor:
        chunk = recording[:, index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES]
        return F.pad(chunk, (0, WINDOW_SAMPLES - chunk.shape[-1]))

    def silent(_: int) -> torch.Tensor:
        return torch.zeros_like(recording[:, :WINDOW_SAMPLES])

    def scribe(size: str, name: str, hear: Callable[[int], torch.Tensor] = heard, expect: str | None = None) -> Scribe:
        if size not in models:
            print(f"loading muscriptor-{size} …", file=sys.stderr)
            models[size] = TranscriptionModel.load_model(size, device=args.device)
        return Scribe(models[size], name, windows, hear, expect, args.slots)

    own = scribe(args.p, f"muscriptor-{args.p}")

    def other(spec: str) -> Scribe:
        kind, _, value = spec.partition(":")
        if spec in SIZES:
            return scribe(spec, f"muscriptor-{spec}")
        if spec == "silence":
            return scribe(args.p, f"{own.name} · silence", silent)
        if kind == "instruments":
            names = resolve_instrument_names(value.split(","))
            return scribe(args.p, f"{own.name} · expecting {', '.join(names)}", expect=instrument_group_from_names(names))
        raise SystemExit(f"unknown measure {spec!r}: give a model size, silence, instruments:NAMES or temperature:T")

    measures = lm.measures(own, args.q, other)
    seconds = recording.shape[-1] / SAMPLE_RATE
    manifest = [
        {
            "name": f"scribe-{_slug(args.audio.stem)}-{args.at:g}",
            "title": f"{args.audio.stem} · {args.at:g}–{args.at + seconds:.0f} s · live",
            "labels": "labels.json",
            "start": [],
            "measures": [{"name": measure.name, "url": f"{index}/"} for index, measure in enumerate(measures)],
        }
    ]
    model = Model(own.start, windows.ends, own.vocabulary, measures)
    files = {
        "labels.json": (labels_file(models[args.p], windows), "application/json"),
        "audio.wav": (wav_file(recording), "audio/wav"),
    }
    uvicorn.run(app(model, manifest, files), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
