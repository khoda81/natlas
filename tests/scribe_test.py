# /// script
# requires-python = ">=3.12"
# dependencies = ["natlas-core", "muscriptor==0.3.0", "numpy>=1.26", "protobuf>=7.36.1", "starlette>=0.40", "uvicorn>=0.30", "websockets>=13"]
# [tool.uv.sources]
# natlas-core = { path = "../core" }
# ///
"""Which events keep a transcription a valid MuScriptor encoding, with its real tokenizer and no model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from muscriptor.tokenizer.mt3 import MT3Tokenizer  # noqa: E402

import scribe  # noqa: E402

TOKENIZER = MT3Tokenizer()
EOS = TOKENIZER.eos_id


def token(kind: str, value: int = 0) -> int:
    return TOKENIZER._token_index[(kind, value)]  # noqa: SLF001


def windows(at: float = 5.0, count: int = 3) -> scribe.Windows:
    return scribe.Windows(SimpleNamespace(_tokenizer=TOKENIZER), at, count, scribe.MAX_EVENTS)


class TimeAreaTests(unittest.TestCase):
    def test_time_shifts_share_a_width_that_keeps_their_area(self):
        import numpy as np

        # Two shifts of p 0.5 and 0.1, a pitch of 0.4, and a masked shift.
        time = np.array([True, True, False, True])
        rows = np.log(np.array([[0.5, 0.1, 0.4, 0.0]]))
        with np.errstate(divide="ignore"):
            widths = scribe.time_area(rows.astype(np.float32), time)[0]
        width = -(0.5 * np.log(0.5) + 0.1 * np.log(0.1)) / 0.6
        np.testing.assert_allclose(widths[:3], [-width, -width, np.log(0.4)], rtol=1e-6)
        self.assertEqual(widths[3], -np.inf)
        # Their area, p times width, is theirs.
        self.assertAlmostEqual(0.6 * width, -(0.5 * np.log(0.5) + 0.1 * np.log(0.1)), places=6)


class ValidTests(unittest.TestCase):
    def allowed(self, path: list[int], at: float = 5.0, count: int = 3) -> set[tuple[str, int]]:
        mask = windows(at, count).valid(tuple(path))
        return {(event.type, event.value) for event, ok in zip(TOKENIZER._vocab, mask, strict=True) if ok}  # noqa: SLF001

    def test_the_recordings_start_forces_an_empty_tie_prologue(self):
        start = windows(at=0.0)
        window, written = start.locate(())
        self.assertEqual(start.forced(window, written), token("tie"))
        later = windows(at=5.0)
        window, written = later.locate(())
        self.assertIsNone(later.forced(window, written))

    def test_a_prologue_names_instruments_then_their_pitches_once_each_ascending(self):
        opening = self.allowed([])
        self.assertEqual({kind for kind, _ in opening}, {"program", "tie"})
        after_program = self.allowed([token("program", 24)])
        self.assertEqual({kind for kind, _ in after_program}, {"pitch"})
        after_pitch = self.allowed([token("program", 24), token("pitch", 51)])
        self.assertNotIn(("pitch", 51), after_pitch)
        self.assertIn(("pitch", 52), after_pitch)
        self.assertNotIn(("program", 24), after_pitch)
        self.assertIn(("program", 25), after_pitch)
        self.assertIn(("tie", 0), after_pitch)
        self.assertNotIn(("shift", 7), after_pitch)

    def test_a_pitch_needs_an_instrument_and_a_switch_each_used_before_the_next(self):
        body = [token("tie")]
        self.assertFalse({kind for kind, _ in self.allowed(body)} & {"pitch", "tie"})
        self.assertIn(("EOS", 0), self.allowed(body))
        instrument = self.allowed([*body, token("program", 24)])
        self.assertNotIn(("EOS", 0), instrument)
        self.assertNotIn(("program", 0), instrument)
        self.assertNotIn(("pitch", 60), instrument)
        on = [*body, token("program", 24), token("velocity", 1)]
        self.assertIn(("pitch", 60), self.allowed(on))
        self.assertNotIn(("velocity", 0), self.allowed(on))
        played = self.allowed([*on, token("pitch", 60)])
        self.assertIn(("EOS", 0), played)
        self.assertNotIn(("velocity", 1), played)
        self.assertNotIn(("program", 24), played)
        off = self.allowed([*on, token("pitch", 60), token("velocity", 0)])
        self.assertEqual({value for kind, value in off if kind == "pitch"}, {60})

    def test_time_moves_forward_within_the_window_except_the_last(self):
        shifted = self.allowed([token("tie"), token("shift", 20)])
        self.assertNotIn(("shift", 20), shifted)
        self.assertIn(("shift", 21), shifted)
        self.assertIn(("shift", 499), shifted)
        self.assertNotIn(("shift", 500), shifted)
        last = [token("tie"), EOS, token("tie"), EOS, token("tie")]
        self.assertIn(("shift", 700), self.allowed(last))

    def test_a_forced_later_prologue_leaves_its_instrument_set(self):
        played = [token("tie"), token("program", 24), token("velocity", 1), token("pitch", 60), EOS]
        later = windows()
        window, _ = later.locate(tuple(played))
        self.assertEqual(window.prologue, (token("program", 24), token("pitch", 60), token("tie")))
        body = self.allowed([*played, *window.prologue])
        self.assertNotIn(("program", 24), body)
        self.assertIn(("velocity", 1), body)


if __name__ == "__main__":
    unittest.main()
