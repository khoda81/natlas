import { describe, expect, it } from "bun:test";
import { NoteLabels, type NotesFile } from "../src/notes";

// MuScriptor's vocabulary layout: special tokens, shifts, then the note events.
const ranges: [string, number][] = [["PAD", 1], ["EOS", 1], ["UNK", 1], ["shift", 1001], ["pitch", 128], ["velocity", 2], ["tie", 1], ["program", 130], ["drum", 128]];
const events: [string, number][] = ranges.flatMap(([type, count]) => Array.from({ length: count }, (_, value): [string, number] => [type, value]));
const file: NotesFile = { kind: "notes", frameRate: 100, window: 5, windows: 3, at: 30, events, instruments: { 24: "acoustic guitar" } };
const labels = new NoteLabels(file);
const id = (type: string, value = 0) => events.findIndex(([t, v]) => t === type && v === value);
const [program, pitch, tie, shift, velocity, drum, eos] = [
  (v: number) => id("program", v),
  (v: number) => id("pitch", v),
  id("tie"),
  (v: number) => id("shift", v),
  (v: number) => id("velocity", v),
  (v: number) => id("drum", v),
  id("EOS"),
];

// The start of a chunk as MuScriptor wrote it: E3 sustained into the chunk,
// ended at 0.05 s as C3 starts, which ends at 0.28 s.
const path = [program(24), pitch(52), tie, shift(5), velocity(0), pitch(52), velocity(1), pitch(48), shift(28), velocity(0), pitch(48)];

describe("NoteLabels", () => {
  it("reads a path as notes, the prologue's carried in from the start", () => {
    const { notes, score } = labels.transcribe(path);
    expect(notes).toEqual([
      { program: 24, pitch: 52, start: 0, end: 5, carried: true },
      { program: 24, pitch: 48, start: 5, end: 28, carried: false },
    ]);
    expect(score).toMatchObject({ window: 0, tick: 28, program: 24, velocity: 0, prologue: false });
  });

  it("marks what the last event touched, and leaves a started note open", () => {
    const started = labels.transcribe([...path.slice(0, 8)]);
    expect(started.notes.at(-1)).toEqual({ program: 24, pitch: 48, start: 5, end: null, carried: false });
    expect([...started.touched]).toEqual([started.notes.at(-1)!]);
  });

  it("reads a pitch by the state before it", () => {
    const at = (prefix: number[]) => prefix.reduce((score, label) => labels.after(score, label), labels.transcribe([]).score);
    expect(labels.glyph(at([program(24)]), pitch(52))).toMatchObject({ text: "E3", mark: "tie" });
    expect(labels.glyph(at(path.slice(0, 5)), pitch(52))).toMatchObject({ text: "E3", mark: "offset" });
    expect(labels.glyph(at(path.slice(0, 7)), pitch(48))).toMatchObject({ text: "C3", mark: "onset" });
    expect(labels.glyph(at([]), program(24))).toMatchObject({ text: "acoustic guitar", mark: "instrument" });
    expect(labels.glyph(at([]), shift(123))).toEqual({ text: "31.23 s", mark: "time" });
    expect(labels.glyph(at([eos]), shift(123))).toEqual({ text: "36.23 s", mark: "time" });
    expect(labels.glyph(at([]), eos)).toEqual({ text: "⇥ 35 s", mark: "end" });
    expect(labels.glyph(at([eos, eos]), eos)).toEqual({ text: "end", mark: "end" });
  });

  it("carries notes into the next window through its prologue, and ends the rest at its start", () => {
    // C3 is still sounding at the first window's EOS; the next prologue ties only E3, which never started.
    const crossing = [...path.slice(0, 8), eos, program(24), pitch(52), tie, shift(10), velocity(1), pitch(60)];
    const { notes, score, now } = labels.transcribe(crossing);
    expect(notes).toEqual([
      { program: 24, pitch: 52, start: 0, end: 5, carried: true },
      { program: 24, pitch: 48, start: 5, end: 500, carried: false },
      { program: 24, pitch: 60, start: 510, end: null, carried: false },
    ]);
    expect(score.window).toBe(1);
    expect(now).toBe(510);
    // Tied into the next window, C3 sounds on through it.
    const tied = labels.transcribe([...path.slice(0, 8), eos, program(24), pitch(48), tie, shift(20), velocity(0), pitch(48)]);
    expect(tied.notes[1]).toEqual({ program: 24, pitch: 48, start: 5, end: 520, carried: false });
  });

  it("leaves events past a window's end to the next window, except in the last", () => {
    const past = [tie, shift(600), velocity(1), program(24), pitch(60), drum(36)];
    expect(labels.transcribe(past).notes).toEqual([]);
    expect(labels.transcribe([eos, tie, eos, tie, ...past.slice(1)]).notes).toEqual([
      { program: 24, pitch: 60, start: 1600, end: null, carried: false },
    ]);
  });

  it("drops a chunk whose prologue a shift spoils, as MuScriptor does", () => {
    const { notes, hits, score } = labels.transcribe([program(24), pitch(52), shift(5), velocity(1), pitch(60), drum(36)]);
    expect(notes).toEqual([]);
    expect(hits).toEqual([]);
    expect(score.spoiled).toBe(true);
  });

  it("hits drums at the current time and stops at the last window's end", () => {
    const { hits, score } = labels.transcribe([eos, tie, eos, tie, shift(40), drum(36), shift(90), drum(38), eos, drum(42)]);
    expect(hits).toEqual([
      { pitch: 36, tick: 1040 },
      { pitch: 38, tick: 1090 },
    ]);
    expect(score.ended).toBe(true);
  });
});
