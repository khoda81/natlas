// Labels that are a transcription's events, as MuScriptor writes music. A
// recording is cut into windows of a few seconds, and each window's events
// are a tie prologue naming the notes sustained into it, ended by a tie event,
// then time shifts from the window's start, instrument and on/off switches,
// pitches that start or end a note in the current instrument, drum hits, and
// an EOS that moves on to the next window. A path reads as a piano roll,
// which the panel draws and can play.

import { clock, DRUM_PROGRAM, pitchName, tint } from "./music";
import { type Glyph, type Labels, type Reading, States } from "./reading";
import { ScorePanel } from "./roll";
import type { IntervalTree, Label, NodeId } from "./tree";

type Event = readonly [type: string, value: number];

/** A notes labels file. */
export interface NotesFile {
  kind: "notes";
  /** Shift steps per second. */
  frameRate: number;
  /** Seconds per window. */
  window: number;
  /** Windows in the recording; the last one's EOS ends a path. */
  windows: number;
  /** Where the first window starts in the recording, in seconds. */
  at?: number;
  /** The recording from the first window on, relative to the labels file. */
  audio?: string;
  /** Each label's event, indexed by label. */
  events: Event[];
  /** Instrument names by program. */
  instruments?: Record<string, string>;
  /** Drum names by pitch. */
  drums?: Record<string, string>;
}

/** What the decoder knows after a path: all that a label's reading depends on. */
export interface Score {
  window: number;
  /** Shift steps from the window's start. */
  tick: number;
  program: number | null;
  /** 1 while pitches start notes, 0 while they end them. */
  velocity: number | null;
  prologue: boolean;
  /** A shift inside the prologue spoils the window: the rest of it is dropped. */
  spoiled: boolean;
  /** The last window's EOS came. */
  ended: boolean;
}

/** Times are shift steps from the first window's start. */
export interface Note {
  program: number;
  pitch: number;
  start: number;
  /** Null while it still sounds. */
  end: number | null;
  /** Sounding before the first window, as its prologue says. */
  carried: boolean;
}

export interface Hit {
  pitch: number;
  tick: number;
}

export interface Transcript {
  notes: Note[];
  hits: Hit[];
  score: Score;
  /** The time written up to, in shift steps from the first window's start. */
  now: number;
  /** What the path's last event started, ended or hit. */
  touched: Set<Note | Hit>;
}

const START: Score = { window: 0, tick: 0, program: null, velocity: null, prologue: true, spoiled: false, ended: false };
export class NoteLabels implements Labels {
  private readonly instruments: Map<number, string>;
  private readonly drums: Map<number, string>;
  /** Shift steps per window. */
  readonly span: number;

  constructor(
    readonly file: NotesFile,
    /** Where the labels file came from, which its audio is relative to. */
    readonly base: URL | null = null,
  ) {
    const numbered = (names: Record<string, string> | undefined) =>
      new Map(Object.entries(names ?? {}).map(([key, name]) => [Number(key), name]));
    this.instruments = numbered(file.instruments);
    this.drums = numbered(file.drums);
    this.span = file.window * file.frameRate;
  }

  event(label: Label): Event {
    return this.file.events[label] ?? ["UNK", label];
  }

  /** The decoder's state after ``label``. */
  after(score: Score, label: Label): Score {
    if (score.ended) return score;
    const [type, value] = this.event(label);
    if (type === "EOS") {
      if (score.window + 1 >= this.file.windows) return { ...score, ended: true };
      return { ...START, window: score.window + 1 };
    }
    if (score.prologue) {
      if (type === "tie") return { ...score, prologue: false, velocity: null };
      if (type === "shift") return { ...score, prologue: false, spoiled: true };
      if (type === "program") return { ...score, program: value };
      return score;
    }
    if (score.spoiled) return score;
    if (type === "shift") return value > 0 ? { ...score, tick: value } : score;
    if (type === "program") return { ...score, program: value };
    if (type === "velocity") return { ...score, velocity: value };
    return score;
  }

  instrument(program: number): string {
    return program === DRUM_PROGRAM ? "drums" : (this.instruments.get(program) ?? `program ${program}`);
  }

  drum(pitch: number): string {
    return this.drums.get(pitch) ?? pitchName(pitch);
  }

  /** Seconds into the recording at ``tick`` shift steps from the first window's start. */
  seconds(tick: number): number {
    return (this.file.at ?? 0) + tick / this.file.frameRate;
  }

  /** ``label`` read in ``score``, the state after its parent's path. */
  glyph(score: Score, label: Label): Glyph {
    const [type, value] = this.event(label);
    switch (type) {
      case "EOS":
        if (score.ended || score.window + 1 >= this.file.windows) return { text: "end", mark: "end" };
        return { text: `⇥ ${clock(this.seconds((score.window + 1) * this.span), 0)}`, mark: "end" };
      case "shift":
        return { text: clock(this.seconds(score.window * this.span + value)), mark: "time" };
      case "tie":
        return { text: "tie", mark: "tie" };
      case "program":
        return { text: this.instrument(value), mark: "instrument", tint: tint(value) };
      case "velocity":
        return { text: value > 0 ? "on" : "off" };
      case "drum":
        return { text: this.drum(value), mark: "hit", tint: tint(DRUM_PROGRAM) };
      case "pitch": {
        const name = pitchName(value);
        if (score.ended || score.spoiled || score.program === null) return { text: name };
        if (score.prologue) return { text: name, mark: "tie", tint: tint(score.program) };
        if (score.velocity === null) return { text: name };
        return { text: name, mark: score.velocity > 0 ? "onset" : "offset", tint: tint(score.program) };
      }
      default:
        return { text: type.toLowerCase() };
    }
  }

  /** The notes a path writes, as MuScriptor's own decoder reads its windows in turn. */
  transcribe(path: readonly Label[]): Transcript {
    let score = START;
    const open = new Map<string, Note>();
    const notes: Note[] = [];
    const hits: Hit[] = [];
    // The prologue's notes: on its tie, any other sounding note ends.
    const tied = new Set<string>();
    let touched = new Set<Note | Hit>();
    const windowStart = () => score.window * this.span;
    const endAll = (at: number) => {
      for (const note of open.values()) {
        note.end = at;
        touched.add(note);
      }
      open.clear();
    };
    for (const label of path) {
      if (score.ended) break;
      const [type, value] = this.event(label);
      touched = new Set();
      const last = score.window + 1 >= this.file.windows;
      if (type === "EOS") {
        // A window that never closed its prologue ends what sounds at its start.
        if (score.prologue) endAll(windowStart());
        // The recording ends with the last window.
        else if (last) endAll(windowStart() + this.span);
        tied.clear();
      } else if (score.prologue) {
        if (type === "pitch" && score.program !== null) {
          const key = `${score.program}:${value}`;
          tied.add(key);
          let note = open.get(key);
          if (note === undefined && score.window === 0) {
            note = { program: score.program, pitch: value, start: 0, end: null, carried: true };
            open.set(key, note);
            notes.push(note);
          }
          if (note !== undefined) touched.add(note);
        } else if (type === "tie") {
          for (const [key, note] of open) {
            if (tied.has(key)) continue;
            note.end = windowStart();
            open.delete(key);
            touched.add(note);
          }
          tied.clear();
        } else if (type === "shift") endAll(windowStart());
      } else if (!score.spoiled) {
        const tick = windowStart() + score.tick;
        // Events past the window belong to the next one, which writes them itself.
        const inside = last || score.tick < this.span;
        if (type === "drum" && inside) {
          const hit = { pitch: value, tick };
          hits.push(hit);
          touched.add(hit);
        } else if (type === "pitch" && inside && score.program !== null && score.velocity !== null) {
          const key = `${score.program}:${value}`;
          const sounding = open.get(key);
          if (sounding !== undefined) {
            sounding.end = tick;
            open.delete(key);
            touched.add(sounding);
          }
          if (score.velocity > 0) {
            const note: Note = { program: score.program, pitch: value, start: tick, end: null, carried: false };
            open.set(key, note);
            notes.push(note);
            touched.add(note);
          }
        }
      }
      score = this.after(score, label);
    }
    const now = score.ended ? this.file.windows * this.span : score.window * this.span + Math.min(score.tick, this.span);
    // A note ended where it started, as a spoiled prologue leaves its guesses, sounds nowhere.
    return { notes: notes.filter((note) => note.end !== note.start), hits, score, now, touched };
  }

  read(tree: IntervalTree): Reading {
    const scores = new States<Score>(tree, START, (score, label) => this.after(score, label));
    const panel = new ScorePanel(this);
    return {
      glyph: (node: NodeId, label: Label) => this.glyph(scores.at(node), label),
      show: (element: HTMLElement, node: NodeId, label: Label | null) => {
        const path = tree.path(node);
        if (label !== null) path.push(label);
        panel.show(element, this.transcribe(path));
      },
      key: (event: KeyboardEvent) => {
        if (event.key !== " ") return false;
        panel.toggle(event.shiftKey ? "both" : "notes");
        return true;
      },
    };
  }
}
