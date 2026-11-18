// Labels that are a transcription's events, as MuScriptor writes music. A
// recording is cut into windows of a few seconds, and each window's events
// are a tie prologue naming the notes sustained into it, ended by a tie event,
// then time shifts from the window's start, instrument and on/off switches,
// pitches that start or end a note in the current instrument, drum hits, and
// an EOS that moves on to the next window. A path reads as a piano roll,
// which the panel draws and can play.

import { clock, DRUM_PROGRAM, pitchName, tint } from "./music";
import { type Glyph, type Labels, type Measure, type Page, type Reading, States, type Trail } from "./reading";
import { ScorePanel } from "./roll";
import type { IntervalTree, Branch, NodeId } from "../core/tree";

type Event = readonly [type: string, value: number];

/** A notes labels file. */
export interface NotesFile {
  kind: "notes";
  /** Shift steps per second. */
  frameRate: number;
  /** Seconds per window. */
  window: number;
  /** Each label's event, indexed by label. */
  events: Event[];
  /** Instrument names by program. */
  instruments?: Record<string, string>;
  /** Drum names by pitch. */
  drums?: Record<string, string>;
}

/** What a reading transcribes: p's recording, or silence. */
export interface Heard {
  /** Where the first window starts in the recording, in seconds. */
  at: number;
  /** Windows in the recording, the last one's EOS ending a path: endless under silence. */
  windows: number;
  /** The recording from its start, where there is one. */
  audio: URL | null;
}

const SILENCE: Heard = { at: 0, windows: Infinity, audio: null };

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
  /** Where time stood before it came to where it is now. */
  trail: Trail | null;
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

/** What a transcript file says it is. */
const PATH_KIND = "natlas.transcript";

const START: Score = { window: 0, tick: 0, program: null, velocity: null, prologue: true, spoiled: false, ended: false, trail: null };
export class NoteLabels implements Labels {
  private readonly instruments: Map<number, string>;
  private readonly drums: Map<number, string>;
  /** Shift steps per window. */
  readonly span: number;

  constructor(
    readonly file: NotesFile,
    readonly heard: Heard = SILENCE,
  ) {
    const numbered = (names: Record<string, string> | undefined) =>
      new Map(Object.entries(names ?? {}).map(([key, name]) => [Number(key), name]));
    this.instruments = numbered(file.instruments);
    this.drums = numbered(file.drums);
    this.span = file.window * file.frameRate;
  }

  event(branch: Branch): Event {
    return this.file.events[branch] ?? ["UNK", branch];
  }

  /** The decoder's state after ``branch``. */
  after(score: Score, branch: Branch): Score {
    if (score.ended) return score;
    const [type, value] = this.event(branch);
    if (type === "EOS") {
      if (score.window + 1 >= this.heard.windows) return { ...score, ended: true };
      return { ...START, window: score.window + 1, trail: this.moved(score, (score.window + 1) * this.span) };
    }
    if (score.prologue) {
      if (type === "tie") return { ...score, prologue: false, velocity: null };
      if (type === "shift") return { ...score, prologue: false, spoiled: true };
      if (type === "program") return { ...score, program: value };
      return score;
    }
    if (score.spoiled) return score;
    if (type === "shift") return value > 0 ? { ...score, tick: value, trail: this.moved(score, score.window * this.span + value) } : score;
    if (type === "program") return { ...score, program: value };
    if (type === "velocity") return { ...score, velocity: value };
    return score;
  }

  /** The trail of time moved from ``score`` to ``to``: where it stood joins it. */
  private moved(score: Score, to: number): Trail | null {
    const at = score.window * this.span + score.tick;
    return at === to ? score.trail : { at, before: score.trail };
  }

  /** ``path`` as a transcript file: its events by name, which outlive the labels' numbering. */
  pathFile(path: readonly Branch[]): string {
    return JSON.stringify({ kind: PATH_KIND, events: path.map((branch) => this.event(branch)) });
  }

  /** The path a transcript file writes, refusing a file of anything else or
   * of an event these labels do not have. */
  parsePath(json: string): Branch[] {
    const file = JSON.parse(json) as { kind?: unknown; events?: unknown } | null;
    if (file?.kind !== PATH_KIND || !Array.isArray(file.events)) throw new Error("not a natlas transcript file");
    const branches = new Map(this.file.events.map(([type, value], branch) => [`${type}:${value}`, branch]));
    return file.events.map((event: unknown) => {
      const branch = Array.isArray(event) && event.length === 2 ? branches.get(`${event[0]}:${event[1]}`) : undefined;
      if (branch === undefined) throw new Error(`this transcriber writes no event ${JSON.stringify(event)}`);
      return branch;
    });
  }

  instrument(program: number): string {
    return program === DRUM_PROGRAM ? "drums" : (this.instruments.get(program) ?? `program ${program}`);
  }

  drum(pitch: number): string {
    return this.drums.get(pitch) ?? pitchName(pitch);
  }

  /** Seconds into the recording at ``tick`` shift steps from the first window's start. */
  seconds(tick: number): number {
    return this.heard.at + tick / this.file.frameRate;
  }

  /** ``branch`` read in ``score``, the state after its parent's path. */
  glyph(score: Score, branch: Branch): Glyph {
    const [type, value] = this.event(branch);
    switch (type) {
      case "EOS":
        if (score.ended || score.window + 1 >= this.heard.windows) return { text: "end", mark: "end" };
        return { text: `⇥ ${clock(this.seconds((score.window + 1) * this.span), 0)}`, mark: "end" };
      case "shift": {
        const start = score.window * this.span;
        const seconds = this.seconds(start + value);
        // Under a second, in milliseconds: shifts are steps of 10 ms.
        const text = seconds < 1 ? `${Math.round(seconds * 1000)} ms` : clock(seconds);
        return { text, mark: "time", time: { from: start + score.tick, to: start + value, trail: score.trail } };
      }
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
  transcribe(path: readonly Branch[]): Transcript {
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
    for (const branch of path) {
      if (score.ended) break;
      const [type, value] = this.event(branch);
      touched = new Set();
      const last = score.window + 1 >= this.heard.windows;
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
      score = this.after(score, branch);
    }
    const now = score.ended ? this.heard.windows * this.span : score.window * this.span + Math.min(score.tick, this.span);
    // A note ended where it started, as a spoiled prologue leaves its guesses, sounds nowhere.
    return { notes: notes.filter((note) => note.end !== note.start), hits, score, now, touched };
  }

  read(tree: IntervalTree, page: Page, p: Measure): Reading {
    const at = Number(p.values.get("at") || 0);
    const heard = p.recording === null
      ? { ...SILENCE, at }
      : { at, windows: Math.ceil((p.recording.seconds - at) / this.file.window), audio: p.recording.url };
    return new NoteLabels(this.file, heard).reading(tree, page);
  }

  private reading(tree: IntervalTree, page: Page): Reading {
    const scores = new States<Score>(tree, START, (score, branch) => this.after(score, branch));
    const panel = new ScorePanel(this, page);
    return {
      glyph: (node: NodeId, branch: Branch) => this.glyph(scores.at(node), branch),
      show: (element: HTMLElement, node: NodeId, branch: Branch | null) => {
        const path = tree.path(node);
        if (branch !== null) path.push(branch);
        panel.show(element, path);
      },
      key: (event: KeyboardEvent) => {
        if (event.key !== " ") return false;
        panel.toggle(event.shiftKey ? "both" : "notes");
        return true;
      },
      timeline: () => panel.timeline(),
    };
  }
}
