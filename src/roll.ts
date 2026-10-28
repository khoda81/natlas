// The path panel for notes: a piano roll of a transcript, which follows the
// window being written, can be scrolled, zoomed and resized, names its notes,
// and plays them alone or over the recording.

import { clock, DRUM_PROGRAM, pitchName, tint } from "./music";
import type { Hit, Note, NoteLabels, Transcript } from "./notes";

type Playing = "notes" | "audio" | "both";

const GUTTER = 38;
const AXIS = 16;
const DRUM_LANE = 16;
/** Rows never get thinner than this; past it the roll scrolls. */
const MIN_ROW = 12;
/** Rows this tall name every pitch, and the notes in them. */
const NAMED_ROW = 11;
const HEIGHT = 260;
const HEIGHT_KEY = "natlas.roll.height";
/** The least pitch range shown, so a few notes do not fill the roll. */
const MIN_PITCHES = 25;
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

interface Geometry {
  width: number;
  height: number;
  top: number;
  bottom: number;
  right: number;
  lane: number;
  row: number;
  /** The highest pitch whose row is in view. */
  high: number;
  /** Rows in view. */
  rows: number;
}

export class ScorePanel {
  private readonly root = document.createElement("div");
  private readonly frame = document.createElement("div");
  private readonly canvas = document.createElement("canvas");
  private readonly tip = document.createElement("div");
  private readonly legend = document.createElement("span");
  private readonly time = document.createElement("span");
  private readonly followButton = document.createElement("button");
  private readonly buttons = new Map<Playing, HTMLButtonElement>();
  private transcript: Transcript | null = null;
  /** Pitches seen: the range only grows, so the roll holds still between hovers. */
  private low = Infinity;
  private high = -Infinity;
  private drums = false;
  /** The highest pitch in view, or null to center the range. */
  private topPitch: number | null = null;
  /** What the roll shows, in shift steps from the first window's start. */
  private view: { start: number; span: number };
  /** While following, the view ends where the window being written does. */
  private follow = true;
  private boxes: { x0: number; x1: number; y0: number; y1: number; item: Note | Hit }[] = [];
  private drag: { id: number; x: number; y: number; moved: boolean } | null = null;
  private audio: AudioContext | null = null;
  private recording: Promise<AudioBuffer> | null = null;
  private playing: { mode: Playing; started: number; from: number; stop: () => void } | null = null;

  constructor(private readonly labels: NoteLabels) {
    this.view = { start: 0, span: 2 * labels.span };
    this.root.className = "score";
    const bar = document.createElement("div");
    bar.className = "score-bar";
    const play = (mode: Playing, text: string, title: string) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = text;
      button.title = title;
      button.addEventListener("click", () => this.toggle(mode));
      this.buttons.set(mode, button);
      bar.append(button);
    };
    play("notes", "▶ notes", "play the transcription in view (space)");
    if (labels.file.audio !== undefined) {
      play("both", "▶ with audio", "play the transcription over the recording (shift + space)");
      play("audio", "▶ audio", "play the recording in view");
    }
    this.followButton.type = "button";
    this.followButton.textContent = "⇥ follow";
    this.followButton.title = "keep the window being written in view (double-click the roll)";
    this.followButton.addEventListener("click", () => this.setFollow(true));
    this.legend.className = "score-legend";
    this.time.className = "score-time";
    bar.append(this.followButton, this.legend, this.time);

    this.frame.className = "score-frame";
    this.frame.style.height = `${storedHeight()}px`;
    this.tip.className = "score-tip";
    this.tip.hidden = true;
    this.frame.append(this.canvas, this.tip);
    this.root.append(bar, this.frame);
    new ResizeObserver(() => {
      if (this.frame.clientHeight > 0) storeHeight(this.frame.offsetHeight);
      this.draw();
    }).observe(this.frame);
    this.bindPointer();
    this.setFollow(true);
  }

  show(panel: HTMLElement, transcript: Transcript): void {
    this.transcript = transcript;
    for (const note of transcript.notes) {
      this.low = Math.min(this.low, note.pitch);
      this.high = Math.max(this.high, note.pitch);
    }
    this.drums ||= transcript.hits.length > 0;
    if (panel.firstChild !== this.root) {
      panel.dataset.kind = "notes";
      panel.replaceChildren(this.root);
    }
    this.updateBar();
    // Bring what the last event touched into view.
    const geometry = this.geometry();
    for (const item of transcript.touched) {
      if (!("program" in item)) continue;
      if (item.pitch > geometry.high || item.pitch <= geometry.high - geometry.rows) {
        this.topPitch = item.pitch + Math.floor(geometry.rows / 2);
      }
    }
    this.draw();
  }

  toggle(mode: Playing): void {
    const was = this.playing?.mode;
    this.playing?.stop();
    if (was !== mode && this.transcript !== null) void this.play(mode, this.transcript);
  }

  private setFollow(follow: boolean): void {
    this.follow = follow;
    this.followButton.classList.toggle("on", follow);
    this.draw();
  }

  private updateBar(): void {
    const transcript = this.transcript!;
    const programs = [...new Set(transcript.notes.map((note) => note.program))];
    if (transcript.hits.length > 0) programs.push(DRUM_PROGRAM);
    this.legend.replaceChildren(
      ...programs.map((program) => {
        const item = document.createElement("span");
        const swatch = document.createElement("i");
        swatch.style.background = tint(program);
        item.append(swatch, this.labels.instrument(program));
        return item;
      }),
    );
    const at = transcript.score.ended ? "ended" : `at ${clock(this.labels.seconds(transcript.now))}`;
    const hits = transcript.hits.length > 0 ? ` · ${transcript.hits.length} hits` : "";
    this.time.textContent = `${at} · ${transcript.notes.length} notes${hits}`;
  }

  /** Pitches the roll can scroll through. */
  private range(): { low: number; high: number } {
    if (this.low > this.high) return { low: 60 - (MIN_PITCHES >> 1), high: 60 + (MIN_PITCHES >> 1) };
    const pad = Math.max(2, (MIN_PITCHES - (this.high - this.low + 1)) / 2);
    return { low: Math.max(0, Math.floor(this.low - pad)), high: Math.min(127, Math.ceil(this.high + pad)) };
  }

  private geometry(): Geometry {
    const width = Math.max(1, this.frame.clientWidth);
    const height = Math.max(1, this.frame.clientHeight);
    const lane = this.drums ? DRUM_LANE : 0;
    const top = 2;
    const bottom = Math.max(top + MIN_ROW, height - AXIS - lane);
    const { low, high } = this.range();
    const span = high - low + 1;
    const row = Math.max(MIN_ROW, (bottom - top) / span);
    const rows = Math.max(1, Math.floor((bottom - top) / row));
    const center = Math.round((low + high + rows) / 2);
    // Scrolling moves by fractions of a row; the roll starts at a whole one.
    const wanted = Math.min(high, Math.max(low + rows - 1, this.topPitch ?? center));
    if (this.topPitch !== null) this.topPitch = wanted;
    return { width, height, top, bottom, right: width - 6, lane, row, high: Math.round(wanted), rows };
  }

  /** The view, following the window being written when asked to. */
  private inView(): { start: number; span: number } {
    const transcript = this.transcript;
    if (this.follow && transcript !== null) {
      const { span } = this.labels;
      const end = transcript.score.ended ? transcript.now : (transcript.score.window + 1) * span;
      this.view.start = Math.max(0, end - this.view.span);
    }
    return this.view;
  }

  private bindPointer(): void {
    const canvas = this.canvas;
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const geometry = this.geometry();
        const view = this.inView();
        const unit = event.deltaMode === 1 ? 16 : 1;
        const dx = event.deltaX * unit;
        const dy = event.deltaY * unit;
        const plot = geometry.right - GUTTER;
        if (event.ctrlKey || event.metaKey) {
          // Zoom time about the pointer, or about the view's end while following.
          const total = this.labels.file.windows * this.labels.span;
          const span = Math.min(total, Math.max(this.labels.file.frameRate / 2, view.span * Math.exp(dy * 0.01)));
          const fixed = this.follow ? 1 : Math.min(1, Math.max(0, (event.offsetX - GUTTER) / plot));
          view.start = Math.max(0, view.start + (view.span - span) * fixed);
          view.span = span;
        } else if (event.shiftKey || Math.abs(dx) > Math.abs(dy)) {
          this.panTime(((event.shiftKey ? dy || dx : dx) / plot) * view.span);
        } else this.scrollPitch(-dy / geometry.row, geometry);
        this.draw();
      },
      { passive: false },
    );
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      canvas.setPointerCapture(event.pointerId);
      this.drag = { id: event.pointerId, x: event.offsetX, y: event.offsetY, moved: false };
    });
    canvas.addEventListener("pointermove", (event) => {
      const drag = this.drag;
      if (drag?.id === event.pointerId) {
        const geometry = this.geometry();
        const dx = event.offsetX - drag.x;
        const dy = event.offsetY - drag.y;
        if (dx !== 0) this.panTime((-dx / (geometry.right - GUTTER)) * this.inView().span);
        if (dy !== 0) this.scrollPitch(dy / geometry.row, geometry);
        drag.x = event.offsetX;
        drag.y = event.offsetY;
        drag.moved = true;
        this.tip.hidden = true;
        this.draw();
        return;
      }
      this.updateTip(event.offsetX, event.offsetY);
    });
    const release = (event: PointerEvent) => {
      if (this.drag?.id === event.pointerId) this.drag = null;
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", () => {
      this.tip.hidden = true;
    });
    canvas.addEventListener("dblclick", () => this.setFollow(true));
  }

  private panTime(by: number): void {
    const view = this.inView();
    const total = this.labels.file.windows * this.labels.span;
    view.start = Math.max(0, Math.min(total - view.span, view.start + by));
    if (this.follow) this.setFollow(false);
  }

  private scrollPitch(by: number, geometry: Geometry): void {
    this.topPitch = (this.topPitch ?? geometry.high) + by;
  }

  private updateTip(x: number, y: number): void {
    const box = this.boxes.findLast((box) => x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1);
    if (box === undefined) {
      this.tip.hidden = true;
      return;
    }
    const { labels } = this;
    const item = box.item;
    const at = (tick: number) => clock(labels.seconds(tick));
    if ("program" in item) {
      const length = item.end === null ? null : (item.end - item.start) / labels.file.frameRate;
      const since = item.carried ? "from before" : at(item.start);
      const until = item.end === null ? "still sounding" : `${at(item.end)} · ${length!.toFixed(2)} s`;
      this.tip.textContent = `${pitchName(item.pitch)} · ${labels.instrument(item.program)}\n${since} → ${until}`;
    } else this.tip.textContent = `${labels.drum(item.pitch)} · ${at(item.tick)}`;
    this.tip.hidden = false;
    const width = this.frame.clientWidth;
    this.tip.style.left = `${Math.min(x + 14, width - this.tip.offsetWidth - 4)}px`;
    this.tip.style.top = `${y + 16 + this.tip.offsetHeight > this.frame.clientHeight ? y - this.tip.offsetHeight - 8 : y + 16}px`;
  }

  private draw(): void {
    const transcript = this.transcript;
    if (transcript === null || !this.root.isConnected) return;
    const geometry = this.geometry();
    const { width, height, top, bottom, right, lane, row, high, rows } = geometry;
    const ratio = devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * ratio)) this.canvas.width = Math.round(width * ratio);
    if (this.canvas.height !== Math.round(height * ratio)) this.canvas.height = Math.round(height * ratio);
    const ctx = this.canvas.getContext("2d")!;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const style = getComputedStyle(this.root);
    const color = (name: string) => style.getPropertyValue(name).trim();
    const [ink, muted, rule, chip, accent] = ["--ink", "--ink-muted", "--rule", "--chip", "--accent"].map(color) as [
      string,
      string,
      string,
      string,
      string,
    ];

    const { labels } = this;
    const { frameRate } = labels.file;
    const view = this.inView();
    const plot = right - GUTTER;
    const x = (tick: number) => GUTTER + ((tick - view.start) / view.span) * plot;
    const y = (pitch: number) => top + (high - pitch) * row;
    const low = high - rows + 1;
    const now = transcript.now;
    const below = bottom + lane;
    this.boxes = [];

    ctx.clearRect(0, 0, width, height);
    ctx.font = `10px ${MONO}`;
    ctx.textBaseline = "middle";

    // Black keys' rows shaded; pitches named down the gutter.
    for (let pitch = high; pitch >= low; pitch--) {
      const at = y(pitch);
      if ([1, 3, 6, 8, 10].includes(pitch % 12)) {
        ctx.fillStyle = chip;
        ctx.fillRect(GUTTER, at, plot, row);
      }
      if (pitch % 12 === 0) {
        ctx.fillStyle = rule;
        ctx.fillRect(GUTTER, at + row - 0.5, plot, 1);
      }
      if (row >= NAMED_ROW || pitch % 12 === 0) {
        ctx.fillStyle = pitch % 12 === 0 ? ink : muted;
        ctx.fillText(pitchName(pitch), 3, at + row / 2);
      }
    }

    // Time: a tick every so many seconds, a stronger line at each window.
    const seconds = view.span / frameRate;
    const step = niceStep(seconds / Math.max(1, plot / 90));
    const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    const offset = labels.file.at ?? 0;
    const first = Math.ceil((offset + view.start / frameRate) / step) * step;
    for (let at = first; at <= offset + (view.start + view.span) / frameRate + 1e-9; at += step) {
      const px = Math.round(x((at - offset) * frameRate)) + 0.5;
      ctx.fillStyle = rule;
      ctx.fillRect(px - 0.5, top, 1, below - top);
      ctx.fillStyle = muted;
      const text = clock(at, digits);
      const textWidth = ctx.measureText(text).width;
      ctx.fillText(text, Math.min(right - textWidth, Math.max(GUTTER, px - textWidth / 2)), height - AXIS / 2);
    }
    const { span } = labels;
    for (let index = Math.ceil(view.start / span); index * span <= view.start + view.span; index++) {
      ctx.fillStyle = muted;
      ctx.globalAlpha = 0.7;
      ctx.fillRect(Math.round(x(index * span)) - 1, top, 2, below - top);
      ctx.globalAlpha = 1;
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, top, plot, below - top);
    ctx.clip();
    // What has not been written yet.
    if (x(now) < right) {
      ctx.fillStyle = chip;
      ctx.globalAlpha = 0.6;
      ctx.fillRect(x(now), top, right - x(now), below - top);
      ctx.globalAlpha = 1;
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, top, plot, bottom - top);
    ctx.clip();
    for (const note of transcript.notes) {
      const end = note.end ?? now;
      if (end < view.start || note.start > view.start + view.span || note.pitch > high || note.pitch < low) continue;
      const x0 = x(note.start);
      const x1 = Math.max(x0 + 3, x(end));
      const y0 = y(note.pitch) + 0.5;
      const h = Math.max(2, row - 1);
      const color = tint(note.program);
      if (note.end === null) {
        // Still sounding: fades toward the cursor, where it may yet end.
        const gradient = ctx.createLinearGradient(x0, 0, x1, 0);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, "transparent");
        ctx.fillStyle = gradient;
      } else ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x0, y0, x1 - x0, h, Math.min(3, h / 2));
      ctx.fill();
      if (note.carried) {
        // Sounding from before the first window.
        ctx.fillStyle = ink;
        ctx.fillRect(x0, y0, 2, h);
      }
      if (transcript.touched.has(note)) {
        ctx.strokeStyle = ink;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (row >= NAMED_ROW) {
        const name = pitchName(note.pitch);
        if (ctx.measureText(name).width + 6 <= x1 - Math.max(x0, GUTTER)) {
          ctx.fillStyle = "rgb(0 0 0 / 0.72)";
          ctx.fillText(name, Math.max(x0, GUTTER) + 3, y0 + h / 2);
        }
      }
      this.boxes.push({ x0, x1, y0, y1: y0 + h, item: note });
    }
    ctx.restore();
    if (lane > 0) {
      const middle = bottom + lane / 2;
      for (const hit of transcript.hits) {
        if (hit.tick < view.start || hit.tick > view.start + view.span) continue;
        const at = x(hit.tick);
        const touched = transcript.touched.has(hit);
        const size = touched ? 5.5 : 4;
        ctx.fillStyle = touched ? ink : tint(DRUM_PROGRAM);
        ctx.beginPath();
        ctx.moveTo(at, middle - size);
        ctx.lineTo(at + size, middle);
        ctx.lineTo(at, middle + size);
        ctx.lineTo(at - size, middle);
        ctx.fill();
        this.boxes.push({ x0: at - size, x1: at + size, y0: middle - size, y1: middle + size, item: hit });
      }
    }
    // The cursor: where the next event happens.
    ctx.fillStyle = accent;
    ctx.fillRect(Math.round(x(now)) - 1, top, 2, below - top);
    if (this.playing !== null && this.audio !== null) {
      const played = this.playing.from + (this.audio.currentTime - this.playing.started) * frameRate;
      ctx.fillStyle = ink;
      ctx.fillRect(Math.round(x(played)) - 0.5, top, 1, below - top);
    }
    ctx.restore();
    if (lane > 0) {
      ctx.fillStyle = muted;
      ctx.fillText("drums", 3, bottom + lane / 2);
    }
  }

  private context(): AudioContext {
    this.audio ??= new AudioContext();
    void this.audio.resume();
    return this.audio;
  }

  private loadRecording(audio: AudioContext): Promise<AudioBuffer> {
    const file = this.labels.file.audio!;
    const url = this.labels.base === null ? file : new URL(file, this.labels.base);
    this.recording ??= fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`could not load the recording: ${response.status}`);
        return response.arrayBuffer();
      })
      .then((bytes) => audio.decodeAudioData(bytes));
    return this.recording;
  }

  /** Play from the view's start to the cursor, or across the view for the recording alone. */
  private async play(mode: Playing, transcript: Transcript): Promise<void> {
    const audio = this.context();
    const { frameRate } = this.labels.file;
    const view = this.inView();
    const now = transcript.now;
    const from = view.start < now ? view.start : Math.max(0, now - view.span);
    const until = mode === "audio" ? from + view.span : now;
    const recording = mode === "notes" ? null : await this.loadRecording(audio);
    const out = audio.createDynamicsCompressor();
    out.connect(audio.destination);
    const sources: AudioScheduledSourceNode[] = [];
    const started = audio.currentTime + 0.05;
    let length = (until - from) / frameRate;
    if (mode !== "audio") {
      const notes = audio.createGain();
      notes.gain.value = mode === "both" ? 0.7 : 1;
      notes.connect(out);
      const synthesized = synthesize(audio, notes, transcript, frameRate, from, started);
      length = Math.max(length, synthesized.length);
      sources.push(...synthesized.sources);
    }
    if (recording !== null) {
      const source = audio.createBufferSource();
      source.buffer = recording;
      const gain = audio.createGain();
      gain.gain.value = mode === "both" ? 0.5 : 1;
      source.connect(gain).connect(out);
      source.start(started, from / frameRate, Math.max(0.5, length));
      sources.push(source);
    }
    let frame = 0;
    const stop = () => {
      for (const source of sources) {
        try {
          source.stop();
        } catch {
          // Never started, or already stopped.
        }
      }
      out.disconnect();
      cancelAnimationFrame(frame);
      if (this.playing?.stop === stop) this.playing = null;
      this.buttons.get(mode)?.classList.remove("on");
      this.draw();
    };
    this.playing = { mode, started, from, stop };
    this.buttons.get(mode)?.classList.add("on");
    const tick = () => {
      if (this.playing?.stop !== stop) return;
      if (audio.currentTime > started + length + 0.3) return stop();
      this.draw();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }
}

function storedHeight(): number {
  try {
    const stored = Number(localStorage.getItem(HEIGHT_KEY));
    return stored > 0 ? stored : HEIGHT;
  } catch {
    return HEIGHT;
  }
}

function storeHeight(height: number): void {
  try {
    localStorage.setItem(HEIGHT_KEY, String(Math.round(height)));
  } catch {
    // Remembering the height is only a convenience.
  }
}

function niceStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(raw));
  for (const multiple of [1, 2, 5, 10]) if (multiple * power >= raw) return multiple * power;
  return 10 * power;
}

let noise: AudioBuffer | null = null;

/**
 * Schedule ``transcript`` from shift step ``from`` on ``out`` at time ``when``,
 * a plucked triangle per note and a filtered noise burst or a falling sine per
 * drum hit, and return how long it lasts. A note still sounding at the cursor
 * rings a little past it, so a lone onset is heard.
 */
function synthesize(
  audio: AudioContext,
  out: AudioNode,
  transcript: Transcript,
  frameRate: number,
  from: number,
  when: number,
): { length: number; sources: AudioScheduledSourceNode[] } {
  const now = (transcript.now - from) / frameRate;
  let length = now;
  const sources: AudioScheduledSourceNode[] = [];
  for (const note of transcript.notes) {
    const onset = (note.start - from) / frameRate;
    const end = note.end === null ? Math.max(now, onset + 0.4) : Math.max((note.end - from) / frameRate, onset + 0.08);
    if (end <= 0) continue;
    const start = Math.max(0, onset);
    length = Math.max(length, end);
    const oscillator = audio.createOscillator();
    oscillator.type = "triangle";
    oscillator.frequency.value = 440 * 2 ** ((note.pitch - 69) / 12);
    const gain = audio.createGain();
    const level = 0.22;
    gain.gain.setValueAtTime(0, when + start);
    gain.gain.linearRampToValueAtTime(level, when + start + 0.006);
    gain.gain.setTargetAtTime(level * 0.35, when + start + 0.006, 0.25);
    gain.gain.setTargetAtTime(0, when + end, 0.05);
    oscillator.connect(gain).connect(out);
    oscillator.start(when + start);
    oscillator.stop(when + end + 0.4);
    sources.push(oscillator);
  }
  for (const hit of transcript.hits) {
    if (hit.tick >= from) sources.push(drum(audio, out, hit.pitch, when + (hit.tick - from) / frameRate));
  }
  return { length, sources };
}

function drum(audio: AudioContext, out: AudioNode, pitch: number, at: number): AudioScheduledSourceNode {
  const gain = audio.createGain();
  gain.connect(out);
  if (pitch === 35 || pitch === 36 || [41, 43, 45, 47, 48, 50].includes(pitch)) {
    // Kick and toms: a sine falling in pitch.
    const kick = pitch <= 36;
    const oscillator = audio.createOscillator();
    oscillator.frequency.setValueAtTime(kick ? 140 : 90 + (pitch - 41) * 18, at);
    oscillator.frequency.exponentialRampToValueAtTime(kick ? 42 : 60 + (pitch - 41) * 10, at + 0.15);
    gain.gain.setValueAtTime(kick ? 0.7 : 0.4, at);
    gain.gain.exponentialRampToValueAtTime(0.001, at + (kick ? 0.3 : 0.35));
    oscillator.connect(gain);
    oscillator.start(at);
    oscillator.stop(at + 0.4);
    return oscillator;
  }
  noise ??= whiteNoise(audio);
  const source = audio.createBufferSource();
  source.buffer = noise;
  const filter = audio.createBiquadFilter();
  const cymbal = [42, 44, 46, 49, 51, 52, 53, 55, 57, 59].includes(pitch);
  filter.type = cymbal ? "highpass" : "bandpass";
  filter.frequency.value = cymbal ? 7000 : pitch >= 37 && pitch <= 40 ? 1800 : 3200;
  const decay = pitch === 42 || pitch === 44 ? 0.05 : cymbal ? 0.35 : 0.14;
  gain.gain.setValueAtTime(cymbal ? 0.25 : 0.45, at);
  gain.gain.exponentialRampToValueAtTime(0.001, at + decay);
  source.connect(filter).connect(gain);
  source.start(at);
  source.stop(at + decay + 0.05);
  return source;
}

function whiteNoise(audio: AudioContext): AudioBuffer {
  const buffer = audio.createBuffer(1, audio.sampleRate / 2, audio.sampleRate);
  const samples = buffer.getChannelData(0);
  for (let index = 0; index < samples.length; index++) samples[index] = Math.random() * 2 - 1;
  return buffer;
}
