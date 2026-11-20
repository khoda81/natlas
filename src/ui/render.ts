// Canvas 2D drawing of one laid-out frame.

import { decimalTicks, layoutTicks } from "@qbar/qbar";
import type { Palette } from "./color";
import type { Box, Camera, Frame, Known, Selection, Viewport } from "../core/layout";
import type { NodeId } from "../core/tree";
import type { Glyph, Mark, Timeline, Trail } from "../readings/reading";

export interface Units {
  name: string;
  /** Nats per unit. */
  nats: number;
}

export const NATS: Units = { name: "nats", nats: 1 };
export const BITS: Units = { name: "bits", nats: Math.LN2 };

export interface DrawState {
  hover: Selection | null;
  units: Units;
  /** How the child along ``branch`` of node ``node`` reads. */
  glyph: (node: NodeId, branch: number) => Glyph;
  /** Why the origin shows no children, if it does not; ``sweep`` animates
   * a faint band behind the text while they load. */
  note: { text: string; sweep: number | null } | null;
  /** Where the reading draws its time, in canvas pixels, while it does:
   * a box whose glyph moves time is a window onto it. */
  timeline: Timeline | null;
  /** The known children of a laid-out node in view, however thin. */
  known: (node: NodeId) => readonly Known[];
}

const FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const BASE_SIZE = 12;
const FONT = `${BASE_SIZE}px ${FAMILY}`;
// Label sizes are text sizes, in CSS pixels like the page's own text, so they
// scale with it across devices and zoom, never with the view.
/** The smallest a label reads at. A label that needs less is shortened, or not drawn. */
const LABEL_MIN_SIZE = 8;
/** Labels grow to fit their box, up to this size. */
const LABEL_MAX_SIZE = 96;
/** Fraction of the box a label may fill across its text. */
const LABEL_FILL = 0.6;
/** Room left at each end of a label, in ems. */
const LABEL_PAD = 0.25;
/** How strongly the labels a box carries from its ancestors show before its own. */
const CARRIED_ALPHA = 0.6;
/** Carried labels are framed, in ems: room inside a frame at each side, the
 * space between a label's pieces, and the frame's half height. */
const TOKEN_PAD = 0.2;
const TOKEN_GAP = 0.3;
const TOKEN_HALF = 0.6;
/** How far apart, in ems, the lines of a wrapped label are: frames clear each other. */
const LABEL_LINE = 1.4;
const graphemes = new Intl.Segmenter(undefined, { granularity: "grapheme" });
/** A mark's width, and the space after it, in ems. */
const MARK_WIDTH = 0.8;
const MARK_GAP = 0.3;
const MORE: Glyph = { text: "…" };
/** How strongly a time window shades the time its event moves over. */
const TIME_ALPHA = 0.16;
/** The width of a time CDF's line, in CSS pixels. */
const TIME_LINE = 2;
export const RULER_HEIGHT = 22;
const RULER_SCALE = decimalTicks();
/** Ruler marks fade in over this much room to their neighbours of the same step. */
const RULER_MARKS = { from: 16, to: 32 };
/** Labels fade in over this much room; the least holds ten characters. */
const RULER_LABELS = { from: 80, to: 120 };
/** One sweep of the loading band across the view. */
const SWEEP_MS = 1400;
// Canvas coordinates far outside the view lose precision and can stall rasterizing.
const OVERSCAN = 64;
const textWidths = new WeakMap<CanvasRenderingContext2D, { scale: number; widths: Map<string, number> }>();

function measure(ctx: CanvasRenderingContext2D, text: string): number {
  const widths = textWidths.get(ctx)!.widths;
  let width = widths.get(text);
  if (width === undefined) {
    // Ruler values keep changing during navigation; bound their memoization.
    if (widths.size >= 8192) widths.clear();
    widths.set(text, width = ctx.measureText(text).width);
  }
  return width;
}

export interface ScreenRect {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export function screenRect(box: Box, cam: Camera, view: Viewport): ScreenRect {
  const clampX = (x: number) => Math.min(view.width + OVERSCAN, Math.max(-OVERSCAN, x));
  const clampY = (y: number) => Math.min(view.height + OVERSCAN, Math.max(-OVERSCAN, y));
  return {
    x0: clampX((box.x0 - cam.x) * cam.k),
    x1: clampX((box.x1 - cam.x) * cam.k),
    y0: clampY(box.y0 * view.height),
    y1: clampY(box.y1 * view.height),
  };
}

// A physical pixel shows, and the pointer over it selects, the box holding
// the pixel's top-left corner. Picking gives a box its top and left edges but
// not its bottom and right ones, so drawn edges round up to the grid and a
// pointer rounds down to its pixel's corner. Picking at the exact pointer
// would select a neighbour over part of a thin box's pixel, and flicker as
// the pointer moves within it.

// Layout's rows lie on the grid only up to float error: a coordinate within
// this many physical pixels of a grid line is on it, so a row never rounds
// to the next.
const ON_GRID = 1e-6;

/** The grid line a box's edge rounds up to, in CSS pixels. */
function gridEdge(at: number, scale: number): number {
  return Math.ceil(at * scale - ON_GRID) / scale;
}

/** The physical pixels a box shows on, expressed in CSS pixels. */
function pixelRect(rect: ScreenRect, scale: number): ScreenRect {
  const edge = (at: number) => gridEdge(at, scale);
  return { x0: edge(rect.x0), x1: edge(rect.x1), y0: edge(rect.y0), y1: edge(rect.y1) };
}

/** The top-left corner of the physical pixel holding a CSS coordinate. */
export function pixelCorner(at: number, scale: number): number {
  return Math.floor(at * scale + ON_GRID) / scale;
}

export function draw(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  cam: Camera,
  view: Viewport,
  palette: Palette,
  state: DrawState,
): void {
  const scale = ctx.getTransform().a;
  if (textWidths.get(ctx)?.scale !== scale) textWidths.set(ctx, { scale, widths: new Map() });
  const { scheme } = palette;

  ctx.fillStyle = scheme.surface;
  ctx.fillRect(0, 0, view.width, view.height + RULER_HEIGHT);
  ctx.font = FONT;
  ctx.textBaseline = "middle";

  // Every node's children as spans of physical rows: the pixels wholly
  // inside them filled, then the time on them, then their outline over both.
  const pixel = 1 / scale;
  const column = (x: number) => Math.floor(within((x - cam.x) * cam.k, -OVERSCAN, view.width + OVERSCAN) * scale + ON_GRID) / scale;
  const row = (y: number) => Math.round(within(y, -1, 2) * view.height * scale) / scale;
  const spans = new Map<NodeId, Painted[]>();
  for (const span of frame.spans) {
    const painted = { y0: row(span.y0), y1: row(span.y1), left: column(span.x0),
      fill: span.x0 < span.fill ? column(span.fill) : null, reach: column(span.reach) + pixel };
    if (painted.y1 <= painted.y0) continue;
    const node = spans.get(span.node);
    if (node === undefined) spans.set(span.node, [painted]);
    else node.push(painted);
  }
  // Each layer one path, filled once: a call a span would cost more than
  // the pixels.
  ctx.beginPath();
  for (const node of spans.values()) {
    for (const { y0, y1, left, fill } of node) {
      if (fill !== null && left + pixel < fill) ctx.rect(left + pixel, y0, fill - left - pixel, y1 - y0);
    }
  }
  ctx.fillStyle = scheme.tile;
  ctx.fill();
  if (state.timeline !== null) {
    timeCdfs(ctx, spans, state.timeline, view, state.known, (node, branch) => state.glyph(node, branch).time, scheme.window, palette.ink);
  }
  ctx.beginPath();
  for (const node of spans.values()) {
    for (const { y0, y1, left, fill, reach } of node) {
      ctx.rect(left, y0, pixel, y1 - y0);
      const from = fill ?? left;
      if (from < reach) ctx.rect(from, y0, reach - from, y1 - y0);
    }
  }
  ctx.fillStyle = scheme.stroke;
  ctx.fill();

  // Labels on the boxes: a gap no width is known for is the background's.
  const labels: { box: Box; rect: ScreenRect; ink: string }[] = [];
  let visible = spans.size !== 0;
  for (const box of frame.boxes) {
    const rect = screenRect(box, cam, view);
    if (rect.x1 <= rect.x0 || rect.y1 <= rect.y0) continue;
    visible = true;
    const ink = box.kind === "gap" && !Number.isFinite(box.x1) ? scheme.inkMuted : palette.ink;
    if (box.count !== 0) labels.push({ box, rect, ink });
  }

  // A box whose label is too small to draw hands it, after the labels it
  // carries itself, to each of its children; they carry it until one draws
  // it before its own. A box's parent is the box of the branch its node is.
  const carriedBy = new Map<NodeId, readonly Glyph[]>();
  /** The labels the children of ``node`` carry, outermost first. A time
   * window shows its time at any size and carries what it was handed on. */
  const carried = (node: NodeId): readonly Glyph[] => {
    let glyphs = carriedBy.get(node);
    if (glyphs === undefined) {
      const parent = frame.parentBoxes.get(node);
      const glyph = parent && state.glyph(parent.node, parent.branch!);
      // A time window shows what it carries, never its own label.
      glyphs = parent === undefined || glyph === undefined || fitOf(parent) !== null
        ? []
        : windowed(glyph)
        ? carried(parent.node)
        : [...carried(parent.node), glyph];
      carriedBy.set(node, glyphs);
    }
    return glyphs;
  };
  const windowed = (glyph: Glyph) => glyph.time !== undefined && state.timeline !== null;
  const fits = new Map<Box, Fit | null>();
  const fitOf = (box: Box): Fit | null => {
    let fit = fits.get(box);
    if (fit === undefined) {
      // A gap's label counts children; it carries nothing to them.
      const [glyph, before] = box.branch !== null
        ? [state.glyph(box.node, box.branch), carried(box.node)]
        : [gapGlyph(box), []];
      const width = (box.x1 - box.x0) * cam.k - 2;
      const height = (box.y1 - box.y0) * view.height;
      fits.set(box, fit = fitLabel(ctx, before, windowed(glyph) ? null : glyph, width, height, box.branch !== null));
    }
    return fit;
  };

  // A label's size follows its whole box, never the viewport. Within its box,
  // a label moves toward the viewport as far as the box allows: it starts at
  // the box's left, centered, and slides right past the left screen edge and
  // up or down into view, stopping at the box's own edges. A label reading up
  // a box is centered both ways and slides into view the same way. The labels
  // a box carries come first, framed and fainter, and a label wraps between
  // its pieces where lines make it larger.
  for (const { box, rect, ink } of labels) {
    const fit = fitOf(box);
    if (fit === null) continue;
    const { lines, size, vertical } = fit;
    const left = (box.x0 - cam.x) * cam.k;
    const right = left + (box.x1 - box.x0) * cam.k - 2;
    const top = box.y0 * view.height;
    const bottom = box.y1 * view.height;
    const pad = LABEL_PAD * size;
    const span = fit.ems * size;
    // The lines' extent across the text, and the distance from the first's
    // middle to the last's.
    const thick = size * (1 + (lines.length - 1) * LABEL_LINE);
    const reach = (lines.length - 1) * LABEL_LINE * size;
    // Where the first line starts, and the screen rectangle the label covers.
    let x: number, y: number, covers: ScreenRect;
    if (vertical) {
      const middle = within(within((left + right) / 2, thick / 2, view.width - thick / 2), left + thick / 2, right - thick / 2);
      x = middle - reach / 2;
      const along = within(within((top + bottom) / 2, pad + span / 2, view.height - pad - span / 2), top + pad + span / 2, bottom - pad - span / 2);
      y = along + span / 2;
      covers = { x0: middle - thick / 2, y0: y - span, x1: middle + thick / 2, y1: y };
    } else {
      x = within(pad, left + pad, right - pad - span);
      const middle = within(within((top + bottom) / 2, thick / 2, view.height - thick / 2), top + thick / 2, bottom - thick / 2);
      y = middle - reach / 2;
      covers = { x0: x, y0: middle - thick / 2, x1: x + span, y1: middle + thick / 2 };
    }
    if (view.width < covers.x0 || covers.x1 < 0 || view.height < covers.y0 || covers.y1 < 0) continue;
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x0, rect.y0, rect.x1 - rect.x0 - 2, rect.y1 - rect.y0);
    ctx.clip();
    ctx.translate(x, y);
    if (vertical) ctx.rotate(-Math.PI / 2);
    ctx.font = `${size}px ${FAMILY}`;
    lines.forEach((line, row) => {
      let at = 0;
      for (const piece of line) {
        drawPiece(ctx, piece, at, row * LABEL_LINE * size, size, ink, scheme.surface);
        at += (piece.ems + TOKEN_GAP) * size;
      }
    });
    ctx.restore();
  }

  const note = state.note ?? (visible ? null : { text: "No continuation discovered here · press 0 for the root", sweep: null });
  if (note !== null) drawNote(ctx, view, palette, note.text, note.sweep);

  outline(ctx, frame, cam, view, state, scheme.hover);
  ruler(ctx, cam, view, state.units, palette);
}

/** A span as painted, in CSS pixels on the physical grid: its rows, the
 * column its children start at, where its fill ends, if it has one, and
 * just past where its outline reaches. */
interface Painted {
  y0: number;
  y1: number;
  left: number;
  fill: number | null;
  reach: number;
}

/** Each node's time branches as one CDF on the timeline drawn above them,
 * across the same pixels: down a run of them, the line stands at the time
 * each known one lands on, however thin, and crosses straight over the
 * unknown ones between. The time they move over, from where the path is,
 * and behind it each interval between the times the path stood at before,
 * are shaded as the reading says, over a base of their own. Clipped to the
 * node's children in those rows: a window onto the timeline. */
function timeCdfs(
  ctx: CanvasRenderingContext2D,
  spans: Map<NodeId, Painted[]>,
  timeline: Timeline,
  view: Viewport,
  known: (node: NodeId) => readonly Known[],
  timeOf: (node: NodeId, branch: number) => { from: number; to: number; shaded: boolean; trail: Trail | null } | undefined,
  base: string,
  ink: string,
): void {
  // Far-off coordinates lose precision and can stall rasterizing.
  const x = (time: number) => within(timeline.x(time), -OVERSCAN, view.width + OVERSCAN);
  for (const [node, painted] of spans) {
    const children = known(node);
    let first = 0;
    while (first < children.length) {
      const start = timeOf(node, children[first]!.branch);
      if (start === undefined) {
        first++;
        continue;
      }
      // A run of known time branches, up to the next known other branch.
      let last = first;
      while (last + 1 < children.length && timeOf(node, children[last + 1]!.branch) !== undefined) last++;
      const line: [number, number][] = [];
      for (let i = first; i <= last; i++) {
        const { branch, y0, y1 } = children[i]!;
        const to = x(timeOf(node, branch)!.to);
        line.push([to, y0 * view.height], [to, y1 * view.height]);
      }
      first = last + 1;
      const from = x(start.from);
      const [top, bottom] = [line[0]![1], line[line.length - 1]![1]];
      const shown = painted.filter((span) => top < span.y1 && span.y0 < bottom);
      if (shown.length === 0) continue;
      ctx.save();
      ctx.beginPath();
      for (const span of shown) ctx.rect(span.left, span.y0, span.reach - span.left, span.y1 - span.y0);
      ctx.clip();
      ctx.beginPath();
      ctx.rect(-OVERSCAN, top, view.width + 2 * OVERSCAN, bottom - top);
      ctx.clip();
      ctx.fillStyle = base;
      ctx.fill();
      ctx.fillStyle = ctx.strokeStyle = ink;
      ctx.beginPath();
      // Back from ``from``, each interval the trail shades. Time only moves
      // on, so the rest of the trail lies left of the view once one time does.
      let newer = from;
      for (let trail = start.trail; trail !== null && -OVERSCAN < newer; trail = trail.before) {
        const older = x(trail.at);
        if (trail.shaded) ctx.rect(older, top, newer - older, bottom - top);
        newer = older;
      }
      if (start.shaded) {
        ctx.moveTo(from, top);
        for (const [px, py] of line) ctx.lineTo(px, py);
        ctx.lineTo(from, bottom);
        ctx.closePath();
      }
      ctx.globalAlpha = TIME_ALPHA;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      line.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.lineWidth = TIME_LINE;
      ctx.lineJoin = "round";
      ctx.stroke();
      ctx.restore();
    }
  }
}


/** One piece of a label, never broken across lines: the ellipsis for
 * carried labels that did not fit, a carried label, or the box's own; and
 * the ems it takes, its frame included. */
interface Piece {
  kind: "cut" | "carried" | "own";
  glyph: Glyph;
  ems: number;
}

/** A label as drawn: its pieces line by line, its size, which way it
 * reads, and its widest line's ems. */
interface Fit {
  lines: Piece[][];
  size: number;
  vertical: boolean;
  ems: number;
}

/** The largest label a box of ``width`` by ``height`` CSS pixels holds:
 * across it, or up it where it is taller than wide and that is larger, in
 * as many lines, broken between its pieces, as make it largest. The
 * labels it carries come before its own, as many of the innermost as fit at
 * a readable size, after an ellipsis where outer ones do not; where none
 * does, it is its own alone. A label too long to read at any size the box
 * allows keeps as much of its start as fits at the smallest readable size;
 * one that keeps nothing is not drawn. Gap labels, which do not ``grow``,
 * stay at the base size. A box without a label of its own, a ``glyph`` of
 * null, shows only what it carries. */
function fitLabel(
  ctx: CanvasRenderingContext2D,
  carried: readonly Glyph[],
  glyph: Glyph | null,
  width: number,
  height: number,
  grows: boolean,
): Fit | null {
  const largest = grows ? LABEL_MAX_SIZE : BASE_SIZE;
  const tokens = carried.map((glyph): Piece => ({ kind: "carried", glyph, ems: 2 * TOKEN_PAD + glyphEms(ctx, glyph) }));
  const cut: Piece = { kind: "cut", glyph: MORE, ems: glyphEms(ctx, MORE) };
  if (glyph === null) {
    // Without a label of its own, as many of the innermost it carries as fit.
    for (let first = 0; first < tokens.length; first++) {
      const kept = tokens.slice(first);
      const fit = largestFit(0 < first ? [cut, ...kept] : kept, width, height, largest);
      if (fit !== null) return fit;
    }
    return null;
  }
  const own: Piece = { kind: "own", glyph, ems: glyphEms(ctx, glyph) };
  // Pieces only add to a label, so where its own does not fit alone, no
  // carried label fits beside it. Otherwise, outermost first, each carried
  // label is dropped until the rest fit.
  const alone = largestFit([own], width, height, largest);
  if (alone !== null) {
    for (let first = 0; first < tokens.length; first++) {
      const kept = [...tokens.slice(first), own];
      const fit = largestFit(0 < first ? [cut, ...kept] : kept, width, height, largest);
      if (fit !== null) return fit;
    }
    return alone;
  }
  // Ems of text that fit at the smallest readable size.
  const marked = markEms(glyph);
  const room = (across: number, along: number) =>
    LABEL_MIN_SIZE <= across * LABEL_FILL ? along / LABEL_MIN_SIZE - 2 * LABEL_PAD - marked : 0;
  const across = room(height, width);
  const up = width < height ? room(width, height) : 0;
  const text = shortened(ctx, glyph.text, Math.max(across, up));
  if (text === null) return null;
  const piece: Piece = { kind: "own", glyph: { ...glyph, text }, ems: glyphEms(ctx, { ...glyph, text }) };
  return { lines: [[piece]], size: LABEL_MIN_SIZE, vertical: across < up, ems: piece.ems };
}

/** ``pieces`` as the largest label a box of ``width`` by ``height`` holds at
 * a readable size, if any. */
function largestFit(pieces: readonly Piece[], width: number, height: number, largest: number): Fit | null {
  const sideways = wrap(pieces, height, width, largest);
  const upward = width < height ? wrap(pieces, width, height, largest) : null;
  const vertical = upward !== null && sideways.size < upward.size;
  const best = vertical ? upward : sideways;
  return best.size < LABEL_MIN_SIZE ? null : { ...best, vertical };
}

/** The lines, broken only between pieces, at which ``pieces`` are largest
 * in a box ``across`` pixels by ``along``; that size; and their widest
 * line's ems.
 *
 * Filling each line before starting the next gives the fewest lines no
 * wider than a limit. Those lines change only at a limit some line could
 * take its next piece at, so stepping the limit up from the widest piece
 * through those widths visits every way of breaking that could be largest.
 * The step's line reaches the new limit, and the widest line never again
 * falls below it: once no line that wide beats the best, none wider can. */
function wrap(pieces: readonly Piece[], across: number, along: number, largest: number): Omit<Fit, "vertical"> {
  let limit = Math.max(...pieces.map((piece) => piece.ems));
  let best: Omit<Fit, "vertical"> | null = null;
  for (;;) {
    const lines: Piece[][] = [[pieces[0]!]];
    let line = pieces[0]!.ems;
    let widest = line;
    let next = Infinity;
    for (const piece of pieces.slice(1)) {
      // Summed in the same order every time, so the line that set ``limit``
      // reaches it exactly.
      const joined = line + TOKEN_GAP + piece.ems;
      if (joined <= limit) {
        lines.at(-1)!.push(piece);
        line = joined;
      } else {
        next = Math.min(next, joined);
        lines.push([piece]);
        line = piece.ems;
      }
      widest = Math.max(widest, line);
    }
    const size = Math.min(largest, (across * LABEL_FILL) / (1 + (lines.length - 1) * LABEL_LINE), along / (widest + 2 * LABEL_PAD));
    if (best === null || best.size < size) best = { lines, size, ems: widest };
    if (next === Infinity || along / (next + 2 * LABEL_PAD) <= best.size) return best;
    limit = next;
  }
}

/** The ems a glyph takes: its mark, then its text. */
function glyphEms(ctx: CanvasRenderingContext2D, glyph: Glyph): number {
  return markEms(glyph) + measure(ctx, glyph.text) / BASE_SIZE;
}

/** The ems a glyph's mark takes before its text. */
function markEms(glyph: Glyph): number {
  return glyph.mark === undefined ? 0 : MARK_WIDTH + (glyph.text === "" ? 0 : MARK_GAP);
}

/** ``piece`` drawn ``size`` tall from ``x``, centered on ``y``: what the box
 * carries framed and fainter than its own. A frame is filled with ``back``,
 * opaque, so what is under it, such as shaded time, never shows through. */
function drawPiece(ctx: CanvasRenderingContext2D, { kind, glyph, ems }: Piece, x: number, y: number, size: number, ink: string, back: string): void {
  if (kind === "carried") {
    ctx.globalAlpha = 1;
    ctx.fillStyle = back;
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1, size / 16);
    ctx.beginPath();
    ctx.roundRect(x, y - TOKEN_HALF * size, ems * size, 2 * TOKEN_HALF * size, TOKEN_PAD * size);
    ctx.fill();
    ctx.globalAlpha = CARRIED_ALPHA;
    ctx.stroke();
    x += TOKEN_PAD * size;
  } else ctx.globalAlpha = kind === "own" ? 1 : CARRIED_ALPHA;
  if (glyph.mark !== undefined) mark(ctx, glyph.mark, x, y, size, glyph.tint ?? ink, ink);
  ctx.fillStyle = ink;
  ctx.fillText(glyph.text, x + markEms(glyph) * size, y);
}

/** The longest start of ``text``, whole characters followed by an
 * ellipsis, that fits ``room`` ems; null if not even one character does. */
function shortened(ctx: CanvasRenderingContext2D, text: string, room: number): string | null {
  let start = "";
  let fits: string | null = null;
  for (const { segment } of graphemes.segment(text)) {
    start += segment;
    if (room < measure(ctx, start + MORE.text) / BASE_SIZE) break;
    fits = start + MORE.text;
  }
  return fits;
}

/** What a gap or the origin says of itself: how many children it holds, where known. */
function gapGlyph(box: Box): Glyph {
  if (box.kind !== "gap" || box.count === 0) return MORE;
  return { text: box.count === 1 ? "1 child" : `${box.count.toLocaleString()} children` };
}

/** ``kind`` drawn ``size`` tall from ``x``, centered on ``y``: filled with ``tint``, edged in ``ink``. */
function mark(ctx: CanvasRenderingContext2D, kind: Mark, x: number, y: number, size: number, tint: string, ink: string): void {
  const r = (MARK_WIDTH * size) / 2;
  const cx = x + r;
  ctx.fillStyle = tint;
  ctx.strokeStyle = ink;
  ctx.lineWidth = Math.max(1, size / 12);
  ctx.lineCap = "round";
  ctx.beginPath();
  switch (kind) {
    case "onset":
      // A note head, tilted as on a staff.
      ctx.ellipse(cx, y, r, r * 0.72, -0.35, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      break;
    case "offset":
      ctx.ellipse(cx, y, r * 0.92, r * 0.66, -0.35, 0, 2 * Math.PI);
      ctx.strokeStyle = tint;
      ctx.lineWidth = Math.max(1.5, size / 7);
      ctx.stroke();
      break;
    case "tie":
      ctx.arc(cx, y + r * 0.9, r, Math.PI * 1.15, Math.PI * 1.85);
      ctx.strokeStyle = tint;
      ctx.lineWidth = Math.max(1.5, size / 7);
      ctx.stroke();
      break;
    case "hit":
      ctx.moveTo(cx - r * 0.75, y - r * 0.75);
      ctx.lineTo(cx + r * 0.75, y + r * 0.75);
      ctx.moveTo(cx + r * 0.75, y - r * 0.75);
      ctx.lineTo(cx - r * 0.75, y + r * 0.75);
      ctx.strokeStyle = tint;
      ctx.lineWidth = Math.max(1.5, size / 6);
      ctx.stroke();
      break;
    case "time":
      // A clock face.
      ctx.arc(cx, y, r * 0.85, 0, 2 * Math.PI);
      ctx.moveTo(cx, y);
      ctx.lineTo(cx, y - r * 0.6);
      ctx.moveTo(cx, y);
      ctx.lineTo(cx + r * 0.45, y);
      ctx.stroke();
      break;
    case "instrument":
      ctx.roundRect(cx - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7, r * 0.4);
      ctx.fill();
      ctx.stroke();
      break;
    case "end":
      ctx.rect(cx - r * 0.7, y - r * 0.7, r * 1.4, r * 1.4);
      ctx.fillStyle = ink;
      ctx.fill();
      break;
  }
}

function outline(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  cam: Camera,
  view: Viewport,
  state: DrawState,
  ink: string,
): void {
  const selected = state.hover;
  if (selected === null) return;
  const scale = ctx.getTransform().a;
  const pixel = 1 / scale;
  ctx.strokeStyle = ctx.fillStyle = ink;
  // The path down to it, faint and thin.
  const ancestors = new Set(selected.ancestors.map((node) => frame.parentBoxes.get(node)));
  ctx.globalAlpha = 0.5;
  for (const box of frame.boxes) {
    if (!ancestors.has(box) && (box.node !== selected.box.node || box.rank !== selected.box.rank)) continue;
    frameRect(ctx, pixelRect(screenRect(box, cam, view), scale), pixel, false);
  }
  ctx.globalAlpha = 1;
  // A gap has no box to outline; the inspector still describes it.
  const box = selected.box;
  if (box.kind === "gap") return;
  // Its own pixels: a child thinner than a pixel, the row of pixels it is in.
  const exact = screenRect(box, cam, view);
  const own = pixelRect(exact, scale);
  const top = pixelCorner(exact.y0, scale);
  const rect = { ...own, y0: top, y1: Math.max(own.y1, top + pixel) };
  // Dashed where the branch is known but its continuation is not.
  frameRect(ctx, rect, 2 * pixel, !selected.exact);
}

/** A frame ``width`` thick just inside a box's pixels, alike on every
 * side; a box too small to hold one is filled. */
function frameRect(ctx: CanvasRenderingContext2D, rect: ScreenRect, width: number, dashed: boolean): void {
  const [w, h] = [rect.x1 - rect.x0, rect.y1 - rect.y0];
  if (w <= 0 || h <= 0) return;
  if (w <= 2 * width || h <= 2 * width) {
    ctx.fillRect(rect.x0, rect.y0, w, h);
    return;
  }
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([3 * width, 2 * width]);
  ctx.strokeRect(rect.x0 + width / 2, rect.y0 + width / 2, w - width, h - width);
  ctx.setLineDash([]);
}

/** ``value`` moved into [low, high]; ``low`` wins when the range is empty. */
function within(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

/** A message across the view; while ``time`` is given, a band sweeps behind it. */
function drawNote(ctx: CanvasRenderingContext2D, view: Viewport, palette: Palette, text: string, time: number | null): void {
  const { scheme } = palette;
  ctx.save();
  if (time !== null) {
    const band = view.width * 0.4;
    const x = ((time % SWEEP_MS) / SWEEP_MS) * (view.width + band) - band;
    const gradient = ctx.createLinearGradient(x, 0, x + band, 0);
    gradient.addColorStop(0, `${scheme.inkMuted}00`);
    gradient.addColorStop(0.5, `${scheme.inkMuted}1f`);
    gradient.addColorStop(1, `${scheme.inkMuted}00`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, view.width, view.height);
  }
  ctx.font = FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = scheme.inkMuted;
  ctx.fillText(text, view.width / 2, view.height / 2);
  ctx.restore();
}

/** The ruler's caption, shown by the units switch over its right end. */
export function rulerCaption(units: Units): string {
  return `−log q · ${units.name}`;
}

function ruler(ctx: CanvasRenderingContext2D, cam: Camera, view: Viewport, units: Units, palette: Palette): void {
  const { scheme } = palette;
  const top = view.height;
  ctx.fillStyle = scheme.surface;
  ctx.fillRect(0, top, view.width, RULER_HEIGHT);
  // The units switch shows the caption in this font, 8px from the right
  // edge (style.css); ticks stop short of it.
  const end = Math.floor(view.width - 8 - measure(ctx, rulerCaption(units)) - 12);
  if (end < 1) return;
  const ticks = layoutTicks({
    quantile: (p) => (cam.x + (p * end) / cam.k) / units.nats,
    scale: RULER_SCALE,
    width: end,
    minSpacing: RULER_MARKS.from,
  });
  ctx.strokeStyle = scheme.inkMuted;
  ctx.fillStyle = scheme.inkMuted;
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  // Where the next label may start.
  let free = 0;
  for (const { value, step, position, spacing } of ticks) {
    const x = Math.round(position) + 0.5;
    ctx.globalAlpha = fade(spacing, RULER_MARKS);
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + 5);
    ctx.stroke();
    // Labelled ticks are those of steps at least RULER_LABELS.from wide,
    // and steps nest, so centred labels clear each other. Only a label
    // pushed inside the ends can run into its neighbour.
    if (spacing < RULER_LABELS.from) continue;
    const label = RULER_SCALE.format(value, step);
    const width = measure(ctx, label);
    // Glyphs off the physical pixel grid blur.
    const left = pixelCorner(within(x - width / 2, 0, end - width), ctx.getTransform().a);
    if (left < free || end < left + width) continue;
    ctx.globalAlpha = fade(spacing, RULER_LABELS);
    ctx.fillText(label, left, top + 13);
    free = left + width + 8;
  }
  ctx.globalAlpha = 1;
}

/** Opacity rising smoothly from 0 at ``from`` pixels of room to 1 at ``to``. */
function fade(spacing: number, { from, to }: { from: number; to: number }): number {
  const t = within((spacing - from) / (to - from), 0, 1);
  return t * t * (3 - 2 * t);
}
