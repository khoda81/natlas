// Canvas 2D drawing of one laid-out frame.

import { decimalTicks, layoutTicks } from "@qbar/qbar";
import type { Palette } from "./color";
import type { Box, Camera, Frame, Lens, Selection, Viewport } from "../core/layout";
import type { NodeId } from "../core/tree";
import type { Glyph, Mark, Timeline } from "../readings/reading";

export interface Units {
  name: string;
  /** Nats per unit. */
  nats: number;
}

export const NATS: Units = { name: "nats", nats: 1 };
export const BITS: Units = { name: "bits", nats: Math.LN2 };

export interface DrawState {
  hover: Selection | null;
  /** ``node:rank`` keys of the hovered box's ancestor children. */
  ancestors: Set<string>;
  units: Units;
  /** How the child along ``branch`` of node ``node`` reads. */
  glyph: (node: NodeId, branch: number) => Glyph;
  /** The node the child along ``branch`` of ``node`` is. */
  child: (node: NodeId, branch: number) => NodeId;
  /** Why the origin shows no children, if it does not; ``sweep`` animates
   * a faint band behind the text while they load. */
  note: { text: string; sweep: number | null } | null;
  /** Where the reading draws its time, in canvas pixels, while it does:
   * a box whose glyph moves time is a window onto it. */
  timeline: Timeline | null;
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
/** Boxes at least this many CSS pixels across keep a separator that way. */
const GAP_MIN = 4;
/** A mark's width, and the space after it, in ems. */
const MARK_WIDTH = 0.8;
const MARK_GAP = 0.3;
const MORE: Glyph = { text: "…" };
/** How strongly a time window shades the time its event moves over. */
const TIME_ALPHA = 0.28;
/** A time window's line where its event lands, and the arrow pointing to it
 * from the window's edge where it lands outside, in CSS pixels. */
const TIME_LINE = 2;
const TIME_ARROW = 5;
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

export function childKey(node: NodeId, rank: number): string {
  return `${node}:${rank}`;
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

/** The physical pixels a box shows on, expressed in CSS pixels. */
function pixelRect(rect: ScreenRect, scale: number): ScreenRect {
  const edge = (at: number) => Math.ceil(at * scale) / scale;
  return { x0: edge(rect.x0), x1: edge(rect.x1), y0: edge(rect.y0), y1: edge(rect.y1) };
}

/** The top-left corner of the physical pixel holding a CSS coordinate. */
export function pixelCorner(at: number, scale: number): number {
  return Math.floor(at * scale) / scale;
}

export function draw(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  cam: Camera,
  view: Viewport,
  lens: Lens,
  palette: Palette,
  state: DrawState,
): void {
  const scale = ctx.getTransform().a;
  if (textWidths.get(ctx)?.scale !== scale) textWidths.set(ctx, { scale, widths: new Map() });
  const { scheme } = palette;
  const equal = lens.p === lens.q;

  ctx.fillStyle = scheme.surface;
  ctx.fillRect(0, 0, view.width, view.height + RULER_HEIGHT);
  ctx.font = FONT;
  ctx.textBaseline = "middle";

  // Where each node's children end in view: the last one's bottom meets the
  // children of the node's sibling below.
  const ends = new Map<NodeId, number>();
  for (const box of frame.boxes) ends.set(box.node, Math.max(ends.get(box.node) ?? -Infinity, box.y1));
  const labels: { box: Box; rect: ScreenRect; ink: string }[] = [];
  let visible = false;
  for (let i = 0; i < frame.boxes.length; i++) {
    const box = frame.boxes[i]!;
    const rect = screenRect(box, cam, view);
    const width = rect.x1 - rect.x0;
    const height = rect.y1 - rect.y0;
    if (width <= 0 || height <= 0) continue;
    visible = true;

    if (box.kind === "merged") {
      const start = i;
      while (i + 1 < frame.boxes.length && joined(frame.boxes[i]!, frame.boxes[i + 1]!)) i++;
      tail(ctx, frame.boxes, start, i, cam, view, (box) => palette.tile(box.p, box.q, equal).fill, scale);
      continue;
    }

    // A gap with no estimate yet is unknown, as the background already says:
    // only its label shows. One with an estimate is drawn as a silhouette.
    let ink = scheme.inkMuted;
    if (box.kind !== "gap" || Number.isFinite(box.x1)) {
      const tile = palette.tile(box.p, box.q, equal);
      ink = tile.ink;
      ctx.fillStyle = tile.fill;
      // Separators are the box's own last column and row of pixels, left as
      // the background; no other box shows on them. The column parts a
      // child from its children. The row parts only a node's last child
      // from its cousins below: siblings part where their widths do.
      const column = box.kind !== "gap" && GAP_MIN <= width ? 1 / scale : 0;
      const row = box.y1 === ends.get(box.node) && GAP_MIN <= height ? 1 / scale : 0;
      const pixels = pixelRect(rect, scale);
      ctx.fillRect(pixels.x0, pixels.y0, pixels.x1 - pixels.x0 - column, pixels.y1 - pixels.y0 - row);
    }

    if (box.count !== 0) labels.push({ box, rect, ink });
  }

  // A box whose label is too small to draw hands it, after the labels it
  // carries itself, to each of its children; they carry it until one draws
  // it before its own. A box's parent is the box of the branch its node is.
  const parents = new Map<NodeId, Box>();
  for (const box of frame.boxes) {
    if (box.kind !== "merged" && box.branch !== null) parents.set(state.child(box.node, box.branch), box);
  }
  const carriedBy = new Map<NodeId, readonly Glyph[]>();
  /** The labels the children of ``node`` carry, outermost first. A time
   * window shows its time at any size and carries what it was handed on. */
  const carried = (node: NodeId): readonly Glyph[] => {
    let glyphs = carriedBy.get(node);
    if (glyphs === undefined) {
      const parent = parents.get(node);
      const glyph = parent && state.glyph(parent.node, parent.branch!);
      glyphs = parent === undefined || glyph === undefined
        ? []
        : windowed(glyph)
        ? carried(parent.node)
        : fitOf(parent) !== null
        ? []
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
      fits.set(box, fit = fitLabel(ctx, before, glyph, width, height, box.branch !== null));
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
    const glyph = box.branch === null ? null : state.glyph(box.node, box.branch);
    if (glyph?.time !== undefined && state.timeline !== null) {
      timeWindow(ctx, rect, view, glyph.time, state.timeline, ink);
      continue;
    }
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
        drawPiece(ctx, piece, at, row * LABEL_LINE * size, size, ink);
        at += (piece.ems + TOKEN_GAP) * size;
      }
    });
    ctx.restore();
  }

  const note = state.note ?? (visible ? null : { text: "No continuation discovered here · press 0 for the root", sweep: null });
  if (note !== null) drawNote(ctx, view, palette, note.text, note.sweep);

  outline(ctx, frame, cam, view, state, scheme.ink);
  ruler(ctx, cam, view, state.units, palette);
}

/** A box as a window onto a timeline drawn above it, across the same
 * pixels: the time its event moves over shaded, and a line where it lands,
 * or an arrow at the window's edge toward where it lands outside. */
function timeWindow(
  ctx: CanvasRenderingContext2D,
  rect: ScreenRect,
  view: Viewport,
  time: { from: number; to: number },
  timeline: Timeline,
  ink: string,
): void {
  // The box's own pixels, short of its separator, as labels have them.
  const left = rect.x0;
  const right = rect.x1 - 2;
  const top = rect.y0;
  const bottom = rect.y1;
  if (right <= left || bottom <= top) return;
  const from = timeline.x(time.from);
  const to = timeline.x(time.to);
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  ctx.fillStyle = ink;
  // Far-off coordinates lose precision and can stall rasterizing.
  const shaded0 = Math.max(left, Math.min(from, to));
  const shaded1 = Math.min(right, Math.max(from, to));
  if (shaded0 < shaded1) {
    ctx.globalAlpha = TIME_ALPHA;
    ctx.fillRect(shaded0, top, shaded1 - shaded0, bottom - top);
    ctx.globalAlpha = 1;
  }
  if (left <= to && to <= right) {
    ctx.fillRect(Math.round(to) - TIME_LINE / 2, top, TIME_LINE, bottom - top);
  } else {
    const size = Math.min(TIME_ARROW, (bottom - top) / 2, (right - left) / 2);
    // In the middle of the window's part in view.
    const middle = (Math.max(top, 0) + Math.min(bottom, view.height)) / 2;
    const tip = to < left ? left + 1 : right - 1;
    const back = to < left ? tip + size : tip - size;
    ctx.beginPath();
    ctx.moveTo(tip, middle);
    ctx.lineTo(back, middle - size);
    ctx.lineTo(back, middle + size);
    ctx.fill();
  }
  ctx.restore();
}

/** Whether ``below`` continues the band silhouette ``above`` ends: only
 * touching sibling bands share one. An unknown interval, a visible child, or
 * another node keeps its own boundary. Touching runs coalesce, so
 * neighbouring bands share one stored edge and touch exactly. */
function joined(above: Box, below: Box): boolean {
  return above.kind === "merged" && below.kind === "merged" && above.node === below.node
    && above.x0 === below.x0 && above.y1 === below.y0;
}

/** Paint a dense sibling tail once. Separate antialiased rectangles let the
 * background leak through their shared edges. A single contour has only an
 * outside edge; a vertical gradient carries each band's own color without
 * adding interior coverage edges or rounding its probability geometry. */
function tail(
  ctx: CanvasRenderingContext2D,
  boxes: readonly Box[],
  start: number,
  end: number,
  cam: Camera,
  view: Viewport,
  fill: (box: Box) => string,
  scale: number,
): void {
  const first = screenRect(boxes[start]!, cam, view);
  const bottom = screenRect(boxes[end]!, cam, view).y1;
  // Meet ordinary boxes on their physical-pixel edges. Inside the tail,
  // keep the measured contour and color boundaries at full precision.
  const edges = pixelRect({ ...first, y1: bottom }, scale);
  if (edges.y1 <= edges.y0) return;
  const initial = fill(boxes[start]!);
  let previous = initial;
  let gradient: CanvasGradient | null = null;
  ctx.beginPath();
  ctx.moveTo(edges.x0, edges.y0);
  for (let i = start; i <= end; i++) {
    const box = boxes[i]!;
    const rect = screenRect(box, cam, view);
    ctx.lineTo(rect.x1, i === start ? edges.y0 : Math.min(rect.y0, edges.y1));
    ctx.lineTo(rect.x1, i === end ? edges.y1 : Math.min(rect.y1, edges.y1));
    const color = fill(box);
    if (color !== previous) {
      if (gradient === null) {
        gradient = ctx.createLinearGradient(0, first.y0, 0, bottom);
        gradient.addColorStop(0, initial);
      }
      // Paired stops preserve the measured band colors. Interpolating only
      // the tail's first and last colors would invent values in between.
      const at = (rect.y0 - first.y0) / (bottom - first.y0);
      gradient.addColorStop(at, previous);
      gradient.addColorStop(at, color);
      previous = color;
    }
  }
  ctx.lineTo(edges.x0, edges.y1);
  ctx.closePath();
  if (gradient !== null) gradient.addColorStop(1, previous);
  ctx.fillStyle = gradient ?? initial;
  ctx.fill();
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
 * stay at the base size. */
function fitLabel(
  ctx: CanvasRenderingContext2D,
  carried: readonly Glyph[],
  glyph: Glyph,
  width: number,
  height: number,
  grows: boolean,
): Fit | null {
  const largest = grows ? LABEL_MAX_SIZE : BASE_SIZE;
  const own: Piece = { kind: "own", glyph, ems: glyphEms(ctx, glyph) };
  const tokens = carried.map((glyph): Piece => ({ kind: "carried", glyph, ems: 2 * TOKEN_PAD + glyphEms(ctx, glyph) }));
  const cut: Piece = { kind: "cut", glyph: MORE, ems: glyphEms(ctx, MORE) };
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
 * carries framed and fainter than its own. */
function drawPiece(ctx: CanvasRenderingContext2D, { kind, glyph, ems }: Piece, x: number, y: number, size: number, ink: string): void {
  ctx.globalAlpha = kind === "own" ? 1 : CARRIED_ALPHA;
  if (kind === "carried") {
    ctx.strokeStyle = ink;
    ctx.lineWidth = Math.max(1, size / 16);
    ctx.beginPath();
    ctx.roundRect(x, y - TOKEN_HALF * size, ems * size, 2 * TOKEN_HALF * size, TOKEN_PAD * size);
    ctx.stroke();
    x += TOKEN_PAD * size;
  }
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
  if (state.hover === null) return;
  const scale = ctx.getTransform().a;
  const pixel = 1 / scale;
  ctx.strokeStyle = ink;
  ctx.lineWidth = pixel;
  ctx.globalAlpha = 0.35;
  for (const box of frame.boxes) {
    if (box.kind === "merged" || !state.ancestors.has(childKey(box.node, box.rank))) continue;
    const rect = pixelRect(screenRect(box, cam, view), scale);
    ctx.strokeRect(rect.x0 + pixel / 2, rect.y0 + pixel / 2, Math.max(0, rect.x1 - rect.x0 - pixel), Math.max(0, rect.y1 - rect.y0 - pixel));
  }
  ctx.globalAlpha = 1;
  // A gap has no box to outline; the inspector still describes it.
  const box = state.hover.box;
  if (box.kind === "gap") return;
  ctx.lineWidth = 2 * pixel;
  if (!state.hover.exact) ctx.setLineDash([5 * pixel, 3 * pixel]);
  // Outline where the frame draws the branch: its own box, or the rows it
  // touches of the band silhouette holding it. Without a box, the origin.
  const boxes = frame.boxes;
  const at = boxes.findIndex(b => b.node === box.node
    && (b.kind === "merged" ? b.y0 <= box.y0 && box.y1 <= b.y1 : b.rank === box.rank));
  const exact = screenRect(box, cam, view);
  const own = pixelRect(exact, scale);
  let rect = own;
  if (at !== -1) {
    let [first, last] = [at, at];
    while (0 < first && joined(boxes[first - 1]!, boxes[first]!)) first--;
    while (last + 1 < boxes.length && joined(boxes[last]!, boxes[last + 1]!)) last++;
    const drawn = pixelRect({
      ...screenRect(boxes[at]!, cam, view),
      y0: screenRect(boxes[first]!, cam, view).y0,
      y1: screenRect(boxes[last]!, cam, view).y1,
    }, scale);
    rect = { ...drawn, y0: Math.max(drawn.y0, pixelCorner(exact.y0, scale)), y1: Math.min(drawn.y1, own.y1) };
  }
  ctx.strokeRect(rect.x0 + pixel, rect.y0 + pixel, Math.max(0, rect.x1 - rect.x0 - 2 * pixel), Math.max(0, rect.y1 - rect.y0 - 2 * pixel));
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
