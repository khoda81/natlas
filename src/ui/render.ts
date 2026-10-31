// Canvas 2D drawing of one laid-out frame.

import { type Palette, rampPosition } from "./color";
import type { Box, Camera, Frame, Lens, Selection, Viewport } from "../core/layout";
import type { NodeId } from "../core/tree";
import type { Glyph, Mark } from "../readings/reading";

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
  /** Why the origin shows no children, if it does not; ``sweep`` animates
   * a faint band behind the text while they load. */
  note: { text: string; sweep: number | null } | null;
}

const FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const BASE_SIZE = 12;
const FONT = `${BASE_SIZE}px ${FAMILY}`;
const LABEL_MIN_HEIGHT = 8;
/** Labels grow to fit their box's height or width, whichever binds first, within these sizes. */
const LABEL_MAX_SIZE = 96;
/** Fraction of the visible box height a label may fill. */
const LABEL_HEIGHT_FILL = 0.6;
const LABEL_MIN_WIDTH = 14;
const GAP_MIN = 4;
/** A mark's width, and the space after it, in ems. */
const MARK_WIDTH = 0.8;
const MARK_GAP = 0.3;
const MORE: Glyph = { text: "…" };
export const RULER_HEIGHT = 22;
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
  const ramp = equal ? palette.sequential : palette.diverging;

  ctx.fillStyle = scheme.surface;
  ctx.fillRect(0, 0, view.width, view.height + RULER_HEIGHT);
  ctx.font = FONT;
  ctx.textBaseline = "middle";

  const labels: { box: Box; rect: ScreenRect; glyph: Glyph; ink: string; grows: boolean }[] = [];
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
      tail(ctx, frame.boxes, start, i, cam, view, equal, ramp, scale);
      continue;
    }

    // A gap with no estimate yet is unknown, as the background already says:
    // only its label shows. One with an estimate is drawn as a silhouette.
    let ink = scheme.inkMuted;
    if (box.kind !== "gap" || Number.isFinite(box.x1)) {
      // With p = q, a gap's q estimates a typical child's weight.
      const level = ramp.level(rampPosition(equal ? box.q : box.p, box.q, equal));
      ink = ramp.ink[level]!;
      ctx.fillStyle = ramp.fill[level]!;
      // A separator is the box's own last column and row of pixels, left
      // as the background. No other box shows on them.
      const separator = box.kind !== "gap" && GAP_MIN <= width && GAP_MIN <= height ? 1 / scale : 0;
      const pixels = pixelRect(rect, scale);
      ctx.fillRect(pixels.x0, pixels.y0, pixels.x1 - pixels.x0 - separator, pixels.y1 - pixels.y0 - separator);
    }

    if (
      height <= LABEL_MIN_HEIGHT
      || width <= LABEL_MIN_WIDTH
      || box.count === 0
    ) {
      continue;
    }

    labels.push({
      box,
      rect,
      glyph: box.branch !== null ? state.glyph(box.node, box.branch) : gapGlyph(box),
      ink,
      grows: box.branch !== null
    });
  }

  // A label's size follows its whole box, never the viewport. Within its box,
  // a label moves toward the viewport as far as the box allows: it starts at
  // the box's left, centered, and slides right past the left screen edge and
  // up or down into view, stopping at the box's own edges. A label that still
  // does not fit stays its size and is clipped.
  for (const { box, rect, glyph, ink, grows } of labels) {
    const boxLeft = (box.x0 - cam.x) * cam.k;
    const width = (box.x1 - box.x0) * cam.k - 2;
    const top = box.y0 * view.height;
    const bottom = box.y1 * view.height;
    const marked = glyph.mark === undefined ? 0 : MARK_WIDTH + (glyph.text === "" ? 0 : MARK_GAP);
    const textWidth = measure(ctx, glyph.text) / BASE_SIZE + marked;
    // At size s the text spans textWidth·s, plus s/4 of padding on each side.
    const fit = Math.min((bottom - top) * LABEL_HEIGHT_FILL, width / (textWidth + 0.5));
    const size = grows ? Math.min(LABEL_MAX_SIZE, fit) : BASE_SIZE;
    const pad = Math.max(4, size / 4);
    const span = textWidth * size;
    const x = within(pad, boxLeft + pad, boxLeft + width - pad - span);
    const y = within(within((top + bottom) / 2, size / 2, view.height - size / 2), top + size / 2, bottom - size / 2);
    if (x > view.width || x + span < 0 || y - size / 2 > view.height || y + size / 2 < 0) continue;
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x0, rect.y0, rect.x1 - rect.x0 - 2, rect.y1 - rect.y0);
    ctx.clip();
    ctx.font = `${size}px ${FAMILY}`;
    if (glyph.mark !== undefined) mark(ctx, glyph.mark, x, y, size, glyph.tint ?? ink, ink);
    ctx.fillStyle = ink;
    ctx.fillText(glyph.text, x + marked * size, y);
    ctx.restore();
  }

  const note = state.note ?? (visible ? null : { text: "No continuation discovered here · press 0 for the root", sweep: null });
  if (note !== null) drawNote(ctx, view, palette, note.text, note.sweep);

  outline(ctx, frame, cam, view, state, scheme.ink);
  ruler(ctx, cam, view, state.units, palette);
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
  equal: boolean,
  ramp: Palette["sequential"],
  scale: number,
): void {
  const first = screenRect(boxes[start]!, cam, view);
  const bottom = screenRect(boxes[end]!, cam, view).y1;
  // Meet ordinary boxes on their physical-pixel edges. Inside the tail,
  // keep the measured contour and color boundaries at full precision.
  const edges = pixelRect({ ...first, y1: bottom }, scale);
  if (edges.y1 <= edges.y0) return;
  const fill = (box: Box) => ramp.fill[ramp.level(rampPosition(box.p, box.q, equal))]!;
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

function niceStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(raw));
  for (const multiple of [1, 2, 5, 10]) if (multiple * power >= raw) return multiple * power;
  return 10 * power;
}

function format(value: number, step: number): string {
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  if (decimals > 12 || Math.abs(value) >= 1e12) return value.toExponential(3);
  return value.toFixed(decimals);
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
  const pxPerUnit = cam.k * units.nats;
  const step = niceStep(80 / pxPerUnit);
  const top = view.height;
  ctx.fillStyle = scheme.surface;
  ctx.fillRect(0, top, view.width, RULER_HEIGHT);
  ctx.fillStyle = scheme.inkMuted;
  ctx.strokeStyle = scheme.inkMuted;
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  // The units switch shows the caption in this font, 8px from the right
  // edge (style.css); ticks stop short of it.
  const captionLeft = view.width - 8 - measure(ctx, rulerCaption(units));
  const first = Math.ceil(cam.x / units.nats / step);
  const last = Math.floor((cam.x + view.width / cam.k) / units.nats / step);
  ctx.beginPath();
  // A bounded tick count also works when adjacent integer indices are no
  // longer distinguishable at an extreme, but finite, horizontal scale.
  for (let tick = 0; tick <= Math.ceil(view.width / 80) + 1; tick++) {
    const i = first + tick;
    if (!Number.isFinite(i) || i > last || !Number.isFinite(step) || step <= 0) break;
    const value = i * step;
    const x = Math.round((value * units.nats - cam.x) * cam.k) + 0.5;
    const text = format(value, step);
    if (x + 3 + measure(ctx, text) > captionLeft - 12) break;
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + 5);
    ctx.fillText(text, x + 3, top + 12);
  }
  ctx.stroke();
}
