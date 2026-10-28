// Canvas 2D drawing of one laid-out frame.

import { type Palette, rampPosition } from "./color";
import type { Box, Camera, Frame, Lens, Viewport } from "./layout";
import type { Glyph, Mark } from "./reading";
import type { NodeId } from "./tree";

export interface Units {
  name: string;
  /** Nats per unit. */
  nats: number;
}

export const NATS: Units = { name: "nats", nats: 1 };
export const BITS: Units = { name: "bits", nats: Math.LN2 };

export interface DrawState {
  hover: Box | null;
  /** ``node:slot`` keys of the hovered box's ancestor slots. */
  ancestors: Set<string>;
  units: Units;
  /** How the child with ``label`` of node ``node`` reads. */
  glyph: (node: NodeId, label: number) => Glyph;
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
// Canvas coordinates far outside the view lose precision and can stall rasterizing.
const OVERSCAN = 64;

export function slotKey(node: number, slot: number): string {
  return `${node}:${slot}`;
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

export function draw(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  cam: Camera,
  view: Viewport,
  lens: Lens,
  palette: Palette,
  state: DrawState,
): void {
  const { scheme } = palette;
  const equal = lens.p === lens.q;
  const ramp = equal ? palette.sequential : palette.diverging;

  ctx.fillStyle = scheme.surface;
  ctx.fillRect(0, 0, view.width, view.height + RULER_HEIGHT);
  ctx.font = FONT;
  ctx.textBaseline = "middle";

  const labels: { box: Box; rect: ScreenRect; glyph: Glyph; ink: string; grows: boolean }[] = [];
  for (const box of frame.boxes) {
    const rect = screenRect(box, cam, view);
    const width = rect.x1 - rect.x0;
    const height = rect.y1 - rect.y0;
    if (width <= 0 || height <= 0) continue;

    let fill: string;
    let ink: string;
    if (box.kind === "merged") {
      fill = scheme.merged;
      ink = palette.mergedInk;
    } else if (box.kind === "tail") {
      fill = scheme.tail;
      ink = palette.tailInk;
    } else {
      const level = ramp.level(rampPosition(box.p, box.q, equal));
      const leaf = box.kind === "leaf";
      fill = (leaf ? ramp.faded : ramp.fill)[level]!;
      ink = (leaf ? ramp.fadedInk : ramp.ink)[level]!;
    }
    ctx.fillStyle = fill;
    ctx.fillRect(
      rect.x0,
      rect.y0,
      width >= GAP_MIN ? width - 1 : width,
      height >= GAP_MIN ? height - 1 : height,
    );

    if (box.kind !== "merged" && height >= LABEL_MIN_HEIGHT && width >= LABEL_MIN_WIDTH) {
      labels.push({ box, rect, glyph: box.label === null ? MORE : state.glyph(box.node, box.label), ink, grows: box.label !== null });
    }
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
    const textWidth = ctx.measureText(glyph.text).width / BASE_SIZE + marked;
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

  if (frame.boxes.length === 0) {
    ctx.fillStyle = scheme.inkMuted;
    ctx.textAlign = "center";
    ctx.fillText("Nothing discovered here · press 0 for the root", view.width / 2, view.height / 2);
    ctx.textAlign = "left";
  }

  outline(ctx, frame, cam, view, state, scheme.ink);
  ruler(ctx, cam, view, state.units, palette);
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
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.35;
  for (const box of frame.boxes) {
    if (box.kind === "merged" || !state.ancestors.has(slotKey(box.node, box.slot))) continue;
    const rect = screenRect(box, cam, view);
    ctx.strokeRect(rect.x0 + 0.5, rect.y0 + 0.5, rect.x1 - rect.x0 - 2, rect.y1 - rect.y0 - 2);
  }
  ctx.globalAlpha = 1;
  ctx.lineWidth = 2;
  const rect = screenRect(state.hover, cam, view);
  ctx.strokeRect(rect.x0 + 1, rect.y0 + 1, Math.max(0, rect.x1 - rect.x0 - 3), Math.max(0, rect.y1 - rect.y0 - 3));
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
  return value.toFixed(decimals);
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
  const caption = `−log q · ${units.name}`;
  const captionLeft = view.width - 8 - ctx.measureText(caption).width;
  const first = Math.ceil(cam.x / units.nats / step);
  const last = Math.floor((cam.x + view.width / cam.k) / units.nats / step);
  ctx.beginPath();
  for (let i = first; i <= last; i++) {
    const value = i * step;
    const x = Math.round((value * units.nats - cam.x) * cam.k) + 0.5;
    const text = format(value, step);
    if (x + 3 + ctx.measureText(text).width > captionLeft - 12) break;
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + 5);
    ctx.fillText(text, x + 3, top + 12);
  }
  ctx.stroke();
  ctx.fillText(caption, captionLeft, top + 12);
}
