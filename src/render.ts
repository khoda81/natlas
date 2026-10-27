// Canvas 2D drawing of one laid-out frame.

import { type Palette, rampPosition } from "./color";
import type { Box, Camera, Frame, Lens, Viewport } from "./layout";
import type { IntervalTree } from "./tree";

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
}

const FONT = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const LABEL_MIN_HEIGHT = 14;
const LABEL_MIN_WIDTH = 14;
const GAP_MIN = 4;
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
  tree: IntervalTree,
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

  const labels: { rect: ScreenRect; text: string; ink: string }[] = [];
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

    if (box.kind !== "merged" && height >= LABEL_MIN_HEIGHT && Math.min(rect.x1, view.width) - Math.max(rect.x0, 0) >= LABEL_MIN_WIDTH) {
      labels.push({ rect, text: tree.node(box.node).label(box.slot), ink });
    }
  }

  // Labels stick to the visible part of their box.
  for (const { rect, text, ink } of labels) {
    const left = Math.max(rect.x0, 0);
    const top = Math.max(rect.y0, 0);
    const bottom = Math.min(rect.y1, view.height);
    if (bottom - top < LABEL_MIN_HEIGHT) continue;
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, Math.min(rect.x1, view.width) - left - 2, bottom - top);
    ctx.clip();
    ctx.fillStyle = ink;
    ctx.fillText(text, left + 4, (top + bottom) / 2);
    ctx.restore();
  }

  outline(ctx, frame, cam, view, state, scheme.ink);
  ruler(ctx, cam, view, state.units, palette);
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
