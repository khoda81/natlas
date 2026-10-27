// Camera motions. Every motion keeps one screen point fixed; vertical zoom is
// never set directly, it follows depth through ``logView``.

import { type Camera, logView } from "./layout";

export const MIN_SCALE = 1e-3;
export const MAX_SCALE = 1e7;

/** Move the left edge ``dx`` nats deeper, zooming vertically about view fraction ``fy``. */
export function moveDepth(cam: Camera, dx: number, fy: number): void {
  const before = logView(cam);
  cam.x += dx;
  cam.a = fy + (cam.a - fy) * Math.exp(before - logView(cam));
}

/** Shift content down by ``dy`` view heights. */
export function panVertical(cam: Camera, dy: number): void {
  cam.a += dy;
}

/** Multiply the horizontal scale about screen x ``sx`` (px) and view fraction ``fy``. */
export function scaleAt(cam: Camera, factor: number, sx: number, fy: number): void {
  const depth = cam.x + sx / cam.k;
  cam.k = Math.min(MAX_SCALE, Math.max(MIN_SCALE, cam.k * factor));
  moveDepth(cam, depth - sx / cam.k - cam.x, fy);
}

/** Drag: content follows the pointer by (dx, dy) px, zooming about its y. */
export function drag(cam: Camera, dx: number, dy: number, height: number, fy: number): void {
  panVertical(cam, dy / height);
  moveDepth(cam, -dx / cam.k, fy);
}
