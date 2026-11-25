// Camera motions. Every motion keeps one screen point fixed; vertical zoom is
// never set directly, it follows depth through ``logView``.

import { type Camera, logView } from "../core/layout";
import { ROOT } from "../core/tree";

/**
 * Move the left edge ``dx`` nats deeper, zooming vertically about view
 * fraction ``fy``. Only the root's left edge stops it, at depth 0: past any
 * other origin lies its parent.
 */
export function moveDepth(cam: Camera, dx: number, fy: number): void {
  const before = logView(cam);
  cam.x = cam.origin === ROOT ? Math.max(0, cam.x + dx) : cam.x + dx;
  cam.a = fy + (cam.a - fy) * Math.exp(before - logView(cam));
}

/**
 * Move ``share`` of the way to depth ``x`` with the point ``a`` view heights
 * down. The view is an interval of its origin, and the motion maps one
 * interval onto the other: a zoom about the one view fraction both leave in
 * place, so an edge that ends where it starts never moves. Its steps toward
 * the same target all zoom about that same fraction.
 */
export function glide(cam: Camera, to: { x: number; a: number }, share: number): void {
  const dx = to.x - cam.x;
  if (Math.abs(dx) < 1e-12) {
    cam.a += (to.a - cam.a) * share;
    return;
  }
  // The fixed fraction satisfies to.a = f + (cam.a − f) e^dx.
  // Use nonpositive exponents so the ratio stays finite when zooming out.
  const fixed = dx < 0
    ? (to.a - cam.a * Math.exp(dx)) / -Math.expm1(dx)
    : (to.a * Math.exp(-dx) - cam.a) / Math.expm1(-dx);
  moveDepth(cam, dx * share, fixed);
}

/** Shift content down by ``dy`` view heights. */
export function panVertical(cam: Camera, dy: number): void {
  cam.a += dy;
}

/**
 * Multiply the horizontal scale about the left edge. The view's height follows
 * the depth there, so keeping it fixed is what leaves the vertical view alone.
 */
export function scale(cam: Camera, factor: number): void {
  const k = cam.k * factor;
  if (Number.isFinite(k) && k > 0) cam.k = k;
}
