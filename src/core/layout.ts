// Anchored integer-distribution geometry, picking, and what a view lacks,
// implemented by the shared Rust core. These adapters keep camera mutation
// at the UI edge.

import {
  aim as aimAnchor,
  clampPosition,
  extent as measureExtent,
  layout as layoutFrame,
  settle as settleAnchor,
  type LayoutFrame,
} from "./wasm";
import type { Branch, IntervalTree, Lens, NodeId, Rank } from "./tree";

export type { Lens } from "./tree";

/** A point at or below a camera's origin: the node ``path`` leads to from the
 * origin, and the fraction of that node's p mass above the point, as digits
 * in base 256. No digits is the node's top. A value: never changed in place. */
export interface Anchor {
  readonly path: readonly Branch[];
  readonly digits: Uint8Array;
}

/** The top of the node ``path`` leads to from the origin. */
export function top(path: readonly Branch[] = []): Anchor {
  return { path, digits: new Uint8Array() };
}

export interface Camera {
  /** The node the view is measured from, standing as the root does: x is
   * q-surprisal from its children's left edge, and the view spans e^-x of its
   * interval. It bounds the view; nothing above it is needed. */
  origin: NodeId;
  x: number;
  k: number;
  /** A point of the origin, ``a`` view heights below the view's top. */
  anchor: Anchor;
  a: number;
}

export interface Viewport {
  width: number;
  height: number;
}

/** A child with known children, one without, children not yet known, or children too thin to draw apart. */
export type BoxKind = "node" | "leaf" | "gap" | "merged";

export interface Box {
  node: NodeId;
  /** The first rank the box covers. */
  rank: Rank;
  /** Children covered; 0 for a gap of unknown size. */
  count: number;
  kind: BoxKind;
  branch: Branch | null;
  /** Conditional probabilities; p-weighted geometric means for a band. */
  p: number;
  /** NaN while unknown; like p, a representative child weight for a band. */
  q: number;
  lnMass: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** How an Expand reads the tree below its node: down a path, the children
 * holding points (each ln of the mass below it), or the children along
 * branches. */
export type Read = { path: Branch[] } | { points: number[] } | { branches: Branch[] };

/** What a frame lacks, as the Expand request that fills it: ``read`` below
 * ``node``, filling in ``target``, which is ``node`` or a child it holds. */
export interface Miss {
  node: NodeId;
  target: NodeId;
  measure: number;
  read: Read;
  /** Visible pixels; larger first. */
  priority: number;
}

const KINDS: BoxKind[] = ["node", "leaf", "gap", "merged"];
const boxBuffers: Box[][] = [];

/** Rust owns the full frame. Only visible rectangles and requests cross the Wasm boundary. */
export class Frame {
  readonly boxes: Box[];
  private missesCache?: Miss[];
  private disposed = false;

  constructor(readonly raw: LayoutFrame) {
    const packed = raw.boxes();
    const nodes = raw.nodes();
    this.boxes = boxBuffers.pop() ?? [];
    for (let i = 0, box = 0; i < packed.length; i += 11, box++) {
      const b = this.boxes[box] ??= { node: 0n, rank: 0, count: 0, kind: "node", branch: null,
        p: 0, q: 0, lnMass: 0, x0: 0, x1: 0, y0: 0, y1: 0 };
      b.node = nodes[box]!; b.rank = packed[i]!; b.count = packed[i + 1]!;
      b.kind = KINDS[packed[i + 2]!]!; b.branch = packed[i + 3] === -1 ? null : packed[i + 3]!;
      b.p = packed[i + 4]!; b.q = packed[i + 5]!; b.lnMass = packed[i + 6]!;
      b.x0 = packed[i + 7]!; b.x1 = packed[i + 8]!; b.y0 = packed[i + 9]!; b.y1 = packed[i + 10]!;
    }
    this.boxes.length = packed.length / 11;
  }

  get misses(): Miss[] {
    return this.missesCache ??= this.raw.misses();
  }

  get originTop(): number { return this.raw.originTop(); }
  get originBottom(): number { return this.raw.originBottom(); }

  /** All frame views are invalid after disposal; buffers may be reused. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.raw.free();
    if (boxBuffers.length < 2 && this.boxes.length <= 32768) boxBuffers.push(this.boxes);
  }
}

export interface Selection {
  box: Box;
  /** False in null space: the branch is known, its exact continuation is not. */
  exact: boolean;
}

export interface LayoutOptions {
  /** Children thinner than this many CSS pixels join a band; 1 / DPR is one physical pixel. */
  mergePx: number;
}

/** ln of the view's height in origin heights: the view spans e^-x of the origin. */
export function logView(cam: Camera): number {
  return -cam.x;
}

export function layout(
  tree: IntervalTree,
  lens: Lens,
  cam: Camera,
  view: Viewport,
  options: Partial<LayoutOptions> = {},
): Frame {
  return new Frame(layoutFrame(tree.raw, lens.p, lens.q, cam, view, options.mergePx ?? 1));
}

/** Name the camera's point by the deepest node that holds it and covers the
 * view, where p still places its path: nothing on screen moves, and the same
 * view of the same tree always gets the same name. */
export function settle(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.anchor = settleAnchor(tree.raw, lens.p, lens.q, cam) as Anchor;
}

/** Move the camera's point to what the view shows ``fy`` view heights down, so
 * a zoom about ``fy`` changes depth alone and zooming back returns exactly.
 * Nothing on screen moves. */
export function aim(tree: IntervalTree, lens: Lens, cam: Camera, fy: number): void {
  const aimed = aimAnchor(tree.raw, lens.p, lens.q, cam, fy) as { anchor: Anchor; a: number };
  cam.anchor = aimed.anchor;
  cam.a = aimed.a;
}

/** The origin's interval edges in view heights. */
export function extent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  return measureExtent(tree.raw, lens.p, lens.q, cam) as { top: number; bottom: number };
}

export function clamp(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.a = clampPosition(tree.raw, lens.p, lens.q, cam);
}

/** The child a pixel ``pixel`` CSS pixels square, with its top-left corner at
 * ``x``, ``y``, shows, however thin, found in the frame drawn with ``cam``. */
export function pick(tree: IntervalTree, lens: Lens, frame: Frame, cam: Camera, view: Viewport, x: number, y: number, pixel: number): Selection | null {
  return frame.raw.pick(tree.raw, lens.p, lens.q, cam, view, x, y, pixel) as Selection | null;
}
