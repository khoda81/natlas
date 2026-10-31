// Anchored integer-distribution geometry, picking, and what a view lacks,
// implemented by the shared Rust core. These adapters keep camera mutation
// at the UI edge.

import {
  clampPosition,
  extent as measureExtent,
  layout as layoutFrame,
  repair as repairAnchor,
  type LayoutFrame,
} from "./wasm";
import type { Branch, IntervalTree, Lens, NodeId, Rank } from "./tree";

export type { Lens } from "./tree";

/** The top edge of a node's child at ``rank``; rank 0 is the node's top. */
export interface Edge {
  node: NodeId;
  rank: Rank;
}

export interface Camera {
  /** The node the view is measured from, standing as the root does: x is
   * q-surprisal from its children's left edge, and the view spans e^-x of its
   * interval. It bounds the view; nothing above it is needed. */
  origin: NodeId;
  x: number;
  k: number;
  /** A known edge at or below the origin, ``a`` view heights below the view's top. */
  anchor: Edge;
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
  p: number;
  /** NaN while unknown. */
  q: number;
  lnMass: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** What a frame lacks, as the Expand request that fills it: ``node``, and
 * ``branch`` below it when given, name the target, whose window should be
 * listed down to ``horizon``. Asking again finds nothing new while the
 * target's first missing rank stays ``from``. */
export interface Miss {
  node: NodeId;
  branch: Branch | null;
  measure: number;
  from: Rank;
  lnTop: number;
  lnBottom: number;
  horizon: number;
  /** Visible pixels; larger first. */
  priority: number;
}

const KINDS: BoxKind[] = ["node", "leaf", "gap", "merged"];
const boxBuffers: Box[][] = [];

/** Rust owns the full frame. Only visible rectangles and requests cross the Wasm boundary. */
export class Frame {
  readonly boxes: Box[];
  readonly cover: { node: NodeId; y: number } | null;
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
    this.cover = raw.cover() as Frame["cover"];
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
  /** Children thinner than this many pixels join a band. */
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

/** Move the anchor up to the deepest edge p still places: a lens or a replaced distribution can take one away. */
export function repair(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.anchor = repairAnchor(tree.raw, lens.p, lens.q, cam) as Edge;
}

/** The origin's interval edges in view heights. */
export function extent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  return measureExtent(tree.raw, lens.p, lens.q, cam) as { top: number; bottom: number };
}

export function reanchor(cam: Camera, frame: Frame): void {
  const position = frame.raw.reanchor() as { anchor: Edge; a: number } | null;
  if (position !== null) {
    cam.anchor = position.anchor;
    cam.a = position.a;
  }
}

export function clamp(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.a = clampPosition(tree.raw, lens.p, lens.q, cam);
}

/** The child under a screen point, however thin, found in the frame drawn with ``cam``. */
export function pick(tree: IntervalTree, lens: Lens, frame: Frame, cam: Camera, view: Viewport, x: number, y: number): Selection | null {
  return frame.raw.pick(tree.raw, lens.p, lens.q, cam, view, x, y) as Selection | null;
}
