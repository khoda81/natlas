// Anchored integer-distribution geometry and best-effort picking, implemented
// by the shared Rust core. These adapters keep camera mutation at the UI edge.

import {
  clampPosition,
  layout as layoutFrame,
  normalizeBoundary,
  place as placeNode,
  repairBoundary,
  extent as measureExtent,
  type LayoutFrame,
} from "./wasm";
import type { Branch, IntervalTree, Lens, NodeId } from "./tree";

export type { Lens } from "./tree";

export interface Boundary {
  node: NodeId;
  slot: number;
}

export interface Camera {
  /** The node the view is measured from, standing as the root does: x is
   * q-surprisal from its children's left edge, and the view spans e^-x of its
   * interval. It bounds the view; nothing above it is needed. */
  origin: NodeId;
  x: number;
  k: number;
  anchor: Boundary;
  a: number;
}

export interface Viewport {
  width: number;
  height: number;
}

export type BoxKind = "node" | "leaf" | "tail" | "merged";

export interface Box {
  node: NodeId;
  slot: number;
  count: number;
  kind: BoxKind;
  branch: Branch | null;
  p: number;
  q: number;
  lnMass: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

const KINDS: BoxKind[] = ["node", "leaf", "tail", "merged"];
const boxBuffers: Box[][] = [];

/** Rust owns the full frame. Only visible rectangles cross the Wasm boundary. */
export class Frame {
  readonly boxes: Box[];
  readonly cover: { node: NodeId; y: number } | null;
  private walkedCache?: { node: NodeId; lnMass: number; x: number }[];
  private unexpandedCache?: { node: NodeId; branch: Branch; lnMass: number; x1: number }[];
  private disposed = false;

  constructor(readonly raw: LayoutFrame) {
    const packed = raw.boxes();
    const nodes = raw.nodes();
    this.boxes = boxBuffers.pop() ?? [];
    for (let i = 0, box = 0; i < packed.length; i += 11, box++) {
      const b = this.boxes[box] ??= { node: 0n, slot: 0, count: 0, kind: "node", branch: null,
        p: 0, q: 0, lnMass: 0, x0: 0, x1: 0, y0: 0, y1: 0 };
      b.node = nodes[box]!; b.slot = packed[i]!; b.count = packed[i + 1]!;
      b.kind = KINDS[packed[i + 2]!]!; b.branch = packed[i + 3] === -1 ? null : packed[i + 3]!;
      b.p = packed[i + 4]!; b.q = packed[i + 5]!; b.lnMass = packed[i + 6]!;
      b.x0 = packed[i + 7]!; b.x1 = packed[i + 8]!; b.y0 = packed[i + 9]!; b.y1 = packed[i + 10]!;
    }
    this.boxes.length = packed.length / 11;
    this.cover = raw.cover() as Frame["cover"];
  }

  get walked(): { node: NodeId; lnMass: number; x: number }[] {
    return this.walkedCache ??= this.raw.walked();
  }

  get unexpanded(): { node: NodeId; branch: Branch; lnMass: number; x1: number }[] {
    return this.unexpandedCache ??= this.raw.unexpanded();
  }

  // Diagnostic views are deliberately lazy: never export hidden vocabulary
  // entries to paint, hit-test, reanchor, or calculate discovery demand.
  get known(): Box[] { return this.raw.known(); }
  get boundaries(): { boundary: Boundary; y: number }[] { return this.raw.boundaries(); }
  get originTop(): number { return this.raw.originTop(); }
  get originBottom(): number { return this.raw.originBottom(); }

  toJSON() {
    return { boxes: this.boxes, known: this.known, walked: this.walked, unexpanded: this.unexpanded,
      boundaries: this.boundaries, originTop: this.originTop, originBottom: this.originBottom, cover: this.cover };
  }

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

export function normalize(tree: IntervalTree, lens: Lens, origin: NodeId, boundary: Boundary): Boundary {
  return normalizeBoundary(tree.raw, lens.p, lens.q, origin, boundary) as Boundary;
}

export function repair(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.anchor = repairBoundary(tree.raw, lens.p, lens.q, cam) as Boundary;
}

/** `id`'s ln p mass and right edge in nats below `origin`. */
export function place(tree: IntervalTree, lens: Lens, origin: NodeId, id: NodeId): { lnMass: number; x1: number } {
  return placeNode(tree.raw, lens.p, lens.q, origin, id) as { lnMass: number; x1: number };
}

/** The origin's interval edges in view heights. */
export function extent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  return measureExtent(tree.raw, lens.p, lens.q, cam) as { top: number; bottom: number };
}

export function reanchor(cam: Camera, frame: Frame): void {
  const position = frame.raw.reanchor() as { anchor: Boundary; a: number } | null;
  if (position !== null) {
    cam.anchor = position.anchor;
    cam.a = position.a;
  }
}

export function clamp(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.a = clampPosition(tree.raw, lens.p, lens.q, cam);
}

export function pick(frame: Frame, cam: Camera, view: Viewport, x: number, y: number): Selection | null {
  return frame.raw.pick(cam.x, cam.k, view.width, view.height, x, y) as Selection | null;
}
