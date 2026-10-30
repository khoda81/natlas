// Anchored integer-distribution geometry and best-effort picking, implemented
// by the shared Rust core. These adapters keep camera mutation at the UI edge.

import {
  clampPosition,
  layout as layoutFrame,
  normalizeBoundary,
  place as placeNode,
  repairBoundary,
  rootExtent as measureRootExtent,
  type LayoutFrame,
} from "./wasm";
import type { IntervalTree, Label, Lens, NodeId } from "./tree";

export type { Lens } from "./tree";

export interface Boundary {
  node: NodeId;
  slot: number;
}

export interface Camera {
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
  label: Label | null;
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
  private unexpandedCache?: { node: NodeId; label: Label; lnMass: number; x1: number }[];
  private disposed = false;

  constructor(readonly raw: LayoutFrame) {
    const packed = raw.boxes();
    this.boxes = boxBuffers.pop() ?? [];
    for (let i = 0, box = 0; i < packed.length; i += 12, box++) {
      const b = this.boxes[box] ??= { node: 0, slot: 0, count: 0, kind: "node", label: null,
        p: 0, q: 0, lnMass: 0, x0: 0, x1: 0, y0: 0, y1: 0 };
      b.node = packed[i]!; b.slot = packed[i + 1]!; b.count = packed[i + 2]!;
      b.kind = KINDS[packed[i + 3]!]!; b.label = packed[i + 4] === -1 ? null : packed[i + 4]!;
      b.p = packed[i + 5]!; b.q = packed[i + 6]!; b.lnMass = packed[i + 7]!;
      b.x0 = packed[i + 8]!; b.x1 = packed[i + 9]!; b.y0 = packed[i + 10]!; b.y1 = packed[i + 11]!;
    }
    this.boxes.length = packed.length / 12;
    this.cover = raw.cover() as Frame["cover"];
  }

  get walked(): { node: NodeId; lnMass: number; x: number }[] {
    return this.walkedCache ??= this.raw.walked();
  }

  get unexpanded(): { node: NodeId; label: Label; lnMass: number; x1: number }[] {
    return this.unexpandedCache ??= this.raw.unexpanded();
  }

  // Diagnostic views are deliberately lazy: never export hidden vocabulary
  // entries to paint, hit-test, reanchor, or calculate discovery demand.
  get known(): Box[] { return this.raw.known(); }
  get boundaries(): { boundary: Boundary; y: number }[] { return this.raw.boundaries(); }
  get rootTop(): number { return this.raw.rootTop(); }
  get rootBottom(): number { return this.raw.rootBottom(); }

  toJSON() {
    return { boxes: this.boxes, known: this.known, walked: this.walked, unexpanded: this.unexpanded,
      boundaries: this.boundaries, rootTop: this.rootTop, rootBottom: this.rootBottom, cover: this.cover };
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

export function logView(cam: Camera): number {
  return Math.min(0, -cam.x);
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

export function normalize(tree: IntervalTree, lens: Lens, boundary: Boundary): Boundary {
  return normalizeBoundary(tree.raw, lens.p, lens.q, boundary) as Boundary;
}

export function repair(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.anchor = repairBoundary(tree.raw, lens.p, lens.q, cam) as Boundary;
}

export function place(tree: IntervalTree, lens: Lens, id: NodeId): { lnMass: number; x1: number } {
  return placeNode(tree.raw, lens.p, lens.q, id) as { lnMass: number; x1: number };
}

export function rootExtent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  return measureRootExtent(tree.raw, lens.p, lens.q, cam) as { top: number; bottom: number };
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
