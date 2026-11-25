// Anchored integer-distribution geometry, picking, and what a view lacks,
// implemented by the shared Rust core. These adapters keep camera mutation
// at the UI edge.

import {
  aim as aimAnchor,
  clampPosition,
  extent as measureExtent,
  layout as layoutFrame,
  project as projectFrame,
  reach as reachOf,
  settle as settleAnchor,
  LayoutFrame,
  view as viewOf,
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
export type BoxKind = "node" | "leaf" | "gap";

export interface Box {
  node: NodeId;
  /** The first rank the box covers. */
  rank: Rank;
  /** Children covered; 0 for a gap of unknown size. */
  count: number;
  kind: BoxKind;
  branch: Branch | null;
  /** Conditional probabilities. */
  p: number;
  /** NaN while unknown. */
  q: number;
  lnMass: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** What a camera looks at, as a session's view: an interval of ``node``'s p
 * mass, its top as digits in base 256 of the fraction of the mass above it
 * and ``lnHeight`` ln of its height, worth ``rows`` pixel rows, seen ``right``
 * nats of −ln q past where the node's children start. */
export interface View {
  node: NodeId;
  top: Uint8Array;
  lnHeight: number;
  rows: number;
  right: number;
}

/** Branches down from a node. */
export interface Path {
  node: NodeId;
  branches: Branch[];
}

const KINDS: BoxKind[] = ["node", "leaf", "gap"];

/** Physical rows of one node's children drawn alike, from ``y0`` to ``y1``
 * view heights: from ``x0``, where the children start, every pixel wholly
 * inside them up to ``fill`` is filled, and every pixel from there to
 * ``reach``, where the widest ends, holds an edge of them and is outline, as
 * is the column at ``x0``. Where ``fill`` is ``x0``, an edge of the node's
 * children, or a hole in them, falls in every row: outline throughout. */
export interface Span {
  node: NodeId;
  y0: number;
  y1: number;
  x0: number;
  fill: number;
  reach: number;
}

/** A known child of a laid-out node: its branch and its edges in view heights. */
export interface Known {
  branch: Branch;
  y0: number;
  y1: number;
}
const boxBuffers: Box[][] = [];

/** Rust owns the full frame. Only visible rectangles cross the Wasm boundary. */
export class Frame {
  /** Each child and gap a physical row or more tall: labelled and picked. */
  readonly boxes: Box[];
  /** What is painted: every node's children as spans of rows. */
  readonly spans: Span[];
  /** The visible box of each child, keyed by its node id. */
  readonly parentBoxes = new Map<NodeId, Box>();
  private disposed = false;

  constructor(readonly raw: LayoutFrame) {
    const packed = raw.boxes();
    const nodes = raw.boxNodes();
    this.boxes = boxBuffers.pop() ?? [];
    for (let i = 0, box = 0; i < packed.length; i += 11, box++) {
      const b = this.boxes[box] ??= { node: 0n, rank: 0, count: 0, kind: "node", branch: null,
        p: 0, q: 0, lnMass: 0, x0: 0, x1: 0, y0: 0, y1: 0 };
      b.node = nodes[2 * box]!; b.rank = packed[i]!; b.count = packed[i + 1]!;
      b.kind = KINDS[packed[i + 2]!]!; b.branch = packed[i + 3] === -1 ? null : packed[i + 3]!;
      b.p = packed[i + 4]!; b.q = packed[i + 5]!; b.lnMass = packed[i + 6]!;
      b.x0 = packed[i + 7]!; b.x1 = packed[i + 8]!; b.y0 = packed[i + 9]!; b.y1 = packed[i + 10]!;
      if (b.branch !== null) this.parentBoxes.set(nodes[2 * box + 1]!, b);
    }
    this.boxes.length = packed.length / 11;
    const spans = raw.spans();
    const spanNodes = raw.spanNodes();
    this.spans = [];
    for (let i = 0, span = 0; i < spans.length; i += 5, span++) {
      this.spans.push({ node: spanNodes[span]!, y0: spans[i]!, y1: spans[i + 1]!, x0: spans[i + 2]!, fill: spans[i + 3]!, reach: spans[i + 4]! });
    }
  }

  /** The known children of ``node`` that reach into the view, top to
   * bottom, however thin: even those drawn only in a span. */
  known(tree: IntervalTree, lens: Lens, node: NodeId): Known[] {
    const packed = this.raw.known(tree.raw, lens.p, lens.q, node);
    const known: Known[] = [];
    for (let i = 0; i < packed.length; i += 3) known.push({ branch: packed[i]!, y0: packed[i + 1]!, y1: packed[i + 2]! });
    return known;
  }

  get originTop(): number { return this.raw.originTop(); }
  get originBottom(): number { return this.raw.originBottom(); }

  aaSpans(
    tree: IntervalTree,
    lens: Lens,
    cam: Camera,
    widthPx: number,
    heightPx: number,
    dpr: number,
  ): Int32Array {
    return this.raw.aaSpans(tree.raw, lens.p, lens.q, cam.x, cam.k, widthPx, heightPx, dpr);
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
  /** Child ids along the picked path through the laid-out part of the frame. */
  ancestors: readonly NodeId[];
}

export interface LayoutOptions {
  /** The height of a physical row in CSS pixels, 1 / DPR: a child thinner
   * than one is drawn in the rows it falls in. */
  rowPx: number;
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
  return new Frame(layoutFrame(tree.raw, lens.p, lens.q, cam, view, options.rowPx ?? 1));
}

/** Settle and clamp the camera, returning a frame and the demand it describes. */
export function project(
  tree: IntervalTree,
  lens: Lens,
  cam: Camera,
  viewport: Viewport,
  pixelRatio: number,
): { frame: Frame; demand: View } {
  const raw = projectFrame(tree.raw, lens.p, lens.q, cam, viewport, pixelRatio);
  const { anchor, a, demand } = raw.takeProjection() as { anchor: Anchor; a: number; demand: View };
  cam.anchor = anchor;
  tree.descend(cam.origin, cam.anchor.path);
  cam.a = a;
  return { frame: new Frame(raw), demand };
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

/** What the camera looks at, ``rows`` pixel rows of it, as a session's view:
 * told exactly as a frame places it. */
export function view(tree: IntervalTree, lens: Lens, cam: Camera, viewport: Viewport, rows: number): View {
  return viewOf(tree.raw, lens.p, lens.q, cam, viewport, rows) as View;
}

/** The interval edges in view heights of the node ``depth`` steps down the
 * camera's path from its origin, as far as p places that path. */
/** Where the node ``path`` leads to from ``origin`` sits, as layout places
 * it: ln of its p mass below the origin, and where its children start, in
 * nats right of the origin's, as far as p places the path: ``placed`` of its
 * branches. */
export interface Reach {
  lnMass: number;
  x: number;
  placed: number;
}

export function reach(tree: IntervalTree, lens: Lens, origin: NodeId, path: readonly Branch[]): Reach {
  return reachOf(tree.raw, lens.p, lens.q, origin, Uint32Array.from(path)) as Reach;
}

export function extent(tree: IntervalTree, lens: Lens, cam: Camera, depth: number): { top: number; bottom: number } {
  return measureExtent(tree.raw, lens.p, lens.q, cam, depth) as { top: number; bottom: number };
}

export function clamp(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.a = clampPosition(tree.raw, lens.p, lens.q, cam);
}

/** The child a pixel ``pixel`` CSS pixels square, with its top-left corner at
 * ``x``, ``y``, shows, however thin, found in the frame drawn with ``cam``. */
export function pick(tree: IntervalTree, lens: Lens, frame: Frame, cam: Camera, view: Viewport, x: number, y: number, pixel: number): Selection | null {
  return frame.raw.pick(tree.raw, lens.p, lens.q, cam, view, x, y, pixel) as Selection | null;
}

export function linearPalette(surfaceRgb: number, strokeRgb: number): string[] {
  const rgbs = LayoutFrame.aaPalette(surfaceRgb, strokeRgb);
  return Array.from(rgbs, (c) => `#${c.toString(16).padStart(6, "0")}`);
}
