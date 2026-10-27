// Geometry of the nested interval tree under a camera.
//
// A slot of node N spans the part of N's interval proportional to its p
// weight, vertically, and -ln q nats, horizontally, starting where N's own box
// ends. With p = q the area of N's slot boxes is N's mass times its entropy;
// otherwise it is N's mass times the cross-entropy of q under p.
//
// Depth is unbounded, so absolute vertical coordinates in [0, 1] would run out
// of float precision a few dozen nats in. The camera is instead anchored to a
// slot boundary near the view. Layout walks from that boundary up its
// ancestor chain and outward from every edge it reaches, so each position is
// a same-sign sum of masses from a nearby edge and never a difference of two
// large coordinates. Its precision is bounded by the anchor's distance from
// the view, which re-anchoring after every frame keeps below one view height.

import type { IntervalTree, NodeId, TreeNode } from "./tree";

/** Measure indices sizing slot heights (p) and slot widths (-ln q). */
export interface Lens {
  p: number;
  q: number;
}

/** The top edge of a slot; a slot at or past ``node.slots`` is the node's bottom. */
export interface Boundary {
  node: NodeId;
  slot: number;
}

/**
 * ``x`` is the depth in nats at the left screen edge and ``k`` the horizontal
 * scale in px per nat. The view height in root-mass units is coupled to
 * depth, e^-max(x, 0), so scrolling deeper zooms vertically at the rate that
 * keeps a node near the left edge at constant screen height. ``a`` places the
 * anchor boundary, in view heights below the top of the view.
 */
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
  /** First slot; a merged box covers ``count`` consecutive slots. */
  slot: number;
  count: number;
  kind: BoxKind;
  /** Conditional weights, summed over a merged run. */
  p: number;
  q: number;
  /** Log of the box's absolute p mass. */
  lnMass: number;
  /** Horizontal extent in nats. */
  x0: number;
  x1: number;
  /** Vertical extent in view heights from the top of the view. */
  y0: number;
  y1: number;
}

export interface Frame {
  boxes: Box[];
  /** Slot boundaries inside the view, re-anchoring candidates. */
  boundaries: { boundary: Boundary; y: number }[];
  /** Root interval edges in view heights; bounds, once an ancestor covers the view. */
  rootTop: number;
  rootBottom: number;
}

export interface LayoutOptions {
  /** Consecutive slots thinner than this merge into one coarse box. */
  mergePx: number;
}

const DEFAULTS: LayoutOptions = { mergePx: 1 };

export function logView(cam: Camera): number {
  return Math.min(0, -cam.x);
}

/** w * s, where a zero weight stays zero even at an infinite scale. */
function scaled(weight: number, scale: number): number {
  return weight === 0 ? 0 : weight * scale;
}

function nats(weight: number): number {
  return -Math.log(weight);
}

/** Canonical spelling of a boundary: a node's bottom is its next sibling's top. */
export function normalize(tree: IntervalTree, boundary: Boundary): Boundary {
  let { node, slot } = boundary;
  for (let current = tree.node(node); slot >= current.slots && current.parent !== null; ) {
    slot = current.slot + 1;
    node = current.parent;
    current = tree.node(node);
  }
  return node === 0 && slot >= tree.node(0).slots ? { node: 0, slot: Number.MAX_SAFE_INTEGER } : { node, slot };
}

interface Run {
  node: TreeNode;
  first: number;
  count: number;
  p: number;
  q: number;
  y0: number;
  y1: number;
  x: number;
  lnMass: number;
}

export function layout(
  tree: IntervalTree,
  lens: Lens,
  cam: Camera,
  view: Viewport,
  options: Partial<LayoutOptions> = {},
): Frame {
  const { mergePx } = { ...DEFAULTS, ...options };
  const { p, q } = lens;
  const lambda = logView(cam);
  const right = cam.x + view.width / cam.k;
  const mergeUnits = mergePx / view.height;
  const boxes: Box[] = [];
  const boundaries: Frame["boundaries"] = [];

  const boundary = (node: TreeNode, slot: number, y: number) => {
    if (y >= 0 && y <= 1) boundaries.push({ boundary: normalize(tree, { node: node.id, slot }), y });
  };

  const push = (box: Box) => {
    if (box.x1 > cam.x && box.x0 < right && box.y1 > 0 && box.y0 < 1) boxes.push(box);
  };

  const flush = (run: Run | null) => {
    if (run === null) return;
    push({
      node: run.node.id,
      slot: run.first,
      count: run.count,
      kind: "merged",
      p: run.p,
      q: run.q,
      lnMass: run.lnMass + Math.log(run.p),
      x0: run.x,
      x1: run.x + nats(run.q),
      y0: run.y0,
      y1: run.y1,
    });
  };

  // One visible slot box; recurse into its node, if expanded, from its edge
  // nearest the walk's origin.
  const slotBox = (
    node: TreeNode,
    slot: number,
    y0: number,
    y1: number,
    scale: number,
    x: number,
    lnMass: number,
    downward: boolean,
  ) => {
    const pw = node.weight(p, slot);
    const qw = node.weight(q, slot);
    const x1 = x + nats(qw);
    const child = node.children.get(slot);
    const kind: BoxKind = node.isTail(slot) ? "tail" : child === undefined ? "leaf" : "node";
    const childMass = lnMass + Math.log(pw);
    push({ node: node.id, slot, count: 1, kind, p: pw, q: qw, lnMass: childMass, x0: x, x1, y0, y1 });
    boundary(node, slot, y0);
    boundary(node, slot + 1, y1);
    if (child === undefined || x1 >= right) return;
    const next = tree.node(child);
    if (downward) down(next, 0, y0, scaled(pw, scale), x1, childMass);
    else up(next, next.slots - 1, y1, scaled(pw, scale), x1, childMass);
  };

  // Slots from ``start`` onward, whose top edge is at ``y``.
  const down = (node: TreeNode, start: number, y: number, scale: number, x: number, lnMass: number) => {
    if (x >= right) return;
    let run: Run | null = null;
    for (let slot = start; slot < node.slots && y < 1; slot++) {
      const pw = node.weight(p, slot);
      const y1 = y + scaled(pw, scale);
      if (y1 > 0) {
        if (y1 - y < mergeUnits) {
          run ??= { node, first: slot, count: 0, p: 0, q: 0, y0: y, y1: y, x, lnMass };
          run.count++;
          run.p += pw;
          run.q += node.weight(q, slot);
          run.y1 = y1;
          if (run.y1 - run.y0 >= mergeUnits) {
            flush(run);
            run = null;
          }
        } else {
          flush(run);
          run = null;
          slotBox(node, slot, y, y1, scale, x, lnMass, true);
        }
      }
      y = y1;
    }
    flush(run);
  };

  // Slots from ``start`` back to the first, whose bottom edge is at ``y``.
  const up = (node: TreeNode, start: number, y: number, scale: number, x: number, lnMass: number) => {
    if (x >= right) return;
    let run: Run | null = null;
    for (let slot = start; slot >= 0 && y > 0; slot--) {
      const pw = node.weight(p, slot);
      const y0 = y - scaled(pw, scale);
      if (y0 < 1) {
        if (y - y0 < mergeUnits) {
          run ??= { node, first: slot, count: 0, p: 0, q: 0, y0: y, y1: y, x, lnMass };
          run.first = slot;
          run.count++;
          run.p += pw;
          run.q += node.weight(q, slot);
          run.y0 = y0;
          if (run.y1 - run.y0 >= mergeUnits) {
            flush(run);
            run = null;
          }
        } else {
          flush(run);
          run = null;
          slotBox(node, slot, y0, y, scale, x, lnMass, false);
        }
      }
      y = y0;
    }
    flush(run);
  };

  const chain = tree.chain(cam.anchor.node);
  const depth = chain.length - 1;
  const xs = new Float64Array(chain.length);
  const lnMasses = new Float64Array(chain.length);
  for (let i = 1; i <= depth; i++) {
    const owner = chain[i - 1]!;
    const slot = chain[i]!.slot;
    xs[i] = xs[i - 1]! + nats(owner.weight(q, slot));
    lnMasses[i] = lnMasses[i - 1]! + Math.log(owner.weight(p, slot));
  }

  const anchor = chain[depth]!;
  const slot = Math.min(cam.anchor.slot, anchor.slots);
  let scale = Math.exp(lnMasses[depth]! - lambda);
  let top = cam.a - scaled(anchor.prefix(p, slot), scale);
  let bottom = cam.a + scaled(anchor.suffix(p, slot), scale);
  boundary(anchor, slot, cam.a);
  down(anchor, slot, cam.a, scale, xs[depth]!, lnMasses[depth]!);
  up(anchor, slot - 1, cam.a, scale, xs[depth]!, lnMasses[depth]!);

  for (let i = depth - 1; i >= 0; i--) {
    const owner = chain[i]!;
    const slot = chain[i + 1]!.slot;
    const x = xs[i]!;
    scale = Math.exp(lnMasses[i]! - lambda);
    push({
      node: owner.id,
      slot,
      count: 1,
      kind: "node",
      p: owner.weight(p, slot),
      q: owner.weight(q, slot),
      lnMass: lnMasses[i + 1]!,
      x0: x,
      x1: xs[i + 1]!,
      y0: top,
      y1: bottom,
    });
    boundary(owner, slot, top);
    boundary(owner, slot + 1, bottom);
    down(owner, slot + 1, bottom, scale, x, lnMasses[i]!);
    up(owner, slot - 1, top, scale, x, lnMasses[i]!);
    top -= scaled(owner.prefix(p, slot), scale);
    bottom += scaled(owner.suffix(p, slot + 1), scale);
    // Once this node covers the view and its own box lies left of it, every
    // remaining ancestor box and sibling is off screen.
    if (top <= 0 && bottom >= 1 && x <= cam.x) break;
  }

  return { boxes, boundaries, rootTop: top, rootBottom: bottom };
}

/** Root interval edges in view heights, walking the whole anchor chain. */
export function rootExtent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  const { p } = lens;
  const lambda = logView(cam);
  const chain = tree.chain(cam.anchor.node);
  const depth = chain.length - 1;
  const lnMasses = new Float64Array(chain.length);
  for (let i = 1; i <= depth; i++) {
    lnMasses[i] = lnMasses[i - 1]! + Math.log(chain[i - 1]!.weight(p, chain[i]!.slot));
  }

  const anchor = chain[depth]!;
  const slot = Math.min(cam.anchor.slot, anchor.slots);
  let scale = Math.exp(lnMasses[depth]! - lambda);
  let top = cam.a - scaled(anchor.prefix(p, slot), scale);
  let bottom = cam.a + scaled(anchor.suffix(p, slot), scale);
  for (let i = depth - 1; i >= 0; i--) {
    const owner = chain[i]!;
    const slot = chain[i + 1]!.slot;
    scale = Math.exp(lnMasses[i]! - lambda);
    top -= scaled(owner.prefix(p, slot), scale);
    bottom += scaled(owner.suffix(p, slot + 1), scale);
  }
  return { top, bottom };
}

/** Re-anchor at the in-view boundary nearest the view's center. */
export function reanchor(cam: Camera, frame: Frame): void {
  let best: Frame["boundaries"][number] | undefined;
  for (const candidate of frame.boundaries) {
    if (best === undefined || Math.abs(candidate.y - 0.5) < Math.abs(best.y - 0.5)) best = candidate;
  }
  if (best !== undefined) {
    cam.anchor = best.boundary;
    cam.a = best.y;
  }
}

/** Keep the view inside the root interval. */
export function clamp(tree: IntervalTree, lens: Lens, cam: Camera): void {
  const { top, bottom } = rootExtent(tree, lens, cam);
  if (top > 0) cam.a -= top;
  else if (bottom < 1) cam.a += 1 - bottom;
}
