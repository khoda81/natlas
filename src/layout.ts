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

import type { IntervalTree, Label, Lens, NodeId, Slots, TreeNode } from "./tree";

export type { Lens } from "./tree";

/** The top edge of a slot; a slot at or past the node's slot count is its bottom. */
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
  /** The child's label; null for a tail or a merged run. */
  label: Label | null;
  /** Conditional weights, summed over a merged run; q may be an upper bound. */
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
  /** Nodes whose slots were laid out, with their ln p mass and left edge in nats. */
  walked: { node: NodeId; lnMass: number; x: number }[];
  /**
   * Children not expanded yet whose own children would start on screen, even
   * where their box itself ends at or left of the screen's left edge.
   */
  unexpanded: { node: NodeId; label: Label; lnMass: number; x1: number }[];
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

/** The slot of ``node`` among its parent's slots under ``lens``. */
function slotIn(tree: IntervalTree, lens: Lens, node: TreeNode): number {
  const slot = tree.node(node.parent!).slots(lens).slotOf(node.label);
  if (slot === undefined) throw new Error(`node ${node.id} is not listed under measure ${lens.p}`);
  return slot;
}

/** Canonical spelling of a boundary: a node's bottom is its next sibling's top. */
export function normalize(tree: IntervalTree, lens: Lens, boundary: Boundary): Boundary {
  let { node, slot } = boundary;
  for (let current = tree.node(node); slot >= current.slots(lens).count && current.parent !== null; ) {
    slot = slotIn(tree, lens, current) + 1;
    node = current.parent;
    current = tree.node(node);
  }
  return node === 0 && slot >= tree.root.slots(lens).count ? { node: 0, slot: Number.MAX_SAFE_INTEGER } : { node, slot };
}

/**
 * Move the anchor up to its deepest ancestor that p lists, so a new lens or a
 * shallower listing never leaves it unreachable.
 */
export function repair(tree: IntervalTree, lens: Lens, cam: Camera): void {
  const chain = tree.chain(cam.anchor.node);
  for (let i = 1; i < chain.length; i++) {
    if (chain[i - 1]!.slots(lens).slotOf(chain[i]!.label) === undefined) {
      cam.anchor = { node: chain[i - 1]!.id, slot: 0 };
      return;
    }
  }
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
  const lambda = logView(cam);
  const right = cam.x + view.width / cam.k;
  const mergeUnits = mergePx / view.height;
  const boxes: Box[] = [];
  const walked: Frame["walked"] = [];
  const unexpanded: Frame["unexpanded"] = [];
  const boundaries: Frame["boundaries"] = [];

  const boundary = (node: TreeNode, slot: number, y: number) => {
    if (y >= 0 && y <= 1) boundaries.push({ boundary: normalize(tree, lens, { node: node.id, slot }), y });
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
      label: null,
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
    slots: Slots,
    slot: number,
    y0: number,
    y1: number,
    scale: number,
    x: number,
    lnMass: number,
    downward: boolean,
  ) => {
    const pw = slots.p[slot]!;
    const qw = slots.q[slot]!;
    const x1 = x + nats(qw);
    const label = slots.label(slot);
    const id = label === null ? undefined : node.children.get(label);
    const child = id === undefined ? undefined : tree.node(id);
    const kind: BoxKind = label === null ? "tail" : child?.expanded(lens.p) ? "node" : "leaf";
    const childMass = lnMass + Math.log(pw);
    push({ node: node.id, slot, count: 1, kind, label, p: pw, q: qw, lnMass: childMass, x0: x, x1, y0, y1 });
    boundary(node, slot, y0);
    boundary(node, slot + 1, y1);
    if (x1 >= right) return;
    if (kind === "leaf") unexpanded.push({ node: node.id, label: label!, lnMass: childMass, x1 });
    if (kind !== "node") return;
    if (downward) down(child!, 0, y0, scaled(pw, scale), x1, childMass);
    else up(child!, child!.slots(lens).count - 1, y1, scaled(pw, scale), x1, childMass);
  };

  // Slots from ``start`` onward, whose top edge is at ``y``.
  const down = (node: TreeNode, start: number, y: number, scale: number, x: number, lnMass: number) => {
    if (x >= right) return;
    walked.push({ node: node.id, lnMass, x });
    const slots = node.slots(lens);
    let run: Run | null = null;
    for (let slot = start; slot < slots.count && y < 1; slot++) {
      const pw = slots.p[slot]!;
      const y1 = y + scaled(pw, scale);
      if (y1 > 0) {
        if (y1 - y < mergeUnits) {
          run ??= { node, first: slot, count: 0, p: 0, q: 0, y0: y, y1: y, x, lnMass };
          run.count++;
          run.p += pw;
          run.q += slots.q[slot]!;
          run.y1 = y1;
          if (run.y1 - run.y0 >= mergeUnits) {
            flush(run);
            run = null;
          }
        } else {
          flush(run);
          run = null;
          slotBox(node, slots, slot, y, y1, scale, x, lnMass, true);
        }
      }
      y = y1;
    }
    flush(run);
  };

  // Slots from ``start`` back to the first, whose bottom edge is at ``y``.
  const up = (node: TreeNode, start: number, y: number, scale: number, x: number, lnMass: number) => {
    if (x >= right) return;
    walked.push({ node: node.id, lnMass, x });
    const slots = node.slots(lens);
    let run: Run | null = null;
    for (let slot = start; slot >= 0 && y > 0; slot--) {
      const pw = slots.p[slot]!;
      const y0 = y - scaled(pw, scale);
      if (y0 < 1) {
        if (y - y0 < mergeUnits) {
          run ??= { node, first: slot, count: 0, p: 0, q: 0, y0: y, y1: y, x, lnMass };
          run.first = slot;
          run.count++;
          run.p += pw;
          run.q += slots.q[slot]!;
          run.y0 = y0;
          if (run.y1 - run.y0 >= mergeUnits) {
            flush(run);
            run = null;
          }
        } else {
          flush(run);
          run = null;
          slotBox(node, slots, slot, y0, y, scale, x, lnMass, false);
        }
      }
      y = y0;
    }
    flush(run);
  };

  const { chain, slots: chainSlots, lnMasses } = walk(tree, lens, cam);
  const depth = chain.length - 1;
  const xs = new Float64Array(chain.length);
  for (let i = 1; i <= depth; i++) {
    xs[i] = xs[i - 1]! + nats(chain[i - 1]!.slots(lens).q[chainSlots[i]!]!);
  }

  const anchor = chain[depth]!;
  const anchorSlots = anchor.slots(lens);
  const slot = Math.min(cam.anchor.slot, anchorSlots.count);
  let scale = Math.exp(lnMasses[depth]! - lambda);
  let top = cam.a - scaled(anchorSlots.prefix(slot), scale);
  let bottom = cam.a + scaled(anchorSlots.suffix(slot), scale);
  boundary(anchor, slot, cam.a);
  down(anchor, slot, cam.a, scale, xs[depth]!, lnMasses[depth]!);
  up(anchor, slot - 1, cam.a, scale, xs[depth]!, lnMasses[depth]!);

  for (let i = depth - 1; i >= 0; i--) {
    const owner = chain[i]!;
    const slots = owner.slots(lens);
    const slot = chainSlots[i + 1]!;
    const x = xs[i]!;
    scale = Math.exp(lnMasses[i]! - lambda);
    push({
      node: owner.id,
      slot,
      count: 1,
      kind: "node",
      label: chain[i + 1]!.label,
      p: slots.p[slot]!,
      q: slots.q[slot]!,
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
    top -= scaled(slots.prefix(slot), scale);
    bottom += scaled(slots.suffix(slot + 1), scale);
    // Once this node covers the view and its own box lies left of it, every
    // remaining ancestor box and sibling is off screen.
    if (top <= 0 && bottom >= 1 && x <= cam.x) break;
  }

  return { boxes, walked, unexpanded, boundaries, rootTop: top, rootBottom: bottom };
}

/** The anchor chain, each node's slot in its parent, and each node's ln p mass. */
function walk(tree: IntervalTree, lens: Lens, cam: Camera) {
  const chain = tree.chain(cam.anchor.node);
  const slots = new Int32Array(chain.length);
  const lnMasses = new Float64Array(chain.length);
  for (let i = 1; i < chain.length; i++) {
    slots[i] = slotIn(tree, lens, chain[i]!);
    lnMasses[i] = lnMasses[i - 1]! + Math.log(chain[i - 1]!.slots(lens).p[slots[i]!]!);
  }
  return { chain, slots, lnMasses };
}

/** ``id``'s ln p mass and the right edge of its box in nats, -Σ ln q along its path. */
export function place(tree: IntervalTree, lens: Lens, id: NodeId): { lnMass: number; x1: number } {
  let lnMass = 0;
  let x1 = 0;
  for (const node of tree.chain(id).slice(1)) {
    const slots = tree.node(node.parent!).slots(lens);
    const slot = slotIn(tree, lens, node);
    lnMass += Math.log(slots.p[slot]!);
    x1 += nats(slots.q[slot]!);
  }
  return { lnMass, x1 };
}

/** Root interval edges in view heights, walking the whole anchor chain. */
export function rootExtent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  const lambda = logView(cam);
  const { chain, slots: chainSlots, lnMasses } = walk(tree, lens, cam);
  const depth = chain.length - 1;

  const anchor = chain[depth]!.slots(lens);
  const slot = Math.min(cam.anchor.slot, anchor.count);
  let scale = Math.exp(lnMasses[depth]! - lambda);
  let top = cam.a - scaled(anchor.prefix(slot), scale);
  let bottom = cam.a + scaled(anchor.suffix(slot), scale);
  for (let i = depth - 1; i >= 0; i--) {
    const owner = chain[i]!.slots(lens);
    const slot = chainSlots[i + 1]!;
    scale = Math.exp(lnMasses[i]! - lambda);
    top -= scaled(owner.prefix(slot), scale);
    bottom += scaled(owner.suffix(slot + 1), scale);
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
