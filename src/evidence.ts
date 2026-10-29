/** Probability-space geometry certificates, relative to a nearby ID boundary.
 * All distances are sums of disjoint, nonnegative masses. Never subtract two
 * absolute floating-point CDF coordinates, including at deep zoom near 1.
 */
import type { MassBounds } from "./radix";
import type { IntervalTree, Lens, NodeId, TreeNode } from "./tree";

export interface Interval { lower: number; upper: number }
export interface IDBoundary { node: NodeId; id: number }
export interface GeometryBounds {
  top: Interval;
  bottom: Interval;
  left: Interval;
  right: Interval;
  /** Log probability of the full path or full ID range. */
  mass: MassBounds;
}

const ZERO: MassBounds = { lower: -Infinity, upper: -Infinity };

function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const high = Math.max(a, b);
  return high + Math.log1p(Math.exp(Math.min(a, b) - high));
}

/** One layout's memoized evidence. No independently rounded per-token sums. */
export class GeometryEvidence {
  private readonly masses = new Map<NodeId, MassBounds>([[0, { lower: 0, upper: 0 }]]);
  private readonly widths = new Map<NodeId, Interval>([[0, { lower: 0, upper: 0 }]]);
  private readonly ranges = new Map<string, MassBounds>();
  private readonly chains = new Map<NodeId, TreeNode[]>();

  constructor(private readonly tree: IntervalTree, private readonly lens: Lens) {}

  private p(node: NodeId) {
    const state = this.tree.node(node).listing(this.lens.p)?.radix;
    if (!state?.ready) throw Error("CDF geometry requires the parent distribution");
    return state;
  }

  mass(node: NodeId): MassBounds {
    const cached = this.masses.get(node);
    if (cached) return cached;
    const child = this.tree.node(node);
    const parent = this.mass(child.parent!);
    const local = this.p(child.parent!).bounds(child.label);
    const bounds = { lower: parent.lower + local.lower, upper: parent.upper + local.upper };
    this.masses.set(node, bounds);
    return bounds;
  }

  x(node: NodeId): Interval {
    const cached = this.widths.get(node);
    if (cached) return cached;
    const child = this.tree.node(node);
    const parent = this.x(child.parent!);
    const q = this.tree.node(child.parent!).listing(this.lens.q)?.radix?.bounds(child.label) ?? { lower: -Infinity, upper: 0 };
    const bounds = { lower: parent.lower - q.upper, upper: parent.upper - q.lower };
    this.widths.set(node, bounds);
    return bounds;
  }

  private range(node: NodeId, first: number, end: number): MassBounds {
    if (first === end) return ZERO;
    const key = `${node}:${first}:${end}`;
    const cached = this.ranges.get(key);
    if (cached) return cached;
    const mass = this.mass(node), local = this.p(node).rangeBounds(first, end);
    const bounds = { lower: mass.lower + local.lower, upper: mass.upper + local.upper };
    this.ranges.set(key, bounds);
    return bounds;
  }

  private sum(terms: readonly MassBounds[]): MassBounds {
    return terms.reduce((sum, term) => ({ lower: logAdd(sum.lower, term.lower), upper: logAdd(sum.upper, term.upper) }), ZERO);
  }

  /** Distance from one ancestor branch's beginning/end to a nested boundary. */
  private remainder(chain: TreeNode[], start: number, id: number, tail: boolean): MassBounds {
    const terms: MassBounds[] = [];
    for (let i = start; i < chain.length; i++) {
      const node = chain[i]!;
      const endpoint = i + 1 < chain.length ? chain[i + 1]!.label + (tail ? 1 : 0) : id;
      const end = this.p(node.id).vocabulary!;
      terms.push(tail ? this.range(node.id, endpoint, end) : this.range(node.id, 0, endpoint));
    }
    return this.sum(terms);
  }

  /** Signed distance from A to B. The sign follows fixed ID order. */
  distance(a: IDBoundary, b: IDBoundary): { sign: number; mass: MassBounds } {
    if (a.node === b.node) return {
      sign: Math.sign(b.id - a.id),
      mass: this.range(a.node, Math.min(a.id, b.id), Math.max(a.id, b.id)),
    };
    const chain = (node: NodeId) => {
      let value = this.chains.get(node);
      if (!value) this.chains.set(node, value = this.tree.chain(node));
      return value;
    };
    const ac = chain(a.node), bc = chain(b.node);
    let shared = 0;
    while (shared + 1 < Math.min(ac.length, bc.length) && ac[shared + 1]!.id === bc[shared + 1]!.id) shared++;
    const common = ac[shared]!;
    if (shared === ac.length - 1) {
      const child = bc[shared + 1]!.label;
      const forward = a.id <= child;
      return { sign: forward ? 1 : -1, mass: this.sum([
        forward ? this.range(common.id, a.id, child) : this.range(common.id, child + 1, a.id),
        this.remainder(bc, shared + 1, b.id, !forward),
      ]) };
    }
    if (shared === bc.length - 1 || ac[shared + 1]!.label > bc[shared + 1]!.label) {
      const reverse = this.distance(b, a);
      return { sign: -reverse.sign, mass: reverse.mass };
    }
    return { sign: 1, mass: this.sum([
      this.remainder(ac, shared + 1, a.id, true),
      this.range(common.id, ac[shared + 1]!.label + 1, bc[shared + 1]!.label),
      this.remainder(bc, shared + 1, b.id, false),
    ]) };
  }

  project(anchor: IDBoundary, target: IDBoundary, y: number, logView: number): Interval {
    const { sign, mass } = this.distance(anchor, target);
    if (!sign) return { lower: y, upper: y };
    const lower = Math.exp(mass.lower - logView), upper = Math.exp(mass.upper - logView);
    return sign > 0 ? { lower: y + lower, upper: y + upper } : { lower: y - upper, upper: y - lower };
  }

  box(node: NodeId, first: number, end: number, anchor: IDBoundary, y: number, logView: number, token: boolean): GeometryBounds {
    const left = this.x(node);
    const q = token ? this.tree.node(node).listing(this.lens.q)?.radix?.bounds(first) ?? { lower: -Infinity, upper: 0 } : { lower: -Infinity, upper: 0 };
    return {
      top: this.project(anchor, { node, id: first }, y, logView),
      bottom: this.project(anchor, { node, id: end }, y, logView),
      left,
      right: token ? { lower: left.lower - q.upper, upper: left.upper - q.lower } : { lower: left.lower, upper: Infinity },
      mass: this.range(node, first, end),
    };
  }
}
