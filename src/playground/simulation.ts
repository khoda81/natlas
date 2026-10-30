import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { SummarizeRequestSchema, SummarizeResponseSchema, type Summary } from "../gen/natlas/partition/v1/partition_pb";
import { bits, type Filter } from "./server";

export type Order = "high" | "low";
export type Unit = "bytes" | "requests";
export interface Budgets { p: number; q: number; unit: Unit; order: Order; fanout: number; clip: number }
export interface Cell extends Filter {
  count: number; lower: number; upper: number; a: number; b: number;
  received: boolean; children?: Cell[];
}
export interface Exchange {
  mask: number; value: number; splitMask: number;
  requestBytes: number; responseBytes: number; children: number;
}
export interface Meter {
  requests: number; sent: number; received: number; nextBytes: number; stopped: "budget" | "complete";
  last: Exchange[];
}
export interface Endpoint { exchange(request: Uint8Array): Uint8Array }
const cell = (s: Summary): Cell => ({
  mask: s.mask, value: s.value, count: s.count, lower: s.massLower, upper: s.massUpper,
  a: s.logMaxProbability === undefined ? 0 : -s.logMaxProbability,
  b: s.logMinProbability === undefined ? Infinity : -s.logMinProbability, received: true,
});
const root = (count: number): Cell => ({ mask: 0, value: 0, count, lower: 1, upper: 1, a: 0, b: Infinity, received: false });
const meter = (): Meter => ({ requests: 0, sent: 0, received: 0, nextBytes: 0, stopped: "complete", last: [] });
const clip = (x: number, limit: number) => Math.min(limit, Math.max(0, x));

export function leaves(node: Cell): Cell[] {
  return node.children ? node.children.flatMap(leaves) : [node];
}

/** Return the observed node for this filter, or its nearest observed ancestor.
 * Both replicas use the same bit order; this uses no hidden probabilities. */
export function covering(tree: Cell, target: Filter): Cell {
  let node = tree;
  while (node.children && node.mask !== target.mask) {
    const next = node.children.find((c) => ((target.value & c.mask) >>> 0) === c.value);
    if (!next || ((next.mask & target.mask) >>> 0) !== next.mask) break;
    node = next;
  }
  return node;
}

class Heap {
  private items: { cell: Cell; priority: number }[] = [];
  push(cell: Cell, priority: number): void {
    let i = this.items.length;
    this.items.push({ cell, priority });
    while (i > 0) {
      const parent = (i - 1) >>> 1;
      if (this.items[parent]!.priority >= priority) break;
      [this.items[parent], this.items[i]] = [this.items[i]!, this.items[parent]!];
      i = parent;
    }
  }
  pop(): Cell | undefined {
    const head = this.items[0], end = this.items.pop();
    if (!head || !this.items.length) return head?.cell;
    this.items[0] = end!;
    let i = 0;
    for (;;) {
      let j = 2 * i + 1;
      if (j >= this.items.length) break;
      if (j + 1 < this.items.length && this.items[j + 1]!.priority > this.items[j]!.priority) j++;
      if (this.items[i]!.priority >= this.items[j]!.priority) break;
      [this.items[i], this.items[j]] = [this.items[j]!, this.items[i]!];
      i = j;
    }
    return head.cell;
  }
}

/** A fresh replica for each slider position. The simulated byte gate prepares
 * a reply at the endpoint and admits only complete exchanges under the cap.
 * Unadmitted reply values are never decoded or used by the scheduler. */
export class Simulation {
  readonly p: Cell;
  readonly q: Cell;
  readonly pm = meter();
  readonly qm = meter();
  private readonly bitOrder: number[];

  constructor(readonly size: number, private endpoints: [Endpoint, Endpoint], readonly budget: Budgets) {
    this.p = root(size); this.q = root(size);
    const n = Math.ceil(Math.log2(size));
    this.bitOrder = Array.from({ length: n }, (_, i) => budget.order === "high" ? n - i - 1 : i);
  }

  private split(node: Cell): number {
    return this.bitOrder.filter((bit) => !((node.mask >>> bit) & 1)).slice(0, this.budget.fanout)
      .reduce((mask, bit) => (mask | (1 << bit)) >>> 0, 0);
  }

  private priority(node: Cell, q: boolean): number {
    if (!node.received) return Infinity;
    if (node.count <= 1) return 0;
    if (!q) return node.upper * Math.log2(node.count);
    const p = covering(this.p, node);
    // Further Q subdivision cannot narrow an unsplit P cell's true extrema.
    if (p.mask !== node.mask || !p.children) return 0;
    return p.upper * (clip(node.b, this.budget.clip) - clip(node.a, this.budget.clip));
  }

  async run(yieldWork: () => Promise<boolean> = async () => true): Promise<boolean> {
    for (const side of [0, 1] as const) {
      const q = side === 1, state = q ? this.q : this.p, usage = q ? this.qm : this.pm;
      const limit = q ? this.budget.q : this.budget.p;
      const queue = new Heap();
      queue.push(state, Infinity);
      for (let node = queue.pop(); node; node = queue.pop()) {
        if (this.budget.unit === "requests" && usage.requests >= limit) { usage.stopped = "budget"; break; }
        const splitMask = node.count > 1 ? this.split(node) : 0;
        const request = toBinary(SummarizeRequestSchema, create(SummarizeRequestSchema, {
          mask: node.mask, value: node.value, splitMask, extrema: q,
        }));
        const response = this.endpoints[side].exchange(request);
        const cost = request.length + response.length;
        if (this.budget.unit === "bytes" && usage.sent + usage.received + cost > limit) {
          usage.nextBytes = cost; usage.stopped = "budget"; break;
        }
        const decoded = fromBinary(SummarizeResponseSchema, response);
        if (!decoded.parent) throw Error("missing parent summary");
        Object.assign(node, cell(decoded.parent));
        if (decoded.children.length) node.children = decoded.children.map(cell);
        usage.requests++; usage.sent += request.length; usage.received += response.length;
        usage.last.push({ mask: node.mask, value: node.value, splitMask,
          requestBytes: request.length, responseBytes: response.length, children: decoded.children.length });
        if (usage.last.length > 8) usage.last.shift();
        for (const child of node.children ?? []) {
          const priority = this.priority(child, q);
          if (priority > 0) queue.push(child, priority);
        }
        if (usage.requests % 32 === 0 && !await yieldWork()) return false;
      }
    }
    return true;
  }
}

export interface Group extends Filter { count: number; lower: number; upper: number; a: number; b: number }
export function groups(sim: Simulation): Group[] {
  return leaves(sim.p).map((p) => {
    const q = covering(sim.q, p);
    return { mask: p.mask, value: p.value, count: p.count, lower: p.lower, upper: p.upper, a: q.a, b: q.b };
  });
}

export function cdfBounds(cells: readonly Group[], width: number): [number, number] {
  let certainLo = 0, otherHi = 0, possibleHi = 0, impossibleLo = 0;
  for (const c of cells) {
    if (c.b <= width) certainLo += c.lower; else otherHi += c.upper;
    if (c.a <= width) possibleHi += c.upper; else impossibleLo += c.lower;
  }
  return [Math.max(0, certainLo, 1 - otherHi), Math.min(1, possibleHi, 1 - impossibleLo)];
}

/** Invert the lower/upper CDFs. Unknown allocation of P inside a coarse
 * filter stays unknown even when Q has supplied finer groups. */
export function silhouette(cells: readonly Group[], rows: number): { lower: Float64Array; upper: Float64Array } {
  const bound = (upperWidth: boolean) => {
    const sorted = [...cells].sort((a, b) => (upperWidth ? a.b - b.b : a.a - b.a));
    let totalLo = 0, totalHi = 0;
    for (const c of cells) { totalLo += c.lower; totalHi += c.upper; }
    const cumulative: number[] = [];
    let lo = 0, hi = 0;
    for (const c of sorted) {
      lo += c.lower; hi += c.upper;
      cumulative.push(upperWidth ? Math.max(lo, 1 - (totalHi - hi)) : Math.min(hi, 1 - (totalLo - lo)));
    }
    const result = new Float64Array(rows);
    let index = 0;
    for (let row = 0; row < rows; row++) {
      const y = (row + 0.5) / rows;
      while (index + 1 < sorted.length && cumulative[index]! < y) index++;
      result[row] = upperWidth ? sorted[index]!.b : sorted[index]!.a;
    }
    return result;
  };
  return { lower: bound(false), upper: bound(true) };
}

export function statistics(sim: Simulation, cells: readonly Group[]) {
  let lower = 0, upper = 0, gap = 0;
  for (const c of cells) {
    if (c.lower > 0) lower += c.lower * c.a;
    if (c.upper > 0) upper += c.upper * c.b;
    gap += c.upper * (clip(c.b, sim.budget.clip) - clip(c.a, sim.budget.clip));
  }
  let chain = 0;
  const visit = (p: Cell, q: Cell) => {
    if (p.children && q.children) {
      for (const child of p.children) {
        const peer = q.children.find((c) => c.value === child.value);
        if (peer) visit(child, peer);
      }
    } else if (p.lower > 0) chain += p.lower * -Math.log(q.upper);
  };
  visit(sim.p, sim.q);
  return { lower: Math.max(lower, chain), upper, chain, gap, pGroups: leaves(sim.p).length,
    qGroups: leaves(sim.q).length, singletonMass: cells.filter((c) => c.count === 1).reduce((sum, c) => sum + c.lower, 0) };
}

export function nextBits(mask: number): number { return bits(mask).length; }
