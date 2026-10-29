import { describe, expect, it } from "bun:test";
import { RadixCDF, type QuantileReply } from "../src/radix";
import { type Camera, layout, normalize, repair } from "../src/layout";
import { IntervalTree } from "../src/tree";
import { Refiner } from "../src/refine";
import type { Source } from "../src/source";
import { Code, ConnectError } from "@connectrpc/connect";
import { distributionBounds } from "../src/statistics";
import { GeometryEvidence } from "../src/evidence";

/** Exact finite integer distribution, independent of the client's algorithms. */
function page(weights: bigint[], prefix: readonly number[] = [], node = 7n): QuantileReply {
  if (!weights.length) return { node, firstBranchId: 0, delta: [], terminal: true };
  const total = weights.reduce((a, b) => a + b, 0n);
  const a = prefix.reduce((a, b) => a * 256n + BigInt(b), 0n) * 256n;
  const den = 256n ** BigInt(prefix.length + 1);
  const ids = Array.from({ length: 257 }, (_, j) => {
    let sum = 0n;
    for (let id = 0; id < weights.length; id++) {
      sum += weights[id]!;
      if (sum * den > (a + BigInt(j)) * total) return id;
    }
    return weights.length;
  });
  return { node, firstBranchId: ids[0]!, delta: ids.slice(1).map((id, j) => BigInt(id - ids[j]!)), terminal: false };
}

function fill(state: RadixCDF, weights: bigint[], label: number, tolerance = 1e-8): void {
  if (!state.ready) state.insert([], page(weights));
  for (let count = 0; count < 512; count++) {
    const b = state.bounds(label);
    if (b.upper - b.lower <= tolerance) return;
    const prefix = state.nextBoundary(label);
    if (!prefix) return;
    state.insert(prefix, page(weights, prefix));
  }
  throw Error("Failed to converge");
}

function attach(tree: IntervalTree, measure: number, state: RadixCDF): void {
  tree.setListing(0, measure, { horizon: 0, radix: state });
}

describe("radix CDF", () => {
  it("uses normalization for exact outside boundaries and empty ranges", () => {
    const state = new RadixCDF();
    state.insert([], page([1n]));
    expect(state.boundaryBounds(0)).toEqual({ lo: 0n, hi: 0n, depth: 0 });
    expect(state.boundaryBounds(1)).toEqual({ lo: 1n, hi: 1n, depth: 0 });
    expect(state.bounds(0)).toEqual({ lower: 0, upper: 0 });
    expect(state.rangeBounds(0, 0)).toEqual({ lower: -Infinity, upper: -Infinity });
    expect(state.partition().map((r) => [r.kind, r.first, r.end, r.lnMass])).toEqual([["token", 0, 1, 0]]);
  });
  it("does not invent individual probabilities from indistinguishable sparse pages", () => {
    const uniform = Array<bigint>(512).fill(1n);
    const spikes = uniform.map((_, i) => i % 2 ? 0n : 2n);
    expect(page(uniform)).toEqual(page(spikes));
    const state = new RadixCDF();
    state.insert([], page(uniform));
    const regions = state.partition();
    expect(regions).toHaveLength(1);
    expect(regions[0]).toMatchObject({ kind: "unresolved", first: 0, end: 512, mass: { lower: 0, upper: 0 }, lnMass: 0 });
  });
  it("preserves shared cumulative boundaries and monotone certificates during refinement", () => {
    const weights = [0n, 1n, 23n, 0n, 7n, 969n];
    const state = new RadixCDF();
    state.insert([], page(weights));
    for (let step = 0; step < 12; step++) {
      const previous = weights.map((_, id) => state.bounds(id));
      const regions = state.partition();
      expect(regions.reduce((sum, region) => sum + Math.exp(region.lnMass), 0)).toBeCloseTo(1, 12);
      for (let i = 0; i < regions.length; i++) {
        const region = regions[i]!;
        const truth = Number(weights.slice(region.first, region.end).reduce((a, b) => a + b, 0n)) / 1000;
        expect(Math.exp(region.mass.lower)).toBeLessThanOrEqual(truth + 1e-14);
        expect(Math.exp(region.mass.upper)).toBeGreaterThanOrEqual(truth - 1e-14);
        if (i) expect(region.before).toEqual(regions[i - 1]!.after);
        if (region.kind === "token") expect(region.mass.lower).toBeGreaterThan(-Infinity);
      }
      const prefix = state.nextBoundary(1);
      if (!prefix) break;
      state.insert(prefix, page(weights, prefix));
      for (let id = 0; id < weights.length; id++) {
        const current = state.bounds(id);
        expect(current.lower).toBeGreaterThanOrEqual(previous[id]!.lower - 1e-12);
        expect(current.upper).toBeLessThanOrEqual(previous[id]!.upper + 1e-12);
      }
    }
    const before = state.partition();
    state.pin(1);
    expect(state.partition()).toEqual(before);
  });
  it("weights unequal refinement levels by their actual probability mass", () => {
    const weights = [1n, 2n, 997n];
    const state = new RadixCDF();
    state.insert([], page(weights));
    const prefix = state.nextSample(Math.log(800))!;
    state.insert(prefix, page(weights, prefix));
    expect(state.partition().reduce((a, r) => a + Math.exp(r.lnMass), 0)).toBeCloseTo(1, 12);
    expect(Math.exp(state.partition().find((r) => r.first === 2)!.lnMass)).toBeGreaterThan(0.99);
  });
  it("recovers tiny q near CDF 1 without subtracting floats", () => {
    const weights = [(1n << 1000n) - 1n, 1n];
    const state = new RadixCDF();
    fill(state, weights, 1, 1e-5);
    const b = state.bounds(1);
    const expected = -1000 * Math.LN2;
    expect(b.lower).toBeLessThanOrEqual(expected + 1e-10);
    expect(b.upper).toBeGreaterThanOrEqual(expected - 1e-10);
    expect((b.upper - b.lower) * 100).toBeLessThan(0.5);
    const region = state.partition().find((r) => r.first === 1)!;
    expect(region.kind).toBe("token");
    expect(region.lnMass).toBeCloseTo(expected, 5);
  });
  it("skips zero mass IDs and handles the vocabulary-end sentinel", () => {
    const weights = [0n, 2n, 0n, 1n, 0n];
    const state = new RadixCDF();
    state.insert([], page(weights));
    expect(state.partition().filter((r) => r.kind === "token").map((r) => r.first)).not.toContain(0);
    fill(state, weights, 3);
    expect(Math.exp(state.bounds(3).upper)).toBeCloseTo(1 / 3, 7);
  });
  it("preserves resolved prompt mass without renormalizing any neighbors", () => {
    const weights = [1n, 1000000000000n, 3n];
    const state = new RadixCDF();
    fill(state, weights, 2);
    const before = state.partition();
    state.pin(2);
    expect(state.partition()).toEqual(before);
    expect(state.partition().reduce((a, r) => a + Math.exp(r.lnMass), 0)).toBeCloseTo(1, 12);
    expect(Math.exp(state.partition().find((r) => r.first === 2)!.lnMass)).toBeCloseTo(3 / 1000000000004, 20);
    expect(state.bounds(3)).toEqual({ lower: -Infinity, upper: -Infinity });
  });

  it("rejects malformed pages and inconsistent measures", () => {
    const state = new RadixCDF();
    state.insert([], page([1n, 1n]));
    expect(() => state.insert([], page([1n, 3n]))).toThrow("changed");
    expect(() => state.insert([0], page([3n, 1n]))).toThrow("parent");
    expect(() => state.insert([], { ...page([1n]), delta: [-1n] })).toThrow();
  });
  it("refines the visible CDF window inside an otherwise unnamed ID range", () => {
    const state = new RadixCDF();
    state.insert([], page(Array<bigint>(512).fill(1n)));
    const prefix = state.nextSample(Math.log(8000), new Set(), [{ first: 0, end: 512 }], { id: 0, y: -6001, height: 800 });
    expect(prefix).toEqual([192]);
  });
});

describe("v2 anchored geometry", () => {
  it("keeps ID order and the same labelled edge when q ranks change", () => {
    const tree = new IntervalTree(2),
      p = new RadixCDF(),
      q = new RadixCDF();
    p.insert([], page([1n, 1n, 1n, 1n]));
    q.insert([], page([123n, 1005n, 1004n, 700n]));
    attach(tree, 0, p);
    attach(tree, 1, q);
    const lens = { p: 0, q: 1 };
    const before = tree.root.slots(lens);
    expect(before.labels).toEqual([0, 1, 2, 3]);
    const cam: Camera = { x: 0, k: 100, anchor: normalize(tree, lens, { node: 0, slot: 2 }), a: 0.37 };
    fill(q, [123n, 1005n, 1004n, 700n], 1);
    fill(q, [123n, 1005n, 1004n, 700n], 2);
    tree.touch();
    repair(tree, lens, cam);
    expect(tree.root.slots(lens).labels).toEqual([0, 1, 2, 3]);
    expect(cam.anchor.slot).toBe(2);
    const box = layout(tree, lens, cam, { width: 1000, height: 800 }).boxes.find((b) => b.label === 2)!;
    expect(box.y0).toBeCloseTo(0.37, 12);
  });
  it("repairs a root anchor whose token has zero mass after a measure swap", () => {
    const tree = new IntervalTree(2),
      p = new RadixCDF(),
      q = new RadixCDF();
    p.insert([], page([1n, 1n]));
    q.insert([], page([0n, 1n]));
    attach(tree, 0, p);
    attach(tree, 1, q);
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0, label: 0 }, a: 0 };
    repair(tree, { p: 1, q: 0 }, cam);
    expect(cam.anchor.label).toBe(1);
    expect(layout(tree, { p: 1, q: 0 }, cam, { width: 1000, height: 800 }).boxes.length).toBeGreaterThan(0);
  });

  it("shows subpixel collections without inventing one average width", () => {
    const tree = new IntervalTree(1),
      p = new RadixCDF();
    p.insert([], page(Array(16).fill(1n)));
    attach(tree, 0, p);
    const lens = { p: 0, q: 0 };
    const cam: Camera = { x: 0, k: 10, anchor: { node: 0, slot: 0 }, a: 0 };
    const frame = layout(tree, lens, cam, { width: 1000, height: 8 });
    expect(frame.boxes.every((b) => b.kind === "aggregate" && b.x1 === Infinity && b.label === null)).toBe(true);
    expect(frame.boxes.reduce((s, b) => s + b.p, 0)).toBeCloseTo(1, 12);
  });

  it("keeps p=q height and width coherent while exposing their uncertainty", () => {
    const tree = new IntervalTree(1), p = new RadixCDF();
    p.insert([], page([3n, 1n, 7n]));
    attach(tree, 0, p);
    const lens = { p: 0, q: 0 };
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    for (const box of layout(tree, lens, cam, { width: 1000, height: 800 }).boxes) {
      expect(box.x1 - box.x0).toBeCloseTo(-Math.log(box.p), 12);
      expect(box.uncertainty).toBeDefined();
      const exactWidth = -Math.log([3, 1, 7][box.label!]! / 11);
      expect(box.uncertainty!.right.lower).toBeLessThanOrEqual(exactWidth);
      expect(box.uncertainty!.right.upper).toBeGreaterThanOrEqual(exactWidth);
    }
  });

  it("keeps unsupported Q continuations infinitely wide instead of inventing a midpoint", () => {
    const tree = new IntervalTree(2), p = new RadixCDF(), q = new RadixCDF();
    p.insert([], page([1n, 1n])); q.insert([], page([]));
    attach(tree, 0, p); attach(tree, 1, q);
    const lens = { p: 0, q: 1 };
    const slots = tree.root.slots(lens);
    expect([...slots.widths]).toEqual([Infinity, Infinity]);
    expect([...slots.q]).toEqual([0, 0]);
    expect(distributionBounds(p, q).cross).toEqual({ lower: Infinity, upper: Infinity });
    const frame = layout(tree, lens, { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 }, { width: 1000, height: 800 });
    expect(frame.boxes).toHaveLength(2);
    expect(frame.boxes.every((box) => box.x1 === Infinity)).toBe(true);
  });

  it("keeps an unresolved range's ID boundary when its interior is discovered", () => {
    const tree = new IntervalTree(1), p = new RadixCDF();
    const weights = Array<bigint>(512).fill(1n);
    p.insert([], page(weights));
    attach(tree, 0, p);
    const lens = { p: 0, q: 0 };
    const cam: Camera = { x: 0, k: 100, anchor: normalize(tree, lens, { node: 0, slot: 0 }), a: 0.2 };
    expect(cam.anchor.label).toBe(0);
    const before = layout(tree, lens, cam, { width: 1000, height: 800 });
    expect(before.boxes[0]!.kind).toBe("unresolved");
    expect(before.unexpanded).toHaveLength(0);
    p.insert([0], page(weights, [0]));
    tree.touch();
    repair(tree, lens, cam);
    expect(layout(tree, lens, cam, { width: 1000, height: 800 }).boxes[0]!.y0).toBeCloseTo(0.2, 12);
  });
});

describe("anchored geometry evidence", () => {
  it("bounds distances between cousins, ancestors, and descendant boundaries", () => {
    const tree = new IntervalTree(2);
    const truth = new Map<number, { top: number; mass: number; x: number }>();
    const weights = [3n, 7n, 2n], qw = [9n, 2n, 1n];
    const build = (node: number, depth: number, top: number, mass: number, x: number) => {
      const p = new RadixCDF(), q = new RadixCDF();
      p.insert([], page(weights)); q.insert([], page(qw));
      tree.setListing(node, 0, { horizon: 0, radix: p });
      tree.setListing(node, 1, { horizon: 0, radix: q });
      truth.set(node, { top, mass, x });
      if (!depth) return;
      let offset = 0;
      for (let id = 0; id < 3; id++) {
        build(tree.child(node, id).id, depth - 1, top + mass * offset, mass * Number(weights[id]) / 12, x - Math.log(Number(qw[id]) / 12));
        offset += Number(weights[id]) / 12;
      }
    };
    build(0, 2, 0, 1, 0);
    const evidence = new GeometryEvidence(tree, { p: 0, q: 1 });
    const exact = (node: number, id: number) => truth.get(node)!.top + truth.get(node)!.mass * Number(weights.slice(0, id).reduce((a, b) => a + b, 0n)) / 12;
    for (const a of tree.nodes) for (const b of tree.nodes) for (const ai of [0, 1, 3]) for (const bi of [0, 2, 3]) {
      const bounds = evidence.project({ node: a.id, id: ai }, { node: b.id, id: bi }, 0.37, 0);
      const actual = 0.37 + exact(b.id, bi) - exact(a.id, ai);
      expect(bounds.lower).toBeLessThanOrEqual(actual + 1e-12);
      expect(bounds.upper).toBeGreaterThanOrEqual(actual - 1e-12);
    }
    for (const node of tree.nodes) {
      expect(evidence.mass(node.id).lower).toBeLessThanOrEqual(Math.log(truth.get(node.id)!.mass) + 1e-12);
      expect(evidence.mass(node.id).upper).toBeGreaterThanOrEqual(Math.log(truth.get(node.id)!.mass) - 1e-12);
      expect(evidence.x(node.id).lower).toBeLessThanOrEqual(truth.get(node.id)!.x + 1e-12);
      expect(evidence.x(node.id).upper).toBeGreaterThanOrEqual(truth.get(node.id)!.x - 1e-12);
    }
  });
  it("keeps deep tail coordinates without subtracting absolute CDF floats", () => {
    const tree = new IntervalTree(1), p = new RadixCDF();
    fill(p, [1n, 1n], 0, 1e-14);
    let node = tree.root;
    for (let depth = 0; depth <= 80; depth++) {
      tree.setListing(node.id, 0, { horizon: 0, radix: p });
      if (depth < 80) node = tree.child(node.id, 1);
    }
    const evidence = new GeometryEvidence(tree, { p: 0, q: 0 });
    const bounds = evidence.project({ node: node.id, id: 0 }, { node: node.id, id: 1 }, 0.2, -80 * Math.LN2);
    expect(bounds.lower).toBeCloseTo(0.7, 10);
    expect(bounds.upper).toBeCloseTo(0.7, 10);
    const rootBottom = evidence.project({ node: node.id, id: 0 }, { node: 0, id: 2 }, 0.2, -80 * Math.LN2);
    expect(rootBottom.lower).toBeCloseTo(1.2, 10);
    expect(rootBottom.upper).toBeCloseTo(1.2, 10);
  });
});

class Quantiles implements Source {
  calls: string[] = [];
  known = new Set([7n]);
  async expand(): Promise<never> {
    throw Error("v1 must not be called");
  }
  async quantile(node: { parent: bigint; label: number } | undefined, prefix: Uint8Array): Promise<QuantileReply> {
    this.calls.push(`${node?.parent ?? "root"}/${node?.label ?? ""}:${[...prefix]}`);
    if (node && !this.known.has(node.parent)) throw new ConnectError("forgotten", Code.NotFound);
    const handle = node ? node.parent * 10n + BigInt(node.label) : 7n;
    this.known.add(handle);
    return page([1n, 3n], prefix.length ? [...prefix] : [], handle);
  }
}

describe("v2 refinement", () => {
  it("names prompt tokens smaller than sample bins and recovers forgotten handles", async () => {
    const source = new Quantiles(),
      tree = new IntervalTree(1),
      refiner = new Refiner(tree, [source]);
    await refiner.reach(tree.root, 0, 1);
    const child = tree.child(0, 1);
    await refiner.reach(child, 0, 0);
    source.known = new Set([7n]);
    await refiner.reach(tree.child(child.id, 1), 0, 0);
    expect(source.calls.some((x) => x.startsWith("71/1"))).toBe(true);
    expect(tree.node(2).slots({ p: 0, q: 0 }).labels).toContain(0);
  });
});

describe("quantile statistics", () => {
  it("bounds both indistinguishable distributions without assuming uniform groups", () => {
    const state = new RadixCDF();
    state.insert([], page(Array<bigint>(512).fill(1n)));
    const bounds = distributionBounds(state, state);
    expect(bounds.entropy.lower).toBe(0);
    expect(bounds.entropy.upper).toBeCloseTo(Math.log(512), 12);
    expect(bounds.entropy.lower).toBeLessThan(Math.log(256));
    expect(bounds.cross).toEqual(bounds.entropy);
  });
  it("contains exact entropy and cross-entropy as independent endpoints refine", () => {
    const pw = [3n, 7n, 2n], qw = [9n, 2n, 1n];
    const p = new RadixCDF(), q = new RadixCDF();
    p.insert([], page(pw)); q.insert([], page(qw));
    const entropy = pw.reduce((s, w) => s - Number(w) / 12 * Math.log(Number(w) / 12), 0);
    const cross = pw.reduce((s, w, i) => s - Number(w) / 12 * Math.log(Number(qw[i]) / 12), 0);
    for (let i = 0; i < pw.length; i++) {
      const bounds = distributionBounds(p, q);
      expect(bounds.entropy.lower).toBeLessThanOrEqual(entropy + 1e-12);
      expect(bounds.entropy.upper).toBeGreaterThanOrEqual(entropy - 1e-12);
      expect(bounds.cross.lower).toBeLessThanOrEqual(cross + 1e-12);
      expect(bounds.cross.upper).toBeGreaterThanOrEqual(cross - 1e-12);
      fill(p, pw, i); fill(q, qw, i);
    }
    expect(distributionBounds(p, q).cross.upper - distributionBounds(p, q).cross.lower).toBeLessThan(1e-7);
  });
});
