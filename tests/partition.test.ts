import { describe, expect, it } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { SummarizeRequestSchema, SummarizeResponseSchema } from "../src/gen/natlas/partition/v1/partition_pb";
import { PartitionServer } from "../src/playground/server";
import { cdfBounds, groups, silhouette, Simulation, statistics, type Budgets } from "../src/playground/simulation";

const logs = (p: number[]) => Float64Array.from(p, Math.log);
const options: Budgets = { p: 50, q: 50, unit: "requests", order: "high", fanout: 2, clip: 16 };
async function run(p: number[], q: number[], budget: Partial<Budgets> = {}) {
  const sim = new Simulation(p.length, [new PartitionServer(logs(p)), new PartitionServer(logs(q))], { ...options, ...budget });
  await sim.run();
  return sim;
}

describe("partition wire contract", () => {
  it("returns disjoint mask/value children and individual extrema", () => {
    const server = new PartitionServer(logs([1, 3, 2, 4, 0]));
    const query = toBinary(SummarizeRequestSchema, create(SummarizeRequestSchema, { splitMask: 3, extrema: true }));
    const result = fromBinary(SummarizeResponseSchema, server.exchange(query));
    expect(result.parent!.massLower).toBe(1);
    expect(result.parent!.logMinProbability).toBe(-Infinity);
    expect(result.children.map((c) => c.count)).toEqual([2, 1, 1, 1]);
    expect(result.children[0]!.massLower).toBeLessThanOrEqual(0.1);
    expect(result.children[0]!.massUpper).toBeGreaterThanOrEqual(0.1);
    expect(Math.exp(result.children[1]!.logMinProbability!)).toBeLessThanOrEqual(0.3);
    expect(Math.exp(result.children[1]!.logMaxProbability!)).toBeGreaterThanOrEqual(0.3);
    const noExtrema = toBinary(SummarizeRequestSchema, create(SummarizeRequestSchema, { splitMask: 3 }));
    expect(server.exchange(noExtrema).length).toBeLessThan(server.exchange(query).length);
  });

  it("rejects overlapping or malformed splits", () => {
    const server = new PartitionServer(logs([1, 2]));
    for (const bad of [{ mask: 1, splitMask: 1 }, { mask: 0, value: 1 }, { splitMask: 31 }]) {
      expect(() => server.exchange(toBinary(SummarizeRequestSchema, create(SummarizeRequestSchema, bad)))).toThrow();
    }
  });

  it("meters actual binary messages and admits no partial exchange", async () => {
    const full = await run([1, 2, 3, 4], [4, 3, 2, 1], { p: 1, q: 1 });
    const pCost = full.pm.sent + full.pm.received, qCost = full.qm.sent + full.qm.received;
    const justShort = await run([1, 2, 3, 4], [4, 3, 2, 1], { unit: "bytes", p: pCost - 1, q: qCost - 1 });
    expect(justShort.pm.requests).toBe(0);
    expect(justShort.qm.requests).toBe(0);
    expect(justShort.pm.sent + justShort.pm.received).toBe(0);
    const exact = await run([1, 2, 3, 4], [4, 3, 2, 1], { unit: "bytes", p: pCost, q: qCost });
    expect(exact.pm.requests).toBe(1);
    expect(exact.qm.requests).toBe(1);
    expect(exact.pm.sent + exact.pm.received).toBe(pCost);
    expect(exact.qm.sent + exact.qm.received).toBe(qCost);
  });
});

describe("independent P/Q evidence", () => {
  it("does not allocate unknown P uniformly even after Q is queried", async () => {
    const sim = await run([1, 1, 1, 97], [70, 20, 9, 1], { p: 0 });
    const cells = groups(sim);
    expect(cells.length).toBe(1);
    expect(cells[0]!.lower).toBe(1);
    const [lo, hi] = cdfBounds(cells, 2);
    expect(lo).toBe(0); expect(hi).toBe(1);
    expect(sim.pm.requests).toBe(0);
  });

  it("keeps widths unknown without a Q reply", async () => {
    const sim = await run([1, 2, 3, 4], [4, 3, 2, 1], { q: 0 });
    const envelope = silhouette(groups(sim), 20);
    expect([...envelope.lower].every((x) => x === 0)).toBe(true);
    expect([...envelope.upper].every((x) => x === Infinity)).toBe(true);
  });

  it("encloses independent oracle CDFs, silhouettes, and cross-entropies", async () => {
    let seed = 97;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % 100 + 1; };
    for (const order of ["high", "low"] as const) for (const cap of [0, 1, 3, 7, 50]) {
      const p = Array.from({ length: 37 }, random), q = Array.from({ length: 37 }, random);
      const ps = p.reduce((a, b) => a + b), qs = q.reduce((a, b) => a + b);
      const widths = q.map((x) => -Math.log(x / qs));
      const sim = await run(p, q, { p: cap, q: Math.max(0, cap - 1), order });
      const cells = groups(sim), stats = statistics(sim, cells), envelope = silhouette(cells, 40);
      for (const w of widths) {
        const truth = p.reduce((sum, mass, i) => sum + (widths[i]! <= w ? mass / ps : 0), 0);
        const [lo, hi] = cdfBounds(cells, w);
        expect(lo).toBeLessThanOrEqual(truth + 1e-12);
        expect(hi).toBeGreaterThanOrEqual(truth - 1e-12);
      }
      const ranked = p.map((_, i) => i).sort((a, b) => widths[a]! - widths[b]! || a - b);
      for (let row = 0; row < 40; row++) {
        let cumulative = 0, truth = 0;
        for (const id of ranked) { cumulative += p[id]! / ps; truth = widths[id]!; if (cumulative >= (row + 0.5) / 40) break; }
        expect(envelope.lower[row]!).toBeLessThanOrEqual(truth + 1e-12);
        expect(envelope.upper[row]!).toBeGreaterThanOrEqual(truth - 1e-12);
      }
      const ce = p.reduce((sum, mass, i) => sum + mass / ps * widths[i]!, 0);
      expect(stats.lower).toBeLessThanOrEqual(ce + 1e-12);
      expect(stats.upper).toBeGreaterThanOrEqual(ce - 1e-12);
    }
  });

  it("resolves a singleton partition and preserves positive P / zero Q", async () => {
    const sim = await run([1, 3], [1, 0]);
    const stats = statistics(sim, groups(sim));
    expect(stats.lower).toBe(Infinity); expect(stats.upper).toBe(Infinity);
    expect([...silhouette(groups(sim), 4).upper]).toEqual([expect.any(Number), Infinity, Infinity, Infinity]);
    const same = await run([1, 2, 3, 4], [1, 2, 3, 4]);
    const s = statistics(same, groups(same));
    expect(s.upper - s.lower).toBeLessThan(1e-12);
  });
});
