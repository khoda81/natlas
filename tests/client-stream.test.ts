import { expect, it } from "bun:test";
import { type Camera, project, top } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { digits } from "./build";

it("projects while streaming 10,000 disjoint singleton runs into one WASM node", () => {
  const batches = 20;
  const batchSize = 500;
  const count = 2 * batches * batchSize;
  const viewport = { width: 1200, height: 900 };
  const lens = { p: 0, q: 0 };
  const camera: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
  const lnMass = -Math.log(count);
  const lengths = new Uint32Array(batchSize).fill(1);
  const weights = new Float64Array(batchSize).fill(lnMass);
  const answers = Array.from({ length: batches }, (_, batch) => {
    const ranks = Uint32Array.from({ length: batchSize }, (_, index) => 2 * (batch * batchSize + index));
    const tops = Array.from(ranks, (rank) => digits(rank / count));
    const precisions = Uint32Array.from(tops, (top) => top.length);
    const packedTops = Uint8Array.from(tops.flatMap((top) => [...top]));
    return [ROOT, lens.p, ranks, lengths, ranks, weights, precisions, packedTops, count] as const;
  });

  // The browser has already loaded WASM before a live session begins.
  const warmup = new IntervalTree();
  warmup.raw.insert(...answers[0]!);
  project(warmup, lens, { ...camera }, viewport, 1).frame.dispose();
  warmup.raw.free();

  const tree = new IntervalTree();
  let changed = 0;
  const inserts: number[] = [];
  const projections: number[] = [];
  const start = performance.now();
  for (const answer of answers) {
    const beforeInsert = performance.now();
    changed += Number(tree.raw.insert(...answer));
    const afterInsert = performance.now();
    const { frame } = project(tree, lens, camera, viewport, 1);
    frame.dispose();
    const afterProject = performance.now();
    inserts.push(afterInsert - beforeInsert);
    projections.push(afterProject - afterInsert);
  }
  const total = performance.now() - start;
  const max = Math.max(...inserts);
  const ingest = inserts.reduce((sum, ms) => sum + ms, 0);
  console.log(JSON.stringify({ batches, batchSize, insertMs: inserts, projectMs: projections, maxInsertMs: max, totalInsertMs: ingest, totalMs: total }));
  expect(max).toBeLessThan(5);
  expect(total).toBeLessThan(50);

  expect(changed).toBe(batches);
  expect(tree.entry(ROOT, lens, 0)).toMatchObject({ rank: 0 });
  expect(tree.entry(ROOT, lens, count - 2)).toMatchObject({ rank: count - 2 });
  expect(tree.entry(ROOT, lens, count - 1)).toBeUndefined();
  expect(tree.statistics(ROOT, lens)).toMatchObject({ bounded: true });
});
