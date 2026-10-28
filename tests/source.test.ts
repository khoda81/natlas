import { describe, expect, it } from "bun:test";
import { create, toBinary } from "@bufbuild/protobuf";
import { encode } from "../src/codec";
import { Expander } from "../src/expand";
import { SnapshotSchema } from "../src/gen/natlas/v1/natlas_pb";
import { type Reply, SnapshotSource } from "../src/source";
import { childHandle, IntervalTree, ROOT } from "../src/tree";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];

function snapshot(): SnapshotSource {
  const expansions = [ROOT, childHandle(ROOT, 1)].map((node) => ({
    request: { node, horizon: 10 },
    edges: encode([0, 1, 2, 3, 4], weights.map(Math.log), 10),
  }));
  return new SnapshotSource(toBinary(SnapshotSchema, create(SnapshotSchema, { expansions })));
}

async function labels(reply: Reply): Promise<number[]> {
  const result: number[] = [];
  for await (const edge of reply.edges) result.push(edge.label);
  return result;
}

describe("SnapshotSource", () => {
  it("answers down to the requested horizon, vouching only for its own", async () => {
    const source = snapshot();
    const shallow = source.expand(ROOT, 4);
    expect(shallow.complete).toBe(4);
    expect(await labels(shallow)).toEqual([0, 1, 2]);
    const deep = source.expand(ROOT, 20);
    expect(deep.complete).toBe(10);
    expect(await labels(deep)).toEqual([0, 1, 2, 3, 4]);
  });

  it("knows nothing about nodes it did not record", async () => {
    const reply = snapshot().expand(childHandle(ROOT, 0), 10);
    expect(reply.complete).toBe(0);
    expect(await labels(reply)).toEqual([]);
  });
});

describe("Expander", () => {
  it("fetches wanted listings once, with headroom", async () => {
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [snapshot()]);
    const fetched = new Promise<void>((resolve) => tree.subscribe(resolve));
    const want = { node: tree.root, measure: 0, horizon: 2, priority: 1 };
    expander.request([want]);
    expander.request([want]);
    await fetched;
    const root = tree.root.listing(0)!;
    expect(root.horizon).toBe(6);
    expect(root.labels).toEqual([0, 1, 2, 3]);
    expect(root.leftover).toBeCloseTo(0.15, 6);
  });
});
