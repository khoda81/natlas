import { describe, expect, it } from "bun:test";
import { create, toBinary } from "@bufbuild/protobuf";
import { encode } from "../src/core/codec";
import { Expander } from "../src/data/expand";
import { type Reply, type Source, SnapshotSource } from "../src/data/source";
import { IntervalTree, type NodeId } from "../src/core/tree";
import { SnapshotSchema } from "../src/gen/natlas/v1/natlas_pb";
import { TextLabels } from "../src/readings/labels";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];

function snapshot(labels = TextLabels.none()): SnapshotSource {
  const expansions = [{ request: { path: [], horizon: 10 } }, { request: { path: [1], horizon: 10 } }].map(
    (expansion) => ({ ...expansion, edges: encode([0, 1, 2, 3, 4], weights.map(Math.log), 10) }),
  );
  return new SnapshotSource(toBinary(SnapshotSchema, create(SnapshotSchema, { expansions, formatVersion: 2 })), labels);
}

async function labels(reply: Reply): Promise<number[]> {
  const result: number[] = [];
  for await (const edge of reply.edges) result.push(edge.label);
  return result;
}

describe("SnapshotSource", () => {
  it("rejects handle recordings with migration instructions", () => {
    const bytes = toBinary(SnapshotSchema, create(SnapshotSchema));
    expect(() => new SnapshotSource(bytes)).toThrow("migrate-snapshots.py");
  });
  it("answers down to the requested horizon, vouching only for its own", async () => {
    const source = snapshot();
    const shallow = await source.expand([], 4);
    expect(shallow.complete).toBe(4);
    expect(await labels(shallow)).toEqual([0, 1, 2]);
    const deep = await source.expand([], 20);
    expect(deep.complete).toBe(10);
    expect(await labels(deep)).toEqual([0, 1, 2, 3, 4]);
    expect((await source.expand([1], 10)).complete).toBe(10);
  });

  it("knows nothing below paths it did not record", async () => {
    const source = snapshot();
    for (const path of [[0], [1, 2], [9, 9, 9]]) {
      const reply = await source.expand(path, 10);
      expect(reply.complete).toBe(0);
      expect(await labels(reply)).toEqual([]);
    }
  });

  it("searches recorded paths by their text", async () => {
    const source = snapshot(new TextLabels([new TextEncoder().encode("x"), new TextEncoder().encode("abc")]));
    expect(await source.search("abc")).toEqual([1]);
    expect(await source.search("ab")).toEqual([1]);
    expect(await source.search("abcd")).toEqual([1]);
    expect(source.search("unrelated")).rejects.toThrow("No recorded sequence");
  });
});

/** Wait for measure 0's listing of ``node`` from ``expander``. */
async function fetched(tree: IntervalTree, expander: Expander, node: NodeId, horizon = 2): Promise<void> {
  const landed = new Promise<void>((resolve) => {
    const stop = tree.subscribe(() => (stop(), resolve()));
  });
  expander.request([{ node, measure: 0, horizon, priority: 1 }]);
  await landed;
}

describe("Expander", () => {
  it("refills capacity using current demand without another render", async () => {
    const tree = new IntervalTree(1);
    const calls: number[][] = [];
    const source: Source = {
      async expand(path, horizon) {
        calls.push([...path]);
        return snapshot().expand([], horizon);
      },
    };
    const child = tree.child(0, 1);
    const expander = new Expander(tree, [source], 1);
    expander.request(() => [{ node: tree.listing(0, 0) === undefined ? 0 : child, measure: 0, horizon: 2, priority: 1 }]);
    try {
      await waitUntil(() => tree.listing(child, 0) !== undefined);
      expect(calls).toEqual([[], [1]]);
    } finally { expander.dispose(); }
  });

  it("coalesces path search with discovery and limits simultaneous requests", async () => {
    const tree = new IntervalTree(1);
    let active = 0;
    let peak = 0;
    const calls: number[][] = [];
    const source: Source = {
      async expand(path, horizon) {
        calls.push([...path]);
        peak = Math.max(peak, ++active);
        await Bun.sleep(5);
        active--;
        return snapshot().expand([], horizon);
      },
    };
    const expander = new Expander(tree, [source], 1);
    const child = tree.child(0, 1);
    expander.request([{ node: 0, measure: 0, horizon: 2, priority: 1 }]);
    try {
      const reached = await Promise.all([expander.reach(0, 0, 1), expander.reach(child, 0, 1)]);
      expect(reached).toEqual([true, true]);
      expect(peak).toBe(1);
      expect(calls.filter((path) => path.length === 0)).toHaveLength(1);
    } finally { expander.dispose(); }
  });

  it("stops unreachable recorded searches and ignores responses after disposal", async () => {
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [snapshot()]);
    expect(await expander.reach(0, 0, 99)).toBe(false);
    expect(tree.listing(0, 0)?.horizon).toBe(16);
    expander.dispose();
    let release!: (reply: Reply) => void;
    const pending = new Expander(new IntervalTree(1), [{ expand: () => new Promise((resolve) => { release = resolve; }) }]);
    pending.request([{ node: 0, measure: 0, horizon: 2, priority: 1 }]);
    pending.dispose();
    release(await snapshot().expand([], 8));
    await Bun.sleep(5);
  });
  it("fetches wanted listings once, with headroom", async () => {
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [snapshot()]);
    await fetched(tree, expander, tree.root);
    const root = tree.listing(tree.root, 0)!;
    expect(root.horizon).toBe(6);
    expect(root.labels).toEqual([0, 1, 2, 3]);
    expect(root.leftover).toBeCloseTo(0.15, 6);
  });

  it("names a deep node directly by its complete integer path", async () => {
    const calls: string[] = [];
    const source: Source = {
      async expand(path, horizon) {
        calls.push(`${path.join(",")}@${horizon}`);
        return snapshot().expand([], horizon);
      },
    };
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [source]);
    const deep = tree.child(tree.child(tree.child(tree.root, 1), 2), 3);
    await fetched(tree, expander, deep);
    expect(calls).toEqual(["1,2,3@6"]);
    expect(tree.listing(tree.root, 0)).toBeUndefined();
  });
});

async function waitUntil(condition: () => boolean): Promise<void> {
  const deadline = performance.now() + 1000;
  while (!condition()) {
    if (performance.now() > deadline) throw new Error("timed out waiting for the expander");
    await Bun.sleep(1);
  }
}
