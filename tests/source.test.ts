import { describe, expect, it } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { Expander } from "../src/data/expand";
import { ConnectSource, type Reply, type Source, type Step, SnapshotSource } from "../src/data/source";
import { IntervalTree, type NodeId } from "../src/core/tree";
import { ExpandRequestSchema, ListingSchema, SnapshotSchema } from "../src/gen/natlas/natlas_pb";
import { TextLabels } from "../src/readings/labels";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];

/** Every node along ``paths`` lists ``weights``, recorded down to 10 bits. */
function snapshot(labels = TextLabels.none(), paths: number[][] = [[], [1]]): SnapshotSource {
  const listing = { branches: [0, 1, 2, 3, 4], lnWeights: weights.map(Math.log), lnLeftover: Math.log(0.14) };
  const expansions = paths.map((path) => ({ path, horizon: 10, listing }));
  return new SnapshotSource(toBinary(SnapshotSchema, create(SnapshotSchema, { expansions })), labels);
}

async function steps(source: Source, tree: IntervalTree, from: NodeId, to: NodeId, horizon: number): Promise<Step[]> {
  const result: Step[] = [];
  for await (const step of source.expand(tree, from, to, horizon)) result.push(step);
  return result;
}

/** One node's reply, named by its parent. */
async function one(source: Source, tree: IntervalTree, node: NodeId, horizon: number): Promise<Reply> {
  const [step] = await steps(source, tree, tree.parent(node) ?? node, node, horizon);
  return step!.reply;
}

/** The recording's root listing, whichever node asked. */
async function rootReply(horizon: number): Promise<Reply> {
  return one(snapshot(), new IntervalTree(1), 0n, horizon);
}

describe("SnapshotSource", () => {
  it("answers down to the requested horizon, vouching only for its own", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    const shallow = await one(source, tree, tree.root, 4);
    expect(shallow.complete).toBe(4);
    expect(shallow.branches).toEqual([0, 1, 2]);
    expect(shallow.leftover).toBeCloseTo(0.2, 12);
    const deep = await one(source, tree, tree.root, 20);
    expect(deep.complete).toBe(10);
    expect(deep.branches).toEqual([0, 1, 2, 3, 4]);
    expect(deep.lnWeights).toEqual(weights.map(Math.log));
    expect((await one(source, tree, tree.child(tree.root, 1), 10)).complete).toBe(10);
  });

  it("walks a path, listing each node through the next branch", async () => {
    const source = snapshot(TextLabels.none(), [[], [1], [1, 4]]);
    const tree = new IntervalTree(1);
    const target = tree.descend(tree.root, [1, 4, 0]);
    const walk = await steps(source, tree, tree.root, target, 1);
    expect(walk.map(({ node }) => tree.path(node))).toEqual([[1], [1, 4], [1, 4, 0]]);
    expect(walk[0]!.reply.branches).toEqual([0, 1, 2, 3, 4]);
    expect(walk[0]!.reply.complete).toBeCloseTo(-Math.log2(0.01), 12);
    expect(walk[1]!.reply.branches).toEqual([0]);
    expect(walk[2]!.reply).toEqual({ branches: [], lnWeights: [], leftover: 1, complete: 0 });
  });

  it("finds recorded nodes by their shared ids, however the client numbered them", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    tree.descend(tree.root, [7, 7, 7]);
    const node = tree.child(tree.root, 1);
    expect(node).not.toBe(1);
    expect((await one(source, tree, node, 10)).branches).toEqual([0, 1, 2, 3, 4]);
  });

  it("knows nothing below paths it did not record", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    for (const path of [[0], [1, 2], [9, 9, 9]]) {
      const reply = await one(source, tree, tree.descend(tree.root, path), 10);
      expect(reply).toEqual({ branches: [], lnWeights: [], leftover: 1, complete: 0 });
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

describe("ConnectSource", () => {
  it("names a walk from a known id, one listing per node, and from the root once the server does not know it", async () => {
    const requests: { node: bigint; path: number[]; horizon: number }[] = [];
    const known = new Set<bigint>([0n]);
    const envelope = (flags: number, body: Uint8Array) => {
      const frame = new Uint8Array(5 + body.length);
      new DataView(frame.buffer).setUint8(0, flags);
      new DataView(frame.buffer).setUint32(1, body.length);
      frame.set(body, 5);
      return frame;
    };
    const listing = (branches: number[], lnWeights: number[]) =>
      envelope(0, toBinary(ListingSchema, create(ListingSchema, { branches, lnWeights, lnLeftover: -Infinity })));
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = new Uint8Array(await request.arrayBuffer());
        const { node, path, horizon } = fromBinary(ExpandRequestSchema, body.subarray(5));
        requests.push({ node, path: [...path], horizon });
        const end = (error?: object) => envelope(2, new TextEncoder().encode(JSON.stringify(error ? { error } : {})));
        const reply = known.has(node)
          ? [...path.slice(1).map((branch) => listing([0, branch], [Math.log(0.5), Math.log(2 ** -20)])), listing([3], [0]), end()]
          : [end({ code: "not_found", message: "unknown node" })];
        return new Response(new Blob(reply), { headers: { "content-type": "application/connect+proto" } });
      },
    });
    try {
      const tree = new IntervalTree(1);
      const deep = tree.descend(tree.root, [5, 6, 7]);
      const source = new ConnectSource(`http://localhost:${server.port}/`);
      const parent = tree.parent(deep)!;
      // Not found by its parent's id, the node is walked to from the root.
      const fallback = await steps(source, tree, parent, deep, 8);
      expect(fallback.map(({ node }) => tree.path(node))).toEqual([[5], [5, 6], [5, 6, 7]]);
      expect(fallback.at(-1)).toEqual({ node: deep, reply: { branches: [3], lnWeights: [0], leftover: 0, complete: 8 } });
      expect(requests).toEqual([
        { node: parent, path: [7], horizon: 8 },
        { node: 0n, path: [5, 6, 7], horizon: 8 },
      ]);
      // The fallback walks from the root: every node down the path, the
      // ones above through the next branch, vouching down to it.
      const walk = await steps(source, tree, tree.root, deep, 8);
      expect(walk.map(({ node }) => tree.path(node))).toEqual([[5], [5, 6], [5, 6, 7]]);
      expect(walk[0]!.reply.complete).toBe(20);
      known.add(parent);
      await steps(source, tree, parent, deep, 8);
      expect(requests).toHaveLength(4);
    } finally {
      server.stop(true);
    }
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

/** A source answering every node with the recording's root listing. */
function rooted(onRequest: (tree: IntervalTree, from: NodeId, to: NodeId) => void | Promise<void>): Source {
  return {
    async *expand(tree, from, to, horizon) {
      await onRequest(tree, from, to);
      yield { node: to, reply: await rootReply(horizon) };
    },
  };
}

describe("Expander", () => {
  it("refills capacity using current demand without another render", async () => {
    const tree = new IntervalTree(1);
    const calls: number[][] = [];
    const child = tree.child(0n, 1);
    const expander = new Expander(tree, [rooted((tree, _from, to) => void calls.push(tree.path(to)))], 1);
    expander.request(() => [{ node: tree.listing(0n, 0) === undefined ? 0n : child, measure: 0, horizon: 2, priority: 1 }]);
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
    const source = rooted(async (tree, _from, to) => {
      calls.push(tree.path(to));
      peak = Math.max(peak, ++active);
      await Bun.sleep(5);
      active--;
    });
    const expander = new Expander(tree, [source], 1);
    const child = tree.child(0n, 1);
    expander.request([{ node: 0n, measure: 0, horizon: 2, priority: 1 }]);
    try {
      const reached = await Promise.all([expander.reach(0n, 0, 1), expander.reach(child, 0, 1)]);
      expect(reached).toEqual([true, true]);
      expect(peak).toBe(1);
      expect(calls.filter((path) => path.length === 0)).toHaveLength(1);
    } finally { expander.dispose(); }
  });

  it("walks once from the deepest node it can already place", async () => {
    const recording = snapshot(TextLabels.none(), [[], [1], [1, 2], [1, 2, 3]]);
    const requests: number[][] = [];
    const source: Source = {
      expand(tree, from, to, horizon) {
        requests.push(tree.path(from));
        return recording.expand(tree, from, to, horizon);
      },
    };
    const tree = new IntervalTree(1);
    tree.setListing(tree.root, 0, { ...(await rootReply(10)), horizon: 10 });
    const target = tree.descend(tree.root, [1, 2, 3]);
    const expander = new Expander(tree, [source]);
    try {
      expect(await expander.walk(tree.root, target, 0, 4)).toBe(true);
      expect(requests).toEqual([[]]);
      for (const path of [[1], [1, 2]]) expect(tree.slotOf(tree.descend(tree.root, path), { p: 0, q: 0 }, path.length + 1)).toBeDefined();
      expect(await expander.walk(tree.root, target, 0, 4)).toBe(true);
      expect(requests).toHaveLength(1);
      // Past what was recorded the walk ends, and asks no more.
      const beyond = tree.descend(target, [0, 0]);
      expect(await expander.walk(tree.root, beyond, 0, 4)).toBe(false);
      const asked = requests.length;
      expect(await expander.walk(tree.root, beyond, 0, 4)).toBe(false);
      expect(requests).toHaveLength(asked);
    } finally { expander.dispose(); }
  });

  it("stops unreachable recorded searches and ignores responses after disposal", async () => {
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [snapshot()]);
    expect(await expander.reach(0n, 0, 99)).toBe(false);
    expect(tree.listing(0n, 0)?.horizon).toBe(16);
    expander.dispose();
    let release!: () => void;
    const pending = new Expander(new IntervalTree(1), [rooted(() => new Promise<void>((resolve) => { release = resolve; }))]);
    pending.request([{ node: 0n, measure: 0, horizon: 2, priority: 1 }]);
    await waitUntil(() => release !== undefined);
    pending.dispose();
    release();
    await Bun.sleep(5);
  });

  it("abandons work in flight when the view moves, without reporting it", async () => {
    const tree = new IntervalTree(1);
    const signals: AbortSignal[] = [];
    const errors: unknown[] = [];
    const source: Source = {
      // eslint-disable-next-line require-yield
      async *expand(_tree, _from, _to, _horizon, signal) {
        signals.push(signal!);
        await new Promise((_resolve, reject) => signal!.addEventListener("abort", () => reject(signal!.reason)));
      },
    };
    const expander = new Expander(tree, [source]);
    expander.onError = (error) => errors.push(error);
    const old = tree.child(0n, 1);
    expander.request([{ node: old, measure: 0, horizon: 2, priority: 1 }]);
    await waitUntil(() => signals.length === 1);
    expander.cancel();
    expander.request([{ node: old, measure: 0, horizon: 2, priority: 1 }]);
    await waitUntil(() => signals.length === 2);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[1]!.aborted).toBe(false);
    await Bun.sleep(5);
    expect(errors).toEqual([]);
    expander.dispose();
  });

  it("fetches wanted listings once, with headroom", async () => {
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [snapshot()]);
    await fetched(tree, expander, tree.root);
    const root = tree.listing(tree.root, 0)!;
    expect(root.horizon).toBe(6);
    expect(root.branches).toEqual([0, 1, 2, 3]);
    expect(root.leftover).toBeCloseTo(0.15, 12);
  });
});

async function waitUntil(condition: () => boolean): Promise<void> {
  const deadline = performance.now() + 1000;
  while (!condition()) {
    if (performance.now() > deadline) throw new Error("timed out waiting for the expander");
    await Bun.sleep(1);
  }
}
