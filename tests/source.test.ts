import { describe, expect, it } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { Expander } from "../src/data/expand";
import { ConnectSource, type Reply, type Source, SnapshotSource } from "../src/data/source";
import { IntervalTree, type NodeId } from "../src/core/tree";
import { ExpandRequestSchema, ListingSchema, SnapshotSchema } from "../src/gen/natlas/natlas_pb";
import { TextLabels } from "../src/readings/labels";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];

function snapshot(labels = TextLabels.none()): SnapshotSource {
  const listing = { branches: [0, 1, 2, 3, 4], lnWeights: weights.map(Math.log), lnLeftover: Math.log(0.14) };
  const expansions = [[], [1]].map((path) => ({ path, horizon: 10, listing }));
  return new SnapshotSource(toBinary(SnapshotSchema, create(SnapshotSchema, { expansions })), labels);
}

/** The recording's root listing, whichever node asked. */
function rootReply(horizon: number): Promise<Reply> {
  return snapshot().expand(new IntervalTree(1), 0, horizon);
}

describe("SnapshotSource", () => {
  it("answers down to the requested horizon, vouching only for its own", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    const shallow = await source.expand(tree, tree.root, 4);
    expect(shallow.complete).toBe(4);
    expect(shallow.branches).toEqual([0, 1, 2]);
    expect(shallow.leftover).toBeCloseTo(0.2, 12);
    const deep = await source.expand(tree, tree.root, 20);
    expect(deep.complete).toBe(10);
    expect(deep.branches).toEqual([0, 1, 2, 3, 4]);
    expect(deep.lnWeights).toEqual(weights.map(Math.log));
    expect((await source.expand(tree, tree.child(tree.root, 1), 10)).complete).toBe(10);
  });

  it("finds recorded nodes by their shared ids, however the client numbered them", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    tree.descend(tree.root, [7, 7, 7]);
    const one = tree.child(tree.root, 1);
    expect(one).not.toBe(1);
    expect((await source.expand(tree, one, 10)).branches).toEqual([0, 1, 2, 3, 4]);
  });

  it("knows nothing below paths it did not record", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    for (const path of [[0], [1, 2], [9, 9, 9]]) {
      const reply = await source.expand(tree, tree.descend(tree.root, path), 10);
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
  it("names a node by its parent's id, and from the root once the server does not know it", async () => {
    const requests: { node: bigint; path: number[]; horizon: number }[] = [];
    const known = new Set<bigint>([0n]);
    const envelope = (flags: number, body: Uint8Array) => {
      const frame = new Uint8Array(5 + body.length);
      new DataView(frame.buffer).setUint8(0, flags);
      new DataView(frame.buffer).setUint32(1, body.length);
      frame.set(body, 5);
      return frame;
    };
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = new Uint8Array(await request.arrayBuffer());
        const { node, path, horizon } = fromBinary(ExpandRequestSchema, body.subarray(5));
        requests.push({ node, path: [...path], horizon });
        const end = (error?: object) => envelope(2, new TextEncoder().encode(JSON.stringify(error ? { error } : {})));
        const reply = known.has(node)
          ? [envelope(0, toBinary(ListingSchema, create(ListingSchema, { branches: [3], lnWeights: [0], lnLeftover: -Infinity }))), end()]
          : [end({ code: "not_found", message: "unknown node" })];
        return new Response(new Blob(reply), { headers: { "content-type": "application/connect+proto" } });
      },
    });
    try {
      const tree = new IntervalTree(1);
      const deep = tree.descend(tree.root, [5, 6, 7]);
      const source = new ConnectSource(`http://localhost:${server.port}/`);
      const reply = await source.expand(tree, deep, 8);
      expect(reply).toEqual({ branches: [3], lnWeights: [0], leftover: 0, complete: 8 });
      expect(requests).toEqual([
        { node: tree.hash(tree.parent(deep)!), path: [7], horizon: 8 },
        { node: 0n, path: [5, 6, 7], horizon: 8 },
      ]);
      known.add(tree.hash(tree.parent(deep)!));
      await source.expand(tree, deep, 8);
      expect(requests).toHaveLength(3);
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

describe("Expander", () => {
  it("refills capacity using current demand without another render", async () => {
    const tree = new IntervalTree(1);
    const calls: number[][] = [];
    const source: Source = {
      async expand(tree, node, horizon) {
        calls.push(tree.path(node));
        return rootReply(horizon);
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
      async expand(tree, node, horizon) {
        calls.push(tree.path(node));
        peak = Math.max(peak, ++active);
        await Bun.sleep(5);
        active--;
        return rootReply(horizon);
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
    await waitUntil(() => release !== undefined);
    pending.dispose();
    release(await rootReply(8));
    await Bun.sleep(5);
  });

  it("abandons work in flight when the view moves, without reporting it", async () => {
    const tree = new IntervalTree(1);
    const signals: AbortSignal[] = [];
    const errors: unknown[] = [];
    const source: Source = {
      expand: (_tree, _node, _horizon, signal) => new Promise((_resolve, reject) => {
        signals.push(signal!);
        signal!.addEventListener("abort", () => reject(signal!.reason));
      }),
    };
    const expander = new Expander(tree, [source]);
    expander.onError = (error) => errors.push(error);
    const old = tree.child(0, 1);
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
