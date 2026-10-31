import { describe, expect, it } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { Expander } from "../src/data/expand";
import { ConnectSource, type ExpandRequest, type Source, type Step, SnapshotSource, WHOLE } from "../src/data/source";
import type { Miss } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { ExpandRequestSchema, RunSchema, SnapshotSchema } from "../src/gen/natlas/natlas_pb";
import { TextLabels } from "../src/readings/labels";
import { run } from "./build";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];

/** Every node along ``paths`` records ``weights``, and the 0.14 below them stays unknown. */
function snapshot(labels = TextLabels.none(), paths: number[][] = [[], [1]]): SnapshotSource {
  const recordings = paths.map((path) => ({ path, run: run(weights) }));
  return new SnapshotSource(toBinary(SnapshotSchema, create(SnapshotSchema, { recordings })), labels);
}

async function steps(source: Source, tree: IntervalTree, request: Partial<ExpandRequest>): Promise<Step[]> {
  const result: Step[] = [];
  for await (const step of source.expand(tree, { node: ROOT, path: [], ...WHOLE, horizon: 8, ...request })) result.push(step);
  return result;
}

describe("SnapshotSource", () => {
  it("lists a window down to its horizon, as a server would", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    const [shallow] = await steps(source, tree, { horizon: 2 });
    expect(shallow!.run.branches).toEqual([0]);
    const [deep] = await steps(source, tree, { horizon: 4 });
    expect(deep!.run.branches).toEqual([0, 1, 2]);
    expect(Array.from(deep!.run.lnBelow).at(-1)).toBeCloseTo(Math.log(0.2), 12);
  });

  it("places each branch of a path, naming every node, and stops where the recording does", async () => {
    const source = snapshot(TextLabels.none(), [[], [1], [1, 4]]);
    const tree = new IntervalTree(1);
    const walk = await steps(source, tree, { path: [1, 4, 0] });
    expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [1], [1, 4]]);
    expect(walk.map(({ run }) => Array.from(run.branches))).toEqual([[1], [4], [0]]);
    expect(await steps(source, tree, { node: tree.descend(ROOT, [9, 9]) })).toEqual([]);
  });

  it("descends a window inside one child", async () => {
    const source = snapshot(TextLabels.none(), [[], [0]]);
    const tree = new IntervalTree(1);
    // (0.6, 0.9] lies inside branch 0's (0.5, 1].
    const walk = await steps(source, tree, { lnTop: Math.log(0.9), lnBottom: Math.log(0.6) });
    expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [0]]);
    expect(walk[0]!.run.branches).toEqual([0]);
  });

  it("refuses an older recording, and searches recorded paths by their text", async () => {
    expect(() => new SnapshotSource(new Uint8Array([10, 0]))).toThrow("migrate-snapshots");
    const source = snapshot(new TextLabels([new TextEncoder().encode("x"), new TextEncoder().encode("abc")]));
    expect(await source.search("abc")).toEqual([1]);
    expect(source.search("unrelated")).rejects.toThrow("No recorded sequence");
  });
});

describe("ConnectSource", () => {
  it("names a request by its node's id, and from the root once the server has not learned it", async () => {
    const requests: { node: bigint; path: number[] }[] = [];
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
        const { node, path } = fromBinary(ExpandRequestSchema, body.subarray(5));
        requests.push({ node, path: [...path] });
        const end = (error?: object) => envelope(2, new TextEncoder().encode(JSON.stringify(error ? { error } : {})));
        const reply = node === ROOT
          ? [...path.map((branch) => envelope(0, toBinary(RunSchema, create(RunSchema, { rank: 0, branches: [branch], lnBelow: [0, -1] })))),
            envelope(0, toBinary(RunSchema, create(RunSchema, run([1])))), end()]
          : [end({ code: "not_found", message: "unknown node" })];
        return new Response(new Blob(reply), { headers: { "content-type": "application/connect+proto" } });
      },
    });
    try {
      const tree = new IntervalTree(1);
      const deep = tree.descend(ROOT, [5, 6]);
      const walk = await steps(new ConnectSource(`http://localhost:${server.port}/`), tree, { node: deep, path: [7] });
      expect(requests).toEqual([{ node: deep, path: [7] }, { node: ROOT, path: [5, 6, 7] }]);
      expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [5], [5, 6], [5, 6, 7]]);
    } finally {
      server.stop(true);
    }
  });
});

/** A source counting requests, answering each with ``answer``. */
function counting(answer: (request: ExpandRequest) => Step[] | Promise<Step[]>): Source & { requests: ExpandRequest[] } {
  const requests: ExpandRequest[] = [];
  return {
    requests,
    async *expand(_tree, request) {
      requests.push(request);
      yield* await answer(request);
    },
  };
}

const rootMiss: Miss = { node: ROOT, branch: null, measure: 0, from: 0, lnTop: 0, lnBottom: -Infinity, horizon: 8, priority: 1 };

describe("Expander", () => {
  it("asks once for what a view lacks, and not again for what a source could not give", async () => {
    const tree = new IntervalTree(1);
    const empty = counting(() => []);
    const expander = new Expander(tree, [empty]);
    try {
      expander.request(() => [rootMiss, { ...rootMiss, priority: 0 }]);
      await waitUntil(() => expander.exhausted(ROOT, 0));
      expander.request(() => [rootMiss]);
      await Bun.sleep(5);
      expect(empty.requests).toHaveLength(1);
    } finally { expander.dispose(); }
  });

  it("names a child through its parent and refills capacity from current demand", async () => {
    const tree = new IntervalTree(1);
    const source = counting(({ node, path }) => {
      const target = path.length === 0 ? node : tree.child(node, path[0]!);
      return [{ node: target, run: run([0.5, 0.5]) }];
    });
    const expander = new Expander(tree, [source], 1);
    try {
      expander.request(() => tree.has(ROOT, 0) ? [{ ...rootMiss, branch: 1 }] : [rootMiss]);
      await waitUntil(() => tree.has(tree.child(ROOT, 1), 0));
      expect(source.requests.map(({ node, path }) => [node, path])).toEqual([[ROOT, []], [ROOT, [1]]]);
    } finally { expander.dispose(); }
  });

  it("walks once from the deepest node already placed", async () => {
    const recording = snapshot(TextLabels.none(), [[], [1], [1, 2], [1, 2, 3]]);
    const requests: ExpandRequest[] = [];
    const source: Source = {
      expand(tree, request) {
        requests.push(request);
        return recording.expand(tree, request);
      },
    };
    const tree = new IntervalTree(1);
    tree.insert(ROOT, 0, run(weights));
    const target = tree.descend(ROOT, [1, 2, 3]);
    const expander = new Expander(tree, [source]);
    try {
      expect(await expander.walk(ROOT, target, 0, 4)).toBe(true);
      expect(requests.map(({ node, path }) => [node, path])).toEqual([[tree.descend(ROOT, [1]), [2, 3]]]);
      expect(await expander.walk(ROOT, target, 0, 4)).toBe(true);
      expect(requests).toHaveLength(1);
      // Past what was recorded, the walk ends.
      expect(await expander.walk(ROOT, tree.descend(target, [0, 0]), 0, 4)).toBe(false);
    } finally { expander.dispose(); }
  });

  it("abandons work in flight when the view moves, without reporting it", async () => {
    const tree = new IntervalTree(1);
    const signals: AbortSignal[] = [];
    const errors: unknown[] = [];
    const source: Source = {
      // eslint-disable-next-line require-yield
      async *expand(_tree, _request, signal) {
        signals.push(signal!);
        await new Promise((_resolve, reject) => signal!.addEventListener("abort", () => reject(signal!.reason)));
      },
    };
    const expander = new Expander(tree, [source]);
    expander.onError = (error) => errors.push(error);
    expander.request(() => [rootMiss]);
    await waitUntil(() => signals.length === 1);
    expander.cancel();
    expander.request(() => [rootMiss]);
    await waitUntil(() => signals.length === 2);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[1]!.aborted).toBe(false);
    await Bun.sleep(5);
    expect(errors).toEqual([]);
    expander.dispose();
  });
});

async function waitUntil(condition: () => boolean): Promise<void> {
  const deadline = performance.now() + 1000;
  while (!condition()) {
    if (performance.now() > deadline) throw new Error("timed out waiting for the expander");
    await Bun.sleep(1);
  }
}
