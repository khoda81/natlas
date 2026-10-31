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

async function steps(source: Source, tree: IntervalTree, request: ExpandRequest): Promise<Step[]> {
  const result: Step[] = [];
  for await (const step of source.expand(tree, request)) result.push(step);
  return result;
}

const whole = (horizon: number, node = ROOT): ExpandRequest => ({ node, window: WHOLE, horizon });

describe("SnapshotSource", () => {
  it("lists a window down to its horizon, as a server would", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    const [shallow] = await steps(source, tree, whole(2));
    expect(shallow!.run.branches).toEqual([0]);
    const [deep] = await steps(source, tree, whole(4));
    expect(deep!.run.branches).toEqual([0, 1, 2]);
    expect(Array.from(deep!.run.lnBelow).at(-1)).toBeCloseTo(Math.log(0.2), 12);
  });

  it("places each branch of a path, naming every node, and stops where the recording does", async () => {
    const source = snapshot(TextLabels.none(), [[], [1], [1, 4]]);
    const tree = new IntervalTree(1);
    const walk = await steps(source, tree, { node: ROOT, path: [1, 4, 0], horizon: 8 });
    expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [1], [1, 4]]);
    expect(walk.map(({ run }) => Array.from(run.branches))).toEqual([[1], [4], [0]]);
    expect(await steps(source, tree, whole(8, tree.descend(ROOT, [9, 9])))).toEqual([]);
  });

  it("descends a window inside one child", async () => {
    const source = snapshot(TextLabels.none(), [[], [0]]);
    const tree = new IntervalTree(1);
    // (0.6, 0.9] lies inside branch 0's (0.5, 1].
    const walk = await steps(source, tree, { node: ROOT, window: { lnTop: Math.log(0.9), lnBottom: Math.log(0.6) }, horizon: 8 });
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
  it("names a request by its node's id, and by a path from the root once the server has not learned it", async () => {
    const requests: string[] = [];
    const learned = new Set([ROOT]);
    const names = new IntervalTree(1);
    const envelope = (flags: number, body: Uint8Array) => {
      const frame = new Uint8Array(5 + body.length);
      new DataView(frame.buffer).setUint8(0, flags);
      new DataView(frame.buffer).setUint32(1, body.length);
      frame.set(body, 5);
      return frame;
    };
    const message = (value: Parameters<typeof create<typeof RunSchema>>[1]) => envelope(0, toBinary(RunSchema, create(RunSchema, value)));
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = new Uint8Array(await request.arrayBuffer());
        const { node, target } = fromBinary(ExpandRequestSchema, body.subarray(5));
        const end = (error?: object) => envelope(2, new TextEncoder().encode(JSON.stringify(error ? { error } : {})));
        const respond = (chunks: BlobPart[]) => new Response(new Blob(chunks), { headers: { "content-type": "application/connect+proto" } });
        requests.push(target.case === "path" ? `${node === ROOT ? "root" : "deep"} path ${target.value.branches}` : `${node === ROOT ? "root" : "deep"} window`);
        if (!learned.has(node)) return respond([end({ code: "not_found", message: "unknown node" })]);
        if (target.case !== "path") return respond([message(run([1])), end()]);
        // Placing each branch teaches the server the nodes along the path.
        const { branches } = target.value;
        const placed = branches.map((branch, i) => {
          learned.add(names.descend(node, branches.slice(0, i + 1)));
          return message({ rank: 0, branches: [branch], lnBelow: [0, -1] });
        });
        return respond([...placed, message(run([1])), end()]);
      },
    });
    try {
      const source = new ConnectSource(`http://localhost:${server.port}/`);
      const tree = new IntervalTree(1);
      const deep = tree.descend(ROOT, [5, 6]);
      const walk = await steps(source, tree, { node: deep, path: [7], horizon: 8 });
      expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [5], [5, 6], [5, 6, 7]]);
      // A window on a node the server has not learned: a path teaches it, then the window.
      const other = tree.descend(ROOT, [9]);
      const window = await steps(source, tree, whole(8, other));
      expect(window.map(({ node }) => tree.path(node))).toEqual([[], [9], [9]]);
      expect(requests).toEqual(["deep path 7", "root path 5,6,7", "deep window", "root path 9", "deep window"]);
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

const rootMiss: Miss = { node: ROOT, target: ROOT, measure: 0, from: 0, to: null, lnTop: 0, lnBottom: -Infinity, horizon: 8, priority: 1 };

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

  it("asks for a child by a window on its parent, and refills capacity from current demand", async () => {
    const tree = new IntervalTree(1);
    const child = tree.child(ROOT, 1);
    const source = counting(() => tree.has(ROOT, 0)
      ? [{ node: ROOT, run: run([0.5], [1], 1, 0) }, { node: child, run: run([0.5, 0.5]) }]
      : [{ node: ROOT, run: run([0.5, 0.5]) }]);
    const expander = new Expander(tree, [source], 1);
    try {
      expander.request(() => tree.has(ROOT, 0) ? [{ ...rootMiss, target: child, lnBottom: Math.log(0.5) }] : [rootMiss]);
      await waitUntil(() => tree.has(child, 0));
      expect(source.requests.map((request) => "window" in request ? [request.node, request.window.lnBottom] : null))
        .toEqual([[ROOT, -Infinity], [ROOT, Math.log(0.5)]]);
    } finally { expander.dispose(); }
  });

  it("asks again for a gap that an answer split, keyed by both its ends", async () => {
    const tree = new IntervalTree(1);
    tree.insert(ROOT, 0, run([0.5, 0.25], [0, 1], 0));
    // The answer lands inside the gap: ranks 2–4 stay unknown above it.
    const source = counting(() => [{ node: ROOT, run: run([0.01], [9], 5, 0.1) }]);
    const expander = new Expander(tree, [source]);
    const gap = (to: number | null): Miss => ({ ...rootMiss, from: 2, to, lnTop: Math.log(0.25) });
    try {
      expander.request(() => [gap(tree.raw.entry(ROOT, 0, 0, 9).length === 0 ? null : 5)]);
      await waitUntil(() => source.requests.length === 2);
      expect(source.requests).toHaveLength(2);
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
      expect(requests.map((request) => "path" in request ? [request.node, request.path] : null)).toEqual([[tree.descend(ROOT, [1]), [2, 3]]]);
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
