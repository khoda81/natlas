import { describe, expect, it } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { Expander } from "../src/data/expand";
import { ConnectSource, type ExpandRequest, type Source, type Step, SnapshotSource } from "../src/data/source";
import type { Miss } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { ChildrenSchema, ExpandRequestSchema, SnapshotSchema } from "../src/gen/natlas/natlas_pb";
import { TextLabels } from "../src/readings/labels";
import { answer, at, run } from "./build";
import type { Part } from "../src/core/layout";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];

/** Every node along ``paths`` records ``weights``, and the 0.14 below them stays unknown. */
function snapshot(labels = TextLabels.none(), paths: number[][] = [[], [1]]): SnapshotSource {
  const recordings = paths.map((path) => ({ path, children: answer(weights) }));
  return new SnapshotSource(toBinary(SnapshotSchema, create(SnapshotSchema, { recordings })), labels);
}

async function steps(source: Source, tree: IntervalTree, request: ExpandRequest): Promise<Step[]> {
  const result: Step[] = [];
  for await (const step of source.expand(tree, request)) result.push(step);
  return result;
}

/** Reads of the node itself. */
const points = (points: Part[]) => ({ path: [], within: { points } });
const branches = (branches: number[]) => ({ path: [], within: { branches } });

/** The branches a step places, in rank order. */
const placed = (step: Step | undefined) => step?.runs.flatMap((run) => Array.from(run.branches));

describe("SnapshotSource", () => {
  it("places the children holding points, once each, as a server would", async () => {
    const source = snapshot();
    const tree = new IntervalTree(1);
    // 0.9 and 0.6 lie in branch 0's (0.5, 1], 0.4 in branch 1's (0.3, 0.5],
    // and 0.05 in the unrecorded rest.
    const [step] = await steps(source, tree, { node: ROOT, ...points(at(0.9, 0.6, 0.4, 0.05)) });
    expect(placed(step)).toEqual([0, 1]);
    expect(step!.runs).toHaveLength(1);
  });

  it("places each branch of a path, naming every node, and stops where the recording does", async () => {
    const source = snapshot(TextLabels.none(), [[], [1], [1, 4]]);
    const tree = new IntervalTree(1);
    const walk = await steps(source, tree, { node: ROOT, path: [1, 4, 0, 2] });
    expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [1], [1, 4]]);
    expect(walk.map(placed)).toEqual([[1], [4], [0]]);
    expect(await steps(source, tree, { node: tree.descend(ROOT, [9, 9]), ...points(at(0.4)) })).toEqual([]);
    // Down a path and then points: the path's branch, then the node it reaches.
    const both = await steps(source, tree, { node: ROOT, path: [1], within: { points: at(0.9, 0.4) } });
    expect(both.map(({ node }) => tree.path(node))).toEqual([[], [1]]);
    expect(both.map(placed)).toEqual([[1], [0, 1]]);
  });

  it("descends where all the points fall in one child, and reads branches alone", async () => {
    const source = snapshot(TextLabels.none(), [[], [0]]);
    const tree = new IntervalTree(1);
    const walk = await steps(source, tree, { node: ROOT, ...points(at(0.9, 0.6)) });
    expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [0]]);
    expect(placed(walk[0])).toEqual([0]);
    const [listed] = await steps(source, tree, { node: ROOT, ...branches([4, 2, 9]) });
    expect(placed(listed)).toEqual([2, 4]);
  });

  it("refuses an older recording, and searches recorded paths by their text", async () => {
    expect(() => new SnapshotSource(new Uint8Array([10, 0]))).toThrow("migrate-snapshots");
    const source = snapshot(new TextLabels([new TextEncoder().encode("x"), new TextEncoder().encode("abc")]));
    expect(await source.search("abc")).toEqual([1]);
    expect(source.search("unrelated")).rejects.toThrow("No recorded sequence");
  });
});

describe("ConnectSource", () => {
  it("names a request by its node's id, and from the root in one request once the server has not learned it", async () => {
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
    const message = (value: Parameters<typeof create<typeof ChildrenSchema>>[1]) => envelope(0, toBinary(ChildrenSchema, create(ChildrenSchema, value)));
    const server = Bun.serve({
      port: 0,
      async fetch(request) {
        const body = new Uint8Array(await request.arrayBuffer());
        const { node, path, within } = fromBinary(ExpandRequestSchema, body.subarray(5));
        const end = (error?: object) => envelope(2, new TextEncoder().encode(JSON.stringify(error ? { error } : {})));
        const respond = (chunks: BlobPart[]) => new Response(new Blob(chunks), { headers: { "content-type": "application/connect+proto" } });
        requests.push([node === ROOT ? "root" : "deep", ...(path.length ? [`path ${path}`] : []), ...(within.case ? [within.case] : [])].join(" "));
        if (!learned.has(node)) return respond([end({ code: "not_found", message: "unknown node" })]);
        // Placing each branch teaches the server the nodes along the path.
        const chunks = path.map((branch, i) => {
          learned.add(names.descend(node, path.slice(0, i + 1)));
          return message({ runs: [{ rank: 0, branches: [branch], lnBelow: [0, -1] }] });
        });
        return respond([...chunks, ...(within.case ? [message(answer([1]))] : []), end()]);
      },
    });
    try {
      const source = new ConnectSource(`http://localhost:${server.port}/`);
      const tree = new IntervalTree(1);
      const deep = tree.descend(ROOT, [5, 6]);
      const walk = await steps(source, tree, { node: deep, path: [7] });
      expect(walk.map(({ node }) => tree.path(node))).toEqual([[], [5], [5, 6]]);
      // Points on a node the server has not learned: down its path from the root, then the points.
      const other = tree.descend(ROOT, [9]);
      const sampled = await steps(source, tree, { node: other, ...points(at(0.4)) });
      expect(sampled.map(({ node }) => tree.path(node))).toEqual([[], [9]]);
      expect(requests).toEqual(["deep path 7", "root path 5,6,7", "deep points", "root path 9 points"]);
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

const rootMiss: Miss = { node: ROOT, measure: 0, read: points(at(0.6)), priority: 1 };

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

  it("asks a node again after each answer that changes it", async () => {
    const tree = new IntervalTree(1);
    // The first two answers each place one more child; the third places
    // the second again, changing nothing.
    const answers = [run([0.5], [7], 0, 0.5), run([0.3], [8], 1, 0.2)];
    const source = counting(({ node }) => [{ node, runs: [answers[Math.min(1, source.requests.length - 1)]!] }]);
    const expander = new Expander(tree, [source]);
    try {
      expander.request(() => [rootMiss]);
      await waitUntil(() => source.requests.length === 3);
      await Bun.sleep(5);
      expect(source.requests).toHaveLength(3);
    } finally { expander.dispose(); }
  });

  it("asks for a child down its branch from its parent, and refills capacity from current demand", async () => {
    const tree = new IntervalTree(1);
    const child = tree.child(ROOT, 1);
    const source = counting(() => tree.has(ROOT, 0)
      ? [{ node: ROOT, runs: [run([0.5], [1], 1, 0)] }, { node: child, runs: [run([0.5, 0.5])] }]
      : [{ node: ROOT, runs: [run([0.5, 0.5])] }]);
    const expander = new Expander(tree, [source], 1);
    try {
      const down: Miss = { ...rootMiss, read: { path: [1], within: { points: at(0.25) } } };
      expander.request(() => tree.has(ROOT, 0) ? [down] : [rootMiss]);
      await waitUntil(() => tree.has(child, 0));
      expect(source.requests).toEqual([{ node: ROOT, ...rootMiss.read }, { node: ROOT, ...down.read }]);
    } finally { expander.dispose(); }
  });

  it("reads a node a source already answered for at once, however many new nodes it is computing", async () => {
    const tree = new IntervalTree(1);
    tree.insert(ROOT, 0, { runs: [run([0.5, 0.2], [1, 2], 0, 0.3)] });
    // New nodes never answer; the root, already held, does.
    const source = counting(({ node, path }) => path.length === 0 ? [{ node, runs: [run([0.1], [3], 2, 0.2)] }] : new Promise<Step[]>(() => {}));
    const expander = new Expander(tree, [source], 1);
    const down = (branch: number): Miss => ({ node: ROOT, measure: 0, read: { path: [branch], within: { points: at(0.5) } }, priority: 9 });
    try {
      expander.request(() => [down(1), down(2), rootMiss]);
      await waitUntil(() => tree.entry(ROOT, { p: 0, q: 0 }, 3) !== undefined);
      // One new node at a time, and the root beside it.
      expect(source.requests.map(({ path }) => path)).toEqual([[1], []]);
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
    tree.insert(ROOT, 0, answer(weights));
    const target = tree.descend(ROOT, [1, 2, 3]);
    const expander = new Expander(tree, [source]);
    try {
      expect(await expander.walk(ROOT, target, 0)).toBe(true);
      expect(requests).toEqual([{ node: tree.descend(ROOT, [1]), path: [2, 3] }]);
      expect(await expander.walk(ROOT, target, 0)).toBe(true);
      expect(requests).toHaveLength(1);
      // Past what was recorded, the walk ends.
      expect(await expander.walk(ROOT, tree.descend(target, [0, 0]), 0)).toBe(false);
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
