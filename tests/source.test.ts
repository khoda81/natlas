import { describe, expect, it } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { Link } from "../src/data/link";
import { type Connection, type Demand, LiveServer, RecordedServer, type Server, type Update } from "../src/data/source";
import type { View } from "../src/core/layout";
import { IntervalTree, type NodeId, ROOT } from "../src/core/tree";
import { DemandSchema, SnapshotSchema, UpdatesSchema } from "../src/gen/natlas/natlas_pb";
import { TextLabels } from "../src/readings/labels";
import { answer, run } from "./build";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];
const P = { p: "m", q: "m" };

/** Every node along ``paths`` records ``weights``, and the 0.14 below them stays unknown. */
function recording(paths: number[][] = [[], [1]]): Uint8Array {
  const recordings = paths.map((path) => ({ path, children: answer(weights) }));
  return toBinary(SnapshotSchema, create(SnapshotSchema, { recordings }));
}

/** The whole of a node, ``rows`` tall, ``right`` nats deep. */
const whole = (node: NodeId, rows: number, right: number): View => ({ node, top: new Uint8Array(), lnHeight: 0, rows, right });

/** Everything a server says to one demand. */
async function told(server: Server, demand: Demand): Promise<Update[]> {
  const updates: Update[] = [];
  const connection = server.connect((batch) => updates.push(...batch), (error) => { throw error; });
  connection.demand(demand);
  await Bun.sleep(1);
  connection.close();
  return updates;
}

/** The branches an answer places, in rank order. */
const placed = (update: Update | undefined) => update?.kind === "answer" ? update.runs.flatMap((run) => Array.from(run.branches)) : undefined;

describe("RecordedServer", () => {
  it("answers a view as a server would, and lacks what it did not record", async () => {
    const tree = new IntervalTree();
    const child = (branch: number) => tree.descend(ROOT, [branch]);
    const updates = await told(new RecordedServer([{ name: "m", snapshot: recording() }]), { specs: P, view: whole(ROOT, 10, 10), path: null });
    // Ten rows: 5, 2 and 1 on branches 0, 1 and 2; the thinner rest sampled.
    const [root] = updates;
    expect(placed(root)!.slice(0, 3)).toEqual([0, 1, 2]);
    const below = updates.find((update) => update.kind === "answer" && update.node === child(1));
    expect(below).toMatchObject({ parent: ROOT, branch: 1 });
    const lacking = updates.flatMap((update) => update.kind === "lacks" ? [update.node] : []);
    expect(lacking).toContain(child(0));
    expect(lacking).not.toContain(child(1));
  });

  it("answers each measure under the spec the page named it by", async () => {
    const other = toBinary(SnapshotSchema, create(SnapshotSchema, { recordings: [{ path: [], children: answer([0.1, 0.9]) }] }));
    const server = new RecordedServer([{ name: "m", snapshot: recording() }, { name: "T = 2", snapshot: other }]);
    const updates = await told(server, { specs: { p: "T%20%3D%202", q: "m" }, view: whole(ROOT, 10, 10), path: null });
    const roots = updates.filter((update) => update.kind === "answer" && update.node === ROOT);
    const first = (update: Update) => update.kind === "answer" ? Math.round(Math.exp(update.runs[0]!.lnWeights[0]!) * 1e6) / 1e6 : NaN;
    expect(new Map(roots.map((update) => [update.kind === "answer" && update.measure, first(update)]))).toEqual(new Map([["T%20%3D%202", 0.1], ["m", 0.5]]));
  });

  it("places a path as far as the recording goes", async () => {
    const tree = new IntervalTree();
    const path = { node: ROOT, branches: [1, 4, 0, 2] };
    const updates = await told(new RecordedServer([{ name: "m", snapshot: recording([[], [1], [1, 4]]) }]), { specs: P, view: null, path });
    const answers = updates.filter((update) => update.kind === "answer");
    expect(answers.map((update) => update.node)).toEqual([ROOT, tree.descend(ROOT, [1]), tree.descend(ROOT, [1, 4])]);
    expect(answers.map(placed)).toEqual([[1], [4], [0]]);
    expect(updates.at(-1)).toEqual({ kind: "placed", node: ROOT, branches: [1, 4, 0, 2] });
  });

  it("refuses an older recording, and searches the recorded paths", async () => {
    expect(() => new RecordedServer([{ name: "m", snapshot: new Uint8Array([10, 0]) }])).toThrow("migrate-snapshots");
    const labels = new TextLabels([new TextEncoder().encode("x"), new TextEncoder().encode("abc")]);
    const server = new RecordedServer([{ name: "m", snapshot: recording() }], labels);
    expect(await server.search("abc")).toEqual([1]);
    expect(server.search("unrelated")).rejects.toThrow("No recorded sequence");
  });
});

describe("LiveServer", () => {
  it("says what the page looks at over one socket, and hears the server's updates", async () => {
    const demands: Demand[] = [];
    const server = Bun.serve({
      port: 0,
      fetch(request, bun) {
        return bun.upgrade(request) ? undefined : new Response("not a socket", { status: 400 });
      },
      websocket: {
        message(socket, frame) {
          const { p, q, path } = fromBinary(DemandSchema, frame as Uint8Array);
          demands.push({ specs: { p, q }, view: null, path: path === undefined ? null : { node: path.node, branches: [...path.branches] } });
          const answer = create(UpdatesSchema, {
            updates: [
              { kind: { case: "answer", value: { measure: "m", node: 5n, parent: 0n, branch: 1, children: { runs: [run([0.5, 0.5])], count: 2 } } } },
              { kind: { case: "placed", value: { node: 0n, branches: [1] } } },
            ],
          });
          socket.send(toBinary(UpdatesSchema, answer));
        },
      },
    });
    try {
      const updates: Update[] = [];
      const connection: Connection = new LiveServer(`http://localhost:${server.port}/`).connect((batch) => updates.push(...batch), (error) => { throw error; });
      connection.demand({ specs: P, view: null, path: { node: ROOT, branches: [1] } });
      const deadline = performance.now() + 2000;
      while (updates.length < 2 && performance.now() < deadline) await Bun.sleep(5);
      connection.close();
      expect(demands).toEqual([{ specs: P, view: null, path: { node: ROOT, branches: [1] } }]);
      expect(updates[0]).toMatchObject({ kind: "answer", measure: "m", node: 5n, parent: 0n, branch: 1, count: 2 });
      expect(updates[1]).toEqual({ kind: "placed", node: 0n, branches: [1] });
    } finally {
      server.stop(true);
    }
  });
});

/** A server the test speaks for: what it was told, and a way to answer. */
function scripted(): Server & { demands: Demand[]; say(updates: Update[]): void } {
  let hear: (updates: Update[]) => void = () => {};
  const server = {
    demands: [] as Demand[],
    say: (updates: Update[]) => hear(updates),
    connect(updates: (updates: Update[]) => void): Connection {
      hear = updates;
      return { demand: (demand) => server.demands.push(demand), close() {} };
    },
  };
  return server;
}

describe("Link", () => {
  it("says what the page looks at when it changes, and puts answers into the tree", () => {
    const tree = new IntervalTree();
    const server = scripted();
    const link = new Link(tree, server);
    link.look(P, whole(ROOT, 10, 1));
    link.look(P, whole(ROOT, 10, 1));
    expect(server.demands).toHaveLength(1);
    const child = tree.child(ROOT, 1);
    // A node named only by its parent and branch is learned from them.
    const grandchild = tree.child(child, 0);
    server.say([{ kind: "answer", measure: "m", node: grandchild, parent: child, branch: 0, ...answer([1]), count: 1 }]);
    expect(tree.has(grandchild, tree.measure("m"))).toBe(true);
    server.say([{ kind: "lacks", measure: "m", node: child }]);
    expect(link.lacks(child, tree.measure("m"))).toBe(true);
    link.dispose();
  });

  it("places a path first, and says when it is placed or another took its place", async () => {
    const tree = new IntervalTree();
    const server = scripted();
    const link = new Link(tree, server);
    link.look(P, null);
    const first = link.place(ROOT, [2]);
    const second = link.place(ROOT, [1]);
    expect(await first).toBeNull();
    expect(server.demands.at(-1)!.path).toEqual({ node: ROOT, branches: [1] });
    server.say([{ kind: "answer", measure: "m", node: ROOT, parent: null, branch: null, runs: [run([0.5], [1], 0, 0.5)], count: null }]);
    server.say([{ kind: "placed", node: ROOT, branches: [1] }]);
    expect(await second).toBe(true);
    // Placed, the path is no longer said.
    expect(server.demands.at(-1)!.path).toBeNull();
    link.dispose();
  });

  it("names a node the server has not learned down its path from the root", () => {
    const tree = new IntervalTree();
    const deep = tree.descend(ROOT, [3, 1, 4]);
    const server = scripted();
    const link = new Link(tree, server);
    link.look(P, whole(deep, 10, 1));
    server.say([{ kind: "unknown", node: deep }]);
    expect(server.demands.at(-1)).toMatchObject({ view: { node: deep }, path: { node: ROOT, branches: [3, 1, 4] } });
    void link.place(deep, [9]);
    expect(server.demands.at(-1)!.path).toEqual({ node: ROOT, branches: [3, 1, 4, 9] });
    // Once a path down to it is placed, the server knows it.
    server.say([{ kind: "placed", node: ROOT, branches: [3, 1, 4, 9] }]);
    void link.place(deep, [9, 9]);
    expect(server.demands.at(-1)!.path).toEqual({ node: deep, branches: [9, 9] });
    link.dispose();
  });

  it("reports a server's failure", () => {
    const server = scripted();
    const link = new Link(new IntervalTree(), server);
    const errors: unknown[] = [];
    link.onError = (error) => errors.push(error);
    server.say([{ kind: "failed", measure: "m", node: ROOT, message: "too long" }]);
    expect(String(errors[0])).toContain("too long");
    link.dispose();
  });
});
