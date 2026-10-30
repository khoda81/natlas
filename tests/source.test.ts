import { describe, expect, it } from "bun:test";
import { create, toBinary } from "@bufbuild/protobuf";
import { Code, ConnectError } from "@connectrpc/connect";
import { encode } from "../src/codec";
import { Expander } from "../src/expand";
import { SnapshotSchema } from "../src/gen/natlas/v1/natlas_pb";
import { type Child, type Reply, type Source, SnapshotSource } from "../src/source";
import { type Handle, IntervalTree, type TreeNode } from "../src/tree";

const weights = [0.5, 0.2, 0.1, 0.05, 0.01];
const ROOT = 7n;
const CHILD = 8n;

function snapshot(): SnapshotSource {
  const expansions = [
    { request: { horizon: 10 }, node: ROOT },
    { request: { node: { parent: ROOT, label: 1 }, horizon: 10 }, node: CHILD },
  ].map((expansion) => ({ ...expansion, edges: encode([0, 1, 2, 3, 4], weights.map(Math.log), 10) }));
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
    const shallow = await source.expand(undefined, 4);
    expect(shallow.node).toBe(ROOT);
    expect(shallow.complete).toBe(4);
    expect(await labels(shallow)).toEqual([0, 1, 2]);
    const deep = await source.expand(undefined, 20);
    expect(deep.complete).toBe(10);
    expect(await labels(deep)).toEqual([0, 1, 2, 3, 4]);
    expect((await source.expand({ parent: ROOT, label: 1 }, 10)).node).toBe(CHILD);
  });

  it("knows nothing about nodes it did not record, nor below them", async () => {
    const source = snapshot();
    const reply = await source.expand({ parent: ROOT, label: 0 }, 10);
    expect([ROOT, CHILD]).not.toContain(reply.node);
    expect(reply.complete).toBe(0);
    expect(await labels(reply)).toEqual([]);
    const below = await source.expand({ parent: reply.node, label: 1 }, 10);
    expect(below.node).toBe(reply.node);
    expect(below.complete).toBe(0);
  });
});

/** Wait for ``measure``'s listing of ``node`` from ``expander``. */
async function fetched(tree: IntervalTree, expander: Expander, node: TreeNode, horizon = 2): Promise<void> {
  const landed = new Promise<void>((resolve) => {
    const stop = tree.subscribe(() => (stop(), resolve()));
  });
  expander.request([{ node, measure: 0, horizon, priority: 1 }]);
  await landed;
}

describe("Expander", () => {
  it("fetches wanted listings once, with headroom", async () => {
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [snapshot()]);
    const want = { node: tree.root, measure: 0, horizon: 2, priority: 1 };
    const landed = fetched(tree, expander, tree.root);
    expander.request([want]);
    await landed;
    const root = tree.root.listing(0)!;
    if (root.radix) throw Error("Expected ranked listing");
    expect(root.horizon).toBe(6);
    expect(root.labels).toEqual([0, 1, 2, 3]);
    expect(root.leftover).toBeCloseTo(0.15, 6);
    expect(tree.root.handle(0)).toBe(ROOT);
  });
});

/**
 * A server that, like the live one, hands out handles as nodes are expanded,
 * knows only those, and forgets them on a restart. Every node has children
 * 0 to 4.
 */
class Forgetful implements Source {
  known = new Set<Handle>([1n]);
  calls: string[] = [];

  async expand(node: Child | undefined, horizon: number): Promise<Reply> {
    this.calls.push(node === undefined ? `root@${horizon}` : `${node.parent}/${node.label}@${horizon}`);
    if (node !== undefined && !this.known.has(node.parent)) throw new ConnectError("forgotten", Code.NotFound);
    const handle = node === undefined ? 1n : node.parent * 10n + BigInt(node.label);
    this.known.add(handle);
    return { ...(await snapshot().expand(undefined, horizon)), node: handle, complete: horizon };
  }

  restart(): void {
    this.known = new Set([1n]);
    this.calls = [];
  }
}

describe("Expander naming nodes", () => {
  it("names a node's unnamed ancestors from the nearest named one, asking for handles alone", async () => {
    const tree = new IntervalTree(1);
    const server = new Forgetful();
    const expander = new Expander(tree, [server]);
    const deep = tree.child(tree.child(tree.child(0, 1).id, 2).id, 3);
    await fetched(tree, expander, deep);
    expect(server.calls).toEqual(["root@0", "1/1@0", "11/2@0", "112/3@6"]);
    expect(deep.handle(0)).toBe(1123n);
    expect(tree.root.listing(0)).toBeUndefined();
  });

  it("after a restart, names forgotten ancestors again, going up only as far as needed", async () => {
    const tree = new IntervalTree(1);
    const server = new Forgetful();
    const expander = new Expander(tree, [server]);
    const child = tree.child(0, 1);
    const grandchild = tree.child(child.id, 2);
    await fetched(tree, expander, grandchild);
    server.restart();
    const leaf = tree.child(grandchild.id, 3);
    await fetched(tree, expander, leaf, 10);
    expect(server.calls).toEqual(["112/3@14", "11/2@0", "1/1@0", "11/2@0", "112/3@14"]);
    expect(leaf.slots({ p: 0, q: 0 }).labels).toEqual([0, 1, 2, 3, 4]);

    // What the walk named, the server knows again.
    server.calls = [];
    await fetched(tree, expander, tree.child(grandchild.id, 4));
    expect(server.calls).toEqual(["112/4@6"]);
  });
});
