import { describe, expect, it } from "bun:test";
import { Expander, type Want, wants } from "../src/expand";
import { type Camera, clamp, layout, normalize, place, repair } from "../src/layout";
import { RadixCDF, type QuantileReply } from "../src/radix";
import { Refiner } from "../src/refine";
import type { Child, Reply, Source } from "../src/source";
import { IntervalTree, type TreeNode } from "../src/tree";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

async function until(condition: () => boolean, timeout = 3000): Promise<void> {
  const deadline = performance.now() + timeout;
  while (!condition()) {
    if (performance.now() >= deadline) throw Error("Scheduler did not progress");
    await Bun.sleep(1);
  }
}

/** Exact inverse CDF for a small integer distribution, independent of the client. */
function page(weights: bigint[], prefix: Uint8Array): QuantileReply {
  const total = weights.reduce((a, b) => a + b, 0n);
  const a = [...prefix].reduce((a, b) => a * 256n + BigInt(b), 0n) * 256n;
  const denominator = 256n ** BigInt(prefix.length + 1);
  const ids = Array.from({ length: 257 }, (_, j) => {
    let sum = 0n;
    for (let id = 0; id < weights.length; id++) {
      sum += weights[id]!;
      if (sum * denominator > (a + BigInt(j)) * total) return id;
    }
    return weights.length;
  });
  return { node: 7n, firstBranchId: ids[0]!, delta: ids.slice(1).map((id, j) => BigInt(id - ids[j]!)), terminal: false };
}

/** A uniform distribution's inverse CDF is floor(u * vocabulary), including its sentinel. */
function uniformPage(vocabulary: number, prefix: Uint8Array = new Uint8Array()): QuantileReply {
  const a = [...prefix].reduce((a, b) => a * 256n + BigInt(b), 0n) * 256n;
  const denominator = 256n ** BigInt(prefix.length + 1);
  const ids = Array.from({ length: 257 }, (_, j) => Number((a + BigInt(j)) * BigInt(vocabulary) / denominator));
  return { node: 7n, firstBranchId: ids[0]!, delta: ids.slice(1).map((id, j) => BigInt(id - ids[j]!)), terminal: false };
}

class ManualSource implements Source {
  readonly requests: { node: Child | undefined; reply: ReturnType<typeof deferred<Reply>> }[] = [];
  active = 0;
  peak = 0;

  async expand(node: Child | undefined): Promise<Reply> {
    const reply = deferred<Reply>();
    this.requests.push({ node, reply });
    this.peak = Math.max(this.peak, ++this.active);
    try {
      return await reply.promise;
    } finally {
      this.active--;
    }
  }

  finish(index: number): void {
    this.requests[index]!.reply.resolve({
      node: BigInt(index + 100),
      complete: 10,
      edges: (async function* () {})(),
    });
  }
}

/** Resolve known integer distributions so the geometry has an independent oracle. */
function sampled(tree: IntervalTree, node: TreeNode, measure: number, weights: bigint[]): void {
  const state = new RadixCDF();
  state.insert([], page(weights, new Uint8Array()));
  for (let label = 0; label < weights.length; label++) {
    for (let count = 0; state.bounds(label).upper > state.bounds(label).lower; count++) {
      if (count > 32) throw Error("Test distribution did not converge");
      const prefix = state.nextBoundary(label)!;
      state.insert(prefix, page(weights, new Uint8Array(prefix)));
    }
  }
  publish(tree, node, measure, state);
}

function publish(tree: IntervalTree, node: TreeNode, measure: number, state: RadixCDF): void {
  tree.setListing(node.id, measure, { horizon: 0, radix: state });
}

const quantileSource: Source = {
  async expand() { throw Error("v1 must not be called"); },
  async quantile() { throw Error("Demand calculation must not fetch"); },
};

function splitTree(q: bigint[] = [1n, 1n]) {
  const tree = new IntervalTree(2);
  sampled(tree, tree.root, 0, [1n, 1n]);
  sampled(tree, tree.root, 1, [1n, 1n]);
  const left = tree.child(0, 0), right = tree.child(0, 1);
  for (const child of [left, right]) sampled(tree, child, 0, [1n, 1n]);
  sampled(tree, left, 1, [1n, 1n]);
  sampled(tree, right, 1, q);
  return { tree, left, right };
}

describe("visible nats priority", () => {
  const lens = { p: 0, q: 1 };
  const sources = [quantileSource, quantileSource];

  it("ranks equal-height distributions by their visible cross-entropy", () => {
    const { tree, left, right } = splitTree([1n, 255n]);
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const view = { width: 1000, height: 800 };
    const demand = wants(tree, lens, cam, view, layout(tree, lens, cam, view), sources);
    const priority = (node: TreeNode) => demand.find((want) => want.node === node)!.priority;
    expect(priority(left)).toBeCloseTo(0.5 * Math.LN2, 8);
    expect(priority(right)).toBeCloseTo(0.25 * (-Math.log(1 / 256) - Math.log(255 / 256)), 8);
    expect(priority(right)).toBeGreaterThan(priority(left));
  });

  it("counts only the visible fraction of a partially clipped node", () => {
    const { tree, left, right } = splitTree();
    const cam: Camera = { x: Math.LN2, k: 100, anchor: { node: 0, slot: 1 }, a: 0.75 };
    const view = { width: 1000, height: 800 };
    const demand = wants(tree, lens, cam, view, layout(tree, lens, cam, view), sources);
    expect(demand.find((want) => want.node === left)!.priority).toBeCloseTo(0.75 * Math.LN2, 8);
    expect(demand.find((want) => want.node === right)!.priority).toBeCloseTo(0.25 * Math.LN2, 8);
  });

  it("clips box widths to both viewport edges", () => {
    const { tree, left, right } = splitTree();
    const cam: Camera = { x: Math.LN2 * 1.5, k: 100, anchor: { node: 0, slot: 1 }, a: 0.5 };
    const view = { width: 20, height: 800 };
    const demand = wants(tree, lens, cam, view, layout(tree, lens, cam, view), sources);
    // Both nodes' boxes extend beyond the left and right edges; each covers
    // half the view, whose entire horizontal span is 0.2 nats.
    expect(demand.find((want) => want.node === left)!.priority).toBeCloseTo(0.1, 8);
    expect(demand.find((want) => want.node === right)!.priority).toBeCloseTo(0.1, 8);
  });

  it("gives an unfetched continuation the visible space after its offscreen box", () => {
    const tree = new IntervalTree(1);
    sampled(tree, tree.root, 0, [1n, 1n]);
    const same = { p: 0, q: 0 };
    const cam: Camera = { x: Math.LN2, k: 100, anchor: { node: 0, slot: 0 }, a: -0.75 };
    const view = { width: 1000, height: 800 };
    const demand = wants(tree, same, cam, view, layout(tree, same, cam, view), [quantileSource]);
    expect(demand.find((want) => want.node === tree.child(0, 0))!.priority).toBeCloseTo(2.5, 8);
    expect(demand.find((want) => want.node === tree.child(0, 1))!.priority).toBeCloseTo(7.5, 8);
  });

  it("fetches the visible node before offscreen ancestors beyond 60 nats", async () => {
    const tree = new IntervalTree(1);
    const paths = new Map<bigint, number[]>([[0n, []]]);
    const handles = new Map<string, bigint>([["", 0n]]);
    const calls: number[][] = [];
    const source: Source = {
      async expand() { throw Error("v1 must not be called"); },
      async quantile(child, prefix) {
        const path = child ? [...paths.get(child.parent)!, child.label] : [];
        let handle = handles.get(path.join(","));
        if (handle === undefined) {
          handle = BigInt(handles.size);
          handles.set(path.join(","), handle);
          paths.set(handle, path);
        }
        calls.push(path);
        return { ...page([1n, 999n, 3n], prefix), node: handle };
      },
    };
    const expander = new Expander(tree, [source], 1);
    try {
      let target = tree.root;
      for (let i = 0; i < 9; i++) {
        await expander.reach(target, 0, 0);
        target = tree.child(target.id, 0);
      }
      const same = { p: 0, q: 0 };
      const { lnMass, x1 } = place(tree, same, target.id);
      const slot = tree.node(target.parent!).slots(same).slotOf(target.label)!;
      const cam: Camera = {
        x: Math.min(-lnMass, x1), k: 364,
        anchor: normalize(tree, same, { node: target.parent!, slot }), a: 0,
      };
      cam.a = 0.5 - Math.exp(lnMass + cam.x) / 2;
      const view = { width: 1864, height: 870 };
      const demand = () => {
        const current = { ...cam, anchor: { ...cam.anchor } };
        repair(tree, same, current);
        clamp(tree, same, current);
        return wants(tree, same, current, view, layout(tree, same, current, view), [source]);
      };
      const wanted = demand();
      expect(cam.x).toBeGreaterThan(60);
      expect(wanted.find((want) => want.node === tree.root)!.priority).toBe(0);
      expect(wanted.find((want) => want.node === target)!.priority).toBeGreaterThan(0);
      for (const want of wanted) expect(want.priority).toBeLessThanOrEqual(view.width / cam.k + 1e-10);
      const before = calls.length;
      expander.request(demand);
      await until(() => calls.length > before);
      expect(calls[before]).toEqual(Array(9).fill(0));
    } finally {
      expander.dispose();
    }
  });
});

describe("child expansion confidence", () => {
  it("resolves an ambiguous cutoff even after the ordinary p pixel budget is met", async () => {
    const tree = new IntervalTree(2);
    const weights = [2001n, 797999n]; // First child's actual height is 2.001px.
    const calls: number[][] = [];
    const source: Source = {
      async expand() { throw Error("v1 must not be called"); },
      async quantile(_node, prefix) {
        calls.push([...prefix]);
        return page(weights, prefix);
      },
    };
    const refiner = new Refiner(tree, [source, source]);
    const want = {
      node: tree.root, lens: { p: 0, q: 1 }, logHeight: Math.log(800),
      k: 100, right: 10, labels: new Set<number>(),
    };
    await refiner.next(want)!();
    const state = tree.root.listing(0)!.radix!;
    for (const j of [0, 255]) state.insert([j], page(weights, new Uint8Array([j])));
    publish(tree, tree.root, 0, state);
    await refiner.next(want)!();
    expect(state.nextSample(want.logHeight)).toBeUndefined();
    expect(refiner.next(want)).toBeUndefined();
    const task = refiner.next({ ...want, children: new Set([0]) });
    expect(task).toBeDefined();
    await task!();
    expect(calls.at(-1)!.length).toBeGreaterThan(1);
    expect(Math.exp(state.bounds(0).lower + want.logHeight)).toBeGreaterThan(2);
  });

  it("resolves the parent before fetching an ambiguous child, then admits supported mass", () => {
    const tree = new IntervalTree(1);
    const state = new RadixCDF();
    state.insert([], uniformPage(256));
    publish(tree, tree.root, 0, state);
    const lens = { p: 0, q: 0 };
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const view = { width: 1000, height: 800 };
    const demand = () => wants(tree, lens, cam, view, layout(tree, lens, cam, view), [quantileSource]);
    const coarse = demand();
    // Interior tokens have no positive lower mass yet. Normalization F(V)=1
    // already certifies the last token's mass, so it may expand immediately.
    expect(coarse.every((want) => want.node === tree.root || want.node.label === 255)).toBe(true);
    expect(coarse[0]!.quantile!.ranges).toContainEqual({ first: 0, end: 255 });
    state.insert([0], uniformPage(256, new Uint8Array([0])));
    publish(tree, tree.root, 0, state);
    const child = tree.child(0, 0);
    expect(demand().some((want) => want.node === child && want.priority > 0)).toBe(true);
  });

  it("refines coarse stripes without inferring subpixel token continuations", async () => {
    const tree = new IntervalTree(1);
    let children = 0;
    const source: Source = {
      async expand() { throw Error("v1 must not be called"); },
      async quantile(node, prefix) {
        if (node) {
          children++;
          return { node: BigInt(node.label + 100), firstBranchId: 0, delta: [], terminal: true };
        }
        return uniformPage(1024, prefix);
      },
    };
    const expander = new Expander(tree, [source]);
    const lens = { p: 0, q: 0 };
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const view = { width: 500, height: 800 };
    try {
      expander.request(() => wants(tree, lens, cam, view, layout(tree, lens, cam, view), [source]));
      await until(() => {
        const state = tree.root.listing(0)?.radix;
        return state !== undefined && state.nextSample(Math.log(view.height)) === undefined;
      });
      // The coarse page made 256 tokens look 3.125px tall. Their actual height
      // is 800/1024px, so resolving them must not require any child inference.
      expect(children).toBe(0);
      expect(Math.max(...tree.root.listing(0)!.radix!.partition().filter((r) => r.kind === "token").map((r) => Math.exp(r.mass.upper))) * view.height).toBeLessThan(1);
    } finally {
      expander.dispose();
    }
  });
});

describe("completion-driven fetching", () => {
  it("refines a v2 node to its pixel budget after one request and no animation frames", async () => {
    const calls: string[] = [];
    const source: Source = {
      async expand() { throw Error("v1 must not be called"); },
      async quantile(_node, prefix) {
        calls.push([...prefix].join(","));
        return page([1n, 2n, 997n], prefix);
      },
    };
    const tree = new IntervalTree(1);
    const expander = new Expander(tree, [source]);
    try {
      expander.request(() => [{
        node: tree.root, measure: 0, horizon: 0, priority: 800,
        quantile: { node: tree.root, lens: { p: 0, q: 0 }, logHeight: Math.log(800), k: 100, right: 20 },
      }]);
      await until(() => {
        const state = tree.root.listing(0)?.radix;
        return state !== undefined && state.nextSample(Math.log(800)) === undefined && [0, 1, 2].every((id) => {
          const bounds = state.bounds(id);
          return (bounds.upper - bounds.lower) * 100 <= 0.25;
        });
      });
      expect(calls.length).toBeGreaterThan(1);
      expect(new Set(calls).size).toBe(calls.length);
      expect(tree.root.listing(0)!.radix!.partition().reduce((a, r) => a + Math.exp(r.lnMass), 0)).toBeCloseTo(1, 12);
      const settled = calls.length;
      await Bun.sleep(10);
      expect(calls.length).toBe(settled);
    } finally {
      expander.dispose();
    }
  });

  it("refills bounded capacity from current demand without another request call", async () => {
    const tree = new IntervalTree(1), source = new ManualSource();
    tree.root.setHandle(0, 7n);
    const want = (label: number): Want => ({ node: tree.child(0, label), measure: 0, horizon: 2, priority: 10 - label });
    let wanted = [want(0), want(1), want(2), want(3)];
    const expander = new Expander(tree, [source], 2);
    try {
      expander.request(() => wanted);
      await until(() => source.requests.length === 2);
      wanted = [want(4)];
      source.finish(0);
      await until(() => source.requests.length === 3);
      expect(source.requests.map(({ node }) => node?.label)).toEqual([0, 1, 4]);
      expect(source.peak).toBe(2);
      source.finish(1);
      source.finish(2);
      await until(() => tree.child(0, 4).listing(0) !== undefined);
    } finally {
      expander.dispose();
    }
  });

  it("uses replacement demand when a pump is already scheduled", async () => {
    const tree = new IntervalTree(1), source = new ManualSource();
    tree.root.setHandle(0, 7n);
    const want = (label: number): Want => ({ node: tree.child(0, label), measure: 0, horizon: 2, priority: 1 });
    const expander = new Expander(tree, [source]);
    try {
      expander.request([want(0)]);
      expander.request([want(1)]);
      await until(() => source.requests.length === 1);
      expect(source.requests[0]!.node?.label).toBe(1);
      source.finish(0);
    } finally {
      expander.dispose();
    }
  });

  it("disposal stops further requests and suppresses errors from an old dataset", async () => {
    const tree = new IntervalTree(1), source = new ManualSource();
    tree.root.setHandle(0, 7n);
    const expander = new Expander(tree, [source], 1);
    const errors: unknown[] = [];
    expander.onError = (error) => errors.push(error);
    expander.request([0, 1].map((label) => ({ node: tree.child(0, label), measure: 0, horizon: 2, priority: 1 })));
    await until(() => source.requests.length === 1);
    expander.dispose();
    source.requests[0]!.reply.reject(Error("old dataset"));
    await Bun.sleep(10);
    expander.request([{ node: tree.root, measure: 0, horizon: 2, priority: 1 }]);
    await Bun.sleep(10);
    expect(source.requests.length).toBe(1);
    expect(errors).toEqual([]);
  });

  it("retries a failed request after backoff without an input event or render", async () => {
    const tree = new IntervalTree(1);
    const failed = deferred<void>(), retried = deferred<void>();
    let calls = 0;
    const source: Source = {
      async expand() {
        if (++calls === 1) throw Error("temporary failure");
        retried.resolve();
        return { node: 7n, complete: 10, edges: (async function* () {})() };
      },
    };
    const expander = new Expander(tree, [source]);
    expander.onError = () => failed.resolve();
    try {
      expander.request([{ node: tree.root, measure: 0, horizon: 2, priority: 1 }]);
      await failed.promise;
      await Bun.sleep(20);
      expect(calls).toBe(1);
      await retried.promise;
      expect(calls).toBe(2);
    } finally {
      expander.dispose();
    }
  });
});
