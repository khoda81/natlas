import { describe, expect, it } from "bun:test";
import { Expander, type Want } from "../src/expand";
import type { QuantileReply } from "../src/radix";
import type { Child, Reply, Source } from "../src/source";
import { IntervalTree } from "../src/tree";

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
      expect([...tree.root.listing(0)!.radix!.weights().values()].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
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
