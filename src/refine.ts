/** View-directed v2 fetching; one page per task keeps refinement progressive. */
import { Code, ConnectError } from "@connectrpc/connect";
import { RadixCDF } from "./radix";
import type { Source } from "./source";
import type { IntervalTree, Lens, TreeNode } from "./tree";

export interface QuantileWant {
  node: TreeNode;
  lens: Lens;
  logHeight: number;
  k: number;
  right: number;
  /** Undefined on first discovery; otherwise refine only visible IDs. */
  labels?: ReadonlySet<number>;
}

export class Refiner {
  private readonly states = new Map<string, RadixCDF>();
  private readonly pending = new Map<string, Promise<void>>();
  private turn = 0;

  constructor(
    private readonly tree: IntervalTree,
    private readonly sources: readonly Source[],
  ) {}

  private state(node: TreeNode, measure: number): RadixCDF {
    const key = `${node.id}:${measure}`;
    let state = this.states.get(key);
    if (!state) {
      state = new RadixCDF();
      this.states.set(key, state);
    }
    return state;
  }

  private publish(node: TreeNode, measure: number, state: RadixCDF): void {
    const values = [...state.weights()].filter(([, mass]) => mass > 0);
    node.setListing(measure, {
      labels: values.map(([label]) => label),
      lnWeights: values.map(([, mass]) => Math.log(mass)),
      leftover: 0,
      complete: 0,
      horizon: 0,
      radix: state,
    });
  }

  private async page(node: TreeNode, measure: number, prefix: number[]): Promise<void> {
    const key = `${node.id}:${measure}:${prefix.join(",")}`;
    const running = this.pending.get(key);
    if (running) return running;
    const task = this.fetch(node, measure, prefix);
    this.pending.set(key, task);
    try {
      await task;
    } finally {
      this.pending.delete(key);
    }
  }

  private async fetch(node: TreeNode, measure: number, prefix: number[]): Promise<void> {
    const source = this.sources[measure]!;
    if (!source.quantile) throw new Error("A quantile lens requires v2 endpoints for both measures");
    const parent = node.parent === null ? undefined : this.tree.node(node.parent);
    if (parent && parent.handle(measure) === undefined) await this.page(parent, measure, []);
    let reply;
    for (let attempt = 0; ; attempt++) {
      try {
        reply = await source.quantile(
          parent && { parent: parent.handle(measure)!, label: node.label },
          new Uint8Array(prefix),
        );
        break;
      } catch (error) {
        if (!parent || attempt > 0 || ConnectError.from(error).code !== Code.NotFound) throw error;
        parent.setHandle(measure, undefined);
        await this.page(parent, measure, []);
      }
    }
    node.setHandle(measure, reply.node);
    const state = this.state(node, measure);
    state.insert(prefix, reply);
    this.publish(node, measure, state);
    // q changes also reorder p's slots, including other active lenses.
    this.tree.touch(node.id);
  }

  /** Select work without issuing requests or causing redraws when satisfied. */
  next(want: QuantileWant): (() => Promise<void>) | undefined {
    const { node, lens, logHeight, k, right, labels } = want;
    const p = this.state(node, lens.p);
    const q = this.state(node, lens.q);
    if (!p.ready) return () => this.page(node, lens.p, []);
    if (p.terminal) return undefined;
    if (!q.ready) return () => this.page(node, lens.q, []);
    const sample = p.nextSample(logHeight, labels);
    // Alternate p geometry and q endpoints, rather than finishing either
    // distribution before showing the other measure's information.
    if (sample && this.turn++ % 2 === 0) return () => this.page(node, lens.p, sample);
    const weights = p.weights();
    // The sum of these per-edge budgets is <0.5px at any path depth.
    const tolerance = 0.25 / this.tree.chain(node.id).length ** 2;
    const targets = [...(labels ?? weights.keys())].sort((a, b) => (weights.get(b) ?? 0) - (weights.get(a) ?? 0));
    for (const label of targets) {
      const bounds = q.bounds(label);
      if (-bounds.upper >= right || (bounds.upper - bounds.lower) * k <= tolerance) continue;
      const prefix = q.nextBoundary(label);
      if (prefix) return () => this.page(node, lens.q, prefix);
    }
    if (sample) return () => this.page(node, lens.p, sample);
    return undefined;
  }

  /** Name a particular path token, even if p's current samples missed it. */
  async reach(node: TreeNode, measure: number, label: number): Promise<boolean> {
    const state = this.state(node, measure);
    if (!state.ready) await this.page(node, measure, []);
    if (state.terminal) return false;
    for (;;) {
      const bounds = state.bounds(label);
      if (bounds.upper === -Infinity) return false;
      if (bounds.lower > -Infinity && bounds.upper - bounds.lower <= 1e-8) break;
      const prefix = state.nextBoundary(label);
      if (!prefix) return false;
      await this.page(node, measure, prefix);
    }
    state.pin(label);
    this.publish(node, measure, state);
    this.tree.touch(node.id);
    return true;
  }
}
