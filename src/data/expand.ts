// Fetch the listings a frame needs. Nodes are named by their complete integer
// paths, so expansion has no handle exchange or restart-recovery path.

import { decode } from "../core/codec";
import { type Camera, type Frame, logView, type Viewport } from "../core/layout";
import type { Source } from "./source";
import type { IntervalTree, Label, Lens, NodeId } from "../core/tree";

const HEADROOM = 4;
const MAX_HORIZON = 4096;
const RETRY_MS = 2000;
const MIN_PX = 2;

export interface Want {
  node: NodeId;
  measure: number;
  horizon: number;
  /** Larger first. */
  priority: number;
}

export function wants(tree: IntervalTree, lens: Lens, cam: Camera, view: Viewport, frame: Frame): Want[] {
  const right = cam.x + view.width / cam.k;
  const lambda = logView(cam);
  const result: Want[] = [];
  const want = (node: () => NodeId, lnMass: number, x: number, p = true) => {
    const heightPx = Math.exp(lnMass - lambda) * view.height;
    if (heightPx < MIN_PX || !(x < right)) return;
    const target = node();
    if (p) result.push({ node: target, measure: lens.p, horizon: Math.ceil(Math.log2(heightPx)), priority: heightPx });
    if (lens.q !== lens.p) {
      result.push({ node: target, measure: lens.q, horizon: Math.ceil((right - x) / Math.LN2), priority: heightPx });
    }
  };

  if (tree.listing(tree.root, lens.p) === undefined || tree.listing(tree.root, lens.q) === undefined) {
    want(() => tree.root, 0, 0);
  }
  for (const { node, label, lnMass, x1 } of frame.unexpanded) {
    want(() => tree.child(node, label), lnMass, x1);
  }
  for (const box of frame.boxes) {
    if (box.kind === "tail") want(() => box.node, box.lnMass - Math.log(box.p), box.x0);
  }
  if (lens.q !== lens.p) {
    for (const { node, lnMass, x } of frame.walked) want(() => node, lnMass, x, false);
  }
  return result;
}

export class Expander {
  private readonly pending = new Map<string, Promise<void>>();
  private readonly failed = new Map<string, number>();
  private demand: () => Want[] = () => [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  private readonly abort = new AbortController();
  private active = 0;
  private readonly waiting: (() => void)[] = [];
  onError: (error: unknown) => void = (error) => console.error(error);

  constructor(
    private readonly tree: IntervalTree,
    private readonly sources: readonly Source[],
    private readonly limit = 4,
  ) {}

  request(demand: Want[] | (() => Want[])): void {
    this.demand = typeof demand === "function" ? demand : () => demand;
    this.pump();
  }

  dispose(): void {
    this.disposed = true;
    this.abort.abort();
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private schedule(delay = 0): void {
    if (this.disposed) return;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.pump();
    }, delay);
  }

  private pump(): void {
    if (this.disposed) return;
    let retry = Infinity;
    for (const { node, measure, horizon } of this.demand().sort((a, b) => b.priority - a.priority)) {
      if (this.pending.size >= this.limit) break;
      const key = `${node}:${measure}`;
      if (this.pending.has(key)) continue;
      const covered = this.tree.listing(node, measure)?.horizon ?? -1;
      const ask = Math.min(MAX_HORIZON, horizon + HEADROOM);
      if (horizon <= covered || ask <= covered) continue;
      const after = (this.failed.get(key) ?? -Infinity) + RETRY_MS - performance.now();
      if (after > 0) {
        retry = Math.min(retry, after);
        continue;
      }
      void this.fetch(node, measure, ask).catch((error: unknown) => {
        if (!this.disposed) {
          this.failed.set(key, performance.now());
          this.onError(error);
        }
      });
    }
    if (retry < Infinity) this.schedule(retry);
  }

  /** Expand until the label is listed, coalescing with visible discovery. */
  async reach(node: NodeId, measure: number, label: Label): Promise<boolean> {
    const listed = () => this.tree.listing(node, measure)?.labels.includes(label) === true;
    while (!listed()) {
      if (this.disposed) throw new Error("dataset changed while finding a path");
      const pending = this.pending.get(`${node}:${measure}`);
      if (pending !== undefined) {
        await pending;
        continue;
      }
      const previous = this.tree.listing(node, measure);
      if (previous !== undefined && previous.complete < previous.horizon) return false;
      if ((previous?.horizon ?? 0) >= MAX_HORIZON) return false;
      const horizon = Math.min(MAX_HORIZON, Math.max(16, 2 * (previous?.horizon ?? 0)));
      await this.fetch(node, measure, horizon);
    }
    return true;
  }

  private fetch(node: NodeId, measure: number, horizon: number): Promise<void> {
    const key = `${node}:${measure}`;
    const pending = this.pending.get(key);
    if (pending !== undefined) return pending;
    const work = (async () => {
      if (this.active >= this.limit) await new Promise<void>((resolve) => this.waiting.push(resolve));
      else this.active++;
      try {
        if (this.disposed) return;
        const reply = await this.sources[measure]!.expand(this.tree.path(node), horizon, this.abort.signal);
        const edges = [];
        for await (const edge of reply.edges) edges.push(edge);
        if (this.disposed || horizon < (this.tree.listing(node, measure)?.horizon ?? -1)) return;
        this.tree.setListing(node, measure, { ...decode(edges, horizon, reply.complete), horizon });
        this.failed.delete(key);
      } finally {
        const next = this.waiting.shift();
        if (next === undefined) this.active--;
        else next();
      }
    })();
    this.pending.set(key, work);
    const finish = () => {
      this.pending.delete(key);
      this.schedule();
    };
    void work.then(finish, finish);
    return work;
  }
}
