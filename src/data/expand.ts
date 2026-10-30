// Fetch the listings a frame needs, the view's origin first: it fills the view.

import { type Camera, type Frame, logView, type Viewport } from "../core/layout";
import type { Source } from "./source";
import type { Branch, IntervalTree, Lens, NodeId } from "../core/tree";

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

  if (!tree.hasListing(cam.origin, lens.p) || !tree.hasListing(cam.origin, lens.q)) {
    want(() => cam.origin, 0, 0);
  }
  for (const { node, branch, lnMass, x1 } of frame.unexpanded) {
    want(() => tree.child(node, branch), lnMass, x1);
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
  /** Aborts the work in flight for the current view. */
  private abort = new AbortController();
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

  /**
   * Abandon the work in flight: the view moved elsewhere. A server runs one
   * forward pass at a time, so what the old view wanted would otherwise run
   * first. The next demand asks again for whatever it still needs.
   */
  cancel(): void {
    this.abort.abort();
    this.abort = new AbortController();
    this.pending.clear();
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
    if (this.disposed || this.pending.size >= this.limit) return;
    let retry = Infinity;
    for (const { node, measure, horizon } of this.demand().sort((a, b) => b.priority - a.priority)) {
      if (this.pending.size >= this.limit) break;
      const key = `${node}:${measure}`;
      if (this.pending.has(key)) continue;
      const covered = this.tree.horizon(node, measure);
      const ask = Math.min(MAX_HORIZON, horizon + HEADROOM);
      if (horizon <= covered || ask <= covered) continue;
      const after = (this.failed.get(key) ?? -Infinity) + RETRY_MS - performance.now();
      if (after > 0) {
        retry = Math.min(retry, after);
        continue;
      }
      const { signal } = this.abort;
      void this.fetch(node, measure, ask).catch((error: unknown) => {
        if (!signal.aborted) {
          this.failed.set(key, performance.now());
          this.onError(error);
        }
      });
    }
    if (retry < Infinity) this.schedule(retry);
  }

  /** Expand until the branch is listed, coalescing with visible discovery. */
  async reach(node: NodeId, measure: number, branch: Branch): Promise<boolean> {
    const listed = () => this.tree.slotOf(node, { p: measure, q: measure }, branch) !== undefined;
    while (!listed()) {
      if (this.disposed) throw new Error("dataset changed while finding a path");
      const pending = this.pending.get(`${node}:${measure}`);
      if (pending !== undefined) {
        await pending;
        continue;
      }
      const covered = this.tree.horizon(node, measure);
      if (this.tree.complete(node, measure) < covered) return false;
      if (covered >= MAX_HORIZON) return false;
      const horizon = Math.min(MAX_HORIZON, Math.max(16, 2 * Math.max(0, covered)));
      await this.fetch(node, measure, horizon);
    }
    return true;
  }

  private fetch(node: NodeId, measure: number, horizon: number): Promise<void> {
    const key = `${node}:${measure}`;
    const pending = this.pending.get(key);
    if (pending !== undefined) return pending;
    const { signal } = this.abort;
    const work = (async () => {
      if (this.active >= this.limit) await new Promise<void>((resolve) => this.waiting.push(resolve));
      else this.active++;
      try {
        signal.throwIfAborted();
        const reply = await this.sources[measure]!.expand(this.tree, node, horizon, signal);
        signal.throwIfAborted();
        if (horizon < this.tree.horizon(node, measure)) return;
        this.tree.setListing(node, measure, { ...reply, horizon });
        this.failed.delete(key);
      } finally {
        const next = this.waiting.shift();
        if (next === undefined) this.active--;
        else next();
      }
    })();
    this.pending.set(key, work);
    const finish = () => {
      if (this.pending.get(key) === work) this.pending.delete(key);
      this.schedule();
    };
    void work.then(finish, finish);
    return work;
  }
}
