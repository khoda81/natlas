// What frames lack, fetched once: a read-through cache. Layout reads only
// what the tree holds and lists what it lacks as requests; this asks the
// sources for them, a few at a time, and the tree's change notifications
// redraw whatever depended on them.

import type { Miss } from "../core/layout";
import type { IntervalTree, NodeId } from "../core/tree";
import type { ExpandRequest, Source } from "./source";

const RETRY_MS = 2000;

export class Expander {
  /** Requests in flight, by target, measure and lacking ranks. */
  private readonly pending = new Map<string, Promise<void>>();
  /** Answered requests, with their target's revision then: asking again
   * finds nothing new until something changes it. */
  private readonly asked = new Map<string, number>();
  private readonly failed = new Map<string, number>();
  private demand: () => Miss[] = () => [];
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

  /** What the current view lacks, read again whenever a request can start. */
  request(demand: () => Miss[]): void {
    this.demand = demand;
    this.pump();
  }

  /** Whether ``measure``'s source answered with nothing for ``node``'s children. */
  exhausted(node: NodeId, measure: number): boolean {
    return !this.tree.has(node, measure) && this.asked.get(`${node}:${measure}:0:null`) === this.tree.revisionOf(node);
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
    for (const miss of this.demand().sort((a, b) => b.priority - a.priority)) {
      if (this.pending.size >= this.limit) break;
      const { target } = miss;
      const key = `${target}:${miss.measure}:${miss.from}:${miss.to}`;
      if (this.pending.has(key) || this.asked.get(key) === this.tree.revisionOf(target)) continue;
      const after = (this.failed.get(key) ?? -Infinity) + RETRY_MS - performance.now();
      if (after > 0) {
        retry = Math.min(retry, after);
        continue;
      }
      const { signal } = this.abort;
      const request: ExpandRequest = { node: miss.node, window: { lnTop: miss.lnTop, lnBottom: miss.lnBottom }, horizon: miss.horizon };
      void this.fetch(key, target, miss.measure, request).catch((error: unknown) => {
        if (!signal.aborted) {
          this.failed.set(key, performance.now());
          this.onError(error);
        }
      });
    }
    if (retry < Infinity) this.schedule(retry);
  }

  /**
   * Place every node from ``origin`` down to ``target`` and list ``target``
   * down to ``horizon``, in one request from the deepest node already placed.
   * True once all of it is known; false where the source has nothing more.
   */
  async walk(origin: NodeId, target: NodeId, measure: number, horizon: number): Promise<boolean> {
    const lens = { p: measure, q: measure };
    const chain = this.tree.chain(origin, target);
    const last = chain.length - 1;
    const placed = (index: number) => this.tree.entry(chain[index]!, lens, this.tree.branch(chain[index + 1]!)) !== undefined;
    let first = 0;
    while (first < last && placed(first)) first++;
    if (first === last && this.tree.has(target, measure)) return true;
    const path = chain.slice(first + 1).map((node) => this.tree.branch(node));
    await this.fetch(`${target}:${measure}:walk`, target, measure, { node: chain[first]!, path, horizon });
    return chain.every((_, index) => index === last ? this.tree.has(target, measure) : placed(index));
  }

  /** One request, unless the same is in flight; ``target``'s revision marks it answered. */
  private fetch(key: string, target: NodeId, measure: number, request: ExpandRequest): Promise<void> {
    const pending = this.pending.get(key);
    if (pending !== undefined) return pending;
    const { signal } = this.abort;
    const work = (async () => {
      if (this.active >= this.limit) await new Promise<void>((resolve) => this.waiting.push(resolve));
      else this.active++;
      try {
        signal.throwIfAborted();
        for await (const { node, run } of this.sources[measure]!.expand(this.tree, request, signal)) {
          signal.throwIfAborted();
          this.tree.insert(node, measure, run);
        }
        this.asked.set(key, this.tree.revisionOf(target));
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
