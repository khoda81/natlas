// What frames lack, fetched once: a read-through cache. Layout reads only
// what the tree holds and lists what it lacks as requests; this asks the
// sources for them, a few at a time, and the tree's change notifications
// redraw whatever depended on them.

import type { Miss, Read } from "../core/layout";
import type { IntervalTree, NodeId } from "../core/tree";
import type { ExpandRequest, Source } from "./source";

const RETRY_MS = 2000;

/** How a read reads the node it ends at. */
const kind = ({ within }: Read) => within === undefined ? "path" : "points" in within ? "points" : "branches";

/** A request in flight. One the view asked for is abandoned while its
 * answer is outstanding once the view no longer lacks what it reads: its
 * node's parts in view are what it is for. */
interface Flight {
  readonly work: Promise<void>;
  readonly controller: AbortController;
  readonly view: boolean;
}

export class Expander {
  /** Requests in flight, by the node they read down to, measure and how they
   * read it, until their target's own answer is in. */
  private readonly pending = new Map<string, Flight>();
  /** Every request in flight, until its stream ends: one may go on into
   * deeper nodes after its target's answer. */
  private readonly flights = new Set<Flight>();
  /** Answered requests, with their target's revision before the answer:
   * while it stays the same, the answer changed nothing, and asking again
   * would find nothing new. */
  private readonly asked = new Map<string, number>();
  private readonly failed = new Map<string, number>();
  private demand: () => Miss[] = () => [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  /** Requests in flight that a source may have to compute for: those reading
   * a node it has not answered for yet. ``limit`` bounds these alone, since a
   * source computes one at a time. What it already holds it answers at once,
   * and waits for nothing. */
  private active = 0;
  private readonly waiting: (() => void)[] = [];
  onError: (error: unknown) => void = (error) => console.error(error);

  constructor(
    private readonly tree: IntervalTree,
    private readonly sources: readonly Source[],
    private readonly limit = 4,
  ) {}

  /** What the current view lacks, read again whenever a request can start
   * or end. */
  request(demand: () => Miss[]): void {
    this.demand = demand;
    this.pump();
  }

  /** Whether ``measure``'s source answered with nothing for ``node``'s children. */
  exhausted(node: NodeId, measure: number): boolean {
    return !this.tree.has(node, measure) && this.asked.get(`${node}:${measure}:points`) === this.tree.revisionOf(node);
  }

  /**
   * Abandon the work in flight: the view moved elsewhere. A server runs one
   * forward pass at a time, so what the old view wanted would otherwise run
   * first. The next demand asks again for whatever it still needs.
   */
  cancel(): void {
    for (const flight of this.flights) flight.controller.abort();
    this.pending.clear();
  }

  dispose(): void {
    this.disposed = true;
    this.cancel();
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
    const misses = this.demand().map((miss) => {
      const target = this.tree.descend(miss.node, miss.read.path);
      return { miss, target, key: `${target}:${miss.measure}:${kind(miss.read)}` };
    });
    // What the view no longer lacks it no longer wants: a request for it
    // would hold a connection, and a source's turn, from what it does.
    const wanted = new Set(misses.map(({ key }) => key));
    for (const [key, flight] of this.pending) {
      if (flight.view && !wanted.has(key)) flight.controller.abort();
    }
    let retry = Infinity;
    for (const { miss, target, key } of misses.sort((a, b) => b.miss.priority - a.miss.priority)) {
      if (this.pending.has(key) || this.asked.get(key) === this.tree.revisionOf(target)) continue;
      const held = this.tree.has(target, miss.measure);
      if (!held && this.limit <= this.active) continue;
      const after = (this.failed.get(key) ?? -Infinity) + RETRY_MS - performance.now();
      if (after > 0) {
        retry = Math.min(retry, after);
        continue;
      }
      const request: ExpandRequest = { node: miss.node, ...miss.read };
      const { work, controller } = this.fetch(key, target, miss.measure, request, held, true);
      void work.catch((error: unknown) => {
        if (!controller.signal.aborted) {
          this.failed.set(key, performance.now());
          this.onError(error);
        }
      });
    }
    if (retry < Infinity) this.schedule(retry);
  }

  /**
   * Place every node from ``origin`` down to ``target``, in one request from
   * the deepest node already placed; views then sample ``target`` itself.
   * True once all of it is placed; false where the source has nothing more.
   */
  async walk(origin: NodeId, target: NodeId, measure: number): Promise<boolean> {
    const lens = { p: measure, q: measure };
    const chain = this.tree.chain(origin, target);
    const placed = (index: number) => this.tree.entry(chain[index]!, lens, this.tree.branch(chain[index + 1]!)) !== undefined;
    let first = 0;
    while (first < chain.length - 1 && placed(first)) first++;
    if (first === chain.length - 1) return true;
    const path = chain.slice(first + 1).map((node) => this.tree.branch(node));
    const key = `${target}:${measure}:path`;
    await (this.pending.get(key) ?? this.fetch(key, target, measure, { node: chain[first]!, path }, false, false)).work;
    return chain.slice(0, -1).every((_, index) => placed(index));
  }

  /** Take in a request's answer, saying once when it has ``target``'s part:
   * the messages after it are of nodes further down. */
  private async answer(target: NodeId, measure: number, request: ExpandRequest, signal: AbortSignal, reached: () => void): Promise<void> {
    signal.throwIfAborted();
    for await (const { node, ...children } of this.sources[measure]!.expand(this.tree, request, signal)) {
      signal.throwIfAborted();
      this.tree.insert(node, measure, children);
      if (node === target) reached();
    }
  }

  /** One request. ``held`` where it reads only what its source has answered
   * for before, and so waits for no other; ``view`` where the view asked
   * for it, and so ends when the view no longer lacks what it reads. */
  private fetch(key: string, target: NodeId, measure: number, request: ExpandRequest, held: boolean, view: boolean): Flight {
    const controller = new AbortController();
    const { signal } = controller;
    const before = this.tree.revisionOf(target);
    // The request is answered once ``target``'s own message is in, or the
    // stream ends without one. A stream that goes on into deeper nodes no
    // longer stands in the way of asking ``target`` for more.
    let answered = false;
    const reached = () => {
      if (answered) return;
      answered = true;
      this.asked.set(key, before);
      this.failed.delete(key);
      if (this.pending.get(key) === flight) this.pending.delete(key);
      this.schedule();
    };
    const work: Promise<void> = (async () => {
      if (!held) {
        if (this.limit <= this.active) await new Promise<void>((resolve) => this.waiting.push(resolve));
        else this.active++;
      }
      try {
        await this.answer(target, measure, request, signal, reached);
        reached();
      } finally {
        if (!held) {
          const next = this.waiting.shift();
          if (next === undefined) this.active--;
          else next();
        }
      }
    })();
    const flight: Flight = { work, controller, view };
    this.pending.set(key, flight);
    this.flights.add(flight);
    const finish = () => {
      this.flights.delete(flight);
      if (this.pending.get(key) === flight) this.pending.delete(key);
      this.schedule();
    };
    void work.then(finish, finish);
    return flight;
  }
}
