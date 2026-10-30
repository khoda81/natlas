// View-directed fetching. V1 requests weight horizons; v2 requests radix-256
// CDF pages near visible p mass and q endpoints. Cached information is reused
// as the anchored camera pans, zooms, or changes measures.

import { Code, ConnectError } from "@connectrpc/connect";
import { MIN_EXPAND_PX, Refiner, type QuantileWant } from "./refine";
import { Decoder } from "./codec";
import { type Camera, type Frame, logView, type Viewport } from "./layout";
import type { Reply, Source } from "./source";
import type { IntervalTree, Lens, TreeNode } from "./tree";

/** Extra bits asked for beyond a want. */
const HEADROOM = 4;
const MAX_HORIZON = 4096;
/** How long a failed listing waits before it is asked for again. */
const RETRY_MS = 2000;

export interface Want {
  node: TreeNode;
  measure: number;
  horizon: number;
  /** Larger first. */
  priority: number;
  quantile?: QuantileWant;
}

/**
 * What ``frame`` needs. A node ``heightPx`` tall needs p down to one pixel,
 * log2(heightPx) bits: every child not expanded yet whose children would be
 * on screen, and more children of every node whose tail is visible. A node
 * whose children start at ``x`` needs q down to the right edge of the screen,
 * wherever the node itself is, since its q listing places everything to its
 * right.
 */
export function wants(
  tree: IntervalTree,
  lens: Lens,
  cam: Camera,
  view: Viewport,
  frame: Frame,
  sources?: readonly Source[],
): Want[] {
  if (sources?.[lens.p]?.quantile) return quantileWants(tree, lens, cam, view, frame);
  const right = cam.x + view.width / cam.k;
  const lambda = logView(cam);
  const result: Want[] = [];
  const want = (node: () => TreeNode, lnMass: number, x: number, p = true) => {
    const heightPx = Math.exp(lnMass - lambda) * view.height;
    if (heightPx < MIN_EXPAND_PX || !(x < right)) return;
    const target = node();
    if (p)
      result.push({
        node: target,
        measure: lens.p,
        horizon: Math.ceil(Math.log2(heightPx)),
        priority: heightPx,
      });
    if (lens.q !== lens.p) {
      result.push({
        node: target,
        measure: lens.q,
        horizon: Math.ceil((right - x) / Math.LN2),
        priority: heightPx,
      });
    }
  };

  const root = tree.root;
  if (root.listing(lens.p) === undefined || root.listing(lens.q) === undefined) want(() => root, 0, 0);
  for (const { node, label, lnMass, x1 } of frame.unexpanded) want(() => tree.child(node, label), lnMass, x1);
  for (const box of frame.boxes) {
    if (box.kind === "tail") want(() => tree.node(box.node), box.lnMass - Math.log(box.p), box.x0);
  }
  if (lens.q !== lens.p) {
    for (const { node, lnMass, x } of frame.walked) want(() => tree.node(node), lnMass, x, false);
  }
  return result;
}

/** v2 precision follows pixels; priority is the node's visible probability × nats. */
function quantileWants(tree: IntervalTree, lens: Lens, cam: Camera, view: Viewport, frame: Frame): Want[] {
  const result = new Map<number, Want>();
  const right = cam.x + view.width / cam.k;
  // Heights are probabilities relative to the view mass. Use the anchored
  // coordinates directly so offscreen mass never inflates the area score.
  const area = (x0: number, x1: number, y0: number, y1: number) =>
    Math.max(0, Math.min(1, y1) - Math.max(0, y0)) *
    Math.max(0, Math.min(right, x1) - Math.max(cam.x, x0));
  const visible = new Map<number, number>();
  for (const box of frame.boxes) {
    visible.set(box.node, (visible.get(box.node) ?? 0) + area(box.x0, box.x1, box.y0, box.y1));
  }
  const add = (
    node: TreeNode,
    lnMass: number,
    x: number,
    discover = false,
    priority = visible.get(node.id) ?? 0,
  ) => {
    const logHeight = lnMass - logView(cam) + Math.log(view.height);
    if (logHeight < Math.log(MIN_EXPAND_PX) || x >= right) return;
    if (!result.has(node.id))
      result.set(node.id, {
        node,
        measure: lens.p,
        horizon: 0,
        priority,
        quantile: {
          node,
          lens,
          logHeight,
          k: cam.k,
          right: right - x,
          labels: discover ? undefined : new Set(),
        },
      });
  };
  if (!tree.root.listing(lens.p)?.radix?.ready) add(tree.root, 0, 0, true, right - cam.x);
  const walked = new Map(frame.walked.map((entry) => [entry.node, entry]));
  for (const entry of walked.values()) {
    add(tree.node(entry.node), entry.lnMass, entry.x);
    const target = result.get(entry.node)?.quantile;
    if (target && entry.at) target.window = { id: entry.at.id, y: entry.at.y * view.height, height: view.height };
  }
  for (const entry of frame.unexpanded) {
    const parent = tree.node(entry.node);
    const parentFrame = walked.get(parent.id)!;
    const logParentHeight = parentFrame.lnMass - logView(cam) + Math.log(view.height);
    const bounds = parent.listing(lens.p)!.radix!.bounds(entry.label);
    const continuationArea = area(entry.x1, right, entry.y0, entry.y1);
    if (bounds.lower + logParentHeight < Math.log(MIN_EXPAND_PX)) {
      if (bounds.upper + logParentHeight >= Math.log(MIN_EXPAND_PX)) {
        // A sample bin can make a tiny token look large. Spend this area's
        // priority on resolving the parent before admitting model work below it.
        const owner = result.get(parent.id)!;
        owner.priority += continuationArea;
        const target = owner.quantile!;
        const children = (target.children ??= new Set()) as Set<number>;
        children.add(entry.label);
      }
      continue;
    }
    add(
      tree.child(entry.node, entry.label),
      entry.lnMass,
      entry.x1,
      true,
      continuationArea,
    );
  }
  for (const box of frame.boxes) {
    const target = result.get(box.node)?.quantile;
    if (!target?.labels) continue;
    const slots = tree.node(box.node).slots(lens);
    if (slots.sampled && (box.kind === "unresolved" || box.kind === "aggregate")) {
      const first = slots.regions[box.slot]!.first;
      const end = slots.regions[box.slot + box.count - 1]!.end;
      target.ranges = [...(target.ranges ?? []), { first, end }];
      continue;
    }
    for (let i = box.slot; i < box.slot + box.count; i++) {
      const label = slots.label(i);
      if (label !== null) (target.labels as Set<number>).add(label);
    }
  }
  // Even an ancestor entirely left of the view places every visible descendant.
  // Keep that dependency eligible when layout stops its outward walk early,
  // without assigning priority to its potentially enormous offscreen height.
  let ancestorMass = 0;
  let ancestorX = 0;
  const chain = tree.chain(cam.anchor.node);
  for (let i = 1; i < chain.length; i++) {
    const node = chain[i]!;
    const parent = tree.node(node.parent!);
    add(parent, ancestorMass, ancestorX);
    const slots = parent.slots(lens);
    const slot = slots.slotOf(node.label);
    if (slot !== undefined) {
      ancestorMass += Math.log(slots.p[slot]!);
      ancestorX += slots.widths[slot]!;
    }
    const target = result.get(node.parent!)?.quantile;
    if (target?.labels) (target.labels as Set<number>).add(node.label);
  }
  return [...result.values()];
}

export class Expander {
  private readonly refiner: Refiner;
  private inFlight = 0;
  private readonly asked = new Map<string, number>();
  private readonly failed = new Map<string, number>();
  private wanted: () => Want[] = () => [];
  private wake: ReturnType<typeof setTimeout> | undefined;
  private wakeAt = Infinity;
  private disposed = false;
  onError: (error: unknown) => void = (error) => console.error(error);

  constructor(
    private readonly tree: IntervalTree,
    private readonly sources: readonly Source[],
    private readonly limit = 4,
  ) {
    this.refiner = new Refiner(tree, sources);
  }

  /** Replace demand. A provider recomputes geometry after each batch of replies. */
  request(wanted: Want[] | (() => Want[])): void {
    if (this.disposed) return;
    this.wanted = typeof wanted === "function" ? wanted : () => wanted;
    this.schedule();
  }

  /** Retire a dataset's scheduler; accepted requests may finish populating its tree. */
  dispose(): void {
    this.disposed = true;
    this.wanted = () => [];
    if (this.wake !== undefined) clearTimeout(this.wake);
    this.wake = undefined;
    this.wakeAt = Infinity;
  }

  private schedule(delay = 0): void {
    if (this.disposed) return;
    const at = performance.now() + delay;
    if (this.wake !== undefined) {
      if (this.wakeAt <= at) return;
      clearTimeout(this.wake);
    }
    this.wakeAt = at;
    // Yield between batches, including when cached sources resolve immediately,
    // so input and painting can run without putting fetching behind display Hz.
    this.wake = setTimeout(() => {
      this.wake = undefined;
      this.wakeAt = Infinity;
      this.pump();
    }, delay);
  }

  private pump(): void {
    if (this.disposed || this.inFlight >= this.limit) return;
    let wanted: Want[];
    try {
      wanted = [...this.wanted()].sort((a, b) => b.priority - a.priority);
    } catch (error) {
      this.onError(error);
      this.schedule(RETRY_MS);
      return;
    }
    let retryAt = Infinity;
    for (const { node, measure, horizon, quantile } of wanted) {
      if (this.inFlight >= this.limit) return;
      const key = `${node.id}:${measure}`;
      const retry = (this.failed.get(key) ?? -Infinity) + RETRY_MS;
      if (retry > performance.now()) {
        retryAt = Math.min(retryAt, retry);
        continue;
      }
      if (quantile) {
        if (this.asked.has(key)) continue;
        let task: (() => Promise<void>) | undefined;
        try {
          task = this.refiner.next(quantile);
        } catch (error) {
          this.failed.set(key, performance.now());
          retryAt = Math.min(retryAt, performance.now() + RETRY_MS);
          this.onError(error);
          continue;
        }
        if (!task) continue;
        this.start(key, 0, task);
        continue;
      }
      const covered = Math.max(node.listing(measure)?.horizon ?? -1, this.asked.get(key) ?? -1);
      if (horizon <= covered) continue;
      const ask = Math.min(MAX_HORIZON, horizon + HEADROOM);
      if (ask <= covered) continue;
      this.start(key, ask, () => this.fetch(node, measure, ask));
    }
    if (Number.isFinite(retryAt)) this.schedule(Math.max(0, retryAt - performance.now()));
  }

  private start(key: string, horizon: number, task: () => Promise<void>): void {
    this.asked.set(key, horizon);
    this.inFlight++;
    task()
      .then(() => this.failed.delete(key))
      .catch((error: unknown) => {
        if (this.disposed) return;
        this.failed.set(key, performance.now());
        this.onError(error);
      })
      .finally(() => {
        this.inFlight--;
        this.asked.delete(key);
        this.schedule();
      });
  }

  /**
   * Fetch ``node``'s listing under ``measure`` deep enough to list ``label``,
   * doubling the horizon until it does; false if no horizon does.
   */
  async reach(node: TreeNode, measure: number, label: number): Promise<boolean> {
    if (this.sources[measure]?.quantile) return this.refiner.reach(node, measure, label);
    const listed = () => {
      const listing = node.listing(measure);
      return listing !== undefined && !listing.radix && listing.labels.includes(label);
    };
    for (let horizon = Math.max(16, 2 * (node.listing(measure)?.horizon ?? 0)); !listed(); horizon *= 2) {
      if (horizon > MAX_HORIZON) return false;
      await this.fetch(node, measure, horizon);
    }
    return true;
  }

  private async fetch(node: TreeNode, measure: number, horizon: number): Promise<void> {
    const reply = await this.open(node, measure, horizon);
    const decoder = new Decoder(horizon, reply.complete);
    for await (const edge of reply.edges) if (!decoder.push(edge)) break;
    // A deeper listing that landed first stays: listings only grow.
    if (horizon < (node.listing(measure)?.horizon ?? -1)) return;
    this.tree.setListing(node.id, measure, { ...decoder.listing(), horizon });
  }

  /**
   * Expand ``node`` under ``measure``, naming it by its parent's handle. An
   * ancestor without one is named first, from the nearest one with a handle,
   * asking for the handle alone; a handle the server has forgotten, as after
   * a restart, is dropped and its node named again the same way.
   */
  private async open(node: TreeNode, measure: number, horizon: number): Promise<Reply> {
    const source = this.sources[measure]!;
    const chain = this.tree.chain(node.id);
    for (let depth = chain.length - 1; ;) {
      const target = chain[depth]!;
      const parent = chain[depth - 1];
      const handle = parent?.handle(measure);
      if (parent !== undefined && handle === undefined) {
        depth--;
        continue;
      }
      const last = depth === chain.length - 1;
      let reply: Reply;
      try {
        reply = await source.expand(parent && { parent: handle!, label: target.label }, last ? horizon : 0);
      } catch (error) {
        if (parent === undefined || ConnectError.from(error).code !== Code.NotFound) throw error;
        parent.setHandle(measure, undefined);
        continue;
      }
      target.setHandle(measure, reply.node);
      if (last) return reply;
      for await (const _ of reply.edges) break;
      depth++;
    }
  }
}
