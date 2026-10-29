// View-directed fetching. V1 requests weight horizons; v2 requests radix-256
// CDF pages near visible p mass and q endpoints. Cached information is reused
// as the anchored camera pans, zooms, or changes measures.

import { Code, ConnectError } from "@connectrpc/connect";
import { Refiner, type QuantileWant } from "./refine";
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

/** Boxes shorter than this do not ask for their children. */
const MIN_PX = 2;

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
    if (heightPx < MIN_PX || !(x < right)) return;
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

/** v2 fetches follow the same anchored frame, with precision set by its pixels. */
function quantileWants(tree: IntervalTree, lens: Lens, cam: Camera, view: Viewport, frame: Frame): Want[] {
  const result = new Map<number, Want>();
  const right = cam.x + view.width / cam.k;
  const add = (node: TreeNode, lnMass: number, x: number, discover = false) => {
    const logHeight = lnMass - logView(cam) + Math.log(view.height);
    if (logHeight < Math.log(MIN_PX) || x >= right) return;
    if (!result.has(node.id))
      result.set(node.id, {
        node,
        measure: lens.p,
        horizon: 0,
        priority: Math.exp(Math.min(700, logHeight)),
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
  if (!tree.root.listing(lens.p)?.radix?.ready) add(tree.root, 0, 0, true);
  for (const entry of frame.walked) add(tree.node(entry.node), entry.lnMass, entry.x);
  for (const entry of frame.unexpanded) add(tree.child(entry.node, entry.label), entry.lnMass, entry.x1, true);
  for (const box of frame.boxes) {
    const target = result.get(box.node)?.quantile;
    if (!target?.labels) continue;
    const slots = tree.node(box.node).slots(lens);
    for (let i = box.slot; i < box.slot + box.count; i++) {
      const label = slots.label(i);
      if (label !== null) (target.labels as Set<number>).add(label);
    }
  }
  // Even an ancestor entirely left of the view places every visible descendant.
  // Do not lose its q request when layout stops its outward walk early.
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
  onError: (error: unknown) => void = (error) => console.error(error);

  constructor(
    private readonly tree: IntervalTree,
    private readonly sources: readonly Source[],
    private readonly limit = 4,
  ) {
    this.refiner = new Refiner(tree, sources);
  }

  /** Start fetching the wants not yet covered, most important first, up to the concurrency limit. */
  request(wants: Want[]): void {
    wants.sort((a, b) => b.priority - a.priority);
    for (const { node, measure, horizon, quantile } of wants) {
      if (this.inFlight >= this.limit) return;
      const key = `${node.id}:${measure}`;
      if ((this.failed.get(key) ?? -Infinity) > performance.now() - RETRY_MS) continue;
      if (quantile) {
        if (this.asked.has(key)) continue;
        let task: (() => Promise<void>) | undefined;
        try {
          task = this.refiner.next(quantile);
        } catch (error) {
          this.failed.set(key, performance.now());
          this.onError(error);
          continue;
        }
        if (!task) continue;
        this.asked.set(key, 0);
        this.inFlight++;
        task()
          .then(() => this.failed.delete(key))
          .catch((error: unknown) => {
            this.failed.set(key, performance.now());
            this.onError(error);
          })
          .finally(() => {
            this.inFlight--;
            this.asked.delete(key);
          });
        continue;
      }
      const covered = Math.max(node.listing(measure)?.horizon ?? -1, this.asked.get(key) ?? -1);
      if (horizon <= covered) continue;
      const ask = Math.min(MAX_HORIZON, horizon + HEADROOM);
      if (ask <= covered) continue;
      this.asked.set(key, ask);
      this.inFlight++;
      this.fetch(node, measure, ask)
        .then(() => this.failed.delete(key))
        .catch((error: unknown) => {
          this.failed.set(key, performance.now());
          this.onError(error);
        })
        .finally(() => {
          this.inFlight--;
          this.asked.delete(key);
        });
    }
  }

  /**
   * Fetch ``node``'s listing under ``measure`` deep enough to list ``label``,
   * doubling the horizon until it does; false if no horizon does.
   */
  async reach(node: TreeNode, measure: number, label: number): Promise<boolean> {
    if (this.sources[measure]?.quantile) return this.refiner.reach(node, measure, label);
    const listed = () => node.listing(measure)?.labels.includes(label) === true;
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
