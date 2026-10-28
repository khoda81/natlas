// Fetches the listings a frame needs. Each visible box wants its node's
// children down to a pixel under p, and, when q differs, every node laid out
// wants every child q could draw short of the screen's right edge. Wants only
// grow a listing: a node is
// fetched again only when it needs a deeper horizon than it has or has asked
// for, with some headroom so that zooming does not refetch every frame.

import { Code, ConnectError } from "@connectrpc/connect";
import { Decoder } from "./codec";
import { type Camera, type Frame, logView, type Viewport } from "./layout";
import type { Source } from "./source";
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
export function wants(tree: IntervalTree, lens: Lens, cam: Camera, view: Viewport, frame: Frame): Want[] {
  const right = cam.x + view.width / cam.k;
  const lambda = logView(cam);
  const result: Want[] = [];
  const want = (node: () => TreeNode, lnMass: number, x: number, p = true) => {
    const heightPx = Math.exp(lnMass - lambda) * view.height;
    if (heightPx < MIN_PX || !(x < right)) return;
    const target = node();
    if (p) result.push({ node: target, measure: lens.p, horizon: Math.ceil(Math.log2(heightPx)), priority: heightPx });
    if (lens.q !== lens.p) {
      result.push({ node: target, measure: lens.q, horizon: Math.ceil((right - x) / Math.LN2), priority: heightPx });
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

export class Expander {
  private inFlight = 0;
  private readonly asked = new Map<string, number>();
  private readonly failed = new Map<string, number>();
  onError: (error: unknown) => void = (error) => console.error(error);

  constructor(
    private readonly tree: IntervalTree,
    private readonly sources: readonly Source[],
    private readonly limit = 4,
  ) {}

  /** Start fetching the wants not yet covered, most important first, up to the concurrency limit. */
  request(wants: Want[]): void {
    wants.sort((a, b) => b.priority - a.priority);
    for (const { node, measure, horizon } of wants) {
      if (this.inFlight >= this.limit) return;
      const key = `${node.id}:${measure}`;
      if ((this.failed.get(key) ?? -Infinity) > performance.now() - RETRY_MS) continue;
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

  private async fetch(node: TreeNode, measure: number, horizon: number): Promise<void> {
    try {
      await this.fetchOnce(node, measure, horizon);
    } catch (error) {
      if (ConnectError.from(error).code !== Code.NotFound) throw error;
      await this.relearn(node);
      await this.fetchOnce(node, measure, horizon);
    }
  }

  /**
   * Fetch ``node``'s listing under ``measure`` deep enough to list ``label``,
   * doubling the horizon until it does; false if no horizon does.
   */
  async reach(node: TreeNode, measure: number, label: number): Promise<boolean> {
    const listed = () => node.listing(measure)?.labels.includes(label) === true;
    for (let horizon = Math.max(16, 2 * (node.listing(measure)?.horizon ?? 0)); !listed(); horizon *= 2) {
      if (horizon > MAX_HORIZON) return false;
      await this.fetch(node, measure, horizon);
    }
    return true;
  }

  /**
   * Teach a server that does not know ``node``, as after a restart, the way
   * there: expand each ancestor again as it was listed, root first. Servers
   * learn handles from their own listings.
   */
  private async relearn(node: TreeNode): Promise<void> {
    const chain = this.tree.chain(node.id);
    for (let i = 0; i + 1 < chain.length; i++) {
      const ancestor = chain[i]!;
      const label = chain[i + 1]!.label;
      const measure = this.sources.findIndex((_, m) => ancestor.listing(m)?.labels.includes(label));
      if (measure < 0) continue;
      const reply = this.sources[measure]!.expand(ancestor.handle, ancestor.listing(measure)!.horizon);
      for await (const _ of reply.edges);
    }
  }

  private async fetchOnce(node: TreeNode, measure: number, horizon: number): Promise<void> {
    const reply = this.sources[measure]!.expand(node.handle, horizon);
    const decoder = new Decoder(horizon, reply.complete);
    for await (const edge of reply.edges) if (!decoder.push(edge)) break;
    // A deeper listing that landed first stays: listings only grow.
    if (horizon < (node.listing(measure)?.horizon ?? -1)) return;
    this.tree.setListing(node.id, measure, { ...decoder.listing(), horizon });
  }
}
