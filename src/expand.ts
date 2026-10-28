// Fetches the listings a frame needs. Each visible box wants its node's
// children down to a pixel under p, and, when q differs, every node laid out
// wants every child q could draw short of the screen's right edge. Wants only
// grow a listing: a node is
// fetched again only when it needs a deeper horizon than it has or has asked
// for, with some headroom so that zooming does not refetch every frame.

import { Decoder } from "./codec";
import { type Camera, type Frame, logView, type Viewport } from "./layout";
import type { Source } from "./source";
import type { IntervalTree, Lens, TreeNode } from "./tree";

/** Extra bits asked for beyond a want. */
const HEADROOM = 4;
const MAX_HORIZON = 4096;

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
 * log2(heightPx) bits: the children of every visible box, and more children of
 * every node whose tail is visible. A node whose children start at ``x`` needs
 * q down to the right edge of the screen, wherever the node itself is, since
 * its q listing places everything to its right.
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
  for (const box of frame.boxes) {
    if (box.label !== null) {
      const { node, label } = box;
      want(() => tree.child(node, label), box.lnMass, box.x1);
    } else if (box.kind === "tail") {
      want(() => tree.node(box.node), box.lnMass - Math.log(box.p), box.x0);
    }
  }
  if (lens.q !== lens.p) {
    for (const { node, lnMass, x } of frame.walked) want(() => tree.node(node), lnMass, x, false);
  }
  return result;
}

export class Expander {
  private inFlight = 0;
  private readonly asked = new Map<string, number>();
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
      const covered = Math.max(node.listing(measure)?.horizon ?? -1, this.asked.get(key) ?? -1);
      if (horizon <= covered) continue;
      const ask = Math.min(MAX_HORIZON, horizon + HEADROOM);
      if (ask <= covered) continue;
      this.asked.set(key, ask);
      this.inFlight++;
      this.fetch(node, measure, ask)
        .catch((error: unknown) => this.onError(error))
        .finally(() => {
          this.inFlight--;
          this.asked.delete(key);
        });
    }
  }

  private async fetch(node: TreeNode, measure: number, horizon: number): Promise<void> {
    const reply = this.sources[measure]!.expand(node.handle, horizon);
    const decoder = new Decoder(horizon, reply.complete);
    for await (const edge of reply.edges) if (!decoder.push(edge)) break;
    this.tree.setListing(node.id, measure, { ...decoder.listing(), horizon });
  }
}
