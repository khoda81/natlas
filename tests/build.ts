// Build trees from listings that went over the wire, as a server would send them.

import { decode, encode } from "../src/codec";
import { type Fetched, IntervalTree, type NodeId } from "../src/tree";

/** Fine enough that rounding stays far below test tolerances. */
export const HORIZON = 40;

/** The listing a client decodes for children with ``weights``, labeled 0, 1, … unless given. */
export function listing(weights: number[], labels = weights.map((_, i) => i), horizon = HORIZON): Fetched {
  return { ...decode(encode(labels, weights.map(Math.log), horizon), horizon), horizon };
}

/** A one-measure tree whose nodes all split their interval with ``weights``, expanded along ``path``. */
export function chainTree(weights: number[], path: number[]): IntervalTree {
  const tree = new IntervalTree(1);
  const children = listing(weights);
  tree.setListing(0, 0, children);
  let node: NodeId = 0;
  for (const label of path) {
    node = tree.child(node, label).id;
    tree.setListing(node, 0, children);
  }
  return tree;
}
