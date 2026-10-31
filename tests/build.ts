// Build trees from listings as a source would send them.

import { IntervalTree, type Listing, type NodeId } from "../src/core/tree";

export const HORIZON = 40;

/** Children with ``weights``, heaviest first, along branches 0, 1, … unless given. */
export function listing(weights: number[], branches = weights.map((_, i) => i), horizon = HORIZON): Listing {
  return {
    branches,
    lnWeights: weights.map(Math.log),
    leftover: Math.max(0, 1 - weights.reduce((a, b) => a + b, 0)),
    horizon,
    complete: horizon,
  };
}

/** A one-measure tree whose nodes all split their interval with ``weights``, expanded along ``path``. */
export function chainTree(weights: number[], path: number[], measures = 1): IntervalTree {
  const tree = new IntervalTree(measures);
  const children = listing(weights);
  tree.setListing(0n, 0, children);
  let node: NodeId = 0n;
  for (const branch of path) {
    node = tree.child(node, branch);
    tree.setListing(node, 0, children);
  }
  return tree;
}
