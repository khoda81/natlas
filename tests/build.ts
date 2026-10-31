// Build trees from runs as a source would send them.

import { IntervalTree, type NodeId, ROOT, type Run } from "../src/core/tree";

/** ln(e^a + e^b). */
function lnAdd(a: number, b: number): number {
  const high = Math.max(a, b);
  return high === -Infinity ? high : high + Math.log1p(Math.exp(Math.min(a, b) - high));
}

/** Children with ``weights``, heaviest first, from ``rank``, along branches
 * 0, 1, … unless given, with ``rest`` of the node's mass below them. */
export function run(weights: number[], branches = weights.map((_, i) => i), rank = 0,
  rest = Math.max(0, 1 - weights.reduce((a, b) => a + b, 0))): Run & { branches: number[]; lnBelow: number[] } {
  const lnBelow = [Math.log(rest)];
  for (const weight of weights.toReversed()) lnBelow.unshift(Math.min(0, lnAdd(lnBelow[0]!, Math.log(weight))));
  return { rank, branches, lnBelow };
}

/** A tree whose nodes along ``path`` all split their interval with ``weights``. */
export function chainTree(weights: number[], path: number[], measures = 1): IntervalTree {
  const tree = new IntervalTree(measures);
  tree.insert(ROOT, 0, run(weights));
  let node: NodeId = ROOT;
  for (const branch of path) {
    node = tree.child(node, branch);
    tree.insert(node, 0, run(weights));
  }
  return tree;
}
