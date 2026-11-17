// Build trees from runs as a source would send them.

import { type Children, IntervalTree, type NodeId, ROOT, type Run } from "../src/core/tree";

/** Children with ``weights`` from ``rank``, along branches 0, 1, … unless
 * given, with ``rest`` of the node's mass below them: the run's top is what
 * the children and the rest leave above it. */
export function run(weights: number[], branches = weights.map((_, i) => i), rank = 0,
  rest = Math.max(0, 1 - weights.reduce((a, b) => a + b, 0))): Run & { branches: number[]; lnWeights: number[] } {
  const above = rank === 0 ? 0 : 1 - rest - weights.reduce((a, b) => a + b, 0);
  return { rank, branches, lnWeights: weights.map(Math.log), top: digits(Math.max(0, above)) };
}

/** An anchor's digits for the point ``fraction`` of a node's mass below its top: exact for any float. */
export function digits(fraction: number): Uint8Array {
  const digits = [];
  for (let rest = fraction; 0 < rest; rest = (rest * 256) % 1) digits.push(Math.floor(rest * 256));
  return Uint8Array.from(digits);
}

/** The points with these masses below them, each a part of its own. */
export function at(...below: number[]): { top: Uint8Array; lnMass: number; count: number }[] {
  return below.map((mass) => ({ top: digits(1 - mass), lnMass: -Infinity, count: 1 }));
}

/** An answer placing those children, as one run, and the node's end
 * where nothing is left below them. */
export function answer(...args: Parameters<typeof run>): Children & { runs: ReturnType<typeof run>[] } {
  const placed = run(...args);
  const rest = args[3] ?? Math.max(0, 1 - args[0].reduce((a, b) => a + b, 0));
  return rest === 0 ? { runs: [placed], count: placed.rank + placed.branches.length } : { runs: [placed] };
}

/** A tree whose nodes along ``path`` all split their interval with ``weights``. */
export function chainTree(weights: number[], path: number[], measures = 1): IntervalTree {
  const tree = new IntervalTree(measures);
  tree.insert(ROOT, 0, answer(weights));
  let node: NodeId = ROOT;
  for (const branch of path) {
    node = tree.child(node, branch);
    tree.insert(node, 0, answer(weights));
  }
  return tree;
}
