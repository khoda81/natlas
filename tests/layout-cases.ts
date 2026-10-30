// Deterministic camera/distribution cases used to freeze the pre-optimization
// geometry, discovery demand, and exact/best-effort selection semantics.
import type { Camera, Lens, Viewport } from "../src/core/layout";
import type { Fetched } from "../src/core/tree";

interface FixtureTree {
  child(parent: number, label: number): number;
  setListing(node: number, measure: number, value: Fetched): void;
}

export function layoutCases<T extends FixtureTree>(Tree: new (measures: number) => T) {
  let seed = 8161;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
  const listing = (weights: number[], labels = weights.map((_, i) => i), tail = 0): Fetched => ({
    labels, lnWeights: weights.map(Math.log), leftover: tail, horizon: 24, complete: 24,
  });
  const cases: { name: string; tree: T; lens: Lens; cam: Camera; view: Viewport; mergePx: number }[] = [];
  for (const count of [0, 3, 73, 2048]) {
    const tree = new Tree(2);
    const weights = Array.from({ length: count }, () => random() ** 4);
    const total = weights.reduce((a, b) => a + b, 0);
    if (count) {
      tree.setListing(0, 0, listing(weights.map(w => w * 0.97 / total), undefined, 0.03));
      tree.setListing(0, 1, listing(weights.map(w => (w + 0.1) / (total + count * 0.1)), weights.map((_, i) => i).reverse()));
      for (let i = 0; i < Math.min(count, 12); i++) {
        const child = tree.child(0, i);
        tree.setListing(child, 0, listing([0.6, 0.24, 0.1], [9, 2, 0], 0.06));
        tree.setListing(child, 1, listing([0.7, 0.3], [2, 0]));
      }
    }
    for (let i = 0; i < 9; i++) {
      cases.push({ name: `${count}/${i}`, tree, lens: { p: i % 2, q: Math.floor(i / 3) % 2 },
        cam: { x: i * 0.7, k: [13, 140, 1000][i % 3]!, a: i % 3 === 0 ? -0.2 : 0, anchor: { node: 0, slot: i % Math.max(1, count) } },
        view: { width: 1234, height: 789 }, mergePx: i % 3 === 0 ? 10 : 1 });
    }
  }
  const vocabulary = new Tree(1);
  vocabulary.setListing(0, 0, listing(Array(150000).fill(1 / 150000)));
  cases.push({ name: "vocabulary/150000", tree: vocabulary, lens: { p: 0, q: 0 },
    cam: { x: 0.2, k: 100, anchor: { node: 0, slot: 74999 }, a: 0.5 },
    view: { width: 1234, height: 789 }, mergePx: 1 });
  const deep = new Tree(1);
  let node = 0;
  for (let depth = 0; depth <= 80; depth++) {
    deep.setListing(node, 0, listing([0.6, 0.3, 0.1]));
    if (depth < 80) node = deep.child(node, depth % 3);
    if ([1, 7, 30, 79].includes(depth)) for (const a of [-0.5, 0.4, 1.1]) {
      cases.push({ name: `deep/${depth}/${a}`, tree: deep, lens: { p: 0, q: 0 },
        cam: { x: depth * 1.1, k: 170, anchor: { node, slot: 1 }, a },
        view: { width: 1234, height: 789 }, mergePx: 1 });
    }
  }
  return cases;
}

export function samplePoints(width: number, height: number) {
  return [-1, 0, 1, width * 0.07, width * 0.4, width - 1, width].flatMap(x =>
    [-1, 0, 1, height * 0.01, height * 0.4, height * 0.5111, height - 1, height].map(y => [x, y] as const));
}
