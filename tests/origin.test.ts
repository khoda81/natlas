import { expect, it } from "bun:test";
import { type Camera, layout } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { answer } from "./build";

const view = { width: 1000, height: 800 };
const lens = { p: 0, q: 1 };

it("opens a deep node on its own: its frame asks for it and then its children, never an ancestor", () => {
  const tree = new IntervalTree(2);
  const origin = tree.descend(ROOT, Array.from({ length: 3000 }, (_, i) => i % 7));
  const cam: Camera = { origin, x: 0, k: 100, anchor: { node: origin, rank: 0 }, a: 0 };
  const asked = () => {
    const frame = layout(tree, lens, cam, view);
    try {
      return frame.misses.map(({ node, target, measure }) => `${tree.path(node).length + (target === node ? 0 : 1)}:${measure}`);
    } finally { frame.dispose(); }
  };
  expect(asked()).toEqual(["3000:0"]);
  tree.insert(origin, 0, answer([0.75, 0.25]));
  // Until q says how wide the children are, they reach past the view.
  expect(asked()).toEqual(["3000:1"]);
  tree.insert(origin, 1, answer([0.5, 0.5]));
  expect(asked()).toEqual(["3001:0", "3001:0"]);
});
