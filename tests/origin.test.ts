import { expect, it } from "bun:test";
import { type Camera, layout } from "../src/core/layout";
import { wants } from "../src/data/expand";
import { IntervalTree } from "../src/core/tree";
import { listing } from "./build";

const view = { width: 1000, height: 800 };
const lens = { p: 0, q: 1 };

it("opens a deep node on its own: demand names it and then its children, never an ancestor", () => {
  const tree = new IntervalTree(2);
  const origin = tree.descend(tree.root, Array.from({ length: 3000 }, (_, i) => i % 7));
  const cam: Camera = { origin, x: 0, k: 100, anchor: { node: origin, slot: 0 }, a: 0 };
  const demand = () => {
    const frame = layout(tree, lens, cam, view);
    try {
      return new Set(wants(tree, lens, cam, view, frame).map(({ node, measure }) => `${tree.path(node).length}:${measure}`));
    } finally { frame.dispose(); }
  };
  expect(demand()).toEqual(new Set(["3000:0", "3000:1"]));
  tree.setListing(origin, 0, listing([0.75, 0.25]));
  tree.setListing(origin, 1, listing([0.5, 0.5]));
  const depths = [...demand()].map((key) => Number(key.split(":")[0]));
  expect(Math.min(...depths)).toBe(3000);
  expect(depths).toContain(3001);
});
