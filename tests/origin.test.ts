import { expect, it } from "bun:test";
import { type Camera, top, view } from "../src/core/layout";
import { IntervalTree, ROOT } from "../src/core/tree";
import { answer } from "./build";

const viewport = { width: 1000, height: 800 };
const lens = { p: 0, q: 1 };

it("looks at a deep node opened on its own, never an ancestor", () => {
  const tree = new IntervalTree();
  const origin = tree.descend(ROOT, Array.from({ length: 3000 }, (_, i) => i % 7));
  const cam: Camera = { origin, x: 0, k: 100, anchor: top(), a: 0 };
  const looked = () => view(tree, lens, cam, viewport, 800);
  expect(looked()).toMatchObject({ node: origin, lnHeight: 0, rows: 800, right: 10 });
  tree.insert(origin, 0, answer([0.75, 0.25]));
  tree.insert(origin, 1, answer([0.5, 0.5]));
  // Its children known, the view is still all of it.
  expect(looked()).toMatchObject({ node: origin, lnHeight: 0 });
});
