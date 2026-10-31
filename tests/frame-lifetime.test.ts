import { expect, it } from "bun:test";
import { IntervalTree, ROOT } from "../src/core/tree";
import { type Camera, layout, pick } from "../src/core/layout";
import { answer } from "./build";

const lens = { p: 0, q: 0 };
const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: { node: ROOT, rank: 0 }, a: 0 };
const view = { width: 1234, height: 789 };
const uniform = (count: number) => answer(Array(count).fill(1 / count), undefined, 0, 0);

it("picks single children out of merged bands, and frames reuse buffers without touching live ones", () => {
  const tree = new IntervalTree(1);
  tree.insert(ROOT, 0, uniform(150000));
  const frame = layout(tree, lens, cam, view);
  // Every row is a band; no branch is drawn alone.
  expect(frame.boxes.every((box) => box.kind === "merged")).toBe(true);
  expect(frame.boxes.length).toBeLessThanOrEqual(view.height);
  const y = (74999.5 / 150000) * view.height;
  const picked = pick(tree, lens, frame, cam, view, 100, y, 1)!;
  expect(picked.box.branch).toBe(74999);
  expect(picked.exact).toBe(true);
  const boxes = frame.boxes.map((box) => ({ ...box }));
  for (let i = 0; i < 20; i++) layout(tree, lens, { ...cam, x: i / 100 }, view).dispose();
  expect(frame.boxes).toEqual(boxes);
  frame.dispose();
  frame.dispose();
});
