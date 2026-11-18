import { expect, it } from "bun:test";
import { IntervalTree, ROOT } from "../src/core/tree";
import { type Camera, layout, pick, top } from "../src/core/layout";
import { answer } from "./build";

const lens = { p: 0, q: 0 };
const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
const view = { width: 1234, height: 789 };
const uniform = (count: number) => answer(Array(count).fill(1 / count), undefined, 0, 0);

it("picks single children out of rows, and frames reuse buffers without touching live ones", () => {
  const tree = new IntervalTree();
  tree.insert(ROOT, 0, uniform(150000));
  const frame = layout(tree, lens, cam, view);
  // Every child is thinner than a row: no box, and at most a span a row.
  expect(frame.boxes.length).toBe(0);
  expect(frame.spans.length).toBeLessThanOrEqual(view.height);
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
