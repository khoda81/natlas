import { describe, expect, it } from "bun:test";
import { moveDepth } from "../src/camera";
import { type Camera, clamp, layout, reanchor, repair, rootExtent } from "../src/layout";
import { chainTree, listing } from "./build";

const view = { width: 1000, height: 800 };
const lens = { p: 0, q: 0 };

describe("layout", () => {
  it("partitions the root interval by p and sizes widths by -ln q", () => {
    const tree = chainTree([0.5, 0.25, 0.25], []);
    const cam: Camera = { x: -0.1, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const frame = layout(tree, lens, cam, view);
    // Weights travel as rounded logs, so they arrive within 1/LOG_SCALE nat.
    const near = (value: number) => expect.closeTo(value, 4);
    const spans = frame.boxes.map((box) => [box.slot, box.y0, box.y1, box.x1 - box.x0]);
    expect(spans).toEqual([
      [0, 0, near(0.5), near(Math.log(2))],
      [1, near(0.5), near(0.75), near(Math.log(4))],
      [2, near(0.75), near(1), near(Math.log(4))],
    ]);
    expect(frame.rootTop).toBe(0);
    expect(frame.rootBottom).toBeCloseTo(1, 12);
  });

  it("stays exact at depths beyond absolute float precision", () => {
    // 80 halvings put the deepest node 2^-80 of the root tall, at an offset
    // from the root top that no double can resolve to that size.
    const path = Array.from({ length: 80 }, (_, i) => i % 2);
    const tree = chainTree([0.5, 0.5], path);
    const deepest = path.length;
    const cam: Camera = { x: path.length * Math.log(2), k: 100, anchor: { node: deepest, slot: 1 }, a: 0.5 };
    const frame = layout(tree, lens, cam, view);
    // Absolute coordinates would be off by ~1e8 view heights here; anchored
    // ones carry only the rounding of exp(ln mass - ln view).
    const near = (actual: number | undefined, expected: number) => expect(actual).toBeCloseTo(expected, 12);
    const children = frame.boxes.filter((box) => box.node === deepest).sort((a, b) => a.slot - b.slot);
    expect(children).toHaveLength(2);
    near(children[0]?.y0, 0);
    near(children[0]?.y1, 0.5);
    near(children[1]?.y0, 0.5);
    near(children[1]?.y1, 1);
    // The deepest node's own box, one level up, fills the view.
    const own = frame.boxes.find((box) => box.node === deepest - 1 && box.slot === path.at(-1));
    near(own?.y0, 0);
    near(own?.y1, 1);
  });

  it("re-anchoring does not move anything", () => {
    const path = Array.from({ length: 30 }, (_, i) => (i * 7) % 3);
    const tree = chainTree([0.6, 0.3, 0.1], path);
    const cam: Camera = { x: 12.3, k: 140, anchor: { node: 30, slot: 0 }, a: 0.41 };
    const before = layout(tree, lens, cam, view);
    reanchor(cam, before);
    const after = layout(tree, lens, cam, view);
    const key = (box: { node: number; slot: number }) => `${box.node}:${box.slot}`;
    const positions = new Map(before.boxes.map((box) => [key(box), box.y0]));
    for (const box of after.boxes) {
      const y0 = positions.get(key(box));
      if (y0 !== undefined) expect(box.y0).toBeCloseTo(y0, 12);
    }
  });

  it("zooms about the fixed view fraction", () => {
    const cam: Camera = { x: 1, k: 100, anchor: { node: 2, slot: 1 }, a: 0.3 };
    moveDepth(cam, 1.5, 0.3);
    expect(cam.a).toBeCloseTo(0.3, 15);
  });

  it("clamps the view inside the root interval", () => {
    const tree = chainTree([0.5, 0.5], [0]);
    const cam: Camera = { x: 0.5, k: 100, anchor: { node: 1, slot: 0 }, a: 0.4 };
    clamp(tree, lens, cam);
    expect(rootExtent(tree, lens, cam).top).toBeCloseTo(0, 15);
    expect(cam.a).toBeCloseTo(0, 15);
  });

  it("moves the anchor up to what p lists", () => {
    // Measure 1 lists only the root's first child, so node 2 is out of reach under it.
    const tree = chainTree([0.5, 0.5], [1, 0]);
    for (const node of tree.nodes) tree.setListing(node.id, 1, listing([1]));
    const cam: Camera = { x: 1, k: 100, anchor: { node: 2, slot: 1 }, a: 0.4 };
    repair(tree, { p: 1, q: 1 }, cam);
    expect(cam.anchor).toEqual({ node: 0, slot: 0 });
    expect(() => layout(tree, { p: 1, q: 1 }, cam, view)).not.toThrow();
  });
});
