import { describe, expect, it } from "bun:test";
import { type Camera, clamp, layout, pick, reanchor, repair, rootExtent } from "../src/core/layout";
import { moveDepth, scale } from "../src/ui/camera";
import { chainTree, listing } from "./build";
import { IntervalTree } from "../src/core/tree";

const view = { width: 1000, height: 800 };
const lens = { p: 0, q: 0 };

describe("layout", () => {
  it("keeps the root selectable before discovery and leaves the canvas bounds alone", () => {
    const tree = new IntervalTree(1);
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const frame = layout(tree, lens, cam, view);
    expect(pick(frame, cam, view, 500, 400)?.box.node).toBe(tree.root);
    expect(pick(frame, cam, view, 500, 400)?.exact).toBe(false);
    expect(pick(frame, cam, view, -1, 400)).toBeNull();
    expect(pick(frame, cam, view, 500, -1)).toBeNull();
    expect(frame.cover).toEqual({ node: tree.root, y: 0 });
  });

  it("selects the containing known branch when its box is left of undiscovered space", () => {
    const tree = chainTree([0.5, 0.5], []);
    const cam: Camera = { x: 2, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const frame = layout(tree, lens, cam, view);
    const hit = pick(frame, cam, view, 500, 400)!;
    expect(hit.exact).toBe(false);
    expect(hit.box.label).toBe(0);
  });

  it("preserves area while merging and still picks the individual integer branches", () => {
    const weights = [0.5, 0.488, 0.006, 0.004, 0.002];
    const tree = chainTree(weights, []);
    const cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const frame = layout(tree, lens, cam, view, { mergePx: 10 });
    const run = frame.boxes.find((box) => box.kind === "merged")!;
    const expected = weights.slice(2).reduce((area, weight) => area - weight * Math.log(weight), 0);
    expect(run.p * (run.x1 - run.x0)).toBeCloseTo(expected, 6);
    const hit = pick(frame, cam, view, 100, 0.991 * view.height)!;
    expect(hit.box.label).toBe(2);
    expect(hit.box.kind).not.toBe("merged");
  });

  it("finds a smaller covering node even when the camera anchor is still its parent", () => {
    const tree = chainTree([0.5, 0.5], [0, 1]);
    const cam: Camera = { x: Math.log(4), k: 100, anchor: { node: 0, slot: 0 }, a: -1 };
    const frame = layout(tree, lens, cam, view);
    expect(frame.cover?.node).toBe(tree.child(tree.child(0, 0), 1));
  });

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

  it("stops at the root's left edge", () => {
    const cam: Camera = { x: 0.5, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    moveDepth(cam, -2, 0.5);
    expect(cam.x).toBe(0);
    moveDepth(cam, -1, 0.5);
    expect(cam.x).toBe(0);
  });

  it("scales horizontally without moving anything vertically", () => {
    const tree = chainTree([0.6, 0.3, 0.1], [0, 1]);
    const cam: Camera = { x: 1.2, k: 100, anchor: { node: 2, slot: 0 }, a: 0.3 };
    const before = layout(tree, lens, cam, view);
    scale(cam, 3);
    const after = layout(tree, lens, cam, view);
    expect(cam.x).toBe(1.2);
    const key = (box: { node: number; slot: number }) => `${box.node}:${box.slot}`;
    const rows = new Map(before.boxes.map((box) => [key(box), [box.y0, box.y1]]));
    const shared = after.boxes.filter((box) => rows.has(key(box)));
    expect(shared.length).toBeGreaterThan(0);
    for (const box of shared) expect([box.y0, box.y1]).toEqual(rows.get(key(box))!);
  });

  it("does not bound horizontal scale", () => {
    const cam: Camera = { x: 1, k: 10, anchor: { node: 0, slot: 0 }, a: 0 };
    scale(cam, 1e12);
    expect(cam.k).toBe(1e13);
    scale(cam, 1e-24);
    expect(cam.k).toBe(1e-11);
  });

  it("clamps the view inside the root interval", () => {
    const tree = chainTree([0.5, 0.5], [0]);
    const cam: Camera = { x: 0.5, k: 100, anchor: { node: 1, slot: 0 }, a: 0.4 };
    clamp(tree, lens, cam);
    expect(rootExtent(tree, lens, cam).top).toBeCloseTo(0, 15);
    expect(cam.a).toBeCloseTo(0, 15);
  });

  it("reports an unexpanded child standing like the root, though its box is off screen", () => {
    // The root's first child fills the view with its box ending at the left
    // edge, as after a double-click; its children are what should show.
    const tree = chainTree([0.5, 0.5], []);
    const cam: Camera = { x: Math.log(2), k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
    const frame = layout(tree, lens, cam, view);
    expect(frame.boxes.some((box) => box.node === 0 && box.slot === 0)).toBe(false);
    expect(frame.unexpanded).toContainEqual({ node: 0, label: 0, lnMass: expect.closeTo(-Math.log(2), 9), x1: expect.closeTo(Math.log(2), 9) });
  });

  it("moves the anchor up to what p lists", () => {
    // Measure 1 lists only the root's first child, so node 2 is out of reach under it.
    const tree = chainTree([0.5, 0.5], [1, 0], 2);
    for (const node of tree.chain(2)) tree.setListing(node, 1, listing([1]));
    const cam: Camera = { x: 1, k: 100, anchor: { node: 2, slot: 1 }, a: 0.4 };
    repair(tree, { p: 1, q: 1 }, cam);
    expect(cam.anchor).toEqual({ node: 0, slot: 0 });
    expect(() => layout(tree, { p: 1, q: 1 }, cam, view)).not.toThrow();
  });
});
