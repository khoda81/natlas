import { describe, expect, it } from "bun:test";
import { type Camera, clamp, extent, layout, pick, reanchor, repair } from "../src/core/layout";
import { moveDepth, scale } from "../src/ui/camera";
import { chainTree, answer } from "./build";
import { IntervalTree, ROOT } from "../src/core/tree";

const view = { width: 1000, height: 800 };
const lens = { p: 0, q: 0 };
const at = (node: bigint, rank = 0) => ({ node, rank });

describe("layout", () => {
  it("keeps children separate when their height exceeds one physical pixel", () => {
    const tree = chainTree([0.997, 0.0015, 0.0009, 0.0006], []);
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: at(ROOT), a: 0 };
    const low = layout(tree, lens, cam, view, { mergePx: 1 });
    const high = layout(tree, lens, cam, view, { mergePx: 1 / 2 });
    // At 800 CSS pixels high, the last two children are 0.72 and 0.48 CSS
    // pixels tall. Only the latter is still subpixel on a 2× display.
    expect(low.boxes.find(box => box.rank === 2)).toMatchObject({ kind: "merged", count: 2 });
    expect(high.boxes.find(box => box.rank === 2)).toMatchObject({ kind: "leaf", branch: 2 });
    expect(high.boxes.find(box => box.rank === 3)).toMatchObject({ kind: "merged", count: 1 });
    for (const frame of [low, high]) {
      expect(pick(tree, lens, frame, cam, view, 100, 799.8, 1)?.box.branch).toBe(3);
      frame.dispose();
    }
  });

  it("colors a band by its children's mean log ratio and keeps its mass in lnMass", () => {
    const tree = new IntervalTree(2);
    const p = [0.5, 0.488, 0.006, 0.004, 0.002];
    const q = [0.5, 0.48, 0.004, 0.008, 0.008];
    tree.insert(ROOT, 0, answer(p));
    tree.insert(ROOT, 1, answer(q));
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: at(ROOT), a: 0 };
    const frame = layout(tree, { p: 0, q: 1 }, cam, view, { mergePx: 10 });
    const expected = p.slice(2).reduce((sum, value, i) => sum + value * Math.log(value / q[i + 2]!), 0) / 0.012;
    const band = frame.boxes.find(box => box.kind === "merged")!;
    expect(Math.log(band.p / band.q)).toBeCloseTo(expected, 10);
    expect(Math.exp(band.lnMass)).toBeCloseTo(0.012, 12);
    frame.dispose();
  });

  it("keeps the origin selectable before anything arrives, and asks for it", () => {
    const tree = new IntervalTree(1);
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: at(ROOT), a: 0 };
    const frame = layout(tree, lens, cam, view);
    expect(pick(tree, lens, frame, cam, view, 500, 400, 1)).toMatchObject({ box: { node: ROOT }, exact: false });
    expect(pick(tree, lens, frame, cam, view, -1, 400, 1)).toBeNull();
    // Two points per visible pixel, through the whole root.
    expect(frame.misses).toMatchObject([{ node: ROOT, target: ROOT, measure: 0, priority: 800 }]);
    const { read } = frame.misses[0]!;
    expect("points" in read && read.points.length).toBe(1600);
    expect(frame.cover).toEqual({ node: ROOT, y: 0 });
  });

  it("splits heights by p in p's order and sizes widths by -ln q", () => {
    const tree = new IntervalTree(1);
    tree.insert(ROOT, 0, answer([0.5, 0.25, 0.25], [7, 3, 9]));
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: at(ROOT), a: 0 };
    const near = (value: number) => expect.closeTo(value, 12);
    expect(layout(tree, lens, cam, view).boxes.map((box) => [box.branch, box.y0, box.y1, box.x1 - box.x0])).toEqual([
      [7, 0, near(0.5), near(Math.log(2))],
      [3, near(0.5), near(0.75), near(Math.log(4))],
      [9, near(0.75), near(1), near(Math.log(4))],
    ]);
  });

  it("counts the children in a gap where the source says where the node ends, and draws them at their mean weight", () => {
    const tree = new IntervalTree(1);
    tree.insert(ROOT, 0, { ...answer([0.5, 0.25], [4, 7]), count: 9 });
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: at(ROOT), a: 0 };
    const gap = layout(tree, lens, cam, view).boxes.find((box) => box.kind === "gap")!;
    expect([gap.rank, gap.count]).toEqual([2, 7]);
    expect(gap.x1 - gap.x0).toBeCloseTo(Math.log(7 / 0.25), 12);
  });

  it("selects the containing known branch where its continuation is unknown", () => {
    const tree = chainTree([0.5, 0.5], []);
    const cam: Camera = { origin: ROOT, x: 2, k: 100, anchor: at(ROOT), a: 0 };
    const hit = pick(tree, lens, layout(tree, lens, cam, view), cam, view, 500, 400, 1)!;
    expect(hit.exact).toBe(false);
    expect(hit.box.branch).toBe(0);
  });

  it("stays exact at depths beyond absolute float precision", () => {
    const path = Array.from({ length: 80 }, (_, i) => i % 2);
    const tree = chainTree([0.5, 0.5], path);
    const deepest = tree.descend(ROOT, path);
    const cam: Camera = { origin: ROOT, x: path.length * Math.log(2), k: 100, anchor: at(deepest, 1), a: 0.5 };
    const frame = layout(tree, lens, cam, view);
    const children = frame.boxes.filter((box) => box.node === deepest).sort((a, b) => a.rank - b.rank);
    expect(children.map((box) => [box.y0, box.y1])).toEqual([[expect.closeTo(0, 12), expect.closeTo(0.5, 12)], [expect.closeTo(0.5, 12), expect.closeTo(1, 12)]]);
  });

  it("re-anchoring moves nothing, and horizontal scale moves nothing vertically", () => {
    const path = Array.from({ length: 30 }, (_, i) => (i * 7) % 3);
    const tree = chainTree([0.6, 0.3, 0.1], path);
    const cam: Camera = { origin: ROOT, x: 12.3, k: 140, anchor: at(tree.descend(ROOT, path)), a: 0.41 };
    const key = (box: { node: bigint; rank: number; kind: string }) => `${box.node}:${box.rank}:${box.kind}`;
    const before = new Map(layout(tree, lens, cam, view).boxes.map((box) => [key(box), box.y0]));
    for (const move of [(c: Camera) => reanchor(c, layout(tree, lens, c, view)), (c: Camera) => scale(c, 3)]) {
      const moved = { ...cam };
      move(moved);
      const after = layout(tree, lens, moved, view).boxes.filter((box) => before.has(key(box)));
      expect(after.length).toBeGreaterThan(0);
      for (const box of after) expect(box.y0).toBeCloseTo(before.get(key(box))!, 9);
    }
  });

  it("zooms about the fixed view fraction, stopping only at the root's left edge", () => {
    const cam: Camera = { origin: ROOT, x: 1, k: 100, anchor: at(2n, 1), a: 0.3 };
    moveDepth(cam, 1.5, 0.3);
    expect(cam.a).toBeCloseTo(0.3, 15);
    const root: Camera = { origin: ROOT, x: 0.5, k: 100, anchor: at(ROOT), a: 0 };
    moveDepth(root, -2, 0.5);
    expect(root.x).toBe(0);
    const deeper: Camera = { ...root, origin: 3n, x: 0.5, a: 0 };
    moveDepth(deeper, -2, 0.5);
    expect(deeper.x).toBe(-1.5);
    expect(deeper.a).toBeCloseTo(0.5 - 0.5 * Math.exp(-2), 12);
    scale(cam, 1e12);
    expect(cam.k).toBe(1e14);
  });

  it("clamps the view inside the root's interval", () => {
    const tree = chainTree([0.5, 0.5], [0]);
    const cam: Camera = { origin: ROOT, x: 0.5, k: 100, anchor: at(tree.descend(ROOT, [0])), a: 0.4 };
    clamp(tree, lens, cam);
    expect(extent(tree, lens, cam).top).toBeCloseTo(0, 15);
    expect(cam.a).toBeCloseTo(0, 15);
  });

  it("moves the anchor up to what p places", () => {
    const tree = chainTree([0.5, 0.5], [1, 0], 2);
    tree.insert(ROOT, 1, answer([1]));
    const deep = tree.descend(ROOT, [1, 0]);
    const cam: Camera = { origin: ROOT, x: 1, k: 100, anchor: at(deep, 1), a: 0.4 };
    repair(tree, { p: 1, q: 1 }, cam);
    expect(cam.anchor).toEqual(at(ROOT));
    expect(() => layout(tree, { p: 1, q: 1 }, cam, view)).not.toThrow();
  });
});
