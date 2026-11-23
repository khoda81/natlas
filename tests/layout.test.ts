import { describe, expect, it } from "bun:test";
import { aim, type Camera, clamp, extent, layout, pick, project, settle, top, view as viewOf } from "../src/core/layout";
import { glide, moveDepth, scale } from "../src/ui/camera";
import { chainTree, answer, digits } from "./build";
import { IntervalTree, ROOT } from "../src/core/tree";

const view = { width: 1000, height: 800 };
const lens = { p: 0, q: 0 };

describe("layout", () => {
  it("keeps children separate when their height exceeds one physical pixel, and paints thinner ones in their rows", () => {
    const tree = chainTree([0.997, 0.0015, 0.0009, 0.0006], []);
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
    const low = layout(tree, lens, cam, view, { rowPx: 1 });
    const high = layout(tree, lens, cam, view, { rowPx: 1 / 2 });
    // At 800 CSS pixels high, the last two children are 0.72 and 0.48 CSS
    // pixels tall: no box of their own, and the last physical row holds
    // both, with a horizontal boundary reaching the wider child. On a 2× display the
    // first of them is a box of its own.
    expect(low.boxes.map(box => box.rank)).toEqual([0, 1]);
    const last = low.spans.find(span => Math.round(span.y0 * view.height) === 799)!;
    expect(last.fill).toBe(last.x0);
    expect(last.reach - last.x0).toBeCloseTo(-Math.log(0.0006), 12);
    expect(high.boxes.find(box => box.rank === 2)).toMatchObject({ kind: "leaf", branch: 2 });
    for (const frame of [low, high]) {
      expect(pick(tree, lens, frame, cam, view, 100, 799.8, 1)?.box.branch).toBe(3);
      frame.dispose();
    }
  });

  it("draws p's heights and q's widths", () => {
    const tree = new IntervalTree();
    tree.insert(ROOT, 0, answer([0.5, 0.5]));
    tree.insert(ROOT, 1, answer([0.25, 0.75]));
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
    const frame = layout(tree, { p: 0, q: 1 }, cam, view);
    const widths = frame.spans.map(span => [Math.round(span.y0 * view.height), Math.round(span.y1 * view.height),
      span.fill - span.x0, span.reach - span.x0]);
    // Each half as wide as −ln q, its outline at its end; outline rows at
    // the node's top and at the boundary between them.
    const [wide, narrow] = [-Math.log(0.25), -Math.log(0.75)];
    expect(widths).toEqual([
      [0, 1, 0, expect.closeTo(wide, 12)],
      [1, 400, expect.closeTo(wide, 12), expect.closeTo(wide, 12)],
      [400, 401, 0, expect.closeTo(wide, 12)],
      [401, 800, expect.closeTo(narrow, 12), expect.closeTo(narrow, 12)],
    ]);
    frame.dispose();
  });

  it("keeps the origin selectable before anything arrives, and looks at all of it", () => {
    const tree = new IntervalTree();
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
    const frame = layout(tree, lens, cam, view);
    expect(pick(tree, lens, frame, cam, view, 500, 400, 1)).toMatchObject({ box: { node: ROOT }, exact: false });
    expect(pick(tree, lens, frame, cam, view, -1, 400, 1)).toBeNull();
    // The whole root, worth every row, ten nats deep.
    expect(viewOf(tree, lens, cam, view, 1600)).toEqual({ node: ROOT, top: new Uint8Array(), lnHeight: 0, rows: 1600, right: 10 });
  });

  it("splits heights by p in p's order and sizes widths by -ln q", () => {
    const tree = new IntervalTree();
    tree.insert(ROOT, 0, answer([0.5, 0.25, 0.25], [7, 3, 9]));
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
    const near = (value: number) => expect.closeTo(value, 12);
    expect(layout(tree, lens, cam, view).boxes.map((box) => [box.branch, box.y0, box.y1, box.x1 - box.x0])).toEqual([
      [7, 0, near(0.5), near(Math.log(2))],
      [3, near(0.5), near(0.75), near(Math.log(4))],
      [9, near(0.75), near(1), near(Math.log(4))],
    ]);
  });

  it("counts the children in a gap where the source says where the node ends, and draws them at their mean weight", () => {
    const tree = new IntervalTree();
    tree.insert(ROOT, 0, { ...answer([0.5, 0.25], [4, 7]), count: 9 });
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
    const gap = layout(tree, lens, cam, view).boxes.find((box) => box.kind === "gap")!;
    expect([gap.rank, gap.count]).toEqual([2, 7]);
    expect(gap.x1 - gap.x0).toBeCloseTo(Math.log(7 / 0.25), 12);
  });

  it("selects the containing known branch where its continuation is unknown", () => {
    const tree = chainTree([0.5, 0.5], []);
    const cam: Camera = { origin: ROOT, x: 2, k: 100, anchor: top(), a: 0 };
    const hit = pick(tree, lens, layout(tree, lens, cam, view), cam, view, 500, 400, 1)!;
    expect(hit.exact).toBe(false);
    expect(hit.box.branch).toBe(0);
  });

  it("stays exact at depths beyond absolute float precision", () => {
    const path = Array.from({ length: 80 }, (_, i) => i % 2);
    const tree = chainTree([0.5, 0.5], path);
    const deepest = tree.descend(ROOT, path);
    const cam: Camera = { origin: ROOT, x: path.length * Math.log(2), k: 100, anchor: { path, digits: digits(0.5) }, a: 0.5 };
    const frame = layout(tree, lens, cam, view);
    const children = frame.boxes.filter((box) => box.node === deepest).sort((a, b) => a.rank - b.rank);
    expect(children.map((box) => [box.y0, box.y1])).toEqual([[expect.closeTo(0, 12), expect.closeTo(0.5, 12)], [expect.closeTo(0.5, 12), expect.closeTo(1, 12)]]);
  });

  it("settling moves nothing, and horizontal scale moves nothing vertically", () => {
    const path = Array.from({ length: 30 }, (_, i) => (i * 7) % 3);
    const tree = chainTree([0.6, 0.3, 0.1], path);
    const cam: Camera = { origin: ROOT, x: 12.3, k: 140, anchor: top(path), a: 0.41 };
    const key = (box: { node: bigint; rank: number; kind: string }) => `${box.node}:${box.rank}:${box.kind}`;
    const before = new Map(layout(tree, lens, cam, view).boxes.map((box) => [key(box), box.y0]));
    for (const move of [(c: Camera) => settle(tree, lens, c), (c: Camera) => scale(c, 3)]) {
      const moved = { ...cam };
      move(moved);
      const after = layout(tree, lens, moved, view).boxes.filter((box) => before.has(key(box)));
      expect(after.length).toBeGreaterThan(0);
      for (const box of after) expect(box.y0).toBeCloseTo(before.get(key(box))!, 9);
    }
  });

  it("zooms about the fixed view fraction, stopping only at the root's left edge", () => {
    const cam: Camera = { origin: ROOT, x: 1, k: 100, anchor: top(), a: 0.3 };
    moveDepth(cam, 1.5, 0.3);
    expect(cam.a).toBeCloseTo(0.3, 15);
    const root: Camera = { origin: ROOT, x: 0.5, k: 100, anchor: top(), a: 0 };
    moveDepth(root, -2, 0.5);
    expect(root.x).toBe(0);
    const deeper: Camera = { ...root, origin: 3n, x: 0.5, a: 0 };
    moveDepth(deeper, -2, 0.5);
    expect(deeper.x).toBe(-1.5);
    expect(deeper.a).toBeCloseTo(0.5 - 0.5 * Math.exp(-2), 12);
    scale(cam, 1e12);
    expect(cam.k).toBe(1e14);
  });

  it("glides the view as an interval, so an edge that ends where it starts never moves", () => {
    // The view's top and bottom below the camera's point, in origin heights.
    const edges = (cam: Camera) => [-cam.a * Math.exp(-cam.x), (1 - cam.a) * Math.exp(-cam.x)];
    const cam: Camera = { origin: 3n, x: 1, k: 100, anchor: top(), a: 0.2 };
    const [topEdge] = edges(cam);
    const to = { x: 3, a: 0.2 * Math.exp(2) };
    for (let eased = 0, step = 1; step <= 10; step++) {
      const next = 1 - (1 - step / 10) ** 2;
      glide(cam, to, (next - eased) / (1 - eased));
      eased = next;
      expect(edges(cam)[0]! / topEdge!).toBeCloseTo(1, 12);
    }
    expect(cam.x).toBeCloseTo(to.x, 12);
    expect(cam.a).toBeCloseTo(to.a, 12);
    // Without a change of depth, it pans.
    glide(cam, { x: 3, a: 0.5 }, 0.5);
    expect(cam.a).toBeCloseTo(to.a + (0.5 - to.a) / 2, 12);
  });

  it("clamps the view inside the root's interval", () => {
    const tree = chainTree([0.5, 0.5], [0]);
    const cam: Camera = { origin: ROOT, x: 0.5, k: 100, anchor: top([0]), a: 0.4 };
    clamp(tree, lens, cam);
    expect(extent(tree, lens, cam, 0).top).toBeCloseTo(0, 15);
    expect(cam.a).toBeCloseTo(0, 15);
  });

  it("projects the frame and physical-row demand from the same clamped camera", () => {
    const tree = chainTree([0.5, 0.5], [0, 0, 0, 0]);
    const cam: Camera = { origin: ROOT, x: 2, k: 100, anchor: top([0, 0, 0]), a: 0.3 };
    const expected = { ...cam };
    settle(tree, lens, expected);
    clamp(tree, lens, expected);
    const dpr = 1.25;
    const { frame, demand } = project(tree, lens, cam, view, dpr);
    const reference = layout(tree, lens, expected, view, { rowPx: 1 / dpr });
    expect(expected.a).toBe(0);
    expect(cam).toEqual(expected);
    expect(frame.boxes).toEqual(reference.boxes);
    expect(frame.spans).toEqual(reference.spans);
    expect(demand).toEqual(viewOf(tree, lens, expected, view, view.height * dpr));
    frame.dispose();
    reference.dispose();
  });

  it("identifies the boxes along the picked path without learning visible leaves", () => {
    const tree = chainTree([0.5, 0.5], [0, 0]);
    const child = tree.child(ROOT, 0);
    const grandchild = tree.child(child, 0);
    const cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
    const { frame } = project(tree, lens, cam, view, 1);
    const hit = pick(tree, lens, frame, cam, view, 160, 50, 1)!;
    expect(hit.box.node).toBe(grandchild);
    expect(hit.ancestors).toEqual([child, grandchild]);
    expect(hit.ancestors.map((node) => frame.parentBoxes.get(node)?.node)).toEqual([ROOT, child]);
    const [leaf] = [...frame.parentBoxes].find(([, box]) => box.node === ROOT && box.branch === 1)!;
    expect(() => tree.path(leaf)).toThrow("unknown node");
    frame.dispose();
  });

  it("ends the anchor's path at what p places", () => {
    const tree = chainTree([0.5, 0.5], [1, 0]);
    tree.insert(ROOT, 1, answer([1]));
    const cam: Camera = { origin: ROOT, x: 1, k: 100, anchor: { path: [1, 0], digits: digits(0.5) }, a: 0.4 };
    settle(tree, { p: 1, q: 1 }, cam);
    expect(cam.anchor).toEqual(top());
    expect(() => layout(tree, { p: 1, q: 1 }, cam, view)).not.toThrow();
  });

  it("aims at what a pointer shows, so zooming out and back in returns to the same camera", () => {
    const path = Array.from({ length: 60 }, (_, i) => (i * 7) % 3);
    const tree = chainTree([0.6, 0.3, 0.1], path);
    // The point half way down the deepest node, a little off the view's middle.
    const cam: Camera = { origin: ROOT, x: 70, k: 100, anchor: { path, digits: digits(0.5) }, a: 0.45 };
    const zoom = (dx: number) => {
      aim(tree, lens, cam, 0.37);
      moveDepth(cam, dx, 0.37);
      settle(tree, lens, cam);
    };
    // The first zoom moves the point under the pointer; after that only depth changes.
    zoom(0);
    expect(cam.a).toBe(0.37);
    expect(cam.anchor.digits).toBeInstanceOf(Uint8Array);
    const deep = { ...cam };
    for (let step = 0; step < 140; step++) zoom(-0.5);
    expect(cam.x).toBe(0);
    expect(cam.anchor.path).toEqual([]);
    expect(cam.anchor.digits.length).toBeGreaterThan(8);
    for (let step = 0; step < 140; step++) zoom(0.5);
    expect(cam).toEqual(deep);
    expect(deep.anchor.path.length).toBeGreaterThan(40);
  });
});
