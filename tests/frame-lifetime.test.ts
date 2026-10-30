import { expect, it } from "bun:test";
import { IntervalTree } from "../src/core/tree";
import { layout, pick } from "../src/core/layout";

const lens = { p: 0, q: 0 };
const cam = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
const view = { width: 1234, height: 789 };
const distribution = (count: number) => ({ labels: Array.from({ length: count }, (_, i) => i),
  lnWeights: Array(count).fill(-Math.log(count)), horizon: 40, complete: 40, leftover: 0 });

it("keeps retained subpixel selection stable while discovery replaces listings", () => {
  const tree = new IntervalTree(1);
  tree.setListing(0, 0, distribution(2000));
  const child = tree.child(0, 17);
  const before = layout(tree, lens, cam, view);
  const y = (17.5 / 2000) * view.height;
  const selected = pick(before, cam, view, 100, y)!;
  expect(selected.box.label).toBe(17);
  expect(selected.box.kind).toBe("leaf");
  tree.setListing(child, 0, distribution(2));
  const after = layout(tree, lens, cam, view);
  expect(pick(after, cam, view, 100, y)!.box.kind).toBe("node");
  expect(pick(before, cam, view, 100, y)).toEqual(selected);
  tree.setListing(0, 0, distribution(3));
  const changed = layout(tree, lens, cam, view);
  expect(pick(before, cam, view, 100, y)).toEqual(selected);
  after.dispose(); changed.dispose();
  // Reuse both native and JS buffers repeatedly without mutating a live frame.
  const snapshot = JSON.stringify(before);
  for (let i = 0; i < 20; i++) layout(tree, lens, { ...cam, x: i / 100 }, view).dispose();
  expect(JSON.stringify(before)).toBe(snapshot);
  before.dispose(); before.dispose();
});

it("queries coverage, membership, bounds and cached statistics without exporting a vocabulary", () => {
  const tree = new IntervalTree(2);
  expect(tree.hasListing(0, 0)).toBe(false);
  tree.setListing(0, 0, { ...distribution(2000), leftover: 0.1 });
  tree.setListing(0, 1, distribution(1000));
  const lens = { p: 0, q: 1 };
  const slots = tree.slots(0, lens);
  const entropy = slots.p.reduce((h, p) => p === 0 ? h : h - p * Math.log(p), 0);
  const cross = slots.p.reduce((h, p, i) => p === 0 ? h : h - p * Math.log(slots.q[i]!), 0);
  tree.raw.listing = () => { throw new Error("full listing export"); };
  tree.raw.slots = () => { throw new Error("full slot export"); };
  expect(tree.hasListing(0, 0)).toBe(true);
  expect(tree.horizon(0, 0)).toBe(40);
  expect(tree.complete(0, 0)).toBe(40);
  expect(tree.slotOf(0, lens, 1999)).toBe(1999);
  expect(tree.slotOf(0, lens, 9999)).toBeUndefined();
  expect(tree.qBound(0, lens, 1)).toBe(false);
  expect(tree.qBound(0, lens, 1999)).toBe(true);
  expect(tree.statistics(0, lens)).toEqual({ entropy: expect.closeTo(entropy, 12), cross: expect.closeTo(cross, 12), bounded: true });
});
