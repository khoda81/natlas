import { describe, expect, it } from "bun:test";
import { IntervalTree, isTail, slotOf } from "../src/core/tree";
import { listing } from "./build";

describe("Slots", () => {
  it("lists p's children in p's order and keeps p's leftover as one tail slot", () => {
    const tree = new IntervalTree(1);
    tree.setListing(tree.root, 0, listing([0.5, 0.3], [10, 20], 2));
    const slots = tree.slots(tree.root, { p: 0, q: 0 });
    expect(slots.labels).toEqual([10, 20]);
    expect(slots.count).toBe(3);
    expect(isTail(slots, 2)).toBe(true);
    expect(slots.p[2]).toBeCloseTo(0.2, 9);
    expect(slotOf(slots, 20)).toBe(1);
  });

  it("bounds q where q has not listed a child", () => {
    const tree = new IntervalTree(2);
    tree.setListing(tree.root, 0, listing([0.5, 0.3], [0, 1], 2));
    tree.setListing(tree.root, 1, listing([0.6, 0.3], [1, 2], 2));
    const slots = tree.slots(tree.root, { p: 0, q: 1 });
    expect(slots.q.map((weight) => Number(weight.toFixed(9)))).toEqual([0.1, 0.6, 0.3]);
    expect(slots.qBound).toEqual([true, false, true]);
  });

  it("bounds a tail's children by the horizon and the lightest listed child", () => {
    const tree = new IntervalTree(1);
    tree.setListing(tree.root, 0, { ...listing([0.5, 0.3], [0, 1], 2), complete: 4 });
    expect(tree.slots(tree.root, { p: 0, q: 0 }).q[2]).toBe(2 ** -4);
  });
});

describe("IntervalTree", () => {
  it("uses the integer label path as node identity", () => {
    const tree = new IntervalTree(2);
    const child = tree.child(tree.root, 5);
    const grandchild = tree.child(child, 2);
    expect(tree.child(tree.root, 5)).toBe(child);
    expect(tree.parent(grandchild)).toBe(child);
    expect(tree.label(grandchild)).toBe(2);
    expect(tree.path(grandchild)).toEqual([5, 2]);
    expect(tree.chain(grandchild)).toEqual([tree.root, child, grandchild]);
  });

  it("tells an unexpanded node from one with nothing within the horizon", () => {
    const tree = new IntervalTree(1);
    expect(tree.expanded(tree.root, 0)).toBe(false);
    tree.setListing(tree.root, 0, { ...listing([]), complete: 0 });
    expect(tree.expanded(tree.root, 0)).toBe(false);
    tree.setListing(tree.root, 0, listing([]));
    expect(tree.expanded(tree.root, 0)).toBe(true);
  });
});
