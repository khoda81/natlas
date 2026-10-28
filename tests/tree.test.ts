import { describe, expect, it } from "bun:test";
import { IntervalTree, Slots } from "../src/tree";
import { listing } from "./build";

describe("Slots", () => {
  it("lists p's children in p's order and keeps p's leftover as one tail slot", () => {
    const slots = new Slots(listing([0.5, 0.3], [10, 20], 2), undefined);
    expect(slots.labels).toEqual([10, 20]);
    expect(slots.count).toBe(3);
    expect(slots.isTail(2)).toBe(true);
    expect(slots.p[2]).toBeCloseTo(0.2, 9);
    expect(slots.suffix(1)).toBeCloseTo(0.5, 9);
    expect(slots.slotOf(20)).toBe(1);
  });

  it("bounds q where q has not listed a child", () => {
    // q lists b and c: anything else weighs at most its leftover, 0.1.
    const p = listing([0.5, 0.3], [0, 1], 2);
    const q = listing([0.6, 0.3], [1, 2], 2);
    const slots = new Slots(p, q);
    expect([...slots.q].map((weight) => Number(weight.toFixed(9)))).toEqual([0.1, 0.6, 0.3]);
    // The tail holds c, which q lists at 0.3.
    expect(slots.qBound).toEqual([true, false, true]);
  });

  it("bounds a tail's children by the horizon and the lightest listed child", () => {
    const p = listing([0.5, 0.3], [0, 1], 2);
    const slots = new Slots(p, p);
    expect(slots.q[2]).toBeCloseTo(0.2, 9);
    const deep = new Slots({ ...p, complete: 4 }, { ...p, complete: 4 });
    expect(deep.q[2]).toBe(2 ** -4);
  });
});

describe("IntervalTree", () => {
  it("names children by label path, and learns handles per measure", () => {
    const tree = new IntervalTree(2);
    const child = tree.child(0, 5);
    const grandchild = tree.child(child.id, 2);
    expect(child.handle(0)).toBeUndefined();
    child.setHandle(1, 42n);
    expect(child.handle(0)).toBeUndefined();
    expect(child.handle(1)).toBe(42n);
    expect(tree.child(0, 5)).toBe(child);
    expect(tree.path(grandchild.id)).toEqual([5, 2]);
  });

  it("tells an unexpanded node from one with nothing within the horizon", () => {
    const tree = new IntervalTree(1);
    expect(tree.root.expanded(0)).toBe(false);
    tree.setListing(0, 0, { ...listing([]), complete: 0 });
    expect(tree.root.expanded(0)).toBe(false);
    tree.setListing(0, 0, listing([]));
    expect(tree.root.expanded(0)).toBe(true);
  });
});
