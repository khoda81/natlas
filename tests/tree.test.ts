import { describe, expect, it } from "bun:test";
import { IntervalTree, ROOT } from "../src/core/tree";
import { answer } from "./build";

describe("IntervalTree", () => {
  it("names nodes by their branches from the root, alike in every tree", () => {
    const tree = new IntervalTree(2);
    const child = tree.child(ROOT, 5);
    const grandchild = tree.child(child, 2);
    expect(tree.child(ROOT, 5)).toBe(child);
    expect(tree.parent(grandchild)).toBe(child);
    expect(tree.branch(grandchild)).toBe(2);
    expect(tree.path(grandchild)).toEqual([5, 2]);
    expect(tree.chain(ROOT, grandchild)).toEqual([ROOT, child, grandchild]);
    expect(tree.descend(ROOT, [5, 2])).toBe(grandchild);
    expect(new IntervalTree(1).descend(ROOT, [5, 2])).toBe(grandchild);
  });

  it("places children by p in p's order and sizes them by q, unknown until q says", () => {
    const tree = new IntervalTree(2);
    tree.insert(ROOT, 0, answer([0.75, 0.25], [9, 4]));
    const lens = { p: 0, q: 1 };
    expect(tree.entry(ROOT, lens, 4)).toEqual({ rank: 1, before: expect.closeTo(0.75, 12), lnP: expect.closeTo(Math.log(0.25), 12), natsQ: Infinity });
    tree.insert(ROOT, 1, answer([0.5, 0.5], [4, 9]));
    expect(tree.entry(ROOT, lens, 4)?.natsQ).toBeCloseTo(Math.log(2), 12);
    expect(tree.entry(ROOT, lens, 7)).toBeUndefined();
    expect(tree.statistics(ROOT, lens)).toEqual({ entropy: expect.closeTo(-0.75 * Math.log(0.75) - 0.25 * Math.log(0.25), 12),
      cross: expect.closeTo(Math.log(2), 12), bounded: false });
  });

  it("tells a node without children from one not yet asked, and notifies only on change", () => {
    const tree = new IntervalTree(1);
    let changes = 0;
    tree.subscribe(() => changes++);
    expect(tree.has(ROOT, 0)).toBe(false);
    tree.insert(ROOT, 0, { runs: [], count: 0 });
    expect(tree.ended(ROOT, 0)).toBe(true);
    tree.insert(ROOT, 0, { runs: [], count: 0 });
    expect(changes).toBe(1);
    expect(() => tree.insert(ROOT, 0, { runs: [{ rank: 0, branches: [1], lnBelow: [0] }] })).toThrow("edge");
  });
});
