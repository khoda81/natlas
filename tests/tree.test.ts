import { describe, expect, it } from "vitest";
import { childHandle, IntervalTree, ROOT } from "../src/tree";
import { hello, run } from "./build";

const children = (labels: string[], weights: number[], rest?: number) => ({
  labels,
  weights: [weights],
  rest: rest === undefined ? undefined : [rest],
});

describe("childHandle", () => {
  it("matches the exporter's handles", () => {
    // Computed by export/natlas_export.py:child_handle.
    expect(childHandle(ROOT, 0)).toBe(0xe220a8397b1dcdafn);
    expect(childHandle(childHandle(ROOT, 3), 0)).toBe(0x4e213bb3324a7b38n);
  });
});

describe("IntervalTree", () => {
  it("normalizes relative weights and keeps the rest as one tail slot", () => {
    const tree = new IntervalTree();
    // Unnormalized: only ratios travel.
    tree.applyAll([hello(["p"]), run(ROOT, children(["a", "b"], [10, 5], 5))]);
    const root = tree.node(0);
    expect(root.slots).toBe(3);
    expect(root.isTail(2)).toBe(true);
    expect(root.weight(0, 0)).toBeCloseTo(0.5, 5);
    expect(root.suffix(0, 1)).toBeCloseTo(0.5, 5);
  });

  it("lets a run that extends the listing replace the rest", () => {
    const tree = new IntervalTree();
    const all = { labels: ["a", "b", "c", "d"], weights: [[0.5, 0.25, 0.15, 0.1]] };
    tree.applyAll([hello(["p"]), run(ROOT, { ...all, rest: [0.25] }, 0, 2)]);
    tree.apply(run(ROOT, all, 2, 4));
    const root = tree.node(0);
    expect(root.slots).toBe(4);
    expect(root.hasTail).toBe(false);
    expect(root.weight(0, 3)).toBeCloseTo(0.1, 5);

    // A replay that stops short of the listing leaves it alone.
    tree.apply(run(ROOT, { ...all, rest: [0.25] }, 0, 2));
    expect(root.hasTail).toBe(false);
  });

  it("measures other measures from their own first child", () => {
    const tree = new IntervalTree();
    tree.applyAll([
      hello(["p", "q"]),
      run(ROOT, { labels: ["a", "b"], weights: [[0.6, 0.3], [0.1, 0.4]], rest: [0.1, 0.5] }),
    ]);
    const root = tree.node(0);
    expect([0, 1, 2].map((slot) => root.weight(1, slot))).toEqual([
      expect.closeTo(0.1, 5),
      expect.closeTo(0.4, 5),
      expect.closeTo(0.5, 5),
    ]);
  });

  it("creates nodes from runs addressed by path handles", () => {
    const tree = new IntervalTree();
    const b = childHandle(ROOT, 1);
    tree.applyAll([hello(["p"]), run(ROOT, children(["a", "b"], [2, 1])), run(b, children([], []))]);
    const leaf = tree.node(1);
    expect(leaf.parent).toBe(0);
    expect(leaf.slot).toBe(1);
    expect(leaf.slots).toBe(0);
    expect(tree.node(0).children.get(1)).toBe(1);
    expect(tree.path(1)).toEqual(["b"]);
  });

  it("verifies replays and rejects conflicts, gaps and unknown nodes", () => {
    const tree = new IntervalTree();
    const root = run(ROOT, children(["a", "b"], [3, 1]));
    tree.applyAll([hello(["p"]), root]);
    expect(() => tree.apply(root)).not.toThrow();
    expect(() => tree.apply(run(ROOT, children(["a", "x"], [3, 1])))).toThrow(/conflicting/);
    expect(() => tree.apply(run(ROOT, children(["a", "b", "c", "d"], [3, 1, 1, 1]), 3))).toThrow(/missing/);
    expect(() => tree.apply(run(childHandle(ROOT, 2), children(["x"], [1])))).toThrow(/unknown node/);
  });

  it("rejects a rest without a child to measure it against", () => {
    const tree = new IntervalTree();
    tree.apply(hello(["p"]));
    expect(() => tree.apply(run(ROOT, { labels: [], weights: [[]], rest: [Number.NaN] }))).toThrow(/rest but no children/);
  });
});
