import { describe, expect, it } from "vitest";
import { IntervalTree } from "../src/tree";
import { hello, node, page } from "./build";

const slots = (labels: string[], weights: number[], tail: number) => ({ labels, weights: [weights], tail: [tail] });

describe("IntervalTree", () => {
  it("keeps an unlisted remainder as one tail slot until it is paged in", () => {
    const tree = new IntervalTree();
    tree.applyAll([hello(["p"]), node(0, 0, 0, 4, slots(["a", "b"], [0.5, 0.25], 0.25))]);
    const root = tree.node(0);
    expect(root.slots).toBe(3);
    expect(root.isTail(2)).toBe(true);
    expect(root.suffix(0, 1)).toBeCloseTo(0.5, 15);

    tree.apply(page(0, 2, slots(["c", "d"], [0.15, 0.1], 0)));
    expect(root.slots).toBe(4);
    expect(root.hasTail).toBe(false);
    expect(root.prefix(0, 4)).toBeCloseTo(1, 15);
  });

  it("verifies replays and rejects conflicts and gaps", () => {
    const tree = new IntervalTree();
    const root = node(0, 0, 0, 2, slots(["a", "b"], [0.75, 0.25], 0));
    tree.applyAll([hello(["p"]), root, node(1, 0, 1, 1, slots(["x"], [1], 0))]);
    expect(() => tree.apply(root)).not.toThrow();
    expect(() => tree.apply(node(1, 0, 0, 1, slots(["x"], [1], 0)))).toThrow(/conflicting/);
    expect(() => tree.apply(node(3, 0, 0, 1, slots(["x"], [1], 0)))).toThrow(/missing nodes/);
    expect(() => tree.apply(page(0, 0, slots(["a"], [0.5], 0.5)))).toThrow(/conflicting slot/);
  });

  it("rejects pages that do not conserve the tail mass", () => {
    const tree = new IntervalTree();
    tree.applyAll([hello(["p"]), node(0, 0, 0, 3, slots(["a"], [0.5], 0.5))]);
    expect(() => tree.apply(page(0, 1, slots(["b"], [0.4], 0.2)))).toThrow(/conserve/);
  });

  it("only expands listed slots", () => {
    const tree = new IntervalTree();
    tree.applyAll([hello(["p"]), node(0, 0, 0, 3, slots(["a"], [0.5], 0.5))]);
    expect(() => tree.apply(node(1, 0, 1, 0, slots([], [], 0)))).toThrow(/unlisted/);
  });
});
