import { describe, expect, it } from "bun:test";
import { parseView, viewKey } from "../src/ui/view-state";

const saved = { path: [0, 0xffffffff], x: 80, k: 1e6, a: -0.25, p: 1, q: 0 };

describe("saved views", () => {
  it("round trips complete paths, precision and the selected measures", () => {
    expect(parseView(JSON.stringify(saved), 2)).toEqual(saved);
  });

  it("rejects malformed storage and stale measure indices", () => {
    for (const value of [null, {}, { ...saved, path: [-1] }, { ...saved, path: [1.5] },
      { ...saved, x: -1 }, { ...saved, k: 0 }, { ...saved, q: 2 }, { ...saved, p: 0.5 }]) {
      expect(parseView(JSON.stringify(value), 2)).toBeNull();
    }
    expect(parseView("broken json", 2)).toBeNull();
    expect(viewKey("live:A")).not.toBe(viewKey("recorded:A"));
  });
});
