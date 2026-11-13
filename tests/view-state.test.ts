import { describe, expect, it } from "bun:test";
import { parseView, serializeView, viewKey } from "../src/ui/view-state";

const saved = { origin: [0], anchor: { path: [0, 0xffffffff], digits: Uint8Array.of(0, 0x80, 0xff) }, x: 80, k: 1e6, a: 0.25, p: 1, q: 0 };

describe("saved views", () => {
  it("round trips the origin, the point below it, precision and the selected measures", () => {
    expect(parseView(serializeView(saved), 2)).toEqual(saved);
    const root = { ...saved, origin: [], anchor: { path: [], digits: new Uint8Array() }, x: 0 };
    expect(parseView(serializeView(root), 2)).toEqual(root);
    // Past an origin that is not the root lies its parent.
    expect(parseView(serializeView({ ...saved, x: -1 }), 2)).not.toBeNull();
  });

  it("rejects malformed storage and stale measure indices", () => {
    const stored = JSON.parse(serializeView(saved)) as Record<string, unknown>;
    for (const value of [null, {}, { ...stored, path: [-1] }, { ...stored, path: [1.5] }, { ...stored, digits: "0" }, { ...stored, digits: "0g" },
      { ...stored, digits: [0] }, { ...stored, origin: [], x: -1 }, { ...stored, origin: undefined }, { ...stored, k: 0 }, { ...stored, q: 2 }, { ...stored, p: 0.5 }]) {
      expect(parseView(JSON.stringify(value), 2)).toBeNull();
    }
    expect(parseView("broken json", 2)).toBeNull();
    expect(viewKey("live:A")).not.toBe(viewKey("recorded:A"));
  });
});
