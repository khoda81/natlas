import { describe, expect, it } from "bun:test";
import { parseView, serializeView, viewKey } from "../src/ui/view-state";

const saved = { origin: [0], anchor: { path: [0, 0xffffffff], digits: Uint8Array.of(0, 0x80, 0xff) }, x: 80, k: 1e6, a: 0.25, p: "medium/audio=3f2a9c/temperature=1.6", q: "medium" };

describe("saved views", () => {
  it("round trips the origin, the point below it, precision and the selected measures", () => {
    expect(parseView(serializeView(saved))).toEqual(saved);
    const root = { ...saved, origin: [], anchor: { path: [], digits: new Uint8Array() }, x: 0 };
    expect(parseView(serializeView(root))).toEqual(root);
    // Past an origin that is not the root lies its parent.
    expect(parseView(serializeView({ ...saved, x: -1 }))).not.toBeNull();
  });

  it("rejects malformed storage and measures named by anything but a spec", () => {
    const stored = JSON.parse(serializeView(saved)) as Record<string, unknown>;
    for (const value of [null, {}, { ...stored, path: [-1] }, { ...stored, path: [1.5] }, { ...stored, digits: "0" }, { ...stored, digits: "0g" },
      { ...stored, digits: [0] }, { ...stored, origin: [], x: -1 }, { ...stored, origin: undefined }, { ...stored, k: 0 }, { ...stored, q: 2 }, { ...stored, p: undefined }]) {
      expect(parseView(JSON.stringify(value))).toBeNull();
    }
    expect(parseView("broken json")).toBeNull();
    expect(viewKey("live:A")).not.toBe(viewKey("recorded:A"));
  });
});
