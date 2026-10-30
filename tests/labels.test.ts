import { describe, expect, it } from "bun:test";
import { TextLabels } from "../src/readings/labels";

const utf8 = (text: string) => new TextEncoder().encode(text);
// 🦢 is F0 9F A6 A2 and چ is DA 86, each split over byte-level tokens as SmolLM2 splits them.
const labels = new TextLabels([utf8("."), Uint8Array.of(0xf0, 0x9f), Uint8Array.of(0xa6), Uint8Array.of(0xa2), utf8(" "), Uint8Array.of(0xda), Uint8Array.of(0x86)]);

describe("TextLabels", () => {
  it("reads a path's bytes as one text, each label adding the characters it finishes", () => {
    expect(labels.along([0, 1, 2, 3, 4, 5, 6])).toEqual([".", "", "", "🦢", " ", "", "چ"]);
    expect(labels.along([0, 1, 2, 3, 4, 5, 6]).join("")).toBe(".🦢 چ");
  });

  it("shows a label after its path's unfinished character, else as hex", () => {
    const before = labels.after(labels.after(new Uint8Array(), 1).unfinished, 2).unfinished;
    expect(labels.shown(before, 3)).toBe("🦢");
    expect(labels.shown(new Uint8Array(), 1)).toBe("‹F0\u20099F›");
    expect(labels.shown(new Uint8Array(), 3)).toBe("‹A2›");
    expect(labels.shown(Uint8Array.of(0xda), 6)).toBe("چ");
  });

  it("parses base64 labels", () => {
    expect(TextLabels.parse(JSON.stringify([btoa("a b"), btoa("\xda")])).along([0, 1])).toEqual(["a b", ""]);
  });
});
