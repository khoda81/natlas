import { describe, expect, it } from "bun:test";
import { layoutTicks } from "@qbar/qbar";
import { pitchName, pitchTicks } from "../src/readings/music";

/** The pitch names a 300px gutter shows from C3 up, at ``row`` pixels a pitch. */
const named = (row: number) =>
  layoutTicks({ quantile: (q) => 47.5 + (q * 300) / row, scale: pitchTicks, width: 300, minSpacing: 11 })
    .map((tick) => pitchName(tick.value))
    .join(" ");

describe("pitchTicks", () => {
  it("names fewer pitches as rows shrink: all, then C major's pentatonic, then C and G, then octaves", () => {
    expect(named(24)).toBe("C3 C♯3 D3 D♯3 E3 F3 F♯3 G3 G♯3 A3 A♯3 B3 C4");
    expect(named(7)).toStartWith("C3 D3 E3 G3 A3 C4 D4 E4 G4 A4 C5");
    expect(named(3)).toStartWith("C3 G3 C4 G4 C5");
    expect(named(1.2)).toStartWith("C3 C4 C5 C6");
  });

  it("selects only MIDI pitches", () => {
    expect(pitchTicks.select(127.5, 140)).toBeUndefined();
    expect(pitchTicks.select(-20, -0.5)).toBeUndefined();
    expect(pitchTicks.select(-20, 3)).toEqual({ value: 0, step: 0, size: 12 });
  });
});
