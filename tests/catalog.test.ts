import { describe, expect, it } from "bun:test";
import { type Catalog, format, formatModel, modelOf, parse } from "../src/data/catalog";

const catalog: Catalog = {
  models: [
    { name: "size", kind: "choice", options: ["small", "large"], default: "small" },
    { name: "device", kind: "choice", options: ["cuda", "cpu"], default: "cuda" },
  ],
  parameters: [
    { name: "temperature", kind: "number", default: 1 },
    { name: "instruments", kind: "text" },
    { name: "audio", kind: "recording" },
  ],
};

describe("specs", () => {
  it("format and parse model and measure parameters with defaults omitted", () => {
    const values = new Map([["size", "large"], ["device", "cpu"], ["temperature", "1"], ["instruments", "piano,drums"], ["audio", ""]]);
    const spec = format(catalog, values);
    expect(spec).toBe("size=large&device=cpu|instruments=piano%2Cdrums");
    expect(modelOf(catalog, spec)).toBe("size=large&device=cpu");
    expect(parse(catalog, spec)).toEqual(values);
    expect(format(catalog, new Map([...values, ["temperature", "3"]]))).toBe("size=large&device=cpu|temperature=3&instruments=piano%2Cdrums");
    // All-default model still keeps its first parameter so spec is never empty.
    expect(formatModel(catalog, new Map([["size", "small"], ["device", "cuda"]]))).toBe("size=small");
  });
});
