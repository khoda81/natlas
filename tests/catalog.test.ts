import { describe, expect, it } from "bun:test";
import { type Catalog, format, modelOf, parse } from "../src/data/catalog";

const catalog: Catalog = {
  models: [
    { name: "size", kind: "choice", options: ["small", "large"], default: "small" },
    { name: "device", kind: "choice", options: ["cuda", "cpu"], default: "cuda" },
  ],
  parameters: [{ name: "temperature", kind: "number", default: 1 }, { name: "instruments", kind: "text" }, { name: "audio", kind: "recording" }],
  presets: [],
};

describe("specs", () => {
  it("name a model by each of its parameters, then the measure's not at their default", () => {
    const values = new Map([["size", "large"], ["device", "cpu"], ["temperature", "1"], ["instruments", "piano,drums"], ["audio", ""]]);
    const spec = format(catalog, values);
    expect(spec).toBe("large/cpu/instruments=piano%2Cdrums");
    expect(modelOf(catalog, spec)).toBe("large/cpu");
    expect(parse(catalog, spec)).toEqual(values);
    expect(format(catalog, new Map([...values, ["temperature", "3"]]))).toBe("large/cpu/temperature=3/instruments=piano%2Cdrums");
  });
});
