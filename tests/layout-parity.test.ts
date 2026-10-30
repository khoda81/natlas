import { describe, expect, it } from "bun:test";
import { createHash } from "node:crypto";
import { IntervalTree } from "../src/core/tree";
import { clamp, layout, pick, reanchor, repair } from "../src/core/layout";
import { wants } from "../src/data/expand";
import { layoutCases, samplePoints } from "./layout-cases";
import reference from "./fixtures/layout-parity.json";

describe(`behavior frozen at ${reference.baseline}`, () => {
  for (const { name, tree, lens, cam: initial, view, mergePx } of layoutCases(IntervalTree)) {
    it(`preserves every geometry field, selection, anchor and discovery request: ${name}`, () => {
      const cam = structuredClone(initial);
      repair(tree, lens, cam);
      clamp(tree, lens, cam);
      const frame = layout(tree, lens, cam, view, { mergePx });
      try {
        const value = { frame, picks: samplePoints(view.width, view.height).map(([x, y]) => pick(frame, cam, view, x, y)),
          demand: wants(tree, lens, cam, view, frame) };
        reanchor(cam, frame);
        expect(createHash("sha256").update(JSON.stringify({ ...value, cam })).digest("hex"))
          .toBe(reference.results.find(result => result.name === name)!.hash);
      } finally { frame.dispose(); }
    });
  }
});
