import { describe, expect, it } from "bun:test";
import { decode, encode, SATURATED } from "../src/core/codec";

const ln = (weights: number[]) => weights.map(Math.log);
const step = (horizon: number) => (horizon * Math.LN2) / 2 ** 31;

describe("encode", () => {
  it("matches the exporter's edges", () => {
    // Computed by export/natlas_codec.py:encode.
    const weights = [0.5, 0.3, 0.3, 0.15, 0.001, 1e-6];
    expect(encode([7, 3, 9, 1, 4, 2], ln(weights), 12)).toEqual([
      { label: 7, odds: 0 },
      { label: 3, odds: 104683117 },
      { label: 9, odds: SATURATED },
      { label: 1, odds: 0 },
      { label: 4, odds: -1291920529 },
    ]);
    expect(encode([0], [Math.log1p(-1e-12)], 48)).toEqual([{ label: 0, odds: 1783446566 }]);
  });

  it("stops at the horizon", () => {
    expect(encode([0, 1, 2], ln([0.5, 0.25, 0.2]), 2).map((edge) => edge.label)).toEqual([0, 1]);
  });
});

describe("decode", () => {
  it("recovers weights to within a step", () => {
    const weights = [0.41, 0.2, 0.13, 0.1, 0.07, 0.05, 0.03, 0.01];
    const horizon = 20;
    const { lnWeights, leftover } = decode(encode(weights.map((_, i) => i), ln(weights), horizon), horizon);
    lnWeights.forEach((lnWeight, i) => expect(Math.abs(lnWeight - Math.log(weights[i]!))).toBeLessThanOrEqual(step(horizon)));
    expect(leftover).toBe(0);
  });

  it("keeps ties exact", () => {
    const { lnWeights, leftover } = decode(encode([0, 1, 2, 3], ln([0.25, 0.25, 0.25, 0.25]), 8), 8);
    expect(lnWeights.slice(1)).toEqual([lnWeights[0]!, lnWeights[0]!, lnWeights[0]!]);
    expect(leftover).toBe(0);
  });

  it("keeps the leftover of a near-certain first child", () => {
    const { leftover } = decode(encode([0], [Math.log1p(-1e-12)], 48), 48);
    expect(leftover / 1e-12).toBeCloseTo(1, 6);
  });

  it("sorts any odds whatsoever", () => {
    let seed = 7;
    const random = () => ((seed = (seed * 48271) % 0x7fffffff), seed);
    const edges = Array.from({ length: 50 }, (_, label) => ({ label, odds: random() - 0x40000000 }));
    const { lnWeights } = decode(edges, 30);
    for (let i = 1; i < lnWeights.length; i++) expect(lnWeights[i]!).toBeLessThanOrEqual(lnWeights[i - 1]!);
  });

  it("stops reading where a listing overflows the interval", () => {
    // 0.5, 0.3, then a tie at 0.3 that does not fit.
    const horizon = 12;
    const { labels, leftover } = decode(encode([7, 3, 9, 1], ln([0.5, 0.3, 0.3, 0.15]), horizon), horizon);
    expect(labels).toEqual([7, 3]);
    expect(leftover).toBeCloseTo(0.2, 6);
  });

  it("leaves everything to the rest without edges", () => {
    expect(decode([], 10)).toEqual({ labels: [], lnWeights: [], leftover: 1, complete: 10 });
  });
});
