import type { RadixCDF } from "./radix";

export interface Interval { lower: number; upper: number }

function entropyMass(logMass: number, logCount = 0): number {
  return logMass === -Infinity ? 0 : Math.exp(logMass) * (logCount - logMass);
}

/** Bounds from disjoint whole-ID groups. No uniform-within-group assumption.
 * A group of m tokens and mass r contributes between -r ln r and r ln(m/r).
 * These are conservative under the shared boundary constraints, not tight optima.
 */
export function distributionBounds(p: RadixCDF, q: RadixCDF | undefined): { entropy: Interval; cross: Interval } {
  const entropy = { lower: 0, upper: 0 };
  const cross = { lower: 0, upper: 0 };
  if (!p.ready) return { entropy: { lower: 0, upper: Infinity }, cross: { lower: 0, upper: Infinity } };
  const regions = p.partition();
  for (const region of regions) {
    const { lower, upper } = region.mass;
    const logCount = Math.log(region.end - region.first);
    entropy.lower += Math.min(entropyMass(lower), entropyMass(upper));
    const maximum = Math.max(lower, Math.min(upper, logCount - 1));
    entropy.upper += entropyMass(maximum, logCount);
    const qb = q?.ready && !q.terminal ? q.rangeBounds(region.first, region.end) : { lower: -Infinity, upper: 0 };
    cross.lower += lower === -Infinity ? 0 : Math.exp(lower) * -qb.upper;
    cross.upper += upper === -Infinity ? 0 : region.kind === "unresolved" ? Infinity : Math.exp(upper) * -qb.lower;
  }
  const vocabulary = regions.at(-1)?.end ?? 1;
  entropy.upper = Math.min(entropy.upper, Math.log(vocabulary));
  return { entropy, cross: p === q ? { ...entropy } : cross };
}
