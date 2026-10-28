// Listing codec (see proto/natlas/v1). An edge carries the log-odds of its
// weight as a fraction of the previous edge's, quantized in steps that scale
// with the request's horizon.

/** Odds of a fraction g = 1: a tie with the previous edge. */
export const SATURATED = 0x7fffffff;
const LOWEST = -0x80000000;

export interface WireEdge {
  label: number;
  odds: number;
}

/** Nats per odds step at ``horizon`` bits. */
export function step(horizon: number): number {
  return (horizon * Math.LN2) / 2 ** 31;
}

/** ln σ(λ), without overflow at either end. */
export function lnSigmoid(lambda: number): number {
  if (lambda === Infinity) return 0;
  return lambda >= 0 ? -Math.log1p(Math.exp(-lambda)) : lambda - Math.log1p(Math.exp(lambda));
}

function lambda(odds: number, unit: number): number {
  return odds === SATURATED ? Infinity : odds * unit;
}

/** A decoded listing: conditional weights in edge order and what they leave. */
export interface Listing {
  labels: number[];
  /** ln of each child's conditional weight; non-increasing. */
  lnWeights: number[];
  /** Weight of the children not listed, at least 0. */
  leftover: number;
  /** Every child not listed weighs less than 2^-complete. */
  complete: number;
}

/**
 * Decodes edges one at a time, as they arrive. The leftover after the first
 * edge is exact; later edges subtract from it, so it is only known to within
 * the rounding they carry, and a leftover within that rounding is spread over
 * the listed children instead. An edge that would take more than is left
 * ends the listing.
 */
export class Decoder {
  readonly labels: number[] = [];
  readonly lnWeights: number[] = [];
  private readonly unit: number;
  private leftover = 1;
  private slack = 0;
  private stopped = false;

  constructor(
    horizon: number,
    private readonly complete = horizon,
  ) {
    this.unit = step(horizon);
  }

  /** Accepts an edge; false once the listing has ended. */
  push(edge: WireEdge): boolean {
    if (this.stopped) return false;
    const l = lambda(edge.odds, this.unit);
    const previous = this.lnWeights.at(-1) ?? 0;
    const lnWeight = previous + lnSigmoid(l);
    const weight = Math.exp(lnWeight);
    if (this.lnWeights.length === 0) {
      this.leftover = Math.exp(lnSigmoid(-l));
    } else if (weight > this.leftover + this.slack) {
      this.stopped = true;
      return false;
    } else {
      this.leftover -= weight;
      this.slack += this.unit * weight;
    }
    this.labels.push(edge.label);
    this.lnWeights.push(lnWeight);
    return true;
  }

  listing(): Listing {
    const tail = this.leftover > this.slack;
    // A leftover within rounding goes to the listed children, which then
    // partition the interval exactly.
    const shift = tail ? 0 : -Math.log1p(-this.leftover);
    return {
      labels: [...this.labels],
      lnWeights: this.lnWeights.map((lnWeight) => lnWeight + shift),
      leftover: tail ? this.leftover : 0,
      complete: this.complete,
    };
  }
}

export function decode(edges: Iterable<WireEdge>, horizon: number, complete = horizon): Listing {
  const decoder = new Decoder(horizon, complete);
  for (const edge of edges) if (!decoder.push(edge)) break;
  return decoder.listing();
}

/**
 * Edges for children sorted by decreasing weight, given as natural logs, down
 * to 2^-horizon. Each fraction is taken against the previous weight as the
 * client will decode it.
 */
export function encode(labels: readonly number[], lnWeights: readonly number[], horizon: number): WireEdge[] {
  const unit = step(horizon);
  const floor = -horizon * Math.LN2;
  const edges: WireEdge[] = [];
  let previous = 0;
  for (let i = 0; i < labels.length; i++) {
    const lnWeight = lnWeights[i]!;
    if (!(lnWeight >= floor)) break;
    const lnG = Math.min(0, lnWeight - previous);
    let odds = SATURATED;
    if (lnG < 0 && unit > 0) {
      const l = lnG - Math.log(-Math.expm1(lnG));
      odds = Math.max(LOWEST, Math.min(SATURATED, Math.round(l / unit)));
    }
    edges.push({ label: labels[i]!, odds });
    previous += lnSigmoid(lambda(odds, unit));
  }
  return edges;
}
