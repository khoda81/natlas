// Listing codec implemented by the shared Rust core.

import { decodeListing, encodeListing } from "./wasm";

export const SATURATED = 0x7fffffff;

export interface WireEdge {
  label: number;
  odds: number;
}

/** A decoded conditional integer distribution, heaviest first. */
export interface Listing {
  labels: number[];
  /** Natural log of each listed child's conditional weight. */
  lnWeights: number[];
  /** Weight left to unlisted children. */
  leftover: number;
  /** Every unlisted child weighs less than 2^-complete. */
  complete: number;
}

export function decode(edges: Iterable<WireEdge>, horizon: number, complete = horizon): Listing {
  const listing = decodeListing(horizon, complete, [...edges]) as Listing;
  return {
    labels: listing.labels,
    lnWeights: listing.lnWeights,
    leftover: listing.leftover,
    complete: listing.complete,
  };
}

export function encode(labels: readonly number[], lnWeights: readonly number[], horizon: number): WireEdge[] {
  return encodeListing([...labels], [...lnWeights], horizon) as WireEdge[];
}
