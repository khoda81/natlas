/** A measure's inverse-CDF pages. All positions are exact base-256 rationals. */
export interface QuantileReply {
  node: bigint;
  firstBranchId: number;
  delta: readonly bigint[];
  terminal: boolean;
}

export interface MassBounds {
  /** Natural logs; -Infinity denotes zero. */
  lower: number;
  upper: number;
}

export interface CDFBoundary {
  /** Bounds on F(id), in units of 256^-depth. Keep these rational until projection. */
  lo: bigint;
  hi: bigint;
  depth: number;
}

interface Boundary extends CDFBoundary {
  /** Next page to fetch to tighten this boundary, if it is not exact. */
  next?: number[];
}

/** A disjoint range of whole IDs, with its cumulative boundaries preserved. */
export interface CDFRegion {
  kind: "token" | "unresolved";
  first: number;
  end: number;
  before: CDFBoundary;
  after: CDFBoundary;
  mass: MassBounds;
  /** A layout representative between the boundary midpoints, never a measurement. */
  lnMass: number;
}

export interface IDRange {
  first: number;
  end: number;
}

/** A visible window relative to one CDF boundary; avoids absolute float CDFs. */
export interface CDFWindow {
  id: number;
  /** The boundary midpoint's screen position in pixels. */
  y: number;
  height: number;
}

/** Difference of rational CDF coordinates, without cancellation near 1. */
function logDifference(a: bigint, ad: number, b: bigint, bd: number): number {
  const depth = Math.max(ad, bd);
  return logInteger((b << BigInt(8 * (depth - bd))) - (a << BigInt(8 * (depth - ad)))) - depth * 8 * Math.LN2;
}

export function boundaryUncertainty(boundary: CDFBoundary): number {
  return logInteger(boundary.hi - boundary.lo) - boundary.depth * 8 * Math.LN2;
}

function midpointMass(a: CDFBoundary, b: CDFBoundary): number {
  // The sums are twice the midpoint; divide the difference by two in log space.
  return logDifference(a.lo + a.hi, a.depth, b.lo + b.hi, b.depth) - Math.LN2;
}

interface Bin {
  prefix: number[];
  first: number;
  last: number;
  logMass: number;
}

export const MAX_PREFIX = 1024;

function logInteger(value: bigint): number {
  if (value <= 0n) return -Infinity;
  const shift = Math.max(0, value.toString(2).length - 53);
  return Math.log(Number(value >> BigInt(shift))) + shift * Math.LN2;
}

function pageKey(prefix: readonly number[]): string {
  return prefix.join(",");
}

export class RadixCDF {
  private readonly pages = new Map<string, readonly number[]>();
  private readonly pins = new Set<number>();
  private readonly massBounds = new Map<number, MassBounds>();
  private leaves: Bin[] | undefined;
  private regions: readonly CDFRegion[] | undefined;
  terminal = false;
  ready = false;
  version = 0;

  get vocabulary(): number | undefined {
    return this.terminal ? 0 : this.pages.get("")?.[256];
  }

  has(prefix: readonly number[]): boolean {
    return this.pages.has(pageKey(prefix));
  }

  /** Validate the complete reply before changing any cached state. */
  insert(prefix: readonly number[], reply: QuantileReply): void {
    if (prefix.length > MAX_PREFIX || prefix.some((v) => !Number.isInteger(v) || v < 0 || v > 255)) {
      throw new Error("Invalid CDF prefix");
    }
    if (reply.terminal) {
      if (reply.delta.length !== 0 || (this.ready && !this.terminal)) throw new Error("Inconsistent terminal reply");
      this.terminal = this.ready = true;
      this.version++;
      return;
    }
    if (this.terminal || reply.delta.length !== 256) throw new Error("Expected 256 CDF deltas");
    let id = reply.firstBranchId;
    if (!Number.isInteger(id) || id < 0 || id > 0xffffffff) throw new Error("Invalid first branch ID");
    const ids = [id];
    for (const delta of reply.delta) {
      if (delta < 0n || delta > 0x100000000n) throw new Error("Invalid CDF delta");
      id += Number(delta);
      if (id > 0x100000000) throw new Error("CDF ID exceeds vocabulary sentinel");
      ids.push(id);
    }
    if (prefix.length > 0) {
      const parent = this.pages.get(pageKey(prefix.slice(0, -1)));
      const j = prefix.at(-1)!;
      if (!parent || ids[0] !== parent[j] || ids[256] !== parent[j + 1]) {
        throw new Error("CDF page disagrees with its parent");
      }
    } else if (ids[256] === 0) throw new Error("Nonterminal distribution has no vocabulary");
    const vocabulary = prefix.length ? this.pages.get("")![256]! : ids[256]!;
    if (ids.slice(0, 256).some((value) => value >= vocabulary))
      throw new Error("Vocabulary sentinel used as a branch ID");
    const previous = this.pages.get(pageKey(prefix));
    if (previous && previous.some((value, i) => value !== ids[i])) throw new Error("Measure changed within a session");
    this.pages.set(pageKey(prefix), ids);
    this.ready = true;
    this.massBounds.clear();
    this.leaves = undefined;
    this.regions = undefined;
    this.version++;
  }

  private boundary(target: number): Boundary {
    const vocabulary = this.pages.get("")?.[256];
    if (!Number.isInteger(target) || target < 0 || (vocabulary !== undefined && target > vocabulary))
      throw new Error("Branch outside the shared vocabulary");
    if (target === 0) return { lo: 0n, hi: 0n, depth: 0 };
    if (target === vocabulary) return { lo: 1n, hi: 1n, depth: 0 };
    let prefix: number[] = [];
    let a = 0n;
    for (;;) {
      const page = this.pages.get(pageKey(prefix));
      if (!page) return { lo: a, hi: a + 1n, depth: prefix.length, next: prefix };
      const j = page.findIndex((id) => id >= target);
      if (j < 0) throw new Error("Branch outside the shared vocabulary");
      const base = a * 256n;
      const depth = prefix.length + 1;
      if (j === 0) return { lo: base, hi: base, depth };
      const lo = base + BigInt(j - 1);
      const next = [...prefix, j - 1];
      if (!this.has(next)) return { lo, hi: lo + 1n, depth, next };
      prefix = next;
      a = lo;
    }
  }

  boundaryBounds(target: number): CDFBoundary {
    const { lo, hi, depth } = this.boundary(target);
    return { lo, hi, depth };
  }

  rangeBounds(first: number, end: number): MassBounds {
    if (first > end) throw new Error("Reversed ID range");
    const a = this.boundary(first), b = this.boundary(end);
    if (first === end) return { lower: -Infinity, upper: -Infinity };
    return {
      lower: logDifference(a.hi, a.depth, b.lo, b.depth),
      upper: Math.min(0, logDifference(a.lo, a.depth, b.hi, b.depth)),
    };
  }

  /** Representative derived from shared CDF midpoints, not independent token estimates. */
  estimate(first: number, end = first + 1): number {
    return midpointMass(this.boundary(first), this.boundary(end));
  }

  /** Known tokens and explicit unresolved ID ranges, in the protocol's order.
   * Only tokens with a certified positive interior receive an individual region.
   * Shared midpoint boundaries partition the interval without global renormalization.
   */
  partition(): readonly CDFRegion[] {
    if (!this.ready || this.terminal) return [];
    if (this.regions) return this.regions;
    const vocabulary = this.pages.get("")![256]!;
    const labels = new Set<number>(this.pins);
    for (const ids of this.pages.values()) for (const id of ids) if (id < vocabulary) labels.add(id);
    const regions: CDFRegion[] = [];
    const add = (first: number, end: number, kind: CDFRegion["kind"]) => {
      if (first === end) return;
      const before = this.boundaryBounds(first), after = this.boundaryBounds(end);
      const mass = this.rangeBounds(first, end);
      if (mass.upper === -Infinity) return;
      regions.push({ kind, first, end, before, after, mass, lnMass: midpointMass(before, after) });
    };
    let cursor = 0;
    for (const label of [...labels].sort((a, b) => a - b)) {
      if (this.bounds(label).lower === -Infinity) continue;
      add(cursor, label, "unresolved");
      add(label, label + 1, "token");
      cursor = label + 1;
    }
    add(cursor, vocabulary, "unresolved");
    this.regions = regions;
    return regions;
  }

  bounds(label: number): MassBounds {
    if (!this.ready) return { lower: -Infinity, upper: 0 };
    if (this.terminal || label >= this.pages.get("")![256]!) return { lower: -Infinity, upper: -Infinity };
    const cached = this.massBounds.get(label);
    if (cached) return cached;
    const bounds = this.rangeBounds(label, label + 1);
    this.massBounds.set(label, bounds);
    return bounds;
  }

  /** One cached page advances both endpoint searches whenever they share a bin. */
  nextBoundary(label: number): number[] | undefined {
    if (!this.ready) return [];
    if (this.terminal) return undefined;
    const a = this.boundary(label);
    const b = this.boundary(label + 1);
    const next = a.next && (!b.next || a.depth <= b.depth) ? a.next : b.next;
    if (next && next.length > MAX_PREFIX) throw new Error("CDF precision limit reached");
    return next;
  }

  private *bins(prefix: number[] = [], logMass = 0): Generator<Bin> {
    const page = this.pages.get(pageKey(prefix));
    if (!page) return;
    const step = logMass - 8 * Math.LN2;
    for (let j = 0; j < 256; j++) {
      const first = page[j]!;
      const last = page[j + 1]!;
      const child = [...prefix, j];
      if (first === last) {
        // Fine pages mostly repeat one ID. Keep certified runs compact.
        let end = j + 1;
        while (end < 256 && page[end] === first && page[end + 1] === first) end++;
        yield { prefix: child, first, last, logMass: step + Math.log(end - j) };
        j = end - 1;
      } else if (this.has(child)) yield* this.bins(child, step);
      else yield { prefix: child, first, last, logMass: step };
    }
  }

  pin(label: number): void {
    const bounds = this.bounds(label);
    if (!Number.isFinite(bounds.lower) || Math.exp((bounds.lower + bounds.upper) / 2) === 0) {
      throw new Error("Token p mass cannot be represented by the camera");
    }
    this.pins.add(label);
    this.regions = undefined;
  }

  /** Largest unresolved p bin relevant to the visible labels, down to 0.5px. */
  nextSample(logHeight: number, labels?: ReadonlySet<number>, ranges: readonly IDRange[] = [], window?: CDFWindow): number[] | undefined {
    if (!this.ready) return [];
    if (this.terminal) return undefined;
    let best: Bin | undefined;
    const sorted = labels ? [...labels].sort((a, b) => a - b) : undefined;
    const anchor = window ? this.boundary(window.id) : undefined;
    const error = anchor ? Math.exp(boundaryUncertainty(anchor) + logHeight) / 2 : 0;
    const position = (coordinate: bigint, depth: number) => {
      const common = Math.max(depth, anchor!.depth);
      const delta = (coordinate << BigInt(8 * (common - depth) + 1)) -
        ((anchor!.lo + anchor!.hi) << BigInt(8 * (common - anchor!.depth)));
      if (delta === 0n) return window!.y;
      return window!.y + (delta < 0n ? -1 : 1) * Math.exp(logInteger(delta < 0n ? -delta : delta) - common * 8 * Math.LN2 - Math.LN2 + logHeight);
    };
    for (const bin of (this.leaves ??= [...this.bins()])) {
      if (bin.first === bin.last || logHeight + bin.logMass <= Math.log(0.5)) continue;
      if (window) {
        const a = bin.prefix.reduce((a, b) => a * 256n + BigInt(b), 0n);
        if (position(a + 1n, bin.prefix.length) + error <= 0 ||
          position(a, bin.prefix.length) - error >= window.height) continue;
      }
      if (sorted) {
        let lo = 0,
          hi = sorted.length;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (sorted[mid]! < bin.first) lo = mid + 1;
          else hi = mid;
        }
        if ((lo === sorted.length || sorted[lo]! > bin.last) &&
          !ranges.some((range) => range.first <= bin.last && range.end > bin.first)) continue;
      }
      if (!best || bin.logMass > best.logMass) best = bin;
    }
    if (best && best.prefix.length > MAX_PREFIX) throw new Error("CDF precision limit reached");
    return best?.prefix;
  }
}
