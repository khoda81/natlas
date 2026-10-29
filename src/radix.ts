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

interface Boundary {
  lo: bigint;
  hi: bigint;
  depth: number;
  /** Next page to fetch to tighten this boundary, if it is not exact. */
  next?: number[];
}

interface Bin {
  prefix: number[];
  first: number;
  last: number;
  mass: number;
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
  private estimates: Map<number, number> | undefined;
  private readonly massBounds = new Map<number, MassBounds>();
  private leaves: Bin[] | undefined;
  terminal = false;
  ready = false;
  version = 0;

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
    this.estimates = undefined;
    this.massBounds.clear();
    this.leaves = undefined;
    this.version++;
  }

  private boundary(target: number): Boundary {
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

  bounds(label: number): MassBounds {
    if (!this.ready) return { lower: -Infinity, upper: 0 };
    if (this.terminal || label >= this.pages.get("")![256]!) return { lower: -Infinity, upper: -Infinity };
    const cached = this.massBounds.get(label);
    if (cached) return cached;
    const a = this.boundary(label);
    const b = this.boundary(label + 1);
    const depth = Math.max(a.depth, b.depth);
    const align = (n: bigint, d: number) => n << BigInt(8 * (depth - d));
    const lower = align(b.lo, b.depth) - align(a.hi, a.depth);
    const upper = align(b.hi, b.depth) - align(a.lo, a.depth);
    const scale = depth * 8 * Math.LN2;
    const bounds = {
      lower: logInteger(lower) - scale,
      upper: Math.min(0, logInteger(upper) - scale),
    };
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

  private *bins(prefix: number[] = [], mass = 1): Generator<Bin> {
    const page = this.pages.get(pageKey(prefix));
    if (!page) return;
    const step = mass / 256;
    for (let j = 0; j < 256; j++) {
      const first = page[j]!;
      const last = page[j + 1]!;
      const child = [...prefix, j];
      if (first === last) {
        // Fine pages mostly repeat one ID. Keep certified runs compact.
        let end = j + 1;
        while (end < 256 && page[end] === first && page[end + 1] === first) end++;
        yield { prefix: child, first, last, mass: step * (end - j) };
        j = end - 1;
      } else if (this.has(child)) yield* this.bins(child, step);
      else yield { prefix: child, first, last, mass: step };
    }
  }

  /** Each unrefined bin contributes its own mass, including between levels. */
  weights(): ReadonlyMap<number, number> {
    if (this.estimates) return this.estimates;
    const values = new Map<number, number>();
    for (const bin of (this.leaves ??= [...this.bins()]))
      values.set(bin.first, (values.get(bin.first) ?? 0) + bin.mass);
    // Travel can explicitly name a token smaller than any current sample bin.
    // Reserve its resolved mass and renormalize only the sampled remainder.
    let pinned = 0;
    for (const label of this.pins) {
      const bounds = this.bounds(label);
      const mass = Math.exp((bounds.lower + bounds.upper) / 2);
      if (mass > 0) {
        values.set(label, mass);
        pinned += mass;
      }
    }
    if (pinned > 1) {
      for (const label of this.pins) if (values.has(label)) values.set(label, values.get(label)! / pinned);
      pinned = 1;
    }
    let rest = 0;
    for (const [label, mass] of values) if (!this.pins.has(label)) rest += mass;
    if (rest > 0) {
      for (const [label, mass] of values)
        if (!this.pins.has(label)) values.set(label, (mass * Math.max(0, 1 - pinned)) / rest);
    } else if (pinned > 0) {
      for (const [label, mass] of values) values.set(label, mass / pinned);
    }
    this.estimates = values;
    return values;
  }

  pin(label: number): void {
    const bounds = this.bounds(label);
    if (!Number.isFinite(bounds.lower) || Math.exp((bounds.lower + bounds.upper) / 2) === 0) {
      throw new Error("Token p mass cannot be represented by the camera");
    }
    this.pins.add(label);
    this.estimates = undefined;
  }

  /** Largest unresolved p bin relevant to the visible labels, down to 0.5px. */
  nextSample(logHeight: number, labels?: ReadonlySet<number>): number[] | undefined {
    if (!this.ready) return [];
    if (this.terminal) return undefined;
    let best: Bin | undefined;
    const sorted = labels ? [...labels].sort((a, b) => a - b) : undefined;
    for (const bin of (this.leaves ??= [...this.bins()])) {
      if (bin.first === bin.last || logHeight + Math.log(bin.mass) <= Math.log(0.5)) continue;
      if (sorted) {
        let lo = 0,
          hi = sorted.length;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (sorted[mid]! < bin.first) lo = mid + 1;
          else hi = mid;
        }
        if (lo === sorted.length || sorted[lo]! > bin.last) continue;
      }
      if (!best || bin.mass > best.mass) best = bin;
    }
    if (best && best.prefix.length > MAX_PREFIX) throw new Error("CDF precision limit reached");
    return best?.prefix;
  }
}
