// Client replica of a nested interval tree, fetched node by node and measure
// by measure. Legacy v1 listings use p-rank order and a residual tail; v2
// preserves ID order, shared CDF boundaries, and explicit unresolved ranges.

import { boundaryUncertainty, type CDFRegion, type MassBounds, type RadixCDF } from "./radix";
import type { Listing } from "./codec";

export type NodeId = number;
/** What one server calls a node; the client only learns it. */
export type Handle = bigint;
/** A child's name among its siblings; its meaning is agreed outside the protocol. */
export type Label = number;

/** Measure indices sizing slot heights (p) and slot widths (-ln q). */
export interface Lens {
  p: number;
  q: number;
}

/** A listing and the horizon it was fetched at. */
export interface RankedFetched extends Listing {
  horizon: number;
  radix?: undefined;
}

/** CDF evidence is never passed off as a complete ranked listing. */
export interface QuantileFetched {
  radix: RadixCDF;
  horizon: 0;
}

export type Fetched = RankedFetched | QuantileFetched;

/** Upper bound on the weight of any child ``listing`` leaves out. */
function unlistedBound(listing: Listing): number {
  const last = listing.lnWeights.at(-1);
  return Math.min(listing.leftover, last === undefined ? 1 : Math.exp(last), 2 ** -listing.complete);
}

/** A node's slots under one lens. */
export class Slots {
  readonly sampled: boolean;
  readonly widthReady: boolean;
  readonly labels: Label[];
  /** Surprisal kept in log space, including below float probability range. */
  readonly widths: Float64Array;
  /** Conditional p weight per slot; the tail holds p's leftover. */
  readonly p: Float64Array;
  /** Conditional q weight per slot, or an upper bound where ``qBound``. */
  readonly q: Float64Array;
  readonly qBound: boolean[];
  readonly regions: readonly CDFRegion[];
  readonly qBounds: readonly MassBounds[];
  private readonly slotLabels: (Label | null)[];
  readonly listed: number;
  readonly count: number;
  private readonly index = new Map<Label, number>();
  private readonly prefixes: Float64Array;
  private readonly suffixes: Float64Array;

  constructor(p: Fetched | undefined, q: Fetched | undefined) {
    this.sampled = p?.radix !== undefined;
    this.widthReady = !this.sampled || q?.radix?.ready === true;
    this.regions = p?.radix?.partition() ?? [];
    this.labels = p?.radix
      ? this.regions.filter((r) => r.kind === "token").map((r) => r.first)
      : p?.labels ?? [];
    this.listed = this.labels.length;
    const tail = p !== undefined && !p.radix && p.leftover > 0;
    this.slotLabels = this.sampled
      ? this.regions.map((r) => r.kind === "token" ? r.first : null)
      : [...this.labels, ...(tail ? [null] : [])];
    this.count = this.slotLabels.length;
    this.p = new Float64Array(this.count);
    this.q = new Float64Array(this.count);
    this.widths = new Float64Array(this.count);
    this.qBound = new Array<boolean>(this.count).fill(true);
    this.slotLabels.forEach((label, slot) => {
      if (label !== null) this.index.set(label, slot);
    });
    const qBounds: MassBounds[] = [];
    this.qBounds = qBounds;

    if (p?.radix) {
      this.regions.forEach((region, slot) => {
        this.p[slot] = Math.exp(region.lnMass);
        const bounds = region.kind === "unresolved" ? { lower: -Infinity, upper: 0 } : q?.radix?.bounds(region.first) ?? {
          lower: -Infinity,
          upper: 0,
        };
        qBounds.push(bounds);
        const midpoint = q?.radix?.ready ? q.radix.estimate(region.first, region.end) : -Infinity;
        // One shared representative in p=q; an unresolved q endpoint stays
        // explicitly uncertain, including when its lower mass bound is zero.
        const estimate = Number.isFinite(midpoint) ? midpoint : bounds.upper;
        this.widths[slot] = region.kind === "unresolved" ? Infinity : -estimate;
        this.q[slot] = region.kind === "unresolved" ? NaN : Math.exp(estimate);
        this.qBound[slot] = bounds.lower !== bounds.upper;
      });
    } else {
      this.labels.forEach((_, slot) => this.p[slot] = Math.exp(p!.lnWeights[slot]!));
      if (tail) this.p[this.listed] = p!.leftover;
      // Whatever q leaves out weighs at most this; a child it lists only under
      // p's tail can be heavier.
      const rankedQ = q?.radix ? undefined : q;
      const unlisted = rankedQ === undefined ? 1 : unlistedBound(rankedQ);
      this.q.fill(unlisted);
      let tailBound = unlisted;
      rankedQ?.labels.forEach((label, rank) => {
        const weight = Math.exp(rankedQ.lnWeights[rank]!);
        const slot = this.index.get(label);
        if (slot === undefined) tailBound = Math.max(tailBound, weight);
        else {
          this.q[slot] = weight;
          this.qBound[slot] = false;
        }
      });
      if (tail) this.q[this.listed] = tailBound;

      for (let slot = 0; slot < this.count; slot++) this.widths[slot] = -Math.log(this.q[slot]!);
    }

    this.prefixes = new Float64Array(this.count + 1);
    this.suffixes = new Float64Array(this.count + 1);
    for (let slot = 0; slot < this.count; slot++) this.prefixes[slot + 1] = this.prefixes[slot]! + this.p[slot]!;
    for (let slot = this.count - 1; slot >= 0; slot--) this.suffixes[slot] = this.suffixes[slot + 1]! + this.p[slot]!;
  }

  isTail(slot: number): boolean {
    return !this.sampled && slot === this.listed && this.count > this.listed;
  }

  label(slot: number): Label | null {
    return this.slotLabels[slot] ?? null;
  }

  slotOf(label: Label): number | undefined {
    return this.index.get(label);
  }

  /** Preserve an ID boundary even while its containing unresolved range splits. */
  slotAtID(id: Label): number | undefined {
    if (!this.sampled) return this.slotOf(id);
    const slot = this.regions.findIndex((region) => region.first <= id && id < region.end);
    return slot < 0 ? undefined : slot;
  }

  /** Half-widths of the cumulative boundary brackets, in conditional p units. */
  edgeError(slot: number): { before: number; after: number } {
    const region = this.regions[slot];
    return region ? {
      before: Math.exp(boundaryUncertainty(region.before)) / 2,
      after: Math.exp(boundaryUncertainty(region.after)) / 2,
    } : { before: 0, after: 0 };
  }

  /** Σ p over slots [0, slot): accumulated forward, so free of cancellation. */
  prefix(slot: number): number {
    return this.prefixes[Math.min(slot, this.count)]!;
  }

  /** Σ p over slots [slot, count): accumulated backward, so free of cancellation. */
  suffix(slot: number): number {
    return this.suffixes[Math.min(slot, this.count)]!;
  }
}

export class TreeNode {
  readonly children = new Map<Label, NodeId>();
  private readonly listings: (Fetched | undefined)[] = [];
  private readonly handles: (Handle | undefined)[] = [];
  private readonly views = new Map<string, Slots>();

  constructor(
    readonly id: NodeId,
    readonly parent: NodeId | null,
    /** Label of the edge from the parent; -1 for the root. */
    readonly label: Label,
  ) {}

  /** The handle ``measure``'s server gave this node, if it has. */
  handle(measure: number): Handle | undefined {
    return this.handles[measure];
  }

  setHandle(measure: number, handle: Handle | undefined): void {
    this.handles[measure] = handle;
  }

  listing(measure: number): Fetched | undefined {
    return this.listings[measure];
  }

  /** Whether ``measure`` has said anything about this node's children. */
  expanded(measure: number): boolean {
    const listing = this.listings[measure];
    return (
      listing !== undefined && (listing.radix ? listing.radix.ready : listing.labels.length > 0 || listing.complete > 0)
    );
  }

  slots(lens: Lens): Slots {
    const key = `${lens.p}:${lens.q}`;
    let slots = this.views.get(key);
    if (slots === undefined) {
      slots = new Slots(this.listings[lens.p], this.listings[lens.q]);
      this.views.set(key, slots);
    }
    return slots;
  }

  clearViews(): void {
    this.views.clear();
  }

  setListing(measure: number, listing: Fetched): void {
    this.listings[measure] = listing;
    this.views.clear();
  }
}

export class IntervalTree {
  readonly nodes: TreeNode[] = [new TreeNode(0, null, -1)];
  private listeners = new Set<() => void>();

  constructor(readonly measures: number) {}

  get root(): TreeNode {
    return this.nodes[0]!;
  }

  node(id: NodeId): TreeNode {
    const node = this.nodes[id];
    if (node === undefined) throw new Error(`unknown node ${id}`);
    return node;
  }

  /** The child of ``parent`` with ``label``, created on first use. */
  child(parent: NodeId, label: Label): TreeNode {
    const owner = this.node(parent);
    const id = owner.children.get(label);
    if (id !== undefined) return this.node(id);
    const node = new TreeNode(this.nodes.length, parent, label);
    this.nodes.push(node);
    owner.children.set(label, node.id);
    return node;
  }

  /** Nodes from the root down to ``id``, inclusive. */
  chain(id: NodeId): TreeNode[] {
    const chain: TreeNode[] = [];
    for (let node: TreeNode | undefined = this.node(id); node;) {
      chain.push(node);
      node = node.parent === null ? undefined : this.nodes[node.parent];
    }
    return chain.reverse();
  }

  /** Edge labels from the root down to ``id``. */
  path(id: NodeId): Label[] {
    return this.chain(id)
      .slice(1)
      .map((node) => node.label);
  }

  /** Refreshes derived slots after a shared v2 CDF gains a page. */
  touch(id?: NodeId): void {
    if (id !== undefined) this.node(id).clearViews();
    else for (const node of this.nodes) node.clearViews();
    for (const listener of this.listeners) listener();
  }

  setListing(id: NodeId, measure: number, listing: Fetched): void {
    this.node(id).setListing(measure, listing);
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
