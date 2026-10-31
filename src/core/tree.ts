// Browser adapter for the Rust integer-sequence distribution tree. Rust owns
// nodes, listings, derived slot caches, paths, and all geometry queries; this
// wrapper only adds change notifications for the fetch/render loop.

import { Tree } from "./wasm";

/** A node's id: hashed from its branches, shared by the client and every server. */
export type NodeId = bigint;
/** The empty sequence; every tree's first node. */
export const ROOT: NodeId = 0n;
/** One of a node's 2^32 branches. What it denotes is up to a reading. */
export type Branch = number;

export interface Lens {
  p: number;
  q: number;
}

/** What a measure said about a node's children: the heaviest, in order. */
export interface Listing {
  branches: ArrayLike<Branch>;
  /** Natural log of each listed child's conditional weight. */
  lnWeights: ArrayLike<number>;
  /** Weight of every child not listed. */
  leftover: number;
  /** The horizon asked for, in bits. */
  horizon: number;
  /** The horizon the source vouches for; a recording can vouch for less. */
  complete: number;
}

export interface Slots {
  readonly branches: Branch[];
  readonly p: number[];
  readonly q: number[];
  readonly qBound: boolean[];
  readonly listed: number;
  readonly count: number;
}

export function slotOf(slots: Slots, branch: Branch): number | undefined {
  const slot = slots.branches.indexOf(branch);
  return slot < 0 ? undefined : slot;
}

export function isTail(slots: Slots, slot: number): boolean {
  return slot === slots.listed && slots.count > slots.listed;
}

export class IntervalTree {
  readonly raw: Tree;
  readonly root: NodeId;
  /** Listings, unlike paths, can change. Cache keys include this revision. */
  revision = 0;
  private readonly listeners = new Set<() => void>();
  private readonly revisions = new Map<NodeId, number>();
  private readonly parents = new Map<NodeId, NodeId | null>([[ROOT, null]]);
  private readonly branches = new Map<NodeId, Branch>([[ROOT, -1]]);
  private readonly paths = new Map<NodeId, Branch[]>();

  constructor(readonly measures: number) {
    this.raw = new Tree(measures);
    this.root = this.raw.root();
  }

  child(parent: NodeId, branch: Branch): NodeId {
    const id = this.raw.child(parent, branch);
    this.parents.set(id, parent);
    this.branches.set(id, branch);
    return id;
  }

  /** The node `path` below `node`, creating only what is missing. */
  descend(node: NodeId, path: readonly Branch[]): NodeId {
    return this.raw.descend(node, Uint32Array.from(path));
  }

  parent(id: NodeId): NodeId | null {
    if (this.parents.has(id)) return this.parents.get(id)!;
    const parent = this.raw.parent(id) ?? null;
    this.parents.set(id, parent);
    return parent;
  }

  /** Branch into this node, or -1 at the root. */
  branch(id: NodeId): Branch {
    const cached = this.branches.get(id);
    if (cached !== undefined) return cached;
    const branch = this.raw.branch(id);
    this.branches.set(id, branch);
    return branch;
  }

  /** Branches from the root. */
  path(id: NodeId): Branch[] {
    let path = this.paths.get(id);
    if (path === undefined) this.paths.set(id, path = [...this.raw.path(id)]);
    return [...path];
  }

  /** Nodes from `origin` down to `id`, inclusive. */
  chain(origin: NodeId, id: NodeId): NodeId[] {
    return [...this.raw.chain(origin, id)];
  }

  listing(id: NodeId, measure: number): Listing | undefined {
    return (this.raw.listing(id, measure) as Listing | null) ?? undefined;
  }

  /** What a source answers from this listing at ``horizon`` bits, placing
   * ``through`` too, as the protocol's Expand cuts it; undefined without one. */
  cut(id: NodeId, measure: number, horizon: number, through?: Branch): Listing | undefined {
    return (this.raw.cut(id, measure, horizon, through) as Listing | null) ?? undefined;
  }

  /** Where the child along ``branch`` sits in ``id``; undefined if p does not list it. */
  edge(id: NodeId, lens: Lens, branch: Branch): { slot: number; before: number; lnP: number; natsQ: number; qBound: boolean } | undefined {
    const edge = this.raw.edge(id, lens.p, lens.q, branch);
    if (edge.length === 0) return undefined;
    return { slot: edge[0]!, before: edge[1]!, lnP: edge[2]!, natsQ: edge[3]!, qBound: edge[4] === 1 };
  }

  expanded(id: NodeId, measure: number): boolean {
    return this.raw.expanded(id, measure);
  }

  hasListing(id: NodeId, measure: number): boolean {
    return this.raw.hasListing(id, measure);
  }

  horizon(id: NodeId, measure: number): number {
    return this.raw.horizon(id, measure);
  }

  complete(id: NodeId, measure: number): number {
    return this.raw.complete(id, measure);
  }

  slotOf(id: NodeId, lens: Lens, branch: Branch): number | undefined {
    const slot = this.raw.slotOf(id, lens.p, lens.q, branch);
    return slot < 0 ? undefined : slot;
  }

  statistics(id: NodeId, lens: Lens): { entropy: number; cross: number; bounded: boolean } {
    return this.raw.statistics(id, lens.p, lens.q);
  }

  qBound(id: NodeId, lens: Lens, slot: number): boolean {
    return this.raw.qBound(id, lens.p, lens.q, slot);
  }

  revisionOf(id: NodeId): number { return this.revisions.get(id) ?? 0; }

  slots(id: NodeId, lens: Lens): Slots {
    return this.raw.slots(id, lens.p, lens.q) as Slots;
  }

  setListing(id: NodeId, measure: number, { branches, lnWeights, leftover, horizon, complete }: Listing): void {
    this.raw.setListing(id, measure, Uint32Array.from(branches), Float64Array.from(lnWeights), leftover, horizon, complete);
    this.revision++;
    this.revisions.set(id, this.revision);
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
