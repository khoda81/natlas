// Browser adapter for the Rust integer-sequence distribution tree. Rust owns
// nodes, listings, derived slot caches, paths, and all geometry queries; this
// wrapper only adds change notifications for the fetch/render loop.

import type { Listing } from "./codec";
import { Tree } from "./wasm";

export type NodeId = number;
/** The integer identifying one branch among its siblings. */
export type Label = number;

export interface Lens {
  p: number;
  q: number;
}

export interface Fetched extends Listing {
  horizon: number;
}

export interface Slots {
  readonly labels: Label[];
  readonly p: number[];
  readonly q: number[];
  readonly qBound: boolean[];
  readonly listed: number;
  readonly count: number;
}

export function slotOf(slots: Slots, label: Label): number | undefined {
  const slot = slots.labels.indexOf(label);
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
  private readonly parents = new Map<NodeId, NodeId | null>([[0, null]]);
  private readonly labels = new Map<NodeId, Label>([[0, -1]]);
  private readonly paths = new Map<NodeId, Label[]>();

  constructor(readonly measures: number) {
    this.raw = new Tree(measures);
    this.root = this.raw.root();
  }

  child(parent: NodeId, label: Label): NodeId {
    const id = this.raw.child(parent, label);
    this.parents.set(id, parent);
    this.labels.set(id, label);
    return id;
  }

  parent(id: NodeId): NodeId | null {
    if (this.parents.has(id)) return this.parents.get(id)!;
    const parent = this.raw.parent(id) as NodeId | null;
    this.parents.set(id, parent);
    return parent;
  }

  /** Edge label into this node, or -1 at the root. */
  label(id: NodeId): Label {
    const cached = this.labels.get(id);
    if (cached !== undefined) return cached;
    const label = this.raw.label(id);
    this.labels.set(id, label);
    return label;
  }

  path(id: NodeId): Label[] {
    let path = this.paths.get(id);
    if (path === undefined) this.paths.set(id, path = [...this.raw.path(id)]);
    return [...path];
  }

  chain(id: NodeId): NodeId[] {
    return this.raw.chain(id) as NodeId[];
  }

  listing(id: NodeId, measure: number): Fetched | undefined {
    return (this.raw.listing(id, measure) as Fetched | null) ?? undefined;
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

  slotOf(id: NodeId, lens: Lens, label: Label): number | undefined {
    const slot = this.raw.slotOf(id, lens.p, lens.q, label);
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

  setListing(id: NodeId, measure: number, listing: Fetched): void {
    this.raw.setListing(id, measure, listing);
    this.revision++;
    this.revisions.set(id, this.revision);
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
