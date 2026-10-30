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
  private readonly listeners = new Set<() => void>();

  constructor(readonly measures: number) {
    this.raw = new Tree(measures);
    this.root = this.raw.root();
  }

  child(parent: NodeId, label: Label): NodeId {
    return this.raw.child(parent, label);
  }

  parent(id: NodeId): NodeId | null {
    return this.raw.parent(id) as NodeId | null;
  }

  /** Edge label into this node, or -1 at the root. */
  label(id: NodeId): Label {
    return this.raw.label(id);
  }

  path(id: NodeId): Label[] {
    return [...this.raw.path(id)];
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

  slots(id: NodeId, lens: Lens): Slots {
    return this.raw.slots(id, lens.p, lens.q) as Slots;
  }

  setListing(id: NodeId, measure: number, listing: Fetched): void {
    this.raw.setListing(id, measure, listing);
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
