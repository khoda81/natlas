// Browser adapter for the Rust integer-sequence tree. Rust owns nodes,
// distributions, and all geometry; this wrapper adds change notifications
// for the fetch/render loop and caches immutable paths.

import { Tree } from "./wasm";

/** A node's id: hashed from its branches, shared by the client and every server. */
export type NodeId = bigint;
/** The empty sequence; every tree's first node. */
export const ROOT: NodeId = 0n;
/** One of a node's 2^32 branches. What it denotes is up to a reading. */
export type Branch = number;
/** A child's position in its measure's order: 0 is the heaviest. */
export type Rank = number;

export interface Lens {
  p: number;
  q: number;
}

/** Consecutive ranks of a node's children, as a source sends them. */
export interface Run {
  rank: Rank;
  branches: ArrayLike<Branch>;
  /** ln of the mass at and below each edge; one more than branches. */
  lnBelow: ArrayLike<number>;
}

/** What a source places of one node's children in one answer. */
export interface Children {
  runs: readonly Run[];
  /** How many of the node's children have weight, where the source knows. */
  count?: number;
}

/** A child as a lens sees it within its parent. */
export interface Entry {
  rank: Rank;
  /** p mass between the parent's top and the child's. */
  before: number;
  lnP: number;
  /** −ln q; infinite while q is unknown. */
  natsQ: number;
}

export class IntervalTree {
  readonly raw: Tree;
  readonly root: NodeId;
  /** Distributions, unlike paths, change. Cache keys include this revision. */
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

  /** Whether `measure` has said anything of the node's children. */
  has(id: NodeId, measure: number): boolean {
    return this.raw.has(id, measure);
  }

  /** Whether `measure` says the node has no children at all. */
  ended(id: NodeId, measure: number): boolean {
    return this.raw.ended(id, measure);
  }

  /** Where p places the child along ``branch`` in ``id``; undefined where it does not. */
  entry(id: NodeId, lens: Lens, branch: Branch): Entry | undefined {
    const entry = this.raw.entry(id, lens.p, lens.q, branch);
    if (entry.length === 0) return undefined;
    return { rank: entry[0]!, before: entry[1]!, lnP: entry[2]!, natsQ: entry[3]! };
  }

  statistics(id: NodeId, lens: Lens): { entropy: number; cross: number; bounded: boolean } {
    return this.raw.statistics(id, lens.p, lens.q);
  }

  revisionOf(id: NodeId): number { return this.revisions.get(id) ?? 0; }

  /** Add one answer for a node. Listeners hear only of answers that changed something. */
  insert(id: NodeId, measure: number, { runs, count }: Children): void {
    for (const run of runs) {
      if (run.lnBelow.length !== run.branches.length + 1) throw new Error("a run needs one more edge than branches");
    }
    const ranks = Uint32Array.from(runs, (run) => run.rank);
    const lengths = Uint32Array.from(runs, (run) => run.branches.length);
    const branches = new Uint32Array(lengths.reduce((sum, length) => sum + length, 0));
    const lnBelow = new Float64Array(branches.length + runs.length);
    // Each run's edges follow its branches' offset, one more per run before it.
    for (let i = 0, at = 0; i < runs.length; at += runs[i]!.branches.length, i++) {
      branches.set(runs[i]!.branches, at);
      lnBelow.set(runs[i]!.lnBelow, at + i);
    }
    if (!this.raw.insert(id, measure, ranks, lengths, branches, lnBelow, count)) return;
    this.revision++;
    this.revisions.set(id, this.revision);
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
