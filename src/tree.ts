// Client replica of a nested interval tree streamed as runs (see proto/natlas/v1).
//
// A node's slots are its listed children followed, while the listing is
// incomplete, by one tail slot holding the rest. The tail is a coarse event
// like any other slot; it just cannot become a node. Runs carry log weights
// relative to the node's first child; a node normalizes them into conditional
// weights each time its listing changes.

import type { Hello, Run, ServerMessage } from "./gen/natlas/v1/natlas_pb";

export type NodeId = number;
export type Handle = bigint;

export const ROOT: Handle = 0n;

const MASK = (1n << 64n) - 1n;
const GOLDEN = 0x9e3779b97f4a7c15n;

/** Handle of the child at ``rank`` of the node with handle ``parent``. */
export function childHandle(parent: Handle, rank: number): Handle {
  let z = (parent + GOLDEN * BigInt(rank + 1)) & MASK;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK;
  return z ^ (z >> 31n);
}

export class TreeNode {
  readonly labels: string[] = [];
  /** Conditional weights per measure, normalized over listed slots and the tail. */
  readonly weights: number[][];
  readonly tail: number[];
  readonly children = new Map<number, NodeId>();
  /** Log weights per measure, in log units below the first child: ln(w[0] / w[r]). */
  private readonly logs: number[][];
  private rest: number[] | null = null;
  private sums: { prefix: Float64Array; suffix: Float64Array }[] = [];

  constructor(
    readonly id: NodeId,
    readonly handle: Handle,
    readonly parent: NodeId | null,
    readonly slot: number,
    measures: number,
    private readonly logScale: number,
  ) {
    this.weights = Array.from({ length: measures }, () => []);
    this.logs = Array.from({ length: measures }, () => []);
    this.tail = new Array<number>(measures).fill(0);
  }

  get listed(): number {
    return this.labels.length;
  }

  get hasTail(): boolean {
    return this.rest !== null;
  }

  /** Listed slots plus the tail slot, if any. */
  get slots(): number {
    return this.listed + (this.hasTail ? 1 : 0);
  }

  isTail(slot: number): boolean {
    return slot === this.listed && this.hasTail;
  }

  weight(measure: number, slot: number): number {
    return slot < this.listed ? this.weights[measure]![slot]! : this.tail[measure]!;
  }

  label(slot: number): string {
    return slot < this.listed ? this.labels[slot]! : "…";
  }

  /** Σ weight over slots [0, slot): accumulated forward, so free of cancellation. */
  prefix(measure: number, slot: number): number {
    return this.cumulative(measure).prefix[Math.min(slot, this.slots)]!;
  }

  /** Σ weight over slots [slot, slots): accumulated backward, so free of cancellation. */
  suffix(measure: number, slot: number): number {
    return this.cumulative(measure).suffix[Math.min(slot, this.slots)]!;
  }

  private cumulative(measure: number) {
    let sums = this.sums[measure];
    if (sums === undefined) {
      const n = this.slots;
      const prefix = new Float64Array(n + 1);
      const suffix = new Float64Array(n + 1);
      for (let slot = 0; slot < n; slot++) prefix[slot + 1] = prefix[slot]! + this.weight(measure, slot);
      for (let slot = n - 1; slot >= 0; slot--) suffix[slot] = suffix[slot + 1]! + this.weight(measure, slot);
      sums = { prefix, suffix };
      this.sums[measure] = sums;
    }
    return sums;
  }

  /** Append or verify one run of children; returns the newly listed range. */
  apply(run: Run): { from: number; to: number } {
    const measures = this.logs.length;
    const { start } = run;
    const n = run.labels.length;
    const end = start + n;
    if (start > this.listed) {
      throw new Error(`missing children before ${start} of node ${this.id}; listed ${this.listed}`);
    }
    // Rank 0 carries no log weights: it is the zero they are measured from.
    const first = Math.max(start, 1);
    const entries = Math.max(0, end - first);
    if (run.gaps.length !== entries) throw new Error(`run for node ${this.id} has ${run.gaps.length} gaps for ${n} children`);
    if (run.others.length !== measures - 1) throw new Error(`run for node ${this.id} has wrong measure count`);
    for (const logs of run.others) {
      if (logs.values.length !== entries) throw new Error(`run for node ${this.id} has ragged weights`);
    }
    if (run.rest !== undefined) {
      if (run.rest.values.length !== measures) throw new Error(`run for node ${this.id} has wrong rest count`);
      if (end === 0) throw new Error(`run for node ${this.id} has a rest but no children`);
    }

    const levels = this.logs.map(() => [] as number[]);
    for (let rank = start; rank < end; rank++) {
      const offset = rank - first;
      if (rank === 0) {
        for (const level of levels) level.push(0);
        continue;
      }
      const previous = rank - 1 < start ? this.logs[0]![rank - 1]! : levels[0]![rank - 1 - start]!;
      levels[0]!.push(previous + run.gaps[offset]!);
      for (let m = 1; m < measures; m++) levels[m]!.push(run.others[m - 1]!.values[offset]!);
    }

    const listed = this.listed;
    for (let rank = start; rank < Math.min(end, listed); rank++) {
      const same =
        this.labels[rank] === run.labels[rank - start] &&
        levels.every((level, m) => level[rank - start] === this.logs[m]![rank]);
      if (!same) throw new Error(`conflicting child ${rank} of node ${this.id}`);
    }
    // A run that stops short of the listing is a replay: its rest is stale.
    if (end < listed) return { from: listed, to: listed };

    for (let rank = listed; rank < end; rank++) {
      this.labels.push(run.labels[rank - start]!);
      for (let m = 0; m < measures; m++) this.logs[m]!.push(levels[m]![rank - start]!);
    }
    this.rest = run.rest === undefined ? null : [...run.rest.values];
    this.normalize();
    return { from: listed, to: end };
  }

  private normalize(): void {
    const unit = 1 / this.logScale;
    this.logs.forEach((logs, m) => {
      // Shift by the heaviest child so nothing overflows.
      const rest = this.rest?.[m];
      const heaviest = Math.min(...logs, rest ?? Infinity);
      const scale = (log: number) => Math.exp((heaviest - log) * unit);
      const weights = logs.map(scale);
      const tail = rest === undefined ? 0 : scale(rest);
      let total = tail;
      for (const weight of weights) total += weight;
      this.weights[m] = weights.map((weight) => weight / total);
      this.tail[m] = total === 0 ? 0 : tail / total;
    });
    this.sums = [];
  }
}

export class IntervalTree {
  title = "";
  measures: string[] = [];
  logScale = 0;
  readonly nodes: TreeNode[] = [];
  private readonly ids = new Map<Handle, NodeId>();
  /** Listed children that are not nodes yet, by handle. */
  private readonly pending = new Map<Handle, { parent: NodeId; slot: number }>();
  private listeners = new Set<() => void>();

  get root(): TreeNode | undefined {
    return this.nodes[0];
  }

  node(id: NodeId): TreeNode {
    const node = this.nodes[id];
    if (node === undefined) throw new Error(`unknown node ${id}`);
    return node;
  }

  /** Nodes from the root down to ``id``, inclusive. */
  chain(id: NodeId): TreeNode[] {
    const chain: TreeNode[] = [];
    for (let node: TreeNode | undefined = this.node(id); node; ) {
      chain.push(node);
      node = node.parent === null ? undefined : this.nodes[node.parent];
    }
    return chain.reverse();
  }

  /** Slot labels from the root down to ``id``. */
  path(id: NodeId): string[] {
    return this.chain(id)
      .slice(1)
      .map((node) => this.node(node.parent!).label(node.slot));
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  applyAll(messages: Iterable<ServerMessage>): void {
    for (const message of messages) this.applyOne(message);
    this.notify();
  }

  apply(message: ServerMessage): void {
    this.applyOne(message);
    this.notify();
  }

  private notify(): void {
    for (const listener of this.listeners) listener();
  }

  private applyOne(message: ServerMessage): void {
    switch (message.kind.case) {
      case "hello":
        return this.applyHello(message.kind.value);
      case "run":
        return this.applyRun(message.kind.value);
      default:
        throw new Error("empty server message");
    }
  }

  private applyHello(hello: Hello): void {
    if (this.measures.length > 0) {
      const same = hello.measures.join("\0") === this.measures.join("\0") && hello.logScale === this.logScale;
      if (!same) throw new Error("measures changed mid-stream");
      return;
    }
    if (hello.measures.length === 0) throw new Error("stream declares no measures");
    if (hello.logScale === 0) throw new Error("stream declares no log scale");
    this.title = hello.title;
    this.measures = [...hello.measures];
    this.logScale = hello.logScale;
  }

  private applyRun(run: Run): void {
    if (this.measures.length === 0) throw new Error("run before hello");
    const node = this.resolve(run.node);
    const { from, to } = node.apply(run);
    for (let slot = from; slot < to; slot++) this.pending.set(childHandle(node.handle, slot), { parent: node.id, slot });
  }

  /** The node with ``handle``, created on its first run. */
  private resolve(handle: Handle): TreeNode {
    const id = this.ids.get(handle);
    if (id !== undefined) return this.node(id);

    let place: { parent: NodeId | null; slot: number } | undefined = this.pending.get(handle);
    if (place === undefined && handle === ROOT && this.nodes.length === 0) place = { parent: null, slot: 0 };
    if (place === undefined) throw new Error(`run for unknown node ${handle.toString(16)}`);

    const node = new TreeNode(this.nodes.length, handle, place.parent, place.slot, this.measures.length, this.logScale);
    this.nodes.push(node);
    this.ids.set(handle, node.id);
    this.pending.delete(handle);
    if (place.parent !== null) this.node(place.parent).children.set(place.slot, node.id);
    return node;
  }
}
