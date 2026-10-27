// Client replica of an append-only nested interval tree (see proto/natlas/v1).
//
// A node's slots are its listed children followed, while the listing is
// incomplete, by one tail slot that aggregates every unlisted child. The tail
// is a coarse event like any other slot; it just cannot become a node.

import type { Hello, Node, Page, ServerMessage } from "./gen/natlas/v1/natlas_pb";

export type NodeId = number;

const MASS_TOLERANCE = 1e-9;

export class TreeNode {
  readonly labels: string[] = [];
  readonly weights: number[][];
  readonly tail: number[];
  readonly children = new Map<number, NodeId>();
  private sums: { prefix: Float64Array; suffix: Float64Array }[] = [];
  private paged = false;

  constructor(
    readonly id: NodeId,
    readonly parent: NodeId | null,
    readonly slot: number,
    readonly size: number,
    measures: number,
  ) {
    this.weights = Array.from({ length: measures }, () => []);
    this.tail = new Array<number>(measures).fill(0);
  }

  get listed(): number {
    return this.labels.length;
  }

  get hasTail(): boolean {
    return this.listed < this.size;
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
    return slot < this.listed ? this.labels[slot]! : `… ${this.size - this.listed}`;
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

  /** Append or verify one page of slots. */
  apply(page: Page): void {
    const measures = this.weights.length;
    if (page.weights.length !== measures) throw new Error(`page for node ${this.id} has wrong measure count`);
    const n = page.labels.length;
    for (const weights of page.weights) {
      if (weights.values.length !== n) throw new Error(`page for node ${this.id} has ragged weights`);
    }
    if (page.start > this.listed) {
      throw new Error(`missing slots before ${page.start} of node ${this.id}; listed ${this.listed}`);
    }
    if (page.start + n > this.size) throw new Error(`page overruns node ${this.id}`);

    const overlap = Math.min(this.listed - page.start, n);
    for (let offset = 0; offset < overlap; offset++) {
      const slot = page.start + offset;
      const same =
        this.labels[slot] === page.labels[offset] &&
        page.weights.every((weights, m) => weights.values[offset] === this.weights[m]![slot]);
      if (!same) throw new Error(`conflicting slot ${slot} of node ${this.id}`);
    }
    if (this.paged && page.start + n <= this.listed) return;

    const tail = page.start + n === this.size ? new Array<number>(measures).fill(0) : page.tail;
    if (tail.length !== measures) throw new Error(`page for node ${this.id} has wrong tail count`);
    if (this.paged) {
      for (let m = 0; m < measures; m++) {
        let mass = tail[m]!;
        for (let offset = overlap; offset < n; offset++) mass += page.weights[m]!.values[offset]!;
        if (Math.abs(mass - this.tail[m]!) > MASS_TOLERANCE * Math.max(1, this.tail[m]!)) {
          throw new Error(`page for node ${this.id} does not conserve tail mass of measure ${m}`);
        }
      }
    }

    for (let offset = overlap; offset < n; offset++) {
      this.labels.push(page.labels[offset]!);
      for (let m = 0; m < measures; m++) this.weights[m]!.push(page.weights[m]!.values[offset]!);
    }
    for (let m = 0; m < measures; m++) this.tail[m] = tail[m]!;
    this.sums = [];
    this.paged = true;
  }
}

export class IntervalTree {
  title = "";
  measures: string[] = [];
  readonly nodes: TreeNode[] = [];
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
      case "node":
        return this.applyNode(message.kind.value);
      case "page":
        return this.node(Number(message.kind.value.node)).apply(message.kind.value);
      default:
        throw new Error("empty server message");
    }
  }

  private applyHello(hello: Hello): void {
    if (this.nodes.length > 0) {
      if (hello.measures.join("\0") !== this.measures.join("\0")) throw new Error("measures changed mid-stream");
      return;
    }
    if (hello.measures.length === 0) throw new Error("stream declares no measures");
    this.title = hello.title;
    this.measures = [...hello.measures];
  }

  private applyNode(message: Node): void {
    if (this.measures.length === 0) throw new Error("node before hello");
    const id = Number(message.id);
    const root = id === 0;
    const parent = root ? null : Number(message.parent);
    const slot = root ? 0 : message.slot;

    if (id < this.nodes.length) {
      const existing = this.nodes[id]!;
      if (existing.parent !== parent || existing.slot !== slot || existing.size !== message.size) {
        throw new Error(`conflicting contents for node ${id}`);
      }
      if (message.page) existing.apply({ ...message.page, start: 0 });
      return;
    }
    if (id > this.nodes.length) throw new Error(`missing nodes before ${id}; expected ${this.nodes.length}`);

    if (parent !== null) {
      const owner = this.node(parent);
      if (slot >= owner.listed) throw new Error(`node ${id} expands unlisted slot ${slot} of ${parent}`);
      if (owner.children.has(slot)) throw new Error(`slot ${slot} of node ${parent} is already expanded`);
      owner.children.set(slot, id);
    }
    const node = new TreeNode(id, parent, slot, message.size, this.measures.length);
    this.nodes.push(node);
    if (message.page) node.apply({ ...message.page, start: 0 });
  }
}
