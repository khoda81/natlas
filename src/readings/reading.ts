// Dataset-specific readings of integer labels. Readers never touch weights or
// distribution geometry: text renders token bytes, music renders event paths.

import type { IntervalTree, Label, NodeId } from "../core/tree";

export type Mark = "onset" | "offset" | "tie" | "hit" | "time" | "instrument" | "end";

export interface Glyph {
  text: string;
  mark?: Mark;
  tint?: string;
}

export interface Labels {
  read(tree: IntervalTree): Reading;
}

export interface Reading {
  glyph(node: NodeId, label: Label): Glyph;
  /** Show the path to ``node``/``label``, with an optional unknown continuation. */
  show(panel: HTMLElement, node: NodeId, label: Label | null, more?: string): void;
  /** Plain text for editing/searching, for readers whose labels represent text. */
  text?(node: NodeId, label: Label | null): string;
  key?(event: KeyboardEvent): boolean;
}

/** Reader state after each node path, reconstructed from the nearest known ancestor. */
export class States<S> {
  private readonly known = new Map<NodeId, S>();

  constructor(
    private readonly tree: IntervalTree,
    private readonly root: S,
    private readonly after: (state: S, label: Label) => S,
  ) {}

  at(id: NodeId): S {
    const pending: Label[] = [];
    const ids: NodeId[] = [];
    let state = this.root;
    for (let node = id; this.tree.parent(node) !== null; node = this.tree.parent(node)!) {
      const known = this.known.get(node);
      if (known !== undefined) {
        state = known;
        break;
      }
      pending.push(this.tree.label(node));
      ids.push(node);
    }
    for (let index = pending.length - 1; index >= 0; index--) {
      state = this.after(state, pending[index]!);
      this.known.set(ids[index]!, state);
    }
    return state;
  }
}
