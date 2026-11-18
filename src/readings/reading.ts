// Dataset-specific readings of integer labels. Readers never touch weights or
// distribution geometry: text renders token bytes, music renders event paths.

import type { IntervalTree, Branch, NodeId } from "../core/tree";

export type Mark = "onset" | "offset" | "tie" | "hit" | "time" | "instrument" | "end";

export interface Glyph {
  text: string;
  mark?: Mark;
  tint?: string;
  /** Where the event moves time from and to, in the reading's time: what a
   * timeline shows in place of the text. */
  time?: { from: number; to: number };
}

/** Where a reading draws its time across the page. */
export interface Timeline {
  /** The client x of a time. */
  x(time: number): number;
  /** The client x range it draws. */
  left: number;
  right: number;
}

export interface Labels {
  /** ``changed`` is told when what glyphs show changes without the tree:
   * a timeline that moved. */
  read(tree: IntervalTree, changed: () => void): Reading;
}

export interface Reading {
  glyph(node: NodeId, branch: Branch): Glyph;
  /** Show the path to ``node``/``branch``, with an optional unknown continuation. */
  show(panel: HTMLElement, node: NodeId, branch: Branch | null, more?: string): void;
  /** Plain text for editing/searching, for readers whose labels represent text. */
  text?(node: NodeId, branch: Branch | null): string;
  key?(event: KeyboardEvent): boolean;
  /** Where its time is on the page now, while it shows it. */
  timeline?(): Timeline | null;
  /** Ranges of branches, each [start, end), it draws as one: a node's
   * children in a group are drawn at their mean width. */
  groups?(): (readonly [Branch, Branch])[];
}

/** Reader state after each node path, reconstructed from the nearest known ancestor. */
export class States<S> {
  private readonly known = new Map<NodeId, S>();

  constructor(
    private readonly tree: IntervalTree,
    private readonly root: S,
    private readonly after: (state: S, branch: Branch) => S,
  ) {}

  at(id: NodeId): S {
    const pending: Branch[] = [];
    const ids: NodeId[] = [];
    let state = this.root;
    for (let node = id; this.tree.parent(node) !== null; node = this.tree.parent(node)!) {
      const known = this.known.get(node);
      if (known !== undefined) {
        state = known;
        break;
      }
      pending.push(this.tree.branch(node));
      ids.push(node);
    }
    for (let index = pending.length - 1; index >= 0; index--) {
      state = this.after(state, pending[index]!);
      this.known.set(ids[index]!, state);
    }
    return state;
  }
}
