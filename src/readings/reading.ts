// Dataset-specific readings of integer labels. Readers never touch weights or
// distribution geometry: text renders token bytes, music renders event paths.

import type { IntervalTree, Branch, NodeId } from "../core/tree";
import type { Values } from "../data/catalog";

export type Mark = "onset" | "offset" | "tie" | "hit" | "time" | "instrument" | "end";

export interface Glyph {
  text: string;
  mark?: Mark;
  tint?: string;
  /** Where the event moves time from and to, in the reading's time, whether
   * that time is shaded, and where time stood before ``from``: what a
   * timeline shows in place of the text. */
  time?: { from: number; to: number; shaded: boolean; trail: Trail | null };
}

/** The times a path's time stood at, newest first, each shaded on to the
 * next or not. A path's is its parent's with at most one more, which it shares. */
export interface Trail {
  at: number;
  shaded: boolean;
  before: Trail | null;
}

/** Where a reading draws its time across the page. */
export interface Timeline {
  /** The client x of a time. */
  x(time: number): number;
  /** The client x range it draws. */
  left: number;
  right: number;
}

/** The measure p, as a reading sees it: what its spec says, and the
 * recording it hears where it hears one. */
export interface Measure {
  values: Values;
  recording: { url: URL; seconds: number } | null;
}

/** What a reading may tell the page it is read on. */
export interface Page {
  /** What glyphs show changed without the tree: a timeline that moved. */
  changed(): void;
  /** Fly the view down a path from the root. */
  go(path: readonly Branch[]): void;
  failed(error: unknown): void;
}

export interface Labels {
  /** Paths as ``p`` reads them, on ``page``. */
  read(tree: IntervalTree, page: Page, p: Measure): Reading;
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
