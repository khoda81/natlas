// How a dataset's labels read. The protocol only carries integers; a labels
// file says what they stand for, and its kind picks the reader: text for a
// language model's tokens, notes for a transcription's MIDI-like events. A
// reader shows each box's label and the path down to a box; neither touches
// the tree's weights.

import type { IntervalTree, Label, NodeId } from "./tree";

/** A shape drawn before a label's text, as tall as the text. */
export type Mark = "onset" | "offset" | "tie" | "hit" | "time" | "instrument" | "end";

export interface Glyph {
  text: string;
  mark?: Mark;
  /** The mark's color; the text keeps the box's ink. */
  tint?: string;
}

export interface Labels {
  /** A reading of ``tree``'s labels, remembering what it has read per node. */
  read(tree: IntervalTree): Reading;
}

export interface Reading {
  /** The child with ``label`` of ``node``, read after the path to ``node``. */
  glyph(node: NodeId, label: Label): Glyph;
  /**
   * Fill ``panel`` with the path down to child ``label`` of ``node``, or to
   * ``node`` itself followed by ``more`` when ``label`` is null.
   */
  show(panel: HTMLElement, node: NodeId, label: Label | null, more: string): void;
  /** Handle a key the view leaves alone, such as playing what the panel shows. */
  key?(event: KeyboardEvent): boolean;
}

/**
 * What a reader knows after each node's path, computed along the path once
 * and kept: walking up stops at the nearest node already read.
 */
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
    for (let node = this.tree.node(id); node.parent !== null; node = this.tree.node(node.parent)) {
      const known = this.known.get(node.id);
      if (known !== undefined) {
        state = known;
        break;
      }
      pending.push(node.label);
      ids.push(node.id);
    }
    for (let index = pending.length - 1; index >= 0; index--) {
      state = this.after(state, pending[index]!);
      this.known.set(ids[index]!, state);
    }
    return state;
  }
}
