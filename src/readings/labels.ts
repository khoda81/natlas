// Label texts. A language model's labels are tokens, and a token can hold
// part of a character (an emoji, or a Persian letter, split over byte-level
// tokens), so a label is bytes and its text depends on the bytes before it:
// it shows the characters its own bytes complete. A labels file is a JSON
// array of base64 strings, indexed by label, as proto3 JSON writes bytes.

import { type ChatSection, type Glyph, type Labels, type Reading, States } from "./reading";
import type { IntervalTree, Branch, NodeId } from "../core/tree";

const TURN_START = "<|im_start|>";
const USER_START = `${TURN_START}user\n`;
const ASSISTANT_START = `${TURN_START}assistant\n`;
const TURN_END = "<|im_end|>";
const THINK_START = "<think>";
const THINK_END = "</think>";
const CHAT_MARKERS = [USER_START, ASSISTANT_START, TURN_START, TURN_END, THINK_START, THINK_END];
const TURN_START_BYTES = new TextEncoder().encode(TURN_START);

interface ChatState {
  section: ChatSection | null;
  /** A suffix that can complete a delimiter or role header on the next branch. */
  pending: string;
}

function chatAfter(before: ChatState, text: string): ChatState {
  text = before.pending + text;
  let section = before.section;
  for (let at = text.indexOf("<"); at !== -1;) {
    const tail = text.slice(at);
    if (CHAT_MARKERS.some(marker => tail.length < marker.length && marker.startsWith(tail))) {
      return { section: tail.startsWith(TURN_START) ? null : section, pending: tail };
    }
    const marker = CHAT_MARKERS.find(marker => tail.startsWith(marker));
    switch (marker) {
      case USER_START: section = "user"; break;
      case ASSISTANT_START: section = "assistant"; break;
      case TURN_START:
      case TURN_END: section = null; break;
      case THINK_START: if (section === "assistant") section = "thinking"; break;
      case THINK_END: if (section === "thinking") section = "assistant"; break;
    }
    at = text.indexOf("<", at + (marker?.length ?? 1));
  }
  return section === before.section && before.pending === "" ? before : { section, pending: "" };
}

/** The bytes a decoder holds back after ``bytes``: an unfinished UTF-8 character, at most 3 bytes. */
function unfinished(bytes: Uint8Array): Uint8Array {
  for (let back = 1; back <= Math.min(3, bytes.length); back++) {
    const byte = bytes[bytes.length - back]!;
    if ((byte & 0xc0) === 0x80) continue;
    const length = byte >= 0xf0 ? 4 : byte >= 0xe0 ? 3 : byte >= 0xc0 ? 2 : 1;
    return length > back ? bytes.slice(-back) : new Uint8Array();
  }
  return new Uint8Array();
}

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  if (a.length === 0) return b;
  const joined = new Uint8Array(a.length + b.length);
  joined.set(a);
  joined.set(b, a.length);
  return joined;
}

function hex(bytes: Uint8Array): string {
  return `‹${Array.from(bytes, (byte) => byte.toString(16).toUpperCase().padStart(2, "0")).join("\u2009")}›`;
}

export class TextLabels implements Labels {
  private readonly decoder = new TextDecoder();
  private readonly chat: boolean;

  constructor(private readonly bytes: readonly Uint8Array[]) {
    this.chat = bytes.some(label => equal(label, TURN_START_BYTES));
  }

  get searchable(): boolean {
    return this.bytes.length > 0;
  }

  /** Labels from a labels file's array of base64 strings. */
  static fromJson(encoded: string[]): TextLabels {
    return new TextLabels(encoded.map((text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0))));
  }

  static parse(json: string): TextLabels {
    return TextLabels.fromJson(JSON.parse(json) as string[]);
  }

  static none(): TextLabels {
    return new TextLabels([]);
  }

  read(tree: IntervalTree): Reading {
    // Per node, the unfinished character its path's bytes end in.
    const unfinished = new States<Uint8Array>(tree, new Uint8Array(), (before, branch) => this.after(before, branch).unfinished);
    const readChat = (before: ChatState, branch: Branch) => {
      const own = this.bytes[branch];
      if (before.pending === "" && !own?.includes(0x3c)) return before; // '<' starts every delimiter.
      return chatAfter(before, own === undefined ? `#${branch}` : this.decoder.decode(own));
    };
    const chat = this.chat ? new States<ChatState>(tree, { section: null, pending: "" }, readChat) : null;
    const paths = new Map<NodeId, { tokens: string[]; text: string }>();
    const along = (node: NodeId) => {
      let texts = paths.get(node);
      if (texts === undefined) {
        const tokens = this.along(tree.path(node));
        paths.set(node, texts = { tokens, text: tokens.join("") });
      }
      return texts;
    };
    return {
      glyph: (node: NodeId, branch: Branch): Glyph => ({ text: visible(this.shown(unfinished.at(node), branch)) }),
      conversation: chat === null ? undefined : (node, branch) => {
        const before = chat.at(node);
        return (branch === undefined ? before : readChat(before, branch)).section;
      },
      show: (panel: HTMLElement, node: NodeId, branch: Branch | null, more?: string) => {
        const tokens = along(node).tokens;
        const current = branch === null ? null : this.shown(unfinished.at(node), branch);
        const count = tokens.length + Number(current !== null) + Number(more !== undefined);
        panel.dataset.kind = "text";
        // Most pointer moves change only the final token. Preserve the exact
        // span markup while avoiding reconstruction/layout of the whole path.
        // Walk siblings, not a repeatedly invalidated live childNodes index;
        // append new spans together so long paths incur one DOM insertion.
        let child = panel.firstChild;
        const added = document.createDocumentFragment();
        for (let i = 0; i < count; i++) {
          const next = child?.nextSibling ?? null;
          const kind = i < tokens.length ? "" : i === tokens.length && current !== null ? "current" : "more";
          const text = kind === "" ? tokens[i]! : kind === "current" ? current! : more!;
          const className = kind === "" ? "token" : `token ${kind}`;
          if (!(child instanceof HTMLElement) || child.tagName !== "SPAN" || child.className !== className) {
            const span = token(text, kind || undefined);
            if (child === null) added.append(span);
            else panel.replaceChild(span, child);
          } else if (child.textContent !== text) child.textContent = text;
          child = next;
        }
        while (child) { const next = child.nextSibling; child.remove(); child = next; }
        if (added.hasChildNodes()) panel.append(added);
      },
      text: (node: NodeId, branch: Branch | null): string => {
        return along(node).text + (branch === null ? "" : this.after(unfinished.at(node), branch).text);
      },
    };
  }

  /**
   * What ``branch`` reads as after bytes ending in ``before``, an unfinished
   * character or nothing, and what it leaves unfinished in turn: empty where
   * it finishes no character.
   */
  after(before: Uint8Array, branch: Branch): { text: string; unfinished: Uint8Array } {
    const own = this.bytes[branch];
    if (own === undefined) return { text: `#${branch}`, unfinished: new Uint8Array() };
    const bytes = concat(before, own);
    const text = this.decoder.decode(bytes, { stream: true });
    this.decoder.decode();
    return { text, unfinished: unfinished(bytes) };
  }

  /**
   * ``branch`` on its own, after ``before``: its bytes in hex where it finishes
   * no character, or holds only bytes that cannot be one.
   */
  shown(before: Uint8Array, branch: Branch): string {
    const { text } = this.after(before, branch);
    const own = this.bytes[branch];
    return own !== undefined && /^\uFFFD*$/.test(text) ? hex(own) : text;
  }

  /** Find an exact recorded text, then a completion, then its deepest known prefix. */
  search(paths: Iterable<readonly Branch[]>, text: string): Branch[] {
    if (text === "") return [];
    const query = new TextEncoder().encode(text);
    let prefix: { path: readonly Branch[]; depth: number } | undefined;
    let completion: { path: readonly Branch[]; length: number } | undefined;
    for (const path of paths) {
      const bytes = this.bytesAlong(path);
      if (bytes === null) continue;
      if (equal(bytes, query)) return [...path];
      if (startsWith(query, bytes) && path.length > (prefix?.depth ?? -1)) prefix = { path, depth: path.length };
      if (startsWith(bytes, query) && bytes.length < (completion?.length ?? Infinity)) {
        completion = { path, length: bytes.length };
      }
    }
    const match = completion?.path ?? prefix?.path;
    if (match === undefined || match.length === 0) throw new Error("No recorded sequence matches this text");
    return [...match];
  }

  private bytesAlong(path: readonly Branch[]): Uint8Array | null {
    let bytes: Uint8Array<ArrayBufferLike> = new Uint8Array();
    for (const branch of path) {
      const own = this.bytes[branch];
      if (own === undefined) return null;
      bytes = concat(bytes, own);
    }
    return bytes;
  }

  /** The text of each label along ``path``, each read after the ones before it. */
  along(path: readonly Branch[]): string[] {
    let before: Uint8Array = new Uint8Array();
    return path.map((branch) => {
      const read = this.after(before, branch);
      before = read.unfinished;
      return read.text;
    });
  }
}

function startsWith(bytes: Uint8Array, prefix: Uint8Array): boolean {
  return prefix.length <= bytes.length && prefix.every((byte, index) => bytes[index] === byte);
}

function equal(a: Uint8Array, b: Uint8Array): boolean {
  return a.length === b.length && startsWith(a, b);
}

/** Whitespace made visible, for a label shown on its own. */
function visible(text: string): string {
  return text.replace(/[ \n\r\t]/g, (c) => ({ " ": "␣", "\n": "↵", "\r": "↩", "\t": "⇥" })[c]!);
}

function token(text: string, kind?: string): HTMLElement {
  const span = document.createElement("span");
  span.className = kind === undefined ? "token" : `token ${kind}`;
  span.textContent = text;
  return span;
}
