// Label texts. A language model's labels are tokens, and a token can hold
// part of a character (an emoji, or a Persian letter, split over byte-level
// tokens), so a label is bytes and its text depends on the bytes before it:
// it shows the characters its own bytes complete. A labels file is a JSON
// array of base64 strings, indexed by label, as proto3 JSON writes bytes.

import type { Label } from "./tree";

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

export class Labels {
  private readonly decoder = new TextDecoder();

  constructor(private readonly bytes: readonly Uint8Array[]) {}

  static parse(json: string): Labels {
    const encoded = JSON.parse(json) as string[];
    return new Labels(encoded.map((text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0))));
  }

  static none(): Labels {
    return new Labels([]);
  }

  /**
   * What ``label`` reads as after bytes ending in ``before``, an unfinished
   * character or nothing, and what it leaves unfinished in turn: empty where
   * it finishes no character.
   */
  after(before: Uint8Array, label: Label): { text: string; unfinished: Uint8Array } {
    const own = this.bytes[label];
    if (own === undefined) return { text: `#${label}`, unfinished: new Uint8Array() };
    const bytes = concat(before, own);
    const text = this.decoder.decode(bytes, { stream: true });
    this.decoder.decode();
    return { text, unfinished: unfinished(bytes) };
  }

  /**
   * ``label`` on its own, after ``before``: its bytes in hex where it finishes
   * no character, or holds only bytes that cannot be one.
   */
  shown(before: Uint8Array, label: Label): string {
    const { text } = this.after(before, label);
    const own = this.bytes[label];
    return own !== undefined && /^\uFFFD*$/.test(text) ? hex(own) : text;
  }

  /** The text of each label along ``path``, each read after the ones before it. */
  along(path: readonly Label[]): string[] {
    let before: Uint8Array = new Uint8Array();
    return path.map((label) => {
      const read = this.after(before, label);
      before = read.unfinished;
      return read.text;
    });
  }
}
