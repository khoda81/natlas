import type { Anchor } from "../core/layout";

/** A portable camera: the origin as branches from the root, the point below
 * it, where the point is in the view, and x from the origin. */
export interface SavedView {
  origin: number[];
  anchor: Anchor;
  x: number;
  k: number;
  a: number;
  /** The measures in view, by spec. */
  p: string;
  q: string;
}

const isPath = (path: unknown): path is number[] =>
  Array.isArray(path) && path.every((branch) => Number.isInteger(branch) && 0 <= branch && branch <= 0xffffffff);

/** The anchor's digits are written in hex, two characters each. */
export function serializeView({ anchor, ...view }: SavedView): string {
  const digits = Array.from(anchor.digits, (digit) => digit.toString(16).padStart(2, "0")).join("");
  return JSON.stringify({ ...view, path: anchor.path, digits });
}

export function parseView(raw: string): SavedView | null {
  try {
    const value = JSON.parse(raw) as Partial<Omit<SavedView, "anchor"> & { path: number[]; digits: string }> | null;
    const { origin, path, digits, x, k, a, p, q } = value ?? {};
    const isMeasure = (measure: unknown): measure is string => typeof measure === "string";
    if (
      !isPath(origin) || !isPath(path) ||
      typeof digits !== "string" || !/^([0-9a-f]{2})*$/.test(digits) ||
      // Only the root's left edge stops a view: past any other origin lies its parent.
      typeof x !== "number" || !Number.isFinite(x) || (origin.length === 0 && x < 0) ||
      typeof k !== "number" || !Number.isFinite(k) || k <= 0 ||
      typeof a !== "number" || !Number.isFinite(a) ||
      !isMeasure(p) || !isMeasure(q)
    ) return null;
    const bytes = Uint8Array.from(digits.match(/../g) ?? [], (digit) => parseInt(digit, 16));
    return { origin, anchor: { path, digits: bytes }, x, k, a, p, q };
  } catch {
    return null;
  }
}

export function viewKey(dataset: string): string {
  return `natlas.view:5:${encodeURIComponent(dataset)}`;
}
