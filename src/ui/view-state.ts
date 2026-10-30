/** A portable camera position: the origin and the covering node as branches
 * from the root, the covering node's position in view heights, and x from
 * the origin. */
export interface SavedView {
  origin: number[];
  /** Extends `origin`. */
  path: number[];
  x: number;
  k: number;
  a: number;
  p: number;
  q: number;
}

const isPath = (path: unknown): path is number[] =>
  Array.isArray(path) && path.every((branch) => Number.isInteger(branch) && branch >= 0 && branch <= 0xffffffff);

export function parseView(raw: string, measures: number): SavedView | null {
  try {
    const value = JSON.parse(raw) as Partial<SavedView> | null;
    if (
      value === null ||
      !isPath(value.origin) || !isPath(value.path) ||
      value.origin.length > value.path.length || !value.origin.every((branch, i) => branch === value.path![i]) ||
      typeof value.x !== "number" || !Number.isFinite(value.x) || value.x < 0 ||
      typeof value.k !== "number" || !Number.isFinite(value.k) || value.k <= 0 ||
      typeof value.a !== "number" || !Number.isFinite(value.a) ||
      typeof value.p !== "number" || !Number.isInteger(value.p) || value.p < 0 || value.p >= measures ||
      typeof value.q !== "number" || !Number.isInteger(value.q) || value.q < 0 || value.q >= measures
    ) return null;
    return value as SavedView;
  } catch {
    return null;
  }
}

export function viewKey(dataset: string): string {
  return `natlas.view:3:${encodeURIComponent(dataset)}`;
}
