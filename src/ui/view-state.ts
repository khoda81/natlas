/** A portable camera position, expressed relative to a complete integer path. */
export interface SavedView {
  path: number[];
  x: number;
  k: number;
  a: number;
  p: number;
  q: number;
}

export function parseView(raw: string, measures: number): SavedView | null {
  try {
    const value = JSON.parse(raw) as Partial<SavedView> | null;
    if (
      value === null ||
      !Array.isArray(value.path) ||
      !value.path.every((label) => Number.isInteger(label) && label >= 0 && label <= 0xffffffff) ||
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
  return `natlas.view:2:${encodeURIComponent(dataset)}`;
}
