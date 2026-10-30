/** Full-data reference for validation and the separately labelled overlay.
 * Neither the scheduler nor the bound calculation imports this module. */
export function probabilities(logs: Float64Array): Float64Array {
  let maximum = -Infinity;
  for (const x of logs) maximum = Math.max(maximum, x);
  const values = logs.map((x) => Math.exp(x - maximum));
  let total = 0;
  for (const x of values) total += x;
  return values.map((x) => x / total);
}

export function reference(pLogs: Float64Array, qLogs: Float64Array, rows: number) {
  const p = probabilities(pLogs), q = probabilities(qLogs);
  const ranks = Uint32Array.from({ length: p.length }, (_, i) => i);
  ranks.sort((a, b) => q[b]! - q[a]! || a - b);
  let entropy = 0, cross = 0;
  for (let i = 0; i < p.length; i++) if (p[i]! > 0) {
    entropy -= p[i]! * Math.log(p[i]!);
    cross -= p[i]! * Math.log(q[i]!);
  }
  const widths = new Float64Array(rows), ids = new Uint32Array(rows);
  const pAt = new Float64Array(rows), qAt = new Float64Array(rows);
  let rank = 0, cumulative = p[ranks[0]!]!;
  for (let row = 0; row < rows; row++) {
    while (rank + 1 < ranks.length && cumulative < (row + 0.5) / rows) cumulative += p[ranks[++rank]!]!;
    const id = ranks[rank]!;
    widths[row] = -Math.log(q[id]!); ids[row] = id; pAt[row] = p[id]!; qAt[row] = q[id]!;
  }
  return { widths, ids, pAt, qAt, entropy, cross, kl: cross - entropy };
}

export interface Capture {
  title: string; prompt: string; tokens: number[]; vocabulary: number; vocabulary_sha256: string;
  labels: string; measures: { name: string; file: string; sha256: string; model_sha256: string }[];
  source: string; source_revision: string; note: string; format: string;
}
