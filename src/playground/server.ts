import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { SummarizeRequestSchema, SummarizeResponseSchema, SummarySchema, type Summary } from "../gen/natlas/partition/v1/partition_pb";

export interface Filter { mask: number; value: number }
export const key = (f: Filter) => `${f.mask >>> 0}:${f.value >>> 0}`;
export const bits = (mask: number) => {
  const result: number[] = [];
  for (let i = 0; i < 32; i++) if ((mask >>> i) & 1) result.push(i);
  return result;
};

/** Conservative positive-sum error, including division rounding. The measure
 * is the mathematical normalization of the represented binary64 weights.
 * This concerns numerical representation, not model calibration. */
function sumBounds(sum: number, count: number): [number, number] {
  const e = (count + 4) * Number.EPSILON;
  return [Math.max(0, sum * (1 - e)), sum / (1 - e)];
}
const downLog = (p: number) => p === 0 ? -Infinity : Math.log(p) - 8 * Number.EPSILON * (Math.abs(Math.log(p)) + 1);
const upLog = (p: number) => p === 0 ? -Infinity : Math.min(0, Math.log(p) + 8 * Number.EPSILON * (Math.abs(Math.log(p)) + 1));

interface Cached { ids: Uint32Array; summary: Summary }

/** Mock endpoint. Its private full vector is never available to the replica.
 * Every request and reply crosses the actual generated protobuf codec. */
export class PartitionServer {
  readonly size: number;
  private readonly weights: Float64Array;
  private readonly total: [number, number];
  private readonly cache = new Map<string, Cached>();

  constructor(logWeights: Float64Array) {
    if (!logWeights.length || logWeights.length > 0xffffffff) throw Error("invalid vocabulary size");
    let maximum = -Infinity;
    for (const log of logWeights) {
      if (Number.isNaN(log) || log === Infinity) throw Error("invalid log weight");
      maximum = Math.max(maximum, log);
    }
    if (maximum === -Infinity) throw Error("zero total mass");
    this.size = logWeights.length;
    // The playground captures are far from exp underflow. Fail explicitly for
    // a different fixture rather than turn tiny positive probabilities into 0.
    this.weights = logWeights.map((log) => {
      const w = Math.exp(log - maximum);
      if (w === 0 && log !== -Infinity) throw Error("fixture exceeds binary64 weight range");
      return w;
    });
    let sum = 0;
    for (const weight of this.weights) sum += weight;
    this.total = sumBounds(sum, this.size);
    this.store({ mask: 0, value: 0 }, Uint32Array.from({ length: this.size }, (_, i) => i));
  }

  private store(filter: Filter, ids: Uint32Array): Cached {
    let sum = 0, min = Infinity, max = 0;
    for (const id of ids) {
      const w = this.weights[id]!;
      sum += w;
      min = Math.min(min, w);
      max = Math.max(max, w);
    }
    const [lo, hi] = sumBounds(sum, ids.length);
    const root = filter.mask === 0;
    const summary = create(SummarySchema, {
      ...filter, count: ids.length,
      massLower: root ? 1 : Math.max(0, lo / this.total[1] * (1 - Number.EPSILON)),
      massUpper: root ? 1 : Math.min(1, hi / this.total[0] * (1 + Number.EPSILON)),
      logMinProbability: downLog(ids.length ? min / this.total[1] : 0),
      logMaxProbability: upLog(max / this.total[0]),
    });
    const cached = { ids, summary };
    this.cache.set(key(filter), cached);
    return cached;
  }

  exchange(bytes: Uint8Array): Uint8Array {
    const request = fromBinary(SummarizeRequestSchema, bytes);
    const { mask, value, splitMask, extrema } = request;
    if ((value & ~mask) !== 0 || (mask & splitMask) !== 0 || bits(splitMask).length > 4)
      throw Error("invalid partition predicate");
    let parent = this.cache.get(key(request));
    if (!parent) {
      const ids: number[] = [];
      for (let id = 0; id < this.size; id++) if (((id & mask) >>> 0) === value) ids.push(id);
      parent = this.store(request, Uint32Array.from(ids));
    }
    const children: Summary[] = [];
    if (splitMask && parent.ids.length) {
      const buckets = new Map<number, number[]>();
      for (const id of parent.ids) {
        const childValue = ((id & splitMask) | value) >>> 0;
        let bucket = buckets.get(childValue);
        if (!bucket) buckets.set(childValue, bucket = []);
        bucket.push(id);
      }
      for (const [childValue, ids] of [...buckets].sort(([a], [b]) => a - b)) {
        const filter = { mask: (mask | splitMask) >>> 0, value: childValue };
        const child = this.cache.get(key(filter)) ?? this.store(filter, Uint32Array.from(ids));
        children.push(child.summary);
      }
    }
    const fields = (s: Summary) => extrema ? s : create(SummarySchema, {
      mask: s.mask, value: s.value, count: s.count, massLower: s.massLower, massUpper: s.massUpper,
    });
    return toBinary(SummarizeResponseSchema, create(SummarizeResponseSchema, {
      parent: fields(parent.summary), children: children.map(fields),
    }));
  }
}
