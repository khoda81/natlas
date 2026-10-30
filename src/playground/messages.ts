import type { Budgets, Exchange, Group, Meter, statistics } from "./simulation";
import type { Capture, reference } from "./reference";

export interface Run {
  type: "run"; id: number; budget: Budgets; pIndex: number; qIndex: number; start: number; end: number;
}
export interface Snapshot {
  type: "result"; id: number; budget: Budgets; pIndex: number; qIndex: number; start: number; end: number;
  p: Meter; q: Meter; groups: Group[]; stats: ReturnType<typeof statistics>;
  lower: Float64Array; upper: Float64Array; reference: ReturnType<typeof reference>;
  milliseconds: number; exchanges: ({ side: "P" | "Q" } & Exchange)[];
}
export type ToWorker = { type: "load"; base: string } | Run;
export type FromWorker = { type: "ready"; capture: Capture; bootstrapBytes: number } | Snapshot |
  { type: "progress"; id: number; p: number; q: number } | { type: "error"; message: string; id?: number };
