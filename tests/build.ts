// Build trees from protocol messages, as a server would stream them.

import { create } from "@bufbuild/protobuf";
import { type ServerMessage, ServerMessageSchema } from "../src/gen/natlas/v1/natlas_pb";
import { childHandle, type Handle, IntervalTree, ROOT } from "../src/tree";

/** Fine enough that rounding stays below test tolerances. */
export const LOG_SCALE = 1 << 16;

export function hello(measures: string[], logScale = LOG_SCALE): ServerMessage {
  return create(ServerMessageSchema, { kind: { case: "hello", value: { title: "test", measures, logScale } } });
}

export interface Children {
  labels: string[];
  /** One weight vector per measure, from rank 0, unnormalized. */
  weights: number[][];
  /** Per measure, the weight of every child after ``end``; omitted when there are none. */
  rest?: number[];
}

/** A run of ranks [start, end) of ``children``, which lists ranks [0, end). */
export function run(node: Handle, children: Children, start = 0, end = children.labels.length): ServerMessage {
  const units = (first: number, weight: number) => Math.round(Math.log(first / weight) * LOG_SCALE);
  const levels = children.weights.map((level) => level.slice(0, end).map((weight) => units(level[0]!, weight)));
  const ranks = Array.from({ length: Math.max(0, end - Math.max(start, 1)) }, (_, i) => Math.max(start, 1) + i);
  return create(ServerMessageSchema, {
    kind: {
      case: "run",
      value: {
        node,
        start,
        labels: children.labels.slice(start, end),
        gaps: ranks.map((rank) => levels[0]![rank]! - levels[0]![rank - 1]!),
        others: levels.slice(1).map((level) => ({ values: ranks.map((rank) => level[rank]!) })),
        rest: children.rest && { values: children.rest.map((weight, m) => units(children.weights[m]![0]!, weight)) },
      },
    },
  });
}

/** A one-measure tree whose nodes all split their interval with ``weights``. */
export function chainTree(weights: number[], path: number[]): IntervalTree {
  const tree = new IntervalTree();
  const children = { labels: weights.map((_, i) => `s${i}`), weights: [weights] };
  const messages = [hello(["p"]), run(ROOT, children)];
  let handle = ROOT;
  for (const slot of path) {
    handle = childHandle(handle, slot);
    messages.push(run(handle, children));
  }
  tree.applyAll(messages);
  return tree;
}
