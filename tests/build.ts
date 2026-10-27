// Build trees from protocol messages, as a server would stream them.

import { create } from "@bufbuild/protobuf";
import { type ServerMessage, ServerMessageSchema } from "../src/gen/natlas/v1/natlas_pb";
import { IntervalTree } from "../src/tree";

export interface Slots {
  labels: string[];
  /** One weight vector per measure. */
  weights: number[][];
  tail?: number[];
}

export function hello(measures: string[]): ServerMessage {
  return create(ServerMessageSchema, { kind: { case: "hello", value: { title: "test", measures } } });
}

export function node(id: number, parent: number, slot: number, size: number, slots: Slots): ServerMessage {
  return create(ServerMessageSchema, {
    kind: {
      case: "node",
      value: {
        id: BigInt(id),
        parent: BigInt(parent),
        slot,
        size,
        page: {
          start: 0,
          labels: slots.labels,
          weights: slots.weights.map((values) => ({ values })),
          tail: slots.tail ?? slots.weights.map(() => 0),
        },
      },
    },
  });
}

export function page(id: number, start: number, slots: Slots): ServerMessage {
  return create(ServerMessageSchema, {
    kind: {
      case: "page",
      value: {
        node: BigInt(id),
        start,
        labels: slots.labels,
        weights: slots.weights.map((values) => ({ values })),
        tail: slots.tail ?? slots.weights.map(() => 0),
      },
    },
  });
}

/** A one-measure tree whose nodes all split their interval with ``weights``. */
export function chainTree(weights: number[], path: number[]): IntervalTree {
  const tree = new IntervalTree();
  const slots = { labels: weights.map((_, i) => `s${i}`), weights: [weights] };
  const messages = [hello(["p"]), node(0, 0, 0, weights.length, slots)];
  path.forEach((slot, depth) => messages.push(node(depth + 1, depth, slot, weights.length, slots)));
  tree.applyAll(messages);
  return tree;
}
