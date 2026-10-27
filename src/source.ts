// Tree sources. A recorded snapshot replays a server stream; a live server
// will deliver the same ServerMessages over a socket.

import { fromBinary } from "@bufbuild/protobuf";
import { SnapshotSchema } from "./gen/natlas/v1/natlas_pb";
import { IntervalTree } from "./tree";

export interface DatasetEntry {
  file: string;
  title: string;
}

export async function manifest(): Promise<DatasetEntry[]> {
  const response = await fetch("data/manifest.json");
  if (!response.ok) return [];
  return (await response.json()) as DatasetEntry[];
}

export function fromSnapshot(bytes: Uint8Array): IntervalTree {
  const tree = new IntervalTree();
  tree.applyAll(fromBinary(SnapshotSchema, bytes).messages);
  return tree;
}

export async function fetchSnapshot(file: string): Promise<IntervalTree> {
  const response = await fetch(`data/${file}`);
  if (!response.ok) throw new Error(`could not load ${file}: ${response.status}`);
  return fromSnapshot(new Uint8Array(await response.arrayBuffer()));
}
