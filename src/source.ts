// Tree sources: one per measure, each answering Expand. A recorded snapshot
// answers from memory; a live server will answer the same calls over Connect.

import { fromBinary } from "@bufbuild/protobuf";
import { decode, encode, type WireEdge } from "./codec";
import { SnapshotSchema } from "./gen/natlas/v1/natlas_pb";
import type { Handle } from "./tree";

export interface Reply {
  /**
   * Every child left out weighs less than 2^-complete. A live server honors
   * the requested horizon; a recording can only vouch for its own.
   */
  complete: number;
  edges: AsyncIterable<WireEdge>;
}

export interface Source {
  expand(node: Handle, horizon: number): Reply;
}

async function* replay(edges: WireEdge[]): AsyncIterable<WireEdge> {
  yield* edges;
}

/** A recorded server: one Expand per node, re-encoded for each request. */
export class SnapshotSource implements Source {
  private readonly recorded = new Map<Handle, { horizon: number; labels: number[]; lnWeights: number[] }>();

  constructor(bytes: Uint8Array) {
    for (const { request, edges } of fromBinary(SnapshotSchema, bytes).expansions) {
      if (request === undefined) throw new Error("expansion without a request");
      const { labels, lnWeights } = decode(edges, request.horizon);
      this.recorded.set(request.node, { horizon: request.horizon, labels, lnWeights });
    }
  }

  expand(node: Handle, horizon: number): Reply {
    const recorded = this.recorded.get(node);
    if (recorded === undefined) return { complete: 0, edges: replay([]) };
    return {
      complete: Math.min(horizon, recorded.horizon),
      edges: replay(encode(recorded.labels, recorded.lnWeights, horizon)),
    };
  }
}

export interface DatasetEntry {
  name: string;
  title: string;
  /** JSON array of label texts, indexed by label. */
  labels?: string;
  measures: { name: string; file: string }[];
}

export interface Dataset {
  title: string;
  measures: string[];
  sources: Source[];
  labels: string[];
}

export async function manifest(): Promise<DatasetEntry[]> {
  const response = await fetch("data/manifest.json");
  if (!response.ok) return [];
  return (await response.json()) as DatasetEntry[];
}

async function fetchOk(file: string): Promise<Response> {
  const response = await fetch(`data/${file}`);
  if (!response.ok) throw new Error(`could not load ${file}: ${response.status}`);
  return response;
}

export async function fetchDataset(entry: DatasetEntry): Promise<Dataset> {
  const labels = entry.labels === undefined ? [] : fetchOk(entry.labels).then((r) => r.json() as Promise<string[]>);
  const sources = Promise.all(
    entry.measures.map(async ({ file }) => new SnapshotSource(new Uint8Array(await (await fetchOk(file)).arrayBuffer()))),
  );
  return { title: entry.title, measures: entry.measures.map(({ name }) => name), sources: await sources, labels: await labels };
}

/** Dropped files: each .pb is one measure, a .json holds the labels. */
export async function droppedDataset(files: File[]): Promise<Dataset> {
  const snapshots = files.filter((file) => file.name.endsWith(".pb"));
  if (snapshots.length === 0) throw new Error("no .pb snapshot among the dropped files");
  const labelFile = files.find((file) => file.name.endsWith(".json"));
  return {
    title: snapshots.map((file) => file.name).join(" · "),
    measures: snapshots.map((file) => file.name.replace(/\.pb$/, "")),
    sources: await Promise.all(snapshots.map(async (file) => new SnapshotSource(new Uint8Array(await file.arrayBuffer())))),
    labels: labelFile === undefined ? [] : (JSON.parse(await labelFile.text()) as string[]),
  };
}
