// Tree sources: one per measure, each answering Expand. A recorded snapshot
// answers from memory; a live server answers over Connect.

import { fromBinary } from "@bufbuild/protobuf";
import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { decode, encode, type WireEdge } from "./codec";
import { Natlas, SnapshotSchema } from "./gen/natlas/v1/natlas_pb";
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

/** A live server: every response honors the requested horizon. */
export class ConnectSource implements Source {
  private readonly client;

  constructor(baseUrl: string) {
    this.client = createClient(Natlas, createConnectTransport({ baseUrl, useBinaryFormat: true }));
  }

  expand(node: Handle, horizon: number): Reply {
    return { complete: horizon, edges: this.client.expand({ node, horizon }) };
  }
}

export interface DatasetEntry {
  name: string;
  title: string;
  /** JSON array of label texts, indexed by label. */
  labels?: string;
  /** Each measure is a recorded snapshot file or a live service's base URL. */
  measures: { name: string; file?: string; url?: string }[];
  /** What the entry's paths are relative to; the static data directory by default. */
  base?: string;
  /** Labels from the root to the node the client opens at. */
  start?: number[];
}

export interface Dataset {
  title: string;
  measures: string[];
  sources: Source[];
  labels: string[];
  start: number[];
}

/** The static data directory, beside the page. */
function dataDirectory(): string {
  return new URL("data/", location.href).href;
}

/** Datasets in the static data directory, then those a live server at ``server`` offers. */
export async function manifest(server?: string): Promise<DatasetEntry[]> {
  const listed = async (base: string) => {
    try {
      const response = await fetch(new URL("manifest.json", base));
      if (!response.ok) return [];
      return ((await response.json()) as DatasetEntry[]).map((entry) => ({ ...entry, base }));
    } catch {
      return [];
    }
  };
  const [live, recorded] = await Promise.all([server === undefined ? [] : listed(server), listed(dataDirectory())]);
  return [...live, ...recorded];
}

async function fetchOk(url: URL): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: ${response.status}`);
  return response;
}

export async function fetchDataset(entry: DatasetEntry): Promise<Dataset> {
  const base = entry.base ?? dataDirectory();
  const labels =
    entry.labels === undefined ? [] : fetchOk(new URL(entry.labels, base)).then((r) => r.json() as Promise<string[]>);
  const sources = Promise.all(
    entry.measures.map(async ({ name, file, url }): Promise<Source> => {
      if (url !== undefined) return new ConnectSource(new URL(url, base).href);
      if (file === undefined) throw new Error(`measure ${name} has neither a file nor a url`);
      return new SnapshotSource(new Uint8Array(await (await fetchOk(new URL(file, base))).arrayBuffer()));
    }),
  );
  return {
    title: entry.title,
    measures: entry.measures.map(({ name }) => name),
    sources: await sources,
    labels: await labels,
    start: entry.start ?? [],
  };
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
    start: [],
  };
}
