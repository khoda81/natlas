// Tree sources: one per measure, each answering Expand. A recorded snapshot
// answers from memory; a live server answers over Connect.

import { fromBinary } from "@bufbuild/protobuf";
import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { decode, encode, type WireEdge } from "./codec";
import { type ExpandResponse, Natlas, SnapshotSchema } from "./gen/natlas/v1/natlas_pb";
import { TextLabels } from "./labels";
import { type NotesFile, NoteLabels } from "./notes";
import type { Labels } from "./reading";
import type { Handle, Label } from "./tree";

/** A node as a request names it: the child with ``label`` of the node the server gave ``parent`` to. */
export interface Child {
  parent: Handle;
  label: Label;
}

export interface Reply {
  /** The node's handle, to name its children by. */
  node: Handle;
  /**
   * Every child left out weighs less than 2^-complete. A live server honors
   * the requested horizon; a recording can only vouch for its own.
   */
  complete: number;
  edges: AsyncIterable<WireEdge>;
}

export interface Source {
  /** Expand ``node``, or the root when undefined. */
  expand(node: Child | undefined, horizon: number): Promise<Reply>;
}

async function* replay(edges: WireEdge[]): AsyncIterable<WireEdge> {
  yield* edges;
}

/** A recorded server: one Expand per node, re-encoded for each request. */
export class SnapshotSource implements Source {
  private readonly recorded = new Map<string, { node: Handle; horizon: number; labels: number[]; lnWeights: number[] }>();
  /** Given to every node not recorded: no recorded node has it, so nothing is recorded under it either. */
  private readonly unrecorded: Handle;

  constructor(bytes: Uint8Array) {
    const handles = new Set<Handle>();
    for (const { request, node, edges } of fromBinary(SnapshotSchema, bytes).expansions) {
      if (request === undefined) throw new Error("expansion without a request");
      const { labels, lnWeights } = decode(edges, request.horizon);
      this.recorded.set(key(request.node), { node, horizon: request.horizon, labels, lnWeights });
      handles.add(node);
      if (request.node !== undefined) handles.add(request.node.parent);
    }
    let unrecorded = 0n;
    while (handles.has(unrecorded)) unrecorded++;
    this.unrecorded = unrecorded;
  }

  async expand(node: Child | undefined, horizon: number): Promise<Reply> {
    const recorded = this.recorded.get(key(node));
    if (recorded === undefined) return { node: this.unrecorded, complete: 0, edges: replay([]) };
    return {
      node: recorded.node,
      complete: Math.min(horizon, recorded.horizon),
      edges: replay(horizon === 0 ? [] : encode(recorded.labels, recorded.lnWeights, horizon)),
    };
  }
}

function key(node: Child | undefined): string {
  return node === undefined ? "root" : `${node.parent}:${node.label}`;
}

/** A live server: every response honors the requested horizon. */
export class ConnectSource implements Source {
  private readonly client;

  constructor(baseUrl: string) {
    this.client = createClient(Natlas, createConnectTransport({ baseUrl, useBinaryFormat: true }));
  }

  async expand(node: Child | undefined, horizon: number): Promise<Reply> {
    const stream = this.client.expand({ node, horizon })[Symbol.asyncIterator]();
    const first = await stream.next();
    if (first.done || first.value.part.case !== "node") throw new Error("an expansion must begin with its node's handle");
    return { node: first.value.part.value, complete: horizon, edges: edgesOf(stream) };
  }
}

/** The rest of an expansion; stopping early cancels the call. */
async function* edgesOf(stream: AsyncIterator<ExpandResponse>): AsyncIterable<WireEdge> {
  try {
    for (let next = await stream.next(); !next.done; next = await stream.next()) {
      if (next.value.part.case !== "edge") throw new Error("a handle after the start of an expansion");
      yield next.value.part.value;
    }
  } finally {
    await stream.return?.();
  }
}

export interface DatasetEntry {
  name: string;
  title: string;
  /** A labels file: a JSON array of label bytes in base64, indexed by label, or a notes file. */
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
  labels: Labels;
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
    entry.labels === undefined
      ? TextLabels.none()
      : (async () => {
          const url = new URL(entry.labels!, base);
          return parseLabels(await (await fetchOk(url)).text(), url);
        })();
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
    labels: labelFile === undefined ? TextLabels.none() : parseLabels(await labelFile.text(), null),
    start: [],
  };
}

/** A labels file, read by its kind; ``url`` is where it came from, which its own references are relative to. */
export function parseLabels(json: string, url: URL | null): Labels {
  const parsed = JSON.parse(json) as string[] | { kind?: string };
  if (Array.isArray(parsed)) return TextLabels.fromJson(parsed);
  if (parsed.kind === "notes") return new NoteLabels(parsed as NotesFile, url);
  throw new Error(`labels of an unknown kind: ${parsed.kind}`);
}
