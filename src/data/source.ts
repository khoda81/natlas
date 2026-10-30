// Integer-sequence sources: one per measure, each expanding complete label
// paths. A recorded snapshot answers from memory; a live server uses Connect.

import { fromBinary } from "@bufbuild/protobuf";
import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { decode, encode, type WireEdge } from "../core/codec";
import { Natlas, SnapshotSchema } from "../gen/natlas/v1/natlas_pb";
import { TextLabels } from "../readings/labels";
import { type NotesFile, NoteLabels } from "../readings/notes";
import type { Labels } from "../readings/reading";
import type { Label } from "../core/tree";

export interface Reply {
  /** A recording can only vouch for its own horizon; a live server honors the request. */
  complete: number;
  edges: AsyncIterable<WireEdge>;
}

export interface Source {
  expand(path: readonly Label[], horizon: number, signal?: AbortSignal): Promise<Reply>;
  /** Map text to a complete label path, when this source has a text mapping. */
  search?(text: string): Promise<Label[]>;
}

async function* replay(edges: WireEdge[]): AsyncIterable<WireEdge> {
  yield* edges;
}

interface Recorded {
  path: Label[];
  horizon: number;
  labels: Label[];
  lnWeights: number[];
}

/** A recorded server: one Expand per label path, re-encoded for each request. */
export class SnapshotSource implements Source {
  private readonly recorded = new Map<string, Recorded>();
  private readonly paths: Label[][] = [];

  constructor(
    bytes: Uint8Array,
    private readonly textLabels = TextLabels.none(),
  ) {
    const snapshot = fromBinary(SnapshotSchema, bytes);
    if (snapshot.formatVersion !== 2) {
      throw new Error("This snapshot uses the old handle format. Run uv run scripts/migrate-snapshots.py to convert recordings to integer paths.");
    }
    for (const { request, edges } of snapshot.expansions) {
      if (request === undefined) throw new Error("expansion without a request");
      const { labels, lnWeights } = decode(edges, request.horizon);
      this.recorded.set(key(request.path), { path: request.path, horizon: request.horizon, labels, lnWeights });
    }
    const paths = new Map<string, Label[]>();
    for (const { path, labels } of this.recorded.values()) {
      paths.set(key(path), path);
      for (const label of labels) {
        const child = [...path, label];
        paths.set(key(child), child);
      }
    }
    this.paths = [...paths.values()];
  }

  async expand(path: readonly Label[], horizon: number): Promise<Reply> {
    const recorded = this.recorded.get(key(path));
    if (recorded === undefined) return { complete: 0, edges: replay([]) };
    return {
      complete: Math.min(horizon, recorded.horizon),
      edges: replay(horizon === 0 ? [] : encode(recorded.labels, recorded.lnWeights, horizon)),
    };
  }

  async search(text: string): Promise<Label[]> {
    return this.textLabels.search(this.paths, text);
  }
}

function key(path: readonly Label[]): string {
  return path.join(",");
}

/** A live server: every expansion honors the requested horizon. */
export class ConnectSource implements Source {
  private readonly client;

  constructor(baseUrl: string) {
    this.client = createClient(Natlas, createConnectTransport({ baseUrl, useBinaryFormat: true }));
  }

  async expand(path: readonly Label[], horizon: number, signal?: AbortSignal): Promise<Reply> {
    const stream = this.client.expand({ path: [...path], horizon }, { signal });
    return { complete: horizon, edges: edgesOf(stream) };
  }

  async search(text: string): Promise<Label[]> {
    return (await this.client.search({ text })).path;
  }
}

async function* edgesOf(stream: AsyncIterable<WireEdge>): AsyncIterable<WireEdge> {
  const iterator = stream[Symbol.asyncIterator]();
  try {
    for (let next = await iterator.next(); !next.done; next = await iterator.next()) yield next.value;
  } finally {
    await iterator.return?.();
  }
}

export interface DatasetEntry {
  name: string;
  title: string;
  /** A JSON array of label bytes in base64, indexed by label, or a notes file. */
  labels?: string;
  /** Each measure is a recorded snapshot file or a live service's base URL. */
  measures: { name: string; file?: string; url?: string }[];
  /** What the entry's paths are relative to; the static data directory by default. */
  base?: string;
}

export interface Dataset {
  /** Includes the source URLs (or dropped file contents), so saved views cannot collide. */
  id: string;
  name: string;
  title: string;
  measures: string[];
  sources: Source[];
  labels: Labels;
  searchable: boolean;
  search(text: string, measure?: number): Promise<Label[]>;
}

/** The static data directory, beside the page. */
function dataDirectory(): string {
  return new URL("data/", location.href).href;
}

/** Datasets in the static data directory, then those a live server offers. */
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

function dataset(
  id: string,
  name: string,
  title: string,
  measures: string[],
  sources: Source[],
  labels: Labels,
): Dataset {
  const searchable = labels instanceof TextLabels && labels.searchable;
  return {
    id,
    name,
    title,
    measures,
    sources,
    labels,
    searchable,
    async search(text, measure = 0) {
      const source = sources[measure];
      const search = source?.search;
      if (search === undefined) throw new Error("this dataset has no text search");
      return search.call(source, text);
    },
  };
}

export async function fetchDataset(entry: DatasetEntry): Promise<Dataset> {
  const base = entry.base ?? dataDirectory();
  const labels =
    entry.labels === undefined
      ? TextLabels.none()
      : parseLabels(await (await fetchOk(new URL(entry.labels, base))).text(), new URL(entry.labels, base));
  const sources = await Promise.all(
    entry.measures.map(async ({ file, url }): Promise<Source> => {
      if (url !== undefined) return new ConnectSource(new URL(url, base).href);
      if (file === undefined) throw new Error("a measure has neither a file nor a url");
      const bytes = new Uint8Array(await (await fetchOk(new URL(file, base))).arrayBuffer());
      return new SnapshotSource(bytes, labels instanceof TextLabels ? labels : TextLabels.none());
    }),
  );
  const id = JSON.stringify([entry.name, ...entry.measures.map(({ file, url }) => new URL(url ?? file!, base).href)]);
  return dataset(id, entry.name, entry.title, entry.measures.map(({ name }) => name), sources, labels);
}

/** Dropped files: each .pb is one measure, a .json holds the labels. */
export async function droppedDataset(files: File[]): Promise<Dataset> {
  const snapshots = files.filter((file) => file.name.endsWith(".pb"));
  if (snapshots.length === 0) throw new Error("no .pb snapshot among the dropped files");
  const labelFile = files.find((file) => file.name.endsWith(".json"));
  const labels = labelFile === undefined ? TextLabels.none() : parseLabels(await labelFile.text(), null);
  const sources = await Promise.all(
    snapshots.map(async (file) => {
      const bytes = new Uint8Array(await file.arrayBuffer());
      return new SnapshotSource(bytes, labels instanceof TextLabels ? labels : TextLabels.none());
    }),
  );
  const name = snapshots.map((file) => file.name).join(" · ");
  const hashes = await Promise.all(files.map(async (file) => {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }));
  return dataset(JSON.stringify([name, ...hashes]), name, name, snapshots.map((file) => file.name.replace(/\.pb$/, "")), sources, labels);
}

/** A labels file, read by its kind; ``url`` resolves its own references. */
export function parseLabels(json: string, url: URL | null): Labels {
  const parsed = JSON.parse(json) as string[] | { kind?: string };
  if (Array.isArray(parsed)) return TextLabels.fromJson(parsed);
  if (parsed.kind === "notes") return new NoteLabels(parsed as NotesFile, url);
  throw new Error(`labels of an unknown kind: ${parsed.kind}`);
}
