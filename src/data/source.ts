// Integer-sequence sources: one per measure. Nodes are named by their shared
// hashed ids. A recorded snapshot answers from memory; a live server uses Connect.

import { fromBinary } from "@bufbuild/protobuf";
import { Code, ConnectError, createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { Natlas, SnapshotSchema } from "../gen/natlas/natlas_pb";
import { TextLabels } from "../readings/labels";
import { type NotesFile, NoteLabels } from "../readings/notes";
import type { Labels } from "../readings/reading";
import { type Branch, IntervalTree, type NodeId, ROOT, type Run } from "../core/tree";

/** A part of a node, as ln of the mass below each of its edges. */
export interface Window {
  lnTop: number;
  lnBottom: number;
}

/** The whole of a node, as a window. */
export const WHOLE: Window = { lnTop: 0, lnBottom: -Infinity };

/** The protocol's Expand, reading below ``node`` down a path or down a
 * window, listing down to ``horizon`` bits. */
export type ExpandRequest = { node: NodeId; horizon: number } & ({ path: Branch[] } | { window: Window });

/** One message of an answer: a node's run. */
export interface Step {
  node: NodeId;
  run: Run;
}

export interface Source {
  /** The protocol's Expand, each message's node named in ``tree``. */
  expand(tree: IntervalTree, request: ExpandRequest, signal?: AbortSignal): AsyncIterable<Step>;
  /** Map text to branches from the root, when this source has a text mapping. */
  search?(text: string, signal?: AbortSignal): Promise<Branch[]>;
}

/** Name each message's node in ``tree``: the start, then the child along the
 * path's next branch, or the single child the window descends into. */
async function* named(tree: IntervalTree, request: ExpandRequest, runs: AsyncIterable<Run> | Iterable<Run>): AsyncIterable<Step> {
  const path = "path" in request ? request.path : [];
  let current = request.node;
  let next: Branch | undefined;
  let depth = 0;
  for await (const run of runs) {
    if (depth > 0) {
      if (next === undefined) throw new Error("the source went on after listing its window");
      current = tree.child(current, next);
    }
    const branch = path[depth];
    if (branch !== undefined && (run.branches.length !== 1 || run.branches[0] !== branch)) {
      throw new Error("the source placed another branch than the path's");
    }
    yield { node: current, run };
    next = branch ?? (run.branches.length === 1 ? run.branches[0] : undefined);
    depth++;
  }
}

/** A recorded server. Its own tree holds the recording and answers as a server would. */
export class SnapshotSource implements Source {
  private readonly recorded = new IntervalTree(1);
  private readonly paths: Branch[][] = [];

  constructor(
    bytes: Uint8Array,
    private readonly textLabels = TextLabels.none(),
  ) {
    const snapshot = fromBinary(SnapshotSchema, bytes);
    if (snapshot.recordings.length === 0) {
      throw new Error("an empty recording, or one in an older format: run scripts/migrate-snapshots.py");
    }
    const paths = new Map<string, Branch[]>();
    for (const { path, run } of snapshot.recordings) {
      if (run === undefined) throw new Error("a recorded node has no run");
      this.recorded.insert(this.recorded.descend(ROOT, path), 0, run);
      paths.set(path.join(","), path);
      for (const branch of run.branches) {
        const child = [...path, branch];
        paths.set(child.join(","), child);
      }
    }
    this.paths = [...paths.values()];
  }

  async *expand(tree: IntervalTree, request: ExpandRequest): AsyncIterable<Step> {
    const { node, horizon } = request;
    const runs = ("path" in request
      ? this.recorded.raw.answerPath(node, 0, Uint32Array.from(request.path), horizon)
      : this.recorded.raw.answerWindow(node, 0, request.window.lnTop, request.window.lnBottom, horizon)) as Run[];
    yield* named(tree, request, runs);
  }

  async search(text: string): Promise<Branch[]> {
    return this.textLabels.search(this.paths, text);
  }
}

/** A live server. */
export class ConnectSource implements Source {
  private readonly client;

  constructor(baseUrl: string) {
    this.client = createClient(Natlas, createConnectTransport({ baseUrl, useBinaryFormat: true }));
  }

  /** Name the request from its node's id. Where the server has not learned
   * that id (it restarted, or never saw it), a path from the root names it,
   * and a window is asked again once the path has. */
  async *expand(tree: IntervalTree, request: ExpandRequest, signal?: AbortSignal): AsyncIterable<Step> {
    let answered = false;
    try {
      for await (const step of this.ask(tree, request, signal)) {
        answered = true;
        yield step;
      }
    } catch (error) {
      if (answered || request.node === ROOT || ConnectError.from(error).code !== Code.NotFound) throw error;
      const path = tree.path(request.node);
      if ("path" in request) {
        yield* this.ask(tree, { node: ROOT, path: [...path, ...request.path], horizon: request.horizon }, signal);
      } else {
        yield* this.ask(tree, { node: ROOT, path, horizon: request.horizon }, signal);
        yield* this.ask(tree, request, signal);
      }
    }
  }

  private ask(tree: IntervalTree, request: ExpandRequest, signal?: AbortSignal): AsyncIterable<Step> {
    const target = "path" in request
      ? { case: "path" as const, value: { branches: request.path } }
      : { case: "window" as const, value: request.window };
    return named(tree, request, this.client.expand({ node: request.node, horizon: request.horizon, target }, { signal }));
  }

  async search(text: string, signal?: AbortSignal): Promise<Branch[]> {
    return (await this.client.search({ text }, { signal })).path;
  }
}

export interface DatasetEntry {
  name: string;
  title: string;
  /** A JSON array of label bytes in base64, indexed by branch, or a notes file. */
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
  search(text: string, measure: number, signal?: AbortSignal): Promise<Branch[]>;
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
    async search(text, measure, signal) {
      // A recording finds the sequences recorded under the measure in view.
      const source = sources[measure];
      const search = source?.search;
      if (search === undefined) throw new Error("this dataset has no text search");
      return search.call(source, text, signal);
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
