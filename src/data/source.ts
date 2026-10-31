// Integer-sequence sources: one per measure. Nodes are named by their shared
// hashed ids. A recorded snapshot answers from memory; a live server uses Connect.

import { fromBinary } from "@bufbuild/protobuf";
import { Code, ConnectError, createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { Natlas, SnapshotSchema } from "../gen/natlas/natlas_pb";
import { TextLabels } from "../readings/labels";
import { type NotesFile, NoteLabels } from "../readings/notes";
import type { Labels } from "../readings/reading";
import { type Branch, IntervalTree, type Listing, type NodeId, ROOT } from "../core/tree";

/** A node's children as a source lists them; `complete` is what it vouches for. */
export type Reply = Omit<Listing, "horizon">;

/** One node's reply in a walk. */
export interface Step {
  node: NodeId;
  reply: Reply;
}

export interface Source {
  /**
   * The children of ``to`` and of every node between ``from`` and it, in
   * order, as the protocol's Expand: each down to 2^-horizon and through the
   * path's next branch. ``from`` is ``to`` itself, its parent, or an ancestor
   * the client can already place the path through.
   */
  expand(tree: IntervalTree, from: NodeId, to: NodeId, horizon: number, signal?: AbortSignal): AsyncIterable<Step>;
  /** Map text to branches from the root, when this source has a text mapping. */
  search?(text: string, signal?: AbortSignal): Promise<Branch[]>;
}

/** The nodes a walk from ``from`` to ``to`` lists: those below ``from``, or ``to`` alone. */
function walked(tree: IntervalTree, from: NodeId, to: NodeId): NodeId[] {
  return from === to ? [to] : tree.chain(from, to).slice(1);
}

/** A recorded server. Its own tree learns the recorded nodes' ids. */
export class SnapshotSource implements Source {
  private readonly recorded = new IntervalTree(1);
  private readonly paths: Branch[][] = [];

  constructor(
    bytes: Uint8Array,
    private readonly textLabels = TextLabels.none(),
  ) {
    const snapshot = fromBinary(SnapshotSchema, bytes);
    const paths = new Map<string, Branch[]>();
    for (const { path, horizon, listing } of snapshot.expansions) {
      if (listing === undefined) throw new Error("a recorded expansion has no listing");
      const node = this.recorded.descend(this.recorded.root, path);
      this.recorded.setListing(node, 0, {
        branches: listing.branches,
        lnWeights: listing.lnWeights,
        leftover: Math.exp(listing.lnLeftover),
        horizon,
        complete: horizon,
      });
      paths.set(path.join(","), path);
      for (const branch of listing.branches) {
        const child = [...path, branch];
        paths.set(child.join(","), child);
      }
    }
    this.paths = [...paths.values()];
  }

  async *expand(tree: IntervalTree, from: NodeId, to: NodeId, horizon: number): AsyncIterable<Step> {
    const nodes = walked(tree, from, to);
    for (const [index, node] of nodes.entries()) {
      const next = nodes[index + 1];
      const recorded = this.recorded.raw.find(tree.hash(node));
      const cut = recorded === undefined ? undefined : this.recorded.cut(recorded, 0, horizon, next === undefined ? undefined : tree.branch(next));
      yield { node, reply: cut ?? { branches: [], lnWeights: [], leftover: 1, complete: 0 } };
    }
  }

  async search(text: string): Promise<Branch[]> {
    return this.textLabels.search(this.paths, text);
  }
}

/** A live server: every expansion honors the requested horizon. */
export class ConnectSource implements Source {
  private readonly client;

  constructor(baseUrl: string) {
    this.client = createClient(Natlas, createConnectTransport({ baseUrl, useBinaryFormat: true }));
  }

  /** Name the walk from ``from``'s id, and from the root when the server has
   * not learned that id (it restarted, or never saw it). */
  async *expand(tree: IntervalTree, from: NodeId, to: NodeId, horizon: number, signal?: AbortSignal): AsyncIterable<Step> {
    let listed = false;
    try {
      for await (const step of this.walk(tree, from, to, horizon, signal)) {
        listed = true;
        yield step;
      }
    } catch (error) {
      if (listed || from === ROOT || ConnectError.from(error).code !== Code.NotFound) throw error;
      yield* this.walk(tree, ROOT, to, horizon, signal);
    }
  }

  private async *walk(tree: IntervalTree, from: NodeId, to: NodeId, horizon: number, signal?: AbortSignal): AsyncIterable<Step> {
    const nodes = walked(tree, from, to);
    const path = from === to ? [] : nodes.map((node) => tree.branch(node));
    let index = 0;
    for await (const listing of this.client.expand({ node: tree.hash(from), path, horizon }, { signal })) {
      const node = nodes[index++];
      if (node === undefined) throw new Error("the server listed more nodes than the path holds");
      const last = listing.lnWeights.at(-1);
      // Every child as heavy as the last listed one is listed.
      const complete = last === undefined ? horizon : Math.max(horizon, -last / Math.LN2);
      yield { node, reply: { branches: listing.branches, lnWeights: listing.lnWeights, leftover: Math.exp(listing.lnLeftover), complete } };
    }
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
