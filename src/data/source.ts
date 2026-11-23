// Servers: where a dataset's distributions come from. The page says what it
// looks at, over a session; the server works out what the view lacks and
// answers. A live server is a WebSocket; a recording answers in the page.

import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { DemandSchema, Natlas, SnapshotSchema, UpdatesSchema } from "../gen/natlas/natlas_pb";
import { TextLabels } from "../readings/labels";
import { type NotesFile, NoteLabels } from "../readings/notes";
import type { Labels } from "../readings/reading";
import type { Path, View } from "../core/layout";
import { Session } from "../core/wasm";
import { type Branch, type NodeId, RecordedTree, ROOT, type Run } from "../core/tree";
import { type Catalog, named } from "./catalog";

/** The measures in view, by spec: p places children, q sizes them. */
export interface Specs {
  p: string;
  q: string;
}

/** What the page looks at now, in place of what it looked at before. */
export interface Demand {
  specs: Specs;
  view: View | null;
  /** Placed before anything else. */
  path: Path | null;
}

/** What a server tells the page; each measure by the spec the page named it by. */
export type Update =
  /** Children of a node under a measure, named below ``parent`` along
   * ``branch``: the page may never have named it. */
  | { kind: "answer"; measure: string; node: NodeId; parent: NodeId | null; branch: Branch | null; runs: Run[]; count: number | null }
  /** The demand names a node the server has not learned. */
  | { kind: "unknown"; node: NodeId }
  /** The demand's path is placed as far as the server places it. */
  | { kind: "placed"; node: NodeId; branches: Branch[] }
  /** The server will never have this distribution. */
  | { kind: "lacks"; measure: string; node: NodeId }
  | { kind: "failed"; measure: string; node: NodeId; message: string };

/** A session with a server. */
export interface Connection {
  demand(demand: Demand): void;
  close(): void;
}

export interface Server {
  /** A session: ``updates`` hears each batch, ``failed`` what went wrong. */
  connect(updates: (updates: Update[]) => void, failed: (error: unknown) => void): Connection;
  /** Map text to branches from the root, where the server has a text mapping. */
  search?(text: string, signal?: AbortSignal): Promise<Branch[]>;
}

/** A live server: one WebSocket session at ``session`` beside its manifest,
 * opened again with the same demand after a drop. */
export class LiveServer implements Server {
  constructor(private readonly base: string) {}

  connect(updates: (updates: Update[]) => void, failed: (error: unknown) => void): Connection {
    let socket: WebSocket | null = null;
    let demand: Uint8Array<ArrayBuffer> | null = null;
    let closed = false;
    let wait = 0;
    const open = () => {
      const url = new URL("session", this.base);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      const opened = socket = new WebSocket(url);
      opened.binaryType = "arraybuffer";
      opened.onopen = () => {
        wait = 0;
        if (demand !== null) opened.send(demand);
      };
      opened.onmessage = (event) => updates(decode(new Uint8Array(event.data as ArrayBuffer)));
      opened.onclose = (event) => {
        if (closed) return;
        // 1007: the server refused the demand; the next one may do.
        if (event.code === 1007) {
          demand = null;
          failed(new Error(`the server refused what the page asked: ${event.reason}`));
        }
        wait = Math.min(4000, wait === 0 ? 250 : wait * 2);
        setTimeout(open, wait);
      };
    };
    open();
    return {
      demand(next) {
        demand = encode(next);
        if (socket?.readyState === WebSocket.OPEN) socket.send(demand);
      },
      close() {
        closed = true;
        socket?.close();
      },
    };
  }

  async search(text: string, signal?: AbortSignal): Promise<Branch[]> {
    const transport = createConnectTransport({ baseUrl: this.base, useBinaryFormat: true });
    return (await createClient(Natlas, transport).search({ text }, { signal })).path;
  }
}

/** A recording uploaded to a server. */
export interface Recording {
  id: string;
  name: string;
  seconds: number;
  /** The recording as the server's model hears it. */
  url: URL;
}

/** The recordings a live server holds, which specs name by id. */
export class Recordings {
  /** By id: those listed or uploaded so far. */
  readonly known = new Map<string, Recording>();

  constructor(private readonly base: string) {}

  /** Every recording the server holds. */
  async list(): Promise<Recording[]> {
    const response = await fetch(new URL("recordings", this.base));
    if (!response.ok) throw new Error(`could not list the recordings: ${response.status}`);
    return ((await response.json()) as Omit<Recording, "url">[]).map((listed) => this.keep(listed));
  }

  /** Send ``file`` to the server: the same file is the same recording. */
  async upload(file: File): Promise<Recording> {
    const url = new URL("recordings", this.base);
    url.searchParams.set("name", file.name);
    const response = await fetch(url, { method: "POST", body: file });
    if (!response.ok) {
      const { message } = (await response.json().catch(() => ({ message: `${response.status}` }))) as { message: string };
      throw new Error(`could not upload ${file.name}: ${message}`);
    }
    return this.keep((await response.json()) as Omit<Recording, "url">);
  }

  private keep(listed: Omit<Recording, "url">): Recording {
    const recording = { ...listed, url: new URL(`recordings/${encodeURIComponent(listed.id)}`, this.base) };
    this.known.set(recording.id, recording);
    return recording;
  }
}

/** A model a server was asked to load, as it is now. */
export interface Loaded {
  spec: string;
  state: "loading" | "loaded" | "failed";
  /** Why it failed. */
  message?: string;
}

/** What a server holds: the models asked for, and the measures open. */
export interface Held {
  models: Loaded[];
  measures: string[];
}

/** What a server holds, and where it can be told to load and unload
 * models and open measures. */
export interface Holdings {
  held(): Promise<Held>;
  change?: {
    load(model: string): Promise<Held>;
    unload(model: string): Promise<Held>;
    open(spec: string): Promise<Held>;
  };
}

/** A live server's models and measures. */
export class LiveHoldings implements Holdings {
  constructor(private readonly base: string) {}

  async held(): Promise<Held> {
    const response = await fetch(new URL("held", this.base));
    if (!response.ok) throw new Error(`could not ask what the server holds: ${response.status}`);
    return (await response.json()) as Held;
  }

  readonly change = {
    load: (model: string) => this.send("POST", `models/${model}`),
    unload: (model: string) => this.send("DELETE", `models/${model}`),
    open: (spec: string) => this.send("POST", `measures/${spec}`),
  };

  private async send(method: string, path: string): Promise<Held> {
    const response = await fetch(new URL(path, this.base), { method });
    const body = (await response.json().catch(() => ({ message: `${response.status}` }))) as Held & { message?: string };
    if (!response.ok) throw new Error(body.message);
    return body;
  }
}

/** A demand as one WebSocket frame: bytes of their own buffer. */
function encode({ specs, view, path }: Demand): Uint8Array<ArrayBuffer> {
  return new Uint8Array(toBinary(DemandSchema, create(DemandSchema, {
    p: specs.p,
    q: specs.q,
    view: view ?? undefined,
    path: path ?? undefined,
  })));
}

function decode(frame: Uint8Array): Update[] {
  return fromBinary(UpdatesSchema, frame).updates.flatMap(({ kind }): Update[] => {
    switch (kind.case) {
      case "answer": {
        const { measure, node, parent, branch, children } = kind.value;
        const runs = (children?.runs ?? []).map(({ rank, branches, lnWeights, top }) => ({ rank, branches, lnWeights, top }));
        return [{ kind: "answer", measure, node, parent: parent ?? null, branch: branch ?? null, runs, count: children?.count ?? null }];
      }
      case "unknown":
        return [{ kind: "unknown", node: kind.value }];
      case "placed":
        return [{ kind: "placed", node: kind.value.node, branches: kind.value.branches }];
      case "lacks":
        return [{ kind: "lacks", measure: kind.value.measure, node: kind.value.node }];
      case "failed":
        return [{ kind: "failed", measure: kind.value.measure, node: kind.value.node, message: kind.value.message }];
      default:
        return [];
    }
  });
}

/** A recorded server, for static datasets: every measure's recording in one
 * tree of its own, read through a session in the page as a server would,
 * lacking whatever it did not record. Its specs are its measures' names. */
export class RecordedServer implements Server {
  private readonly recorded = new RecordedTree();
  /** The paths recorded under any measure, for text search. */
  private readonly paths: Branch[][];

  constructor(
    /** Each recorded measure's name and snapshot. */
    measures: { name: string; snapshot: Uint8Array }[],
    private readonly textLabels = TextLabels.none(),
  ) {
    const paths = new Map<string, Branch[]>();
    for (const { name, snapshot: bytes } of measures) {
      const measure = this.recorded.measure(name);
      const snapshot = fromBinary(SnapshotSchema, bytes);
      if (snapshot.recordings.length === 0) {
        throw new Error("an empty recording, or one in an older format: run scripts/migrate-snapshots.py");
      }
      for (const { path, children } of snapshot.recordings) {
        if (children === undefined) throw new Error("a recorded node has no children");
        this.recorded.insert(this.recorded.descend(ROOT, path), measure, children);
        paths.set(path.join(","), path);
        for (const run of children.runs) {
          for (const branch of run.branches) {
            const child = [...path, branch];
            paths.set(child.join(","), child);
          }
        }
      }
    }
    this.paths = [...paths.values()];
  }

  connect(updates: (updates: Update[]) => void, failed: (error: unknown) => void): Connection {
    const session = new Session();
    // The spec the page named each measure by, by its number here.
    const spoken = new Map<number, string>();
    let open = true;
    return {
      demand: ({ specs, view, path }) => {
        try {
          const [p, q] = [specs.p, specs.q].map((spec) => {
            const measure = this.recorded.measure(decodeURIComponent(spec.split("/")[0]!));
            spoken.set(measure, spec);
            return measure;
          }) as [number, number];
          session.demand(this.recorded.raw, p, q, view, path);
          const answered = (session.answer(this.recorded.raw) as (Update & { measure?: number })[]).map((update) =>
            update.measure === undefined ? update : { ...update, measure: spoken.get(update.measure)! }) as Update[];
          // Told after the page's task, as a server's would be.
          if (answered.length !== 0) queueMicrotask(() => { if (open) updates(answered); });
        } catch (error) {
          failed(error);
        }
      },
      close: () => {
        open = false;
        session.free();
      },
    };
  }

  async search(text: string): Promise<Branch[]> {
    // A recording finds the sequences it recorded.
    return this.textLabels.search(this.paths, text);
  }
}

export interface DatasetEntry {
  name: string;
  title: string;
  /** A JSON array of label bytes in base64, indexed by branch, or a notes file. */
  labels?: string;
  /** A live server's models and measures: what their specs may say. */
  catalog?: Catalog;
  /** A recording's measures, each a snapshot file. */
  measures?: { name: string; file: string }[];
  /** What the entry's paths are relative to; the static data directory by default. */
  base?: string;
}

export interface Dataset {
  /** Includes the source URLs (or dropped file contents), so saved views cannot collide. */
  id: string;
  name: string;
  title: string;
  catalog: Catalog;
  server: Server;
  labels: Labels;
  holdings: Holdings;
  /** Those a live server holds, where its specs name recordings. */
  recordings: Recordings | null;
  searchable: boolean;
  search(text: string, signal?: AbortSignal): Promise<Branch[]>;
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
  catalog: Catalog,
  server: Server,
  labels: Labels,
  holdings: Holdings,
  recordings: Recordings | null = null,
): Dataset {
  const searchable = labels instanceof TextLabels && labels.searchable;
  return {
    id,
    name,
    title,
    catalog,
    server,
    labels,
    holdings,
    recordings,
    searchable,
    async search(text, signal) {
      if (server.search === undefined) throw new Error("this dataset has no text search");
      return server.search(text, signal);
    },
  };
}

export async function fetchDataset(entry: DatasetEntry): Promise<Dataset> {
  const base = entry.base ?? dataDirectory();
  const labels =
    entry.labels === undefined
      ? TextLabels.none()
      : parseLabels(await (await fetchOk(new URL(entry.labels, base))).text());
  if (entry.catalog !== undefined) {
    const recordings = entry.catalog.parameters.some(({ kind }) => kind === "recording") ? new Recordings(base) : null;
    return dataset(JSON.stringify([entry.name, base]), entry.name, entry.title, entry.catalog, new LiveServer(base), labels, new LiveHoldings(base), recordings);
  }
  const measures = entry.measures ?? [];
  const snapshots = await Promise.all(measures.map(async ({ name, file }) =>
    ({ name, snapshot: new Uint8Array(await (await fetchOk(new URL(file, base))).arrayBuffer()) })));
  const server = new RecordedServer(snapshots, labels instanceof TextLabels ? labels : TextLabels.none());
  const id = JSON.stringify([entry.name, ...measures.map(({ file }) => new URL(file, base).href)]);
  return recorded(id, entry.name, entry.title, measures.map(({ name }) => name), server, labels);
}

/** Dropped files: each .pb is one measure, a .json holds the labels. */
export async function droppedDataset(files: File[]): Promise<Dataset> {
  const snapshots = files.filter((file) => file.name.endsWith(".pb"));
  if (snapshots.length === 0) throw new Error("no .pb snapshot among the dropped files");
  const labelFile = files.find((file) => file.name.endsWith(".json"));
  const labels = labelFile === undefined ? TextLabels.none() : parseLabels(await labelFile.text());
  const measures = await Promise.all(snapshots.map(async (file) =>
    ({ name: file.name.replace(/\.pb$/, ""), snapshot: new Uint8Array(await file.arrayBuffer()) })));
  const server = new RecordedServer(measures, labels instanceof TextLabels ? labels : TextLabels.none());
  const name = snapshots.map((file) => file.name).join(" · ");
  const hashes = await Promise.all(files.map(async (file) => {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }));
  return recorded(JSON.stringify([name, ...hashes]), name, name, measures.map(({ name }) => name), server, labels);
}

/** A recording: its measures, each its own model, all loaded. */
function recorded(id: string, name: string, title: string, names: string[], server: Server, labels: Labels): Dataset {
  const specs = names.map(encodeURIComponent);
  const held = { models: specs.map((spec) => ({ spec, state: "loaded" as const })), measures: specs };
  return dataset(id, name, title, named(names), server, labels, { held: async () => held });
}

/** A labels file, read by its kind. */
export function parseLabels(json: string): Labels {
  const parsed = JSON.parse(json) as string[] | { kind?: string };
  if (Array.isArray(parsed)) return TextLabels.fromJson(parsed);
  if (parsed.kind === "notes") return new NoteLabels(parsed as NotesFile);
  throw new Error(`labels of an unknown kind: ${parsed.kind}`);
}
