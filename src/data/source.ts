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
import type { Lens, Path, View } from "../core/layout";
import { Session } from "../core/wasm";
import { type Branch, IntervalTree, type NodeId, ROOT, type Run } from "../core/tree";

/** What the page looks at now, in place of what it looked at before. */
export interface Demand {
  lens: Lens;
  view: View | null;
  /** Placed before anything else. */
  path: Path | null;
}

/** What a server tells the page. */
export type Update =
  /** Children of a node under a measure, named below ``parent`` along
   * ``branch``: the page may never have named it. */
  | { kind: "answer"; measure: number; node: NodeId; parent: NodeId | null; branch: Branch | null; runs: Run[]; count: number | null }
  /** The demand names a node the server has not learned. */
  | { kind: "unknown"; node: NodeId }
  /** The demand's path is placed as far as the server places it. */
  | { kind: "placed"; node: NodeId; branches: Branch[] }
  /** The server will never have this distribution. */
  | { kind: "lacks"; measure: number; node: NodeId }
  | { kind: "failed"; measure: number; node: NodeId; message: string };

/** A session with a server. */
export interface Connection {
  demand(demand: Demand): void;
  close(): void;
}

export interface Server {
  /** A session: ``updates`` hears each batch, ``failed`` what went wrong. */
  connect(updates: (updates: Update[]) => void, failed: (error: unknown) => void): Connection;
  /** Map text to branches from the root, where the server has a text mapping. */
  search?(text: string, measure: number, signal?: AbortSignal): Promise<Branch[]>;
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

  async search(text: string, measure: number, signal?: AbortSignal): Promise<Branch[]> {
    const transport = createConnectTransport({ baseUrl: new URL(`${measure}/`, this.base).href, useBinaryFormat: true });
    return (await createClient(Natlas, transport).search({ text }, { signal })).path;
  }
}

/** A demand as one WebSocket frame: bytes of their own buffer. */
function encode({ lens, view, path }: Demand): Uint8Array<ArrayBuffer> {
  return new Uint8Array(toBinary(DemandSchema, create(DemandSchema, {
    p: lens.p,
    q: lens.q,
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
 * lacking whatever it did not record. */
export class RecordedServer implements Server {
  private readonly recorded: IntervalTree;
  /** Each measure's recorded paths, for text search. */
  private readonly paths: Branch[][][];

  constructor(snapshots: Uint8Array[], private readonly textLabels = TextLabels.none()) {
    this.recorded = new IntervalTree(snapshots.length);
    this.paths = snapshots.map((bytes, measure) => {
      const snapshot = fromBinary(SnapshotSchema, bytes);
      if (snapshot.recordings.length === 0) {
        throw new Error("an empty recording, or one in an older format: run scripts/migrate-snapshots.py");
      }
      const paths = new Map<string, Branch[]>();
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
      return [...paths.values()];
    });
  }

  connect(updates: (updates: Update[]) => void, failed: (error: unknown) => void): Connection {
    const session = new Session();
    let open = true;
    return {
      demand: ({ lens, view, path }) => {
        try {
          session.demand(this.recorded.raw, lens.p, lens.q, view, path);
          const answered = session.answer(this.recorded.raw) as Update[];
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

  async search(text: string, measure: number): Promise<Branch[]> {
    // A recording finds the sequences recorded under the measure in view.
    return this.textLabels.search(this.paths[measure] ?? [], text);
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
  server: Server;
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
  server: Server,
  labels: Labels,
): Dataset {
  const searchable = labels instanceof TextLabels && labels.searchable;
  return {
    id,
    name,
    title,
    measures,
    server,
    labels,
    searchable,
    async search(text, measure, signal) {
      if (server.search === undefined) throw new Error("this dataset has no text search");
      return server.search(text, measure, signal);
    },
  };
}

export async function fetchDataset(entry: DatasetEntry): Promise<Dataset> {
  const base = entry.base ?? dataDirectory();
  const labels =
    entry.labels === undefined
      ? TextLabels.none()
      : parseLabels(await (await fetchOk(new URL(entry.labels, base))).text(), new URL(entry.labels, base));
  // Every measure is a live server's, at one base, or each a recording.
  const live = entry.measures.filter(({ url }) => url !== undefined);
  let server: Server;
  if (live.length === entry.measures.length) {
    server = new LiveServer(base);
  } else if (live.length === 0) {
    const snapshots = await Promise.all(entry.measures.map(async ({ file }) => {
      if (file === undefined) throw new Error("a measure has neither a file nor a url");
      return new Uint8Array(await (await fetchOk(new URL(file, base))).arrayBuffer());
    }));
    server = new RecordedServer(snapshots, labels instanceof TextLabels ? labels : TextLabels.none());
  } else {
    throw new Error("a dataset's measures are all a live server's or all recordings");
  }
  const id = JSON.stringify([entry.name, ...entry.measures.map(({ file, url }) => new URL(url ?? file!, base).href)]);
  return dataset(id, entry.name, entry.title, entry.measures.map(({ name }) => name), server, labels);
}

/** Dropped files: each .pb is one measure, a .json holds the labels. */
export async function droppedDataset(files: File[]): Promise<Dataset> {
  const snapshots = files.filter((file) => file.name.endsWith(".pb"));
  if (snapshots.length === 0) throw new Error("no .pb snapshot among the dropped files");
  const labelFile = files.find((file) => file.name.endsWith(".json"));
  const labels = labelFile === undefined ? TextLabels.none() : parseLabels(await labelFile.text(), null);
  const bytes = await Promise.all(snapshots.map(async (file) => new Uint8Array(await file.arrayBuffer())));
  const server = new RecordedServer(bytes, labels instanceof TextLabels ? labels : TextLabels.none());
  const name = snapshots.map((file) => file.name).join(" · ");
  const hashes = await Promise.all(files.map(async (file) => {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }));
  return dataset(JSON.stringify([name, ...hashes]), name, name, snapshots.map((file) => file.name.replace(/\.pb$/, "")), server, labels);
}

/** A labels file, read by its kind; ``url`` resolves its own references. */
export function parseLabels(json: string, url: URL | null): Labels {
  const parsed = JSON.parse(json) as string[] | { kind?: string };
  if (Array.isArray(parsed)) return TextLabels.fromJson(parsed);
  if (parsed.kind === "notes") return new NoteLabels(parsed as NotesFile, url);
  throw new Error(`labels of an unknown kind: ${parsed.kind}`);
}
