// What the page looks at, told to a server, and what the server answers put
// into the tree. The server works out what the view lacks and in what order;
// the page only says what it looks at, and which path to place first.

import type { Path, View } from "../core/layout";
import { type Branch, type IntervalTree, type NodeId, ROOT } from "../core/tree";
import type { Connection, Demand, Server, Specs, Update } from "./source";

export class Link {
  private readonly connection: Connection;
  private specs: Specs | null = null;
  private view: View | null = null;
  /** The path to place first, and what waits on it: true once placed,
   * false where the server places no more of it, null if another path took
   * its place. */
  private path: { path: Path; done: (placed: boolean | null) => void } | null = null;
  /** The path as last sent, which a server's ``placed`` echoes. */
  private sent: Path | null = null;
  /** Nodes the server said it has not learned: named from the root until a
   * path down to them is placed. */
  private readonly unknown = new Set<NodeId>();
  private readonly lacking = new Set<string>();
  private said = "";
  onError: (error: unknown) => void = (error) => console.error(error);

  constructor(private readonly tree: IntervalTree, server: Server) {
    this.connection = server.connect((updates) => this.take(updates), (error) => this.onError(error));
  }

  /** What the page looks at now, under the measures ``specs`` names. */
  look(specs: Specs, view: View | null): void {
    this.specs = specs;
    this.view = view;
    this.tell();
  }

  /** Place ``branches`` below ``node`` before anything else. */
  place(node: NodeId, branches: Branch[]): Promise<boolean | null> {
    this.path?.done(null);
    return new Promise((done) => {
      this.path = { path: { node, branches }, done };
      this.tell();
    });
  }

  /** Place no path: nothing waits on one any more. */
  stop(): void {
    this.path?.done(null);
    this.path = null;
    this.tell();
  }

  /** Whether the server will never have ``measure``'s distribution of ``node``. */
  lacks(node: NodeId, measure: number): boolean {
    return this.lacking.has(`${measure}:${node}`);
  }

  dispose(): void {
    this.path?.done(null);
    this.path = null;
    this.connection.close();
  }

  /** Say what the page looks at, if that changed. A node the server has not
   * learned is named down a path from the root, which teaches it. */
  private tell(): void {
    const specs = this.specs;
    if (specs === null) return;
    let path = this.path?.path ?? null;
    if (path !== null && this.unknown.has(path.node)) {
      path = { node: ROOT, branches: [...this.tree.path(path.node), ...path.branches] };
    }
    if (path === null && this.view !== null && this.unknown.has(this.view.node)) {
      path = { node: ROOT, branches: this.tree.path(this.view.node) };
    }
    const demand: Demand = { specs, view: this.view, path };
    const said = JSON.stringify(demand, (_, value: unknown) =>
      typeof value === "bigint" ? `${value}` : value instanceof Uint8Array ? [...value] : value);
    if (said === this.said) return;
    this.said = said;
    this.sent = path;
    this.connection.demand(demand);
  }

  private take(updates: Update[]): void {
    let changed = false;
    for (const update of updates) {
      switch (update.kind) {
        case "answer": {
          if (update.parent !== null && update.branch !== null) {
            const named = this.tree.child(update.parent, update.branch);
            if (named !== update.node) throw new Error(`the server named node ${update.node} as ${named}`);
          }
          this.tree.insert(update.node, this.tree.measure(update.measure), { runs: update.runs, count: update.count ?? undefined });
          break;
        }
        case "unknown":
          this.unknown.add(update.node);
          changed = true;
          break;
        case "placed": {
          // The server has learned every node down a path it placed.
          this.unknown.clear();
          const waiting = this.path;
          const sent = this.sent;
          if (waiting !== null && sent !== null && sent.node === update.node && sameBranches(sent.branches, update.branches)) {
            this.path = null;
            waiting.done(this.placed(waiting.path));
          }
          changed = true;
          break;
        }
        case "lacks":
          this.lacking.add(`${this.tree.measure(update.measure)}:${update.node}`);
          break;
        case "failed":
          this.onError(new Error(update.message));
          break;
      }
    }
    if (changed) this.tell();
  }

  /** Whether p places every branch of the path. */
  private placed({ node, branches }: Path): boolean {
    const lens = { p: this.tree.measure(this.specs!.p), q: this.tree.measure(this.specs!.q) };
    for (const branch of branches) {
      if (this.tree.entry(node, lens, branch) === undefined) return false;
      node = this.tree.child(node, branch);
    }
    return true;
  }
}

function sameBranches(a: ArrayLike<Branch>, b: ArrayLike<Branch>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
