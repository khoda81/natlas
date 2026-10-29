import { drag, moveDepth, panVertical, scale } from "./camera";
import { DIVERGING_NATS, Palette, SEQUENTIAL_NATS } from "./color";
import { Expander, wants } from "./expand";
import {
  type Box,
  type Camera,
  clamp,
  type Frame,
  type Lens,
  layout,
  logView,
  normalize,
  place,
  reanchor,
  repair,
  type Viewport,
} from "./layout";
import { BITS, draw, NATS, RULER_HEIGHT, screenRect, slotKey, type Units } from "./render";
import type { Glyph, Reading } from "./reading";
import { type Dataset, type DatasetEntry, droppedDataset, fetchDataset, manifest } from "./source";
import { IntervalTree, type NodeId, type Slots, type TreeNode } from "./tree";

/** Held arrow keys move the view at these speeds: px of depth and view heights per second. */
const KEY_DEPTH_SPEED = 900;
const KEY_PAN_SPEED = 1.2;
/** A frame after a stall moves at most this far, so the view never jumps. */
const MAX_FRAME_MS = 50;
/** Held arrow keys: depth direction (+1 deeper) and pan direction (+1 content down). */
const ARROWS: Record<string, { depth: number; pan: number }> = {
  ArrowRight: { depth: 1, pan: 0 },
  ArrowLeft: { depth: -1, pan: 0 },
  ArrowUp: { depth: 0, pan: 1 },
  ArrowDown: { depth: 0, pan: -1 },
};
const KEY_SCALE = 1.25;
const FOCUS_MS = 450;
const LOCAL_SERVER = "http://localhost:8421/";

const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

interface Focus {
  box: Box;
  start: number;
  fromX: number;
  toX: number;
  fromCenter: number;
}

class App {
  private readonly canvas = element<HTMLCanvasElement>("canvas");
  private readonly ctx = this.canvas.getContext("2d")!;
  private readonly inspect = element<HTMLElement>("inspect");
  private readonly where = element<HTMLElement>("where");
  /** The box the path panel shows, as ``node:slot``. */
  private shown = "";
  private readonly readout = element<HTMLElement>("readout");
  private readonly legend = element<HTMLElement>("legend");
  private readonly status = element<HTMLElement>("status");
  private readonly pSelect = element<HTMLSelectElement>("p");
  private readonly qSelect = element<HTMLSelectElement>("q");
  private readonly unitsButton = element<HTMLButtonElement>("units");

  private tree: IntervalTree | null = null;
  private expander: Expander | null = null;
  private dataset: Dataset | null = null;
  private reading: Reading | null = null;
  private lens: Lens = { p: 0, q: 0 };
  private cam: Camera = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
  private view: Viewport = { width: 1, height: 1 };
  private frame: Frame | null = null;
  private units: Units = NATS;
  private palette = Palette.current();
  private hover: Box | null = null;
  private pointer: { x: number; y: number } | null = null;
  private dragging: { id: number; x: number; y: number } | null = null;
  private focus: Focus | null = null;
  /** Arrow keys held down, and the time of the last frame that moved by them. */
  private readonly held = new Set<string>();
  private heldSince: number | null = null;
  private scheduled = false;

  constructor() {
    // ResizeObserver runs after animation callbacks, before paint. Resetting
    // the bitmap here would expose a blank canvas until the next frame.
    new ResizeObserver(() => this.invalidate()).observe(this.canvas);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.palette = Palette.current();
      this.updateLegend();
      this.invalidate();
    });
    this.bindPointer();
    this.bindKeys();
    this.pSelect.addEventListener("change", () => this.setLens({ ...this.lens, p: Number(this.pSelect.value) }));
    this.qSelect.addEventListener("change", () => this.setLens({ ...this.lens, q: Number(this.qSelect.value) }));
    element("swap").addEventListener("click", () => this.setLens({ p: this.lens.q, q: this.lens.p }));
    this.unitsButton.addEventListener("click", () => this.toggleUnits());
    element("help-toggle").addEventListener("click", () => this.toggleHelp());
    this.bindDrop();
  }

  async start(): Promise<void> {
    const select = element<HTMLSelectElement>("dataset");
    const entries = await manifest(liveServer());
    for (const entry of entries) select.add(new Option(entry.title, entry.name));
    const load = (name: string) => {
      const entry = entries.find((candidate) => candidate.name === name) ?? entries[0];
      if (entry !== undefined) void this.load(entry);
    };
    select.addEventListener("change", () => {
      location.hash = select.value;
      load(select.value);
    });
    const initial = decodeURIComponent(location.hash.slice(1)) || entries[0]?.name;
    if (initial === undefined) {
      this.setStatus(
        "No datasets in data/manifest.json. Drop .pb snapshots, one per measure, and a labels .json here.",
      );
      return;
    }
    select.value = initial;
    load(initial);
  }

  private async load(entry: DatasetEntry): Promise<void> {
    this.setStatus(`Loading ${entry.title} …`);
    try {
      this.setDataset(await fetchDataset(entry));
      this.setStatus("");
    } catch (error) {
      this.setStatus(String(error));
    }
  }

  private setDataset(dataset: Dataset): void {
    const tree = new IntervalTree(dataset.measures.length);
    this.dataset = dataset;
    this.tree = tree;
    this.reading = dataset.labels.read(tree);
    this.expander = new Expander(tree, dataset.sources);
    // A failed listing is asked for again; its error stays until one lands.
    let failing = false;
    this.expander.onError = (error) => {
      failing = true;
      this.setStatus(String(error));
    };
    tree.subscribe(() => {
      if (failing) this.setStatus("");
      failing = false;
      this.invalidate();
    });
    element("title").textContent = dataset.title;
    for (const select of [this.pSelect, this.qSelect]) {
      select.replaceChildren(...dataset.measures.map((name, index) => new Option(name, String(index))));
    }
    const multiple = dataset.measures.length > 1;
    element("lens").hidden = !multiple;
    this.setLens({ p: 0, q: 0 });
    this.reset();
    this.shown = "";
    this.where.hidden = true;
    void this.travel(dataset.start).catch((error: unknown) => this.setStatus(String(error)));
  }

  /** Fetch the way down ``path`` from the root, then stand at its end. */
  private async travel(path: number[]): Promise<void> {
    const tree = this.tree!;
    const expander = this.expander!;
    let node = tree.root;
    for (const label of path) {
      if (!(await expander.reach(node, this.lens.p, label))) break;
      node = tree.child(node.id, label);
    }
    // The dataset may have changed, or the view moved on, meanwhile.
    if (tree !== this.tree || node.parent === null) return;
    this.standAt(node);
    this.showPath(node.parent, node.label);
  }

  /** Put ``node`` where the root stands, its children filling the view, as a double-click does. */
  private standAt(node: TreeNode): void {
    const tree = this.tree!;
    const { lnMass, x1 } = place(tree, this.lens, node.id);
    const slot = tree.node(node.parent!).slots(this.lens).slotOf(node.label)!;
    this.focus = null;
    this.cam.anchor = normalize(tree, this.lens, { node: node.parent!, slot });
    this.cam.x = rootLike(lnMass, x1);
    this.cam.a = 0.5 - Math.exp(lnMass - logView(this.cam)) / 2;
    this.invalidate();
  }

  private setLens(lens: Lens): void {
    this.lens = lens;
    this.pSelect.value = String(lens.p);
    this.qSelect.value = String(lens.q);
    this.updateLegend();
    this.invalidate();
  }

  private reset(): void {
    const { width, height } = this.view;
    const k = width / Math.log(Math.max(8, height / 4));
    this.cam = { x: 0, k, anchor: { node: 0, slot: 0 }, a: 0 };
    this.focus = null;
    this.invalidate();
  }

  private resize(): void {
    const ratio = devicePixelRatio || 1;
    const { clientWidth: width, clientHeight: height } = this.canvas;
    const first = this.view.width === 1;
    this.view = { width, height: Math.max(1, height - RULER_HEIGHT) };
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    // Even assigning the current size clears the bitmap and drawing state.
    if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
    if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (first && this.tree) this.reset();
  }

  private invalidate(): void {
    if (this.scheduled) return;
    this.scheduled = true;
    requestAnimationFrame((time) => {
      this.scheduled = false;
      this.render(time);
    });
  }

  private render(time: number): void {
    // Resize and draw in the same callback, so a cleared bitmap is never
    // presented between a hover-driven panel resize and its redraw.
    this.resize();
    const tree = this.tree;
    if (tree === null) return;
    if (this.focus) this.stepFocus(time);
    this.stepKeys(time);
    repair(tree, this.lens, this.cam);
    clamp(tree, this.lens, this.cam);
    const frame = layout(tree, this.lens, this.cam, this.view);
    this.frame = frame;
    this.expander?.request(wants(tree, this.lens, this.cam, this.view, frame, this.dataset?.sources));
    // The anchor is fixed while focusing, so the target stays exactly addressable.
    if (!this.focus) reanchor(this.cam, frame);
    this.updateHover();
    const ancestors = new Set<string>();
    if (this.hover) {
      ancestors.add(slotKey(this.hover.node, this.hover.slot));
      for (const node of tree.chain(this.hover.node).slice(1)) {
        const slot = tree.node(node.parent!).slots(this.lens).slotOf(node.label);
        if (slot !== undefined) ancestors.add(slotKey(node.parent!, slot));
      }
    }
    draw(this.ctx, frame, this.cam, this.view, this.lens, this.palette, {
      hover: this.hover,
      ancestors,
      units: this.units,
      glyph: (node, label) => this.glyph(node, label),
    });
    this.updateReadout();
    if (this.focus || this.held.size > 0) this.invalidate();
  }

  // Pointer and wheel.

  private bindPointer(): void {
    const canvas = this.canvas;
    canvas.addEventListener("wheel", (event) => this.onWheel(event), {
      passive: false,
    });
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      canvas.setPointerCapture(event.pointerId);
      this.dragging = {
        id: event.pointerId,
        x: event.offsetX,
        y: event.offsetY,
      };
      this.focus = null;
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      this.pointer = { x: event.offsetX, y: event.offsetY };
      const grab = this.dragging;
      if (grab && grab.id === event.pointerId) {
        drag(
          this.cam,
          event.offsetX - grab.x,
          event.offsetY - grab.y,
          this.view.height,
          event.offsetY / this.view.height,
        );
        grab.x = event.offsetX;
        grab.y = event.offsetY;
      }
      this.invalidate();
    });
    const release = (event: PointerEvent) => {
      if (this.dragging?.id !== event.pointerId) return;
      this.dragging = null;
      canvas.classList.remove("dragging");
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", () => {
      // The path panel may fit its text again.
      this.where.style.minHeight = "";
      if (this.dragging) return;
      this.pointer = null;
      this.invalidate();
    });
    canvas.addEventListener("dblclick", (event) => {
      const box = this.boxAt(event.offsetX, event.offsetY);
      if (box) this.startFocus(box);
    });
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.view.height : 1;
    const dx = event.deltaX * unit;
    const dy = event.deltaY * unit;
    const fy = event.offsetY / this.view.height;
    this.focus = null;
    if (event.ctrlKey || event.metaKey) scale(this.cam, Math.exp(-dy * 0.01));
    else if (event.shiftKey) panVertical(this.cam, -(dy || dx) / this.view.height);
    else moveDepth(this.cam, (dx - dy) / this.cam.k, fy);
    this.invalidate();
  }

  /** Move by the held arrow keys for the time since the last frame, like movement in a game. */
  private stepKeys(time: number): void {
    if (this.held.size === 0) {
      this.heldSince = null;
      return;
    }
    const dt = Math.min(MAX_FRAME_MS, time - (this.heldSince ?? time)) / 1000;
    this.heldSince = time;
    let depth = 0;
    let pan = 0;
    for (const key of this.held) {
      depth += ARROWS[key]!.depth;
      pan += ARROWS[key]!.pan;
    }
    if (depth !== 0) moveDepth(this.cam, (depth * KEY_DEPTH_SPEED * dt) / this.cam.k, 0.5);
    if (pan !== 0) panVertical(this.cam, pan * KEY_PAN_SPEED * dt);
  }

  private bindKeys(): void {
    const release = () => this.held.clear();
    window.addEventListener("blur", release);
    window.addEventListener("keyup", (event) => {
      if (this.held.delete(event.key)) event.preventDefault();
    });
    window.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLSelectElement || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key in ARROWS) {
        // Repeats change nothing: the key moves the view for as long as it is down.
        event.preventDefault();
        if (!this.held.has(event.key)) {
          this.held.add(event.key);
          this.focus = null;
          this.invalidate();
        }
        return;
      }
      if (this.reading?.key?.(event)) {
        event.preventDefault();
        return;
      }
      const cam = this.cam;
      const handled = (() => {
        switch (event.key) {
          case "+":
          case "=":
            return (scale(cam, KEY_SCALE), true);
          case "-":
          case "_":
            return (scale(cam, 1 / KEY_SCALE), true);
          case "0":
          case "Home":
            return (this.reset(), true);
          case "b":
            return (this.toggleUnits(), true);
          case "s":
            return (this.setLens({ p: this.lens.q, q: this.lens.p }), true);
          case "?":
            return (this.toggleHelp(), true);
          case "Escape":
            return ((element("help").hidden = true), true);
          default:
            return false;
        }
      })();
      if (!handled) return;
      event.preventDefault();
      this.focus = null;
      this.invalidate();
    });
  }

  private bindDrop(): void {
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", async (event) => {
      event.preventDefault();
      const files = [...(event.dataTransfer?.files ?? [])];
      if (files.length === 0) return;
      try {
        this.setDataset(await droppedDataset(files));
        this.setStatus("");
      } catch (error) {
        this.setStatus(String(error));
      }
    });
  }

  // Focus animation: glide until the box stands where the root does, its
  // children starting at the left edge and filling the view.

  private startFocus(box: Box): void {
    if (this.tree === null) return;
    this.cam.anchor = normalize(this.tree, this.lens, {
      node: box.node,
      slot: box.slot,
    });
    this.cam.a = box.y0;
    this.focus = {
      box,
      start: performance.now(),
      fromX: this.cam.x,
      toX: rootLike(box.lnMass, box.x1),
      fromCenter: (box.y0 + box.y1) / 2,
    };
    this.invalidate();
  }

  private stepFocus(time: number): void {
    const focus = this.focus!;
    const t = Math.min(1, Math.max(0, (time - focus.start) / FOCUS_MS));
    const eased = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    this.cam.x = focus.fromX + (focus.toX - focus.fromX) * eased;
    const height = Math.exp(focus.box.lnMass - Math.min(0, -this.cam.x));
    const center = focus.fromCenter + (0.5 - focus.fromCenter) * eased;
    this.cam.a = center - height / 2;
    if (t === 1) this.focus = null;
  }

  // Readouts.

  private boxAt(x: number, y: number): Box | null {
    if (y >= this.view.height) return null;
    for (const box of this.frame?.boxes ?? []) {
      const rect = screenRect(box, this.cam, this.view);
      if (x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1) return box;
    }
    return null;
  }

  private updateHover(): void {
    const pointer = this.pointer;
    this.hover = pointer && !this.dragging ? this.boxAt(pointer.x, pointer.y) : null;
    // Keep the inspector clear of the pointer.
    const covered = pointer !== null && pointer.x < this.view.width * 0.45 && pointer.y > this.view.height * 0.55;
    this.inspect.classList.toggle("right", covered);
    this.updateInspect();
    if (this.hover !== null) this.showPath(this.hover.node, this.hover.label, this.hover);
  }

  /**
   * Show the path from the root down to child ``label`` of ``node``, or to
   * ``node`` itself for a tail or merged box, until another box is hovered.
   * While the pointer is over the canvas the panel only grows, so that a
   * shorter text cannot shrink it, resize the canvas, and move the box under
   * the pointer.
   */
  private showPath(node: NodeId, label: number | null, box?: Box): void {
    const key = `${node}:${box?.slot ?? label}:${box?.kind ?? ""}`;
    if (this.reading === null || key === this.shown) return;
    this.shown = key;
    const where = this.where;
    where.style.minHeight = this.pointer === null ? "" : `${where.offsetHeight}px`;
    where.hidden = false;
    this.reading.show(where, node, label, box?.kind === "merged" ? ` ${box.count} children` : " …");
    where.scrollTop = where.scrollHeight;
  }

  /** How the child with ``label`` of ``node`` reads. */
  private glyph(node: NodeId, label: number): Glyph {
    return this.reading?.glyph(node, label) ?? { text: `#${label}` };
  }

  private fmt(nats: number): string {
    const value = nats / this.units.nats;
    if (!Number.isFinite(value)) return "∞";
    return Math.abs(value) >= 100 ? value.toFixed(0) : value.toPrecision(3);
  }

  private updateReadout(): void {
    const unit = this.units.name;
    const view = -Math.min(0, -this.cam.x);
    this.readout.textContent =
      `depth ${this.fmt(this.cam.x)} · view −log v ${this.fmt(view)} · ` +
      `${(this.cam.k * this.units.nats).toPrecision(3)} px/${unit.slice(0, -1)}`;
  }

  private updateInspect(): void {
    const box = this.hover;
    const tree = this.tree;
    if (box === null || tree === null) {
      this.inspect.hidden = true;
      return;
    }
    const node = tree.node(box.node);
    const last =
      box.kind === "merged" ? `${box.count} slots` : box.label === null ? "…" : this.glyph(box.node, box.label).text;
    const chips = [chip(last)];

    const { p, q } = this.lens;
    const names = this.dataset?.measures ?? [];
    const slots = node.slots(this.lens);
    // A tail's q, and q of a child q has not listed, are upper bounds.
    const qBound = box.kind === "tail" || (box.kind !== "merged" && slots.qBound[box.slot] === true);
    const [atMost, atLeast] = qBound ? ["≤ ", "≥ "] : ["", ""];
    const rows: [string, string][] = [
      [`p · ${names[p]}`, `${slots.sampled ? "≈ " : ""}${box.p.toPrecision(4)}  (${this.fmt(-Math.log(box.p))})`],
    ];
    if (p !== q || box.kind === "tail") {
      rows.push([`q · ${names[q]}`, `${atMost}${box.q.toPrecision(4)}  (${atLeast}${this.fmt(box.x1 - box.x0)})`]);
    }
    rows.push(["p · −log q", `${slots.sampled ? "≈ " : atLeast}${this.fmt(box.p * (box.x1 - box.x0))}`]);
    rows.push(["−log P (path)", this.fmt(-box.lnMass)]);
    const stats = nodeStats(slots, this.lens);
    const bound = slots.sampled ? "≈ " : slots.count > slots.listed || slots.qBound.some(Boolean) ? "≥ " : "";
    if (p === q) rows.push(["H at node", `${bound}${this.fmt(stats.entropy)}`]);
    else {
      rows.push(["H(p) at node", `${bound}${this.fmt(stats.entropy)}`]);
      rows.push(["H(p, q) at node", `${bound}${this.fmt(stats.cross)}`]);
      // A difference of two lower bounds bounds nothing.
      rows.push(["KL(p ‖ q) at node", `${bound === "" ? "" : "≈ "}${this.fmt(stats.cross - stats.entropy)}`]);
    }

    const list = document.createElement("dl");
    for (const [term, value] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      list.append(dt, dd);
    }
    const path = document.createElement("div");
    path.className = "path";
    path.append(...chips);
    const kind = document.createElement("div");
    kind.className = "kind";
    kind.textContent = `${box.kind === "node" ? "expanded" : box.kind === "leaf" ? "unexpanded" : box.kind} · ${this.units.name}`;
    this.inspect.replaceChildren(path, kind, list);
    this.inspect.hidden = false;
  }

  private updateLegend(): void {
    const equal = this.lens.p === this.lens.q;
    const stops = equal ? this.palette.scheme.sequential : this.palette.scheme.diverging;
    const gradient = `linear-gradient(to right, ${stops.join(", ")})`;
    const [low, high] = equal
      ? [`p ≤ e^−${SEQUENTIAL_NATS}`, "p = 1"]
      : [`p/q ≤ e^−${DIVERGING_NATS}`, `p/q ≥ e^${DIVERGING_NATS}`];
    this.legend.innerHTML = "";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = gradient;
    const lowText = document.createElement("span");
    lowText.textContent = low;
    const highText = document.createElement("span");
    highText.textContent = high;
    this.legend.append(lowText, swatch, highText);
    this.legend.title = equal
      ? "Color repeats each slot's conditional probability. Area is entropy."
      : "Color is ln(p/q) per slot: red where q under-weights it. Area is cross-entropy.";
  }

  private toggleUnits(): void {
    this.units = this.units === NATS ? BITS : NATS;
    this.unitsButton.textContent = this.units.name;
    this.invalidate();
  }

  private toggleHelp(): void {
    const help = element("help");
    help.hidden = !help.hidden;
  }

  private setStatus(text: string): void {
    this.status.textContent = text;
    this.status.hidden = text === "";
  }
}

/**
 * Depth at which a box of ln p mass ``lnMass`` fills the view, since the view
 * spans e^-x of the root. With p = q that is also where its children start,
 * at ``x1``; otherwise stop there, if sooner, so they stay on screen.
 */
function rootLike(lnMass: number, x1: number): number {
  return Math.min(-lnMass, x1);
}

/**
 * The live server to ask for datasets: ``?server=URL``, else a local one when
 * the page itself is local, so a deployed page never probes the visitor's
 * machine.
 */
function liveServer(): string | undefined {
  const server = new URLSearchParams(location.search).get("server");
  if (server !== null) return new URL(server.endsWith("/") ? server : `${server}/`, location.href).href;
  return ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) ? LOCAL_SERVER : undefined;
}

function chip(text: string): HTMLElement {
  const span = document.createElement("span");
  span.className = "chip current";
  span.textContent = text;
  return span;
}

/** v1 lower bounds or v2 sample estimates of node entropy and cross-entropy. */
function nodeStats(slots: Slots, lens: Lens): { entropy: number; cross: number } {
  let entropy = 0;
  let cross = 0;
  for (let slot = 0; slot < slots.count; slot++) {
    const p = slots.p[slot]!;
    if (p === 0) continue;
    // With p = q, q bounds the tail's heaviest child, which is tighter than the tail as a whole.
    entropy -= p * Math.log(slots.isTail(slot) && lens.p === lens.q ? slots.q[slot]! : p);
    cross += p * slots.widths[slot]!;
  }
  return { entropy, cross };
}

void new App().start();
