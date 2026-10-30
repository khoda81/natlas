import { Expander, wants } from "../data/expand";
import {
  type Box,
  type Camera,
  clamp,
  type Frame,
  type Lens,
  layout,
  logView,
  normalize,
  pick,
  place,
  reanchor,
  repair,
  type Selection,
  type Viewport,
} from "../core/layout";
import { type Dataset, type DatasetEntry, droppedDataset, fetchDataset, manifest } from "../data/source";
import { IntervalTree, isTail, type NodeId, type Slots, slotOf } from "../core/tree";
import type { Glyph, Reading } from "../readings/reading";
import { drag, moveDepth, panVertical, scale } from "./camera";
import { parseView, type SavedView, viewKey } from "./view-state";
import { DIVERGING_NATS, Palette, SEQUENTIAL_NATS } from "./color";
import { BITS, draw, NATS, RULER_HEIGHT, slotKey, type Units } from "./render";

/** Held arrow keys move the view at these speeds: px of depth and view heights per second. */
const KEY_DEPTH_SPEED = 900;
const KEY_PAN_SPEED = 1.2;
/** A frame after a stall moves at most this far, so the view never jumps. */
const MAX_FRAME_MS = 50;
const ARROWS: Record<string, { depth: number; pan: number }> = {
  ArrowRight: { depth: 1, pan: 0 },
  ArrowLeft: { depth: -1, pan: 0 },
  ArrowUp: { depth: 0, pan: 1 },
  ArrowDown: { depth: 0, pan: -1 },
};
const KEY_SCALE = 1.25;
const FOCUS_MS = 450;
const SEARCH_MS = 150;
const SAVE_MS = 200;
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
  private readonly details = element<HTMLElement>("inspect-details");
  private readonly editor = element<HTMLInputElement>("inspect-editor");
  private readonly where = element<HTMLElement>("where");
  /** The box the path panel shows, as node:slot:kind. */
  private shown = "";
  private loading = 0;
  private navigation = 0;
  private restoring = false;
  private readonly result = element<HTMLButtonElement>("search-result");
  private pendingView: { id: string; view: SavedView } | null = null;
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
  private hover: Selection | null = null;
  /** Screen coordinates, converted against the current canvas rect every frame. */
  private pointer: { x: number; y: number } | null = null;
  private dragging: { id: number; x: number; y: number } | null = null;
  private focus: Focus | null = null;
  private readonly held = new Set<string>();
  private heldSince: number | null = null;
  private scheduled = false;
  private searchTimer: number | null = null;
  private searchGeneration = 0;
  private found: { text: string; path: number[] } | null = null;
  private saveTimer: number | null = null;
  private storage = true;

  constructor() {
    // ResizeObserver runs after animation callbacks, before paint. Draw in the
    // same animation callback as resizing so no cleared bitmap is presented.
    new ResizeObserver(() => this.invalidate()).observe(this.canvas);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.palette = Palette.current();
      this.updateLegend();
      this.invalidate();
    });
    this.bindPointer();
    this.bindKeys();
    this.bindEditor();
    this.pSelect.addEventListener("change", () => this.setLens({ ...this.lens, p: Number(this.pSelect.value) }));
    this.qSelect.addEventListener("change", () => this.setLens({ ...this.lens, q: Number(this.qSelect.value) }));
    element("swap").addEventListener("click", () => this.setLens({ p: this.lens.q, q: this.lens.p }));
    this.unitsButton.addEventListener("click", () => this.toggleUnits());
    element("help-toggle").addEventListener("click", () => this.toggleHelp());
    this.bindDrop();
    window.addEventListener("pagehide", () => this.flushView());
  }

  async start(): Promise<void> {
    this.resize();
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
      this.setStatus("No datasets in data/manifest.json. Drop .pb snapshots, one per measure, and a labels .json here.");
      return;
    }
    select.value = initial;
    load(initial);
  }

  private async load(entry: DatasetEntry): Promise<void> {
    const generation = ++this.loading;
    this.setStatus(`Loading ${entry.title} …`);
    try {
      const dataset = await fetchDataset(entry);
      if (generation !== this.loading) return;
      this.setDataset(dataset);
      this.setStatus("");
    } catch (error) {
      if (generation === this.loading) this.setStatus(String(error));
    }
  }

  private setDataset(dataset: Dataset): void {
    this.flushView();
    this.loading++;
    this.expander?.dispose();
    this.cancelNavigation();
    const tree = new IntervalTree(dataset.measures.length);
    this.dataset = dataset;
    this.tree = tree;
    this.reading = dataset.labels.read(tree);
    this.expander = new Expander(tree, dataset.sources);
    let failing = false;
    this.expander.onError = (error) => {
      if (tree !== this.tree) return;
      failing = true;
      this.setStatus(String(error));
    };
    tree.subscribe(() => {
      if (tree !== this.tree) return;
      if (failing) this.setStatus("");
      failing = false;
      this.invalidate();
    });
    element("title").textContent = dataset.title;
    for (const select of [this.pSelect, this.qSelect]) {
      select.replaceChildren(...dataset.measures.map((name, index) => new Option(name, String(index))));
    }
    element("lens").hidden = dataset.measures.length <= 1;
    this.editor.hidden = !dataset.searchable;
    this.editor.value = "";
    this.result.hidden = true;
    this.editor.classList.remove("found");
    this.found = null;
    this.searchGeneration++;
    if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    if (this.saveTimer !== null) clearTimeout(this.saveTimer);
    this.hover = null;
    this.pointer = null;
    this.shown = "";
    this.where.hidden = true;
    this.details.replaceChildren();
    this.inspect.hidden = !dataset.searchable;
    this.setLens({ p: 0, q: 0 });
    this.reset();
    const saved = this.readView(dataset);
    if (saved !== null) void this.restore(saved);
  }

  private cancelNavigation(): void {
    this.navigation++;
    this.restoring = false;
  }

  /** Fetch as much of a path as the current source knows. */
  private async reachPath(path: readonly number[], generation: number): Promise<{ node: NodeId; complete: boolean } | null> {
    const tree = this.tree!;
    const expander = this.expander!;
    const lens = { ...this.lens };
    const current = () => tree === this.tree && generation === this.navigation;
    let node = tree.root;
    for (const label of path) {
      if (!current()) return null;
      const reached = await expander.reach(node, lens.p, label);
      if (!current()) return null;
      if (!reached) return { node, complete: false };
      if (lens.q !== lens.p) await expander.reach(node, lens.q, label);
      if (!current()) return null;
      node = tree.child(node, label);
    }
    return { node, complete: true };
  }

  private async restore(saved: SavedView): Promise<void> {
    this.setLens({ p: saved.p, q: saved.q });
    const generation = this.navigation;
    this.restoring = true;
    try {
      const reached = await this.reachPath(saved.path, generation);
      if (reached === null) return;
      this.focus = null;
      if (reached.complete) {
        this.cam = { x: saved.x, k: saved.k, anchor: { node: reached.node, slot: 0 }, a: saved.a };
      } else {
        this.standAt(reached.node);
        this.setStatus("The saved path is only partly available; showing its deepest known ancestor.");
      }
      this.invalidate();
    } catch (error) {
      if (generation === this.navigation) this.setStatus(String(error));
    } finally {
      if (generation === this.navigation) this.restoring = false;
    }
  }

  /** Put ``node`` where the root stands, its children filling the view. */
  private standAt(node: NodeId): void {
    const tree = this.tree!;
    if (node === tree.root) {
      this.reset();
      this.where.hidden = true;
      return;
    }
    const parent = tree.parent(node)!;
    const label = tree.label(node);
    const { lnMass, x1 } = place(tree, this.lens, node);
    const slot = slotOf(tree.slots(parent, this.lens), label)!;
    this.focus = null;
    this.cam.anchor = normalize(tree, this.lens, { node: parent, slot });
    this.cam.x = rootLike(lnMass, x1);
    this.cam.a = 0.5 - Math.exp(lnMass - logView(this.cam)) / 2;
    this.showPath(parent, label);
    this.invalidate();
  }

  private setLens(lens: Lens): void {
    this.cancelNavigation();
    this.focus = null;
    this.result.hidden = true;
    this.lens = lens;
    this.pSelect.value = String(lens.p);
    this.qSelect.value = String(lens.q);
    this.found = null;
    this.searchGeneration++;
    this.editor.classList.remove("found");
    this.updateLegend();
    this.invalidate();
  }

  private reset(): void {
    this.cancelNavigation();
    const { width, height } = this.view;
    const k = width / Math.log(Math.max(8, height / 4));
    this.cam = { x: 0, k, anchor: { node: 0, slot: 0 }, a: 0 };
    this.focus = null;
    this.invalidate();
  }

  private resize(): void {
    const ratio = devicePixelRatio || 1;
    const { clientWidth: width, clientHeight: height } = this.canvas;
    this.view = { width, height: Math.max(1, height - RULER_HEIGHT) };
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
    if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
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
    this.resize();
    const tree = this.tree;
    if (tree === null) return;
    if (this.focus) this.stepFocus(time);
    this.stepKeys(time);
    repair(tree, this.lens, this.cam);
    clamp(tree, this.lens, this.cam);
    const frame = layout(tree, this.lens, this.cam, this.view);
    this.frame = frame;
    this.expander?.request(() => {
      if (tree !== this.tree) return [];
      const camera = { ...this.cam };
      repair(tree, this.lens, camera);
      clamp(tree, this.lens, camera);
      const current = layout(tree, this.lens, camera, this.view);
      return wants(tree, this.lens, camera, this.view, current);
    });
    if (!this.focus) reanchor(this.cam, frame);
    this.updateHover();
    const ancestors = new Set<string>();
    if (this.hover) {
      const selected = this.hover.box;
      ancestors.add(slotKey(selected.node, selected.slot));
      for (const node of tree.chain(selected.node).slice(1)) {
        const parent = tree.parent(node)!;
        const slot = slotOf(tree.slots(parent, this.lens), tree.label(node));
        if (slot !== undefined) ancestors.add(slotKey(parent, slot));
      }
    }
    draw(this.ctx, frame, this.cam, this.view, this.lens, this.palette, {
      hover: this.hover,
      ancestors,
      units: this.units,
      glyph: (node, label) => this.glyph(node, label),
    });
    this.updateReadout();
    this.saveView(frame);
    if (this.focus || this.held.size > 0) this.invalidate();
  }

  // Pointer and wheel.

  private bindPointer(): void {
    const canvas = this.canvas;
    canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      this.cancelNavigation();
      canvas.setPointerCapture(event.pointerId);
      this.pointer = { x: event.clientX, y: event.clientY };
      this.dragging = { id: event.pointerId, x: event.offsetX, y: event.offsetY };
      this.focus = null;
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      this.pointer = { x: event.clientX, y: event.clientY };
      const grab = this.dragging;
      if (grab && grab.id === event.pointerId) {
        drag(this.cam, event.offsetX - grab.x, event.offsetY - grab.y, this.view.height, event.offsetY / this.view.height);
        grab.x = event.offsetX;
        grab.y = event.offsetY;
      }
      this.invalidate();
    });
    const release = (event: PointerEvent) => {
      if (this.dragging?.id !== event.pointerId) return;
      this.dragging = null;
      this.pointer = event.type === "pointercancel" ? null : { x: event.clientX, y: event.clientY };
      canvas.classList.remove("dragging");
      this.invalidate();
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", () => {
      this.where.style.minHeight = "";
      if (this.dragging) return;
      this.pointer = null;
      this.invalidate();
    });
    canvas.addEventListener("dblclick", (event) => {
      const selection = this.frame ? pick(this.frame, this.cam, this.view, event.offsetX, event.offsetY) : null;
      if (selection) this.startFocus(selection.box);
    });
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    this.cancelNavigation();
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
      if (
        event.target instanceof HTMLSelectElement ||
        event.target instanceof HTMLInputElement ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey
      ) {
        return;
      }
      if (event.key in ARROWS) {
        event.preventDefault();
        if (!this.held.has(event.key)) {
          this.cancelNavigation();
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
            return scale(cam, KEY_SCALE), true;
          case "-":
          case "_":
            return scale(cam, 1 / KEY_SCALE), true;
          case "0":
          case "Home":
            return this.reset(), true;
          case "b":
            return this.toggleUnits(), true;
          case "s":
            return this.setLens({ p: this.lens.q, q: this.lens.p }), true;
          case "?":
            return this.toggleHelp(), true;
          case "Escape":
            return (element("help").hidden = true), true;
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
      const generation = ++this.loading;
      try {
        const dataset = await droppedDataset(files);
        if (generation !== this.loading) return;
        this.setDataset(dataset);
        this.setStatus("");
      } catch (error) {
        if (generation === this.loading) this.setStatus(String(error));
      }
    });
  }

  // Text search and the inspector editor.

  private bindEditor(): void {
    this.result.addEventListener("click", () => void this.search(true, ++this.searchGeneration));
    this.editor.addEventListener("focus", () => this.held.clear());
    this.editor.addEventListener("input", () => {
      this.result.hidden = true;
      this.found = null;
      this.editor.classList.remove("found");
      const generation = ++this.searchGeneration;
      if (this.searchTimer !== null) clearTimeout(this.searchTimer);
      this.searchTimer = window.setTimeout(() => void this.search(false, generation), SEARCH_MS);
    });
    this.editor.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      if (this.searchTimer !== null) clearTimeout(this.searchTimer);
      void this.search(true, ++this.searchGeneration);
    });
    this.editor.addEventListener("blur", () => this.invalidate());
  }

  private async search(anchor: boolean, generation: number): Promise<void> {
    const dataset = this.dataset;
    if (dataset === null || !dataset.searchable) return;
    const text = this.editor.value;
    try {
      const path = this.found?.text === text ? this.found.path : await dataset.search(text, this.lens.p);
      if (generation !== this.searchGeneration || dataset !== this.dataset) return;
      this.found = { text, path: [...path] };
      this.editor.classList.add("found");
      this.editor.title = "Select the result or press Enter to anchor here";
      let preview = this.tree!.root;
      for (const label of path) preview = this.tree!.child(preview, label);
      this.result.textContent = `${this.reading?.text?.(preview, null) || "Root"} · Enter to anchor`;
      this.result.disabled = false;
      this.result.hidden = false;
      if (!anchor) return;
      this.cancelNavigation();
      const reached = await this.reachPath(path, this.navigation);
      if (generation !== this.searchGeneration || dataset !== this.dataset || reached === null) return;
      this.standAt(reached.node);
      this.result.hidden = true;
      this.setStatus(reached.complete ? "" : "Showing the deepest recorded prefix of this sequence.");
      this.editor.blur();
    } catch (error) {
      if (generation !== this.searchGeneration) return;
      this.editor.classList.remove("found");
      this.editor.title = String(error);
      this.result.textContent = String(error);
      this.result.disabled = true;
      this.result.hidden = false;
      if (anchor) this.setStatus(String(error));
    }
  }

  // Focus animation.

  private startFocus(box: Box): void {
    if (this.tree === null) return;
    this.cam.anchor = normalize(this.tree, this.lens, { node: box.node, slot: box.slot });
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

  // Selection and readouts.

  private pointerPosition(): { x: number; y: number } | null {
    if (this.pointer === null) return null;
    const rect = this.canvas.getBoundingClientRect();
    return { x: this.pointer.x - rect.left, y: this.pointer.y - rect.top };
  }

  private updateHover(): void {
    const pointer = this.pointerPosition();
    this.hover = pointer && !this.dragging && this.frame ? pick(this.frame, this.cam, this.view, pointer.x, pointer.y) : null;
    if (pointer !== null) {
      const covered = pointer.x < this.view.width * 0.45 && pointer.y > this.view.height * 0.55;
      this.inspect.classList.toggle("right", covered);
    }
    this.updateInspect();
    if (this.hover) {
      const { box, exact } = this.hover;
      const more = !exact || box.label === null ? " …" : undefined;
      this.showPath(box.node, box.label, more);
    }
  }

  /** Show the selected path until another box is hovered; ``more`` marks unknown continuation. */
  private showPath(node: NodeId, label: number | null, more?: string): void {
    const key = `${node}:${label}:${more ?? ""}`;
    if (this.reading === null || key === this.shown) return;
    this.shown = key;
    const where = this.where;
    where.style.minHeight = this.pointer === null ? "" : `${where.offsetHeight}px`;
    where.hidden = false;
    this.reading.show(where, node, label, more);
    where.scrollTop = where.scrollHeight;
  }

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
    const selection = this.hover;
    const tree = this.tree;
    const searchable = this.dataset?.searchable === true;
    if (selection === null || tree === null) {
      this.details.hidden = true;
      this.inspect.hidden = !searchable && document.activeElement !== this.editor;
      return;
    }
    const { box, exact } = selection;
    if (searchable && document.activeElement !== this.editor && document.activeElement !== this.result) {
      const text = this.reading?.text?.(box.node, box.label) ?? "";
      if (this.editor.value !== text) {
        this.editor.value = text;
        this.result.hidden = true;
        this.found = null;
        this.searchGeneration++;
        if (this.searchTimer !== null) clearTimeout(this.searchTimer);
        this.editor.classList.remove("found");
      }
    }

    const last = box.label === null ? "…" : this.glyph(box.node, box.label).text;
    const chips = [chip(last)];
    const { p, q } = this.lens;
    const names = this.dataset?.measures ?? [];
    const slots = tree.slots(box.node, this.lens);
    const qBound = box.kind === "tail" || (box.kind !== "merged" && slots.qBound[box.slot] === true);
    const [atMost, atLeast] = qBound ? ["≤ ", "≥ "] : ["", ""];
    const rows: [string, string][] = [
      [`p · ${names[p]}`, `${box.p.toPrecision(4)}  (${this.fmt(-Math.log(box.p))})`],
    ];
    if (p !== q || box.kind === "tail") {
      rows.push([`q · ${names[q]}`, `${atMost}${box.q.toPrecision(4)}  (${atLeast}${this.fmt(-Math.log(box.q))})`]);
    }
    rows.push(["p · −log q", `${atLeast}${this.fmt(-box.p * Math.log(box.q))}`]);
    rows.push(["−log P (path)", this.fmt(-box.lnMass)]);
    const stats = nodeStats(slots, this.lens);
    const bound = slots.count > slots.listed || slots.qBound.some(Boolean) ? "≥ " : "";
    if (p === q) rows.push(["H at node", `${bound}${this.fmt(stats.entropy)}`]);
    else {
      rows.push(["H(p) at node", `${bound}${this.fmt(stats.entropy)}`]);
      rows.push(["H(p, q) at node", `${bound}${this.fmt(stats.cross)}`]);
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
    kind.textContent = exact
      ? `${box.kind === "node" ? "expanded" : box.kind === "leaf" ? "unexpanded" : box.kind} · ${this.units.name}`
      : `known subtree · continuation unknown · ${this.units.name}`;
    this.details.replaceChildren(path, kind, list);
    this.details.hidden = false;
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

  // Remember the deepest anchor that covers the screen, not a server-chosen prompt.

  private readView(dataset: Dataset): SavedView | null {
    if (!this.storage) return null;
    try {
      const raw = localStorage.getItem(viewKey(dataset.id));
      return raw === null ? null : parseView(raw, dataset.measures.length);
    } catch (error) {
      this.storage = false;
      console.warn("Could not read the saved natlas view", error);
      return null;
    }
  }

  private saveView(frame: Frame): void {
    if (!this.storage || this.restoring || frame.cover === null || this.dataset === null || this.tree === null) return;
    if (this.saveTimer !== null) clearTimeout(this.saveTimer);
    const { x, k } = this.cam;
    const { node, y } = frame.cover;
    this.pendingView = {
      id: this.dataset.id,
      view: { path: this.tree.path(node), x, k, a: y, ...this.lens },
    };
    this.saveTimer = window.setTimeout(() => this.flushView(), SAVE_MS);
  }

  private flushView(): void {
    if (this.saveTimer !== null) clearTimeout(this.saveTimer);
    this.saveTimer = null;
    const pending = this.pendingView;
    this.pendingView = null;
    if (!this.storage || pending === null) return;
    try {
      localStorage.setItem(viewKey(pending.id), JSON.stringify(pending.view));
    } catch (error) {
      this.storage = false;
      console.warn("Could not save the natlas view", error);
    }
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

/** Depth at which a box of ln p mass fills the view, or its children start, if sooner. */
function rootLike(lnMass: number, x1: number): number {
  return Math.min(-lnMass, x1);
}

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

/** Lower bounds on entropy and cross-entropy over a node's slots; the tail counts as one slot. */
function nodeStats(slots: Slots, lens: Lens): { entropy: number; cross: number } {
  let entropy = 0;
  let cross = 0;
  for (let slot = 0; slot < slots.count; slot++) {
    const p = slots.p[slot]!;
    if (p === 0) continue;
    entropy -= p * Math.log(isTail(slots, slot) && lens.p === lens.q ? slots.q[slot]! : p);
    cross -= p * Math.log(slots.q[slot]!);
  }
  return { entropy, cross };
}

void new App().start().catch((error: unknown) => {
  const status = element<HTMLElement>("status");
  status.textContent = String(error);
  status.hidden = false;
});
