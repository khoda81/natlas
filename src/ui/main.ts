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
import { IntervalTree, type NodeId } from "../core/tree";
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
/** Opening a node fades the previous view out over this long. */
const TRANSITION_MS = 320;
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


export class App {
  private readonly canvas = element<HTMLCanvasElement>("canvas");
  private readonly ctx = this.canvas.getContext("2d")!;
  private readonly inspect = element<HTMLElement>("inspect");
  private readonly details = element<HTMLElement>("inspect-details");
  private readonly where = element<HTMLElement>("where");
  private readonly pathPanel = element<HTMLElement>("path-panel");
  private editing = false;
  private shownText = "";
  private shownPath: { node: NodeId; branch: number | null; more?: string } | null = null;
  /** The box the path panel shows, as node:slot:kind. */
  private shown = "";
  private whereHeight: number | null = null;
  private windowHeight = window.innerHeight;
  private inspected = "";
  private readoutText = "";
  private inspector: { chip: HTMLElement; kind: HTMLElement; rows: [HTMLElement, HTMLElement][] } | null = null;
  private readonly glyphs = new Map<string, Glyph>();
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
  private cam: Camera = { origin: 0, x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
  private view: Viewport = { width: 1, height: 1 };
  private frame: Frame | null = null;
  private geometry: { frame: Frame; key: string } | null = null;
  private units: Units = NATS;
  private palette = Palette.current();
  private hover: Selection | null = null;
  private lastPick: { frame: Frame; x: number; y: number; selection: Selection | null } | null = null;
  /** Screen coordinates, converted against the current canvas rect every frame. */
  private pointer: { x: number; y: number } | null = null;
  private dragging: { id: number; x: number; y: number } | null = null;
  private focus: Focus | null = null;
  /** The view before the origin last changed, fading out over the new one. */
  private transition: { image: HTMLCanvasElement; start: number } | null = null;
  private readonly reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
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
    if (this.geometry?.frame !== this.frame) this.geometry?.frame.dispose();
    this.frame?.dispose();
    this.frame = null;
    this.lastPick = null;
    this.geometry = null;
    this.inspected = "";
    this.inspector = null;
    this.glyphs.clear();
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
    this.editing = false;
    this.shownText = "";
    this.shownPath = null;
    this.where.replaceChildren();
    this.whereHeight = null;
    this.where.style.minHeight = "";
    this.where.contentEditable = dataset.searchable ? "plaintext-only" : "false";
    this.where.tabIndex = dataset.searchable ? 0 : -1;
    this.where.setAttribute("aria-label", dataset.searchable ? "find text in the sequence distribution" : "text down to the hovered box");
    if (dataset.searchable) {
      this.where.dataset.kind = "text";
      this.where.dataset.placeholder = "Find text…";
      this.where.setAttribute("role", "textbox");
      this.where.setAttribute("aria-multiline", "true");
      this.where.setAttribute("aria-controls", "search-result");
    } else {
      this.where.removeAttribute("role");
      this.where.removeAttribute("aria-multiline");
      this.where.removeAttribute("aria-controls");
    }
    this.result.hidden = true;
    this.where.classList.remove("found");
    this.found = null;
    this.searchGeneration++;
    if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    if (this.saveTimer !== null) clearTimeout(this.saveTimer);
    this.hover = null;
    this.pointer = null;
    this.shown = "";
    this.pathPanel.hidden = this.where.hidden = !dataset.searchable;
    this.details.replaceChildren();
    this.inspect.hidden = true;
    this.setLens({ p: 0, q: 0 });
    this.reset();
    const saved = this.readView(dataset);
    if (saved !== null) void this.restore(saved);
  }

  private cancelNavigation(): void {
    this.navigation++;
    this.restoring = false;
  }

  /** Fetch as much of a path below ``node`` as the current source knows. */
  private async reachPath(node: NodeId, path: readonly number[], generation: number): Promise<{ node: NodeId; complete: boolean } | null> {
    const tree = this.tree!;
    const expander = this.expander!;
    const lens = { ...this.lens };
    const current = () => tree === this.tree && generation === this.navigation;
    for (const branch of path) {
      if (!current()) return null;
      const reached = await expander.reach(node, lens.p, branch);
      if (!current()) return null;
      if (!reached) return { node, complete: false };
      if (lens.q !== lens.p) await expander.reach(node, lens.q, branch);
      if (!current()) return null;
      node = tree.child(node, branch);
    }
    return { node, complete: true };
  }

  /** Open the saved origin at once, then reach down to the saved covering node. */
  private async restore(saved: SavedView): Promise<void> {
    this.setLens({ p: saved.p, q: saved.q });
    const tree = this.tree!;
    const origin = tree.descend(tree.root, saved.origin);
    this.open(origin, saved.k);
    const generation = this.navigation;
    this.restoring = true;
    try {
      const reached = await this.reachPath(origin, saved.path.slice(saved.origin.length), generation);
      if (reached === null) return;
      this.focus = null;
      if (reached.complete) {
        this.cam = { origin, x: saved.x, k: saved.k, anchor: { node: reached.node, slot: 0 }, a: saved.a };
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

  /** Put ``node``, at or below the origin, where the origin stands. */
  private standAt(node: NodeId): void {
    const tree = this.tree!;
    const origin = this.cam.origin;
    this.focus = null;
    if (node === origin) {
      this.cam = { origin, x: 0, k: this.cam.k, anchor: { node: origin, slot: 0 }, a: 0 };
      this.showPath(origin, null);
      this.invalidate();
      return;
    }
    const parent = tree.parent(node)!;
    const branch = tree.branch(node);
    const { lnMass, x1 } = place(tree, this.lens, origin, node);
    const slot = tree.slotOf(parent, this.lens, branch)!;
    this.cam.anchor = normalize(tree, this.lens, origin, { node: parent, slot });
    this.cam.x = rootLike(lnMass, x1);
    this.cam.a = 0.5 - Math.exp(lnMass - logView(this.cam)) / 2;
    this.showPath(parent, branch);
    this.invalidate();
  }

  /**
   * Make ``node`` the origin: it stands as the root does, its children filling
   * the view. Nothing above it is fetched, so this is immediate however deep
   * the node is; its children load while the previous view fades out.
   */
  private open(node: NodeId, k = this.cam.k): void {
    this.cancelNavigation();
    this.expander?.cancel();
    if (this.frame !== null && !this.reducedMotion.matches) {
      const image = this.transition?.image ?? document.createElement("canvas");
      image.width = this.canvas.width;
      image.height = this.canvas.height;
      image.getContext("2d")!.drawImage(this.canvas, 0, 0);
      this.transition = { image, start: performance.now() };
    }
    this.focus = null;
    this.cam = { origin: node, x: 0, k, anchor: { node, slot: 0 }, a: 0 };
    if (this.dataset?.searchable) this.showPath(node, null);
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
    this.where.classList.remove("found");
    this.updateLegend();
    this.invalidate();
  }

  private reset(): void {
    this.cancelNavigation();
    const { width, height } = this.view;
    const k = width / Math.log(Math.max(8, height / 4));
    this.cam = { origin: 0, x: 0, k, anchor: { node: 0, slot: 0 }, a: 0 };
    this.focus = null;
    this.transition = null;
    this.invalidate();
  }

  private resize(): void {
    const ratio = devicePixelRatio || 1;
    const { clientWidth: width, clientHeight: height } = this.canvas;
    if (width !== this.view.width || Math.max(1, height - RULER_HEIGHT) !== this.view.height || this.windowHeight !== window.innerHeight) this.whereHeight = null;
    this.windowHeight = window.innerHeight;
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
    const previous = this.frame;
    const frame = this.currentFrame();
    this.frame = frame;
    if (previous !== null && previous !== frame) previous.dispose();
    this.expander?.request(() => {
      if (tree !== this.tree) return [];
      const camera = { ...this.cam };
      return wants(tree, this.lens, camera, this.view, this.currentFrame(camera));
    });
    if (!this.focus) reanchor(this.cam, frame);
    // Keep the input key after reanchoring. An equivalent anchor can still
    // change floating-point rounding on the next layout and its edge pixels.
    this.updateHover();
    const ancestors = new Set<string>();
    if (this.hover) {
      const selected = this.hover.box;
      ancestors.add(slotKey(selected.node, selected.slot));
      for (const node of tree.chain(this.cam.origin, selected.node).slice(1)) {
        const parent = tree.parent(node)!;
        const slot = tree.slotOf(parent, this.lens, tree.branch(node));
        if (slot !== undefined) ancestors.add(slotKey(parent, slot));
      }
    }
    const origin = this.originNote(tree);
    draw(this.ctx, frame, this.cam, this.view, this.lens, this.palette, {
      hover: this.hover,
      ancestors,
      units: this.units,
      glyph: (node, branch) => this.glyph(node, branch),
      note: origin && { text: origin.text, sweep: origin.loading && !this.reducedMotion.matches ? time : null },
    });
    this.fadeOut(time);
    this.updateReadout();
    this.saveView(frame);
    if (this.focus || this.held.size > 0 || this.transition || origin?.loading) this.invalidate();
  }

  /** What the view says while the origin has no children to draw. */
  private originNote(tree: IntervalTree): { text: string; loading: boolean } | null {
    const { origin } = this.cam;
    const measures = [...new Set([this.lens.p, this.lens.q])];
    const waiting = measures.filter((measure) => !tree.hasListing(origin, measure));
    if (waiting.length > 0) {
      return { text: `Loading ${waiting.map((measure) => this.dataset!.measures[measure]).join(" and ")} …`, loading: true };
    }
    const listing = tree.listing(origin, this.lens.p)!;
    if (listing.branches.length > 0) return null;
    if (listing.leftover === 0) return { text: "The sequence ends here · press 0 for the root", loading: false };
    return listing.complete === 0 ? { text: "This recording does not continue here · press 0 for the root", loading: false } : null;
  }

  /** Draw the previous view over this one, fading and growing as if passed through. */
  private fadeOut(time: number): void {
    const transition = this.transition;
    if (transition === null) return;
    const t = (time - transition.start) / TRANSITION_MS;
    if (t >= 1) {
      this.transition = null;
      return;
    }
    const eased = 1 - (1 - Math.max(0, t)) ** 3;
    const { width, height } = this.canvas;
    const grow = 1 + 0.06 * eased;
    const ctx = this.ctx;
    ctx.save();
    ctx.setTransform(grow, 0, 0, grow, (width * (1 - grow)) / 2, (height * (1 - grow)) / 2);
    ctx.globalAlpha = 1 - eased;
    ctx.drawImage(transition.image, 0, 0);
    ctx.restore();
  }

  private geometryKey(camera = this.cam): string {
    const { origin, x, k, a, anchor } = camera;
    return `${this.tree!.revision}:${this.lens.p}:${this.lens.q}:${origin}:${x}:${k}:${anchor.node}:${anchor.slot}:${a}:${this.view.width}:${this.view.height}`;
  }

  /** Also shared with the completion-driven fetch pump between paint frames. */
  private currentFrame(camera = this.cam): Frame {
    const key = this.geometryKey(camera);
    if (this.geometry?.key === key) return this.geometry.frame;
    repair(this.tree!, this.lens, camera);
    clamp(this.tree!, this.lens, camera);
    const repaired = this.geometryKey(camera);
    if (this.geometry?.key === repaired) return this.geometry.frame;
    const frame = layout(this.tree!, this.lens, camera, this.view);
    if (this.geometry?.frame !== this.frame) this.geometry?.frame.dispose();
    this.geometry = { frame, key: repaired };
    return frame;
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
      this.whereHeight = null;
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
        (event.target instanceof HTMLElement && event.target.isContentEditable) ||
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

  // The top path inspector doubles as the text editor.

  private bindEditor(): void {
    this.result.addEventListener("click", () => void this.search(true, ++this.searchGeneration));
    this.where.addEventListener("focus", () => {
      if (!this.dataset?.searchable || this.editing) return;
      this.held.clear();
      this.editing = true;
      this.where.textContent = this.shownText;
      this.whereHeight = null;
      this.invalidate();
    });
    this.where.addEventListener("input", () => {
      if (!this.dataset?.searchable) return;
      this.whereHeight = null;
      this.result.hidden = true;
      this.result.classList.remove("busy");
      this.found = null;
      this.where.classList.remove("found");
      const generation = ++this.searchGeneration;
      if (this.searchTimer !== null) clearTimeout(this.searchTimer);
      this.searchTimer = window.setTimeout(() => void this.search(false, generation), SEARCH_MS);
    });
    this.where.addEventListener("keydown", (event) => {
      if (!this.dataset?.searchable || event.isComposing) return;
      if (event.key === "Escape") {
        event.preventDefault();
        this.where.blur();
        return;
      }
      if (event.key !== "Enter" || event.shiftKey) return;
      event.preventDefault();
      if (this.searchTimer !== null) clearTimeout(this.searchTimer);
      void this.search(true, ++this.searchGeneration);
    });
    this.where.addEventListener("blur", (event) => {
      if (event.relatedTarget !== this.result) this.finishEditing();
    });
    this.result.addEventListener("blur", (event) => {
      if (event.relatedTarget !== this.where) this.finishEditing();
    });
  }

  private finishEditing(): void {
    if (!this.editing) return;
    this.editing = false;
    this.found = null;
    this.searchGeneration++;
    if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    this.result.hidden = true;
    this.result.classList.remove("busy");
    this.where.classList.remove("found");
    this.shown = "";
    const path = this.shownPath;
    if (path === null) this.where.replaceChildren();
    else this.showPath(path.node, path.branch, path.more);
    this.invalidate();
  }

  private async search(anchor: boolean, generation: number): Promise<void> {
    const dataset = this.dataset;
    if (dataset === null || !dataset.searchable) return;
    const text = this.where.innerText;
    if (anchor && this.found?.text !== text) {
      // Enter was pressed before the search answered; say it is under way.
      this.result.textContent = "Searching…";
      this.result.disabled = true;
      this.result.hidden = false;
      this.result.classList.add("busy");
    }
    try {
      const path = this.found?.text === text ? this.found.path : await dataset.search(text, this.lens.p);
      if (generation !== this.searchGeneration || dataset !== this.dataset) return;
      this.result.classList.remove("busy");
      this.found = { text, path: [...path] };
      this.where.classList.add("found");
      this.where.title = "Select the result or press Enter to anchor here";
      this.result.textContent = "Enter to anchor";
      this.result.disabled = false;
      this.result.hidden = false;
      if (!anchor) return;
      this.editing = false;
      this.shown = "";
      this.result.hidden = true;
      this.setStatus("");
      this.open(this.tree!.descend(this.tree!.root, path));
      this.where.blur();
    } catch (error) {
      if (generation !== this.searchGeneration) return;
      this.result.classList.remove("busy");
      this.where.classList.remove("found");
      this.where.title = String(error);
      this.result.textContent = String(error);
      this.result.disabled = true;
      this.result.hidden = false;
      if (anchor) this.setStatus(String(error));
    }
  }

  // Focus animation.

  private startFocus(box: Box): void {
    if (this.tree === null) return;
    this.cam.anchor = normalize(this.tree, this.lens, this.cam.origin, { node: box.node, slot: box.slot });
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
    if (pointer && !this.dragging && this.frame) {
      if (this.lastPick?.frame !== this.frame || this.lastPick.x !== pointer.x || this.lastPick.y !== pointer.y) {
        this.lastPick = { frame: this.frame, x: pointer.x, y: pointer.y, selection: pick(this.frame, this.cam, this.view, pointer.x, pointer.y) };
      }
      this.hover = this.lastPick.selection;
    } else this.hover = null;
    if (pointer !== null) {
      const covered = pointer.x < this.view.width * 0.45 && pointer.y > this.view.height * 0.55;
      this.inspect.classList.toggle("right", covered);
    }
    this.updateInspect();
    if (this.hover) {
      const { box, exact } = this.hover;
      const more = !exact || box.branch === null ? " …" : undefined;
      this.showPath(box.node, box.branch, more);
    }
  }

  /** Show the selected path until another box is hovered; ``more`` marks unknown continuation. */
  private showPath(node: NodeId, branch: number | null, more?: string): void {
    const key = `${node}:${branch}:${more ?? ""}`;
    if (this.reading === null || this.editing || key === this.shown) return;
    this.shown = key;
    this.shownPath = { node, branch, more };
    this.shownText = this.reading.text?.(node, branch) ?? "";
    this.found = null;
    this.searchGeneration++;
    if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    this.result.hidden = true;
    this.where.classList.remove("found");
    const where = this.where;
    const height = this.pathPanel.hidden || where.hidden ? 0 : this.whereHeight ?? where.offsetHeight;
    where.style.minHeight = this.pointer === null ? "" : `${height}px`;
    this.pathPanel.hidden = where.hidden = false;
    this.reading.show(where, node, branch, more);
    where.scrollTop = Number.MAX_SAFE_INTEGER;
    // The scroll setter has laid out the new path. Reuse this height on the
    // next selection instead of forcing layout after inspector writes, too.
    this.whereHeight = where.offsetHeight;
  }

  private glyph(node: NodeId, branch: number): Glyph {
    const key = `${node}:${branch}`;
    let glyph = this.glyphs.get(key);
    if (glyph === undefined) {
      glyph = this.reading?.glyph(node, branch) ?? { text: `#${branch}` };
      this.glyphs.set(key, glyph);
    }
    return glyph;
  }

  private fmt(nats: number): string {
    const value = nats / this.units.nats;
    if (!Number.isFinite(value)) return "∞";
    return Math.abs(value) >= 100 ? value.toFixed(0) : value.toPrecision(3);
  }

  private updateReadout(): void {
    const unit = this.units.name;
    const view = -Math.min(0, -this.cam.x);
    const text =
      `depth ${this.fmt(this.cam.x)} · view −log v ${this.fmt(view)} · ` +
      `${(this.cam.k * this.units.nats).toPrecision(3)} px/${unit.slice(0, -1)}`;
    if (text !== this.readoutText) this.readout.textContent = this.readoutText = text;
  }

  private updateInspect(): void {
    const selection = this.hover;
    const tree = this.tree;
    if (selection === null || tree === null) {
      this.inspected = "";
      this.details.hidden = true;
      this.inspect.hidden = true;
      return;
    }
    const { box, exact } = selection;

    const key = `${box.node}:${box.slot}:${box.kind}:${box.branch}:${box.p}:${box.q}:${box.lnMass}:${exact}:${this.lens.p}:${this.lens.q}:${this.units.name}:${tree.revisionOf(box.node)}`;
    if (this.inspected === key) return;
    this.inspected = key;

    const last = box.branch === null ? "…" : this.glyph(box.node, box.branch).text;
    const { p, q } = this.lens;
    const names = this.dataset?.measures ?? [];
    const qBound = box.kind === "tail" || (box.kind !== "merged" && tree.qBound(box.node, this.lens, box.slot));
    const [atMost, atLeast] = qBound ? ["≤ ", "≥ "] : ["", ""];
    const rows: [string, string][] = [
      [`p · ${names[p]}`, `${box.p.toPrecision(4)}  (${this.fmt(-Math.log(box.p))})`],
    ];
    if (p !== q || box.kind === "tail") {
      rows.push([`q · ${names[q]}`, `${atMost}${box.q.toPrecision(4)}  (${atLeast}${this.fmt(-Math.log(box.q))})`]);
    }
    rows.push(["p · −log q", `${atLeast}${this.fmt(-box.p * Math.log(box.q))}`]);
    rows.push([this.cam.origin === tree.root ? "−log P (path)" : "−log P (from origin)", this.fmt(-box.lnMass)]);
    const stats = tree.statistics(box.node, this.lens);
    const bound = stats.bounded ? "≥ " : "";
    if (p === q) rows.push(["H at node", `${bound}${this.fmt(stats.entropy)}`]);
    else {
      rows.push(["H(p) at node", `${bound}${this.fmt(stats.entropy)}`]);
      rows.push(["H(p, q) at node", `${bound}${this.fmt(stats.cross)}`]);
      rows.push(["KL(p ‖ q) at node", `${bound === "" ? "" : "≈ "}${this.fmt(stats.cross - stats.entropy)}`]);
    }

    if (this.inspector === null || this.inspector.rows.length !== rows.length) {
      const list = document.createElement("dl");
      const cells = rows.map(() => [document.createElement("dt"), document.createElement("dd")] as [HTMLElement, HTMLElement]);
      for (const pair of cells) list.append(...pair);
      const path = document.createElement("div");
      path.className = "path";
      const token = chip(last);
      path.append(token);
      const kind = document.createElement("div");
      kind.className = "kind";
      this.details.replaceChildren(path, kind, list);
      this.inspector = { chip: token, kind, rows: cells };
    }
    setText(this.inspector.chip, last);
    for (let i = 0; i < rows.length; i++) {
      setText(this.inspector.rows[i]![0], rows[i]![0]);
      setText(this.inspector.rows[i]![1], rows[i]![1]);
    }
    setText(this.inspector.kind, exact
      ? `${box.kind === "node" ? "expanded" : box.kind === "leaf" ? "unexpanded" : box.kind} · ${this.units.name}`
      : `known subtree · continuation unknown · ${this.units.name}`);
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
    const { origin, x, k } = this.cam;
    const { node, y } = frame.cover;
    this.pendingView = {
      id: this.dataset.id,
      view: { origin: this.tree.path(origin), path: this.tree.path(node), x, k, a: y, ...this.lens },
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

function setText(element: HTMLElement, text: string): void {
  if (element.textContent !== text) element.textContent = text;
}
