import { Link } from "../data/link";
import {
  aim,
  type Box,
  type Camera,
  clamp,
  extent,
  type Frame,
  type Lens,
  layout,
  logView,
  pick,
  type Selection,
  settle,
  top,
  view,
  type Viewport,
} from "../core/layout";
import { type Dataset, type DatasetEntry, droppedDataset, fetchDataset, manifest } from "../data/source";
import { IntervalTree, type NodeId, ROOT } from "../core/tree";
import type { Glyph, Reading } from "../readings/reading";
import { glide, moveDepth, panVertical, scale } from "./camera";
import { parseView, type SavedView, serializeView, viewKey } from "./view-state";
import { DIVERGING_NATS, Palette } from "./color";
import { BITS, draw, NATS, RULER_HEIGHT, childKey, pixelCorner, rulerCaption, type Units } from "./render";

/** Held arrow keys move the view at these speeds: px of depth and view heights per second. */
const KEY_DEPTH_SPEED = 300;
const KEY_PAN_SPEED = 0.9;
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
/** The flight to a node closes this share of its distance every ``FLIGHT_MS``. */
const FLIGHT_MS = 140;
const SEARCH_MS = 150;
const SAVE_MS = 200;
const UNITS_KEY = "natlas.units";
const LOCAL_SERVER = "http://localhost:8421/";

const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** A glide to depth ``to.x`` with the camera's point ``to.a`` view heights
 * down, begun at ``start``, ``eased`` of the way there so far. */
interface Focus {
  to: { x: number; a: number };
  start: number;
  eased: number;
}

/**
 * A flight down ``chain`` to its last node. The camera's point stays the
 * top of ``chain[reached]``, the deepest node placed so far, whose ln p mass
 * and right edge below the origin are ``lnMass`` and ``x1``. ``listed`` once
 * the walks have answered; ``time`` is the last frame's.
 */
interface Flight {
  chain: NodeId[];
  reached: number;
  lnMass: number;
  x1: number;
  listed: boolean;
  time: number | null;
}

export class App {
  private readonly canvas = element<HTMLCanvasElement>("canvas");
  private readonly ctx = this.canvas.getContext("2d")!;
  private readonly inspect = element<HTMLElement>("inspect");
  private readonly details = element<HTMLElement>("inspect-details");
  private readonly where = element<HTMLElement>("where");
  private readonly pathPanel = element<HTMLElement>("path-panel");
  /** While the inspector has focus as an editor. */
  private editing = false;
  /** Text edited when the editor last lost focus, given back when it regains it. */
  private draft: string | null = null;
  private shownText = "";
  private shownPath: { node: NodeId; branch: number | null; more?: string } | null = null;
  /** The box the path panel shows, as node:rank:kind. */
  private shown = "";
  private whereHeight: number | null = null;
  private windowHeight = window.innerHeight;
  private inspected = "";
  private readoutText = "";
  private inspector: { chip: HTMLElement; kind: HTMLElement; rows: [HTMLElement, HTMLElement][] } | null = null;
  /** The inspected node's statistics: a pass over all its children, so kept
   * while the node, the lens, and what is known of the node stay the same. */
  private nodeStatistics: { key: string; value: { entropy: number; cross: number; bounded: boolean } } | null = null;
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
  private link: Link | null = null;
  private dataset: Dataset | null = null;
  private reading: Reading | null = null;
  private lens: Lens = { p: 0, q: 0 };
  private cam: Camera = { origin: ROOT, x: 0, k: 100, anchor: top(), a: 0 };
  private view: Viewport = { width: 1, height: 1 };
  private pixelRatio = 1;
  private frame: Frame | null = null;
  private geometry: { frame: Frame; key: string } | null = null;
  private units: Units = storedUnits();
  private palette = Palette.current();
  private hover: Selection | null = null;
  private lastPick: { frame: Frame; x: number; y: number; selection: Selection | null } | null = null;
  /** Screen coordinates, converted against the current canvas rect every frame. */
  private pointer: { x: number; y: number } | null = null;
  private dragging: { id: number; x: number; y: number; moved: boolean } | null = null;
  private focus: Focus | null = null;
  /** The view before the origin last changed, fading out over the new one. */
  private transition: { image: HTMLCanvasElement; start: number } | null = null;
  private readonly reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  /** The origin while its parent is fetched to place it. */
  private placing: NodeId | null = null;
  private flight: Flight | null = null;
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
    this.watchPixelRatio();
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
    this.unitsButton.textContent = rulerCaption(this.units);
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
    this.link?.dispose();
    if (this.geometry?.frame !== this.frame) this.geometry?.frame.dispose();
    this.frame?.dispose();
    this.frame = null;
    this.lastPick = null;
    this.geometry = null;
    this.inspected = "";
    this.inspector = null;
    this.nodeStatistics = null;
    this.glyphs.clear();
    this.cancelNavigation();
    const tree = new IntervalTree(dataset.measures.length);
    this.dataset = dataset;
    this.tree = tree;
    this.reading = dataset.labels.read(tree, () => this.invalidate());
    tree.setGroups(this.reading.groups?.() ?? []);
    this.link = new Link(tree, dataset.server);
    let failing = false;
    this.link.onError = (error) => {
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
    this.draft = null;
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
    this.flight = null;
    this.link?.stop();
  }

  /** Place the path from ``origin`` down to ``target``, from the deepest node
   * already placed: true once all of it is, false where the server places no
   * more of it, null if another path took its place. */
  private async walk(origin: NodeId, target: NodeId): Promise<boolean | null> {
    const tree = this.tree!;
    const chain = tree.chain(origin, target);
    let first = 0;
    while (first < chain.length - 1 && tree.entry(chain[first]!, this.lens, tree.branch(chain[first + 1]!)) !== undefined) first++;
    if (first === chain.length - 1) return true;
    return this.link!.place(chain[first]!, chain.slice(first + 1).map((node) => tree.branch(node)));
  }

  /**
   * Open the node the saved view was in at once: it covered the view, so its
   * children are what the view shows. The saved view itself follows once the
   * walk down from its origin has placed that node.
   */
  private async restore(saved: SavedView): Promise<void> {
    this.setLens({ p: saved.p, q: saved.q });
    const tree = this.tree!;
    const origin = tree.descend(tree.root, saved.origin);
    const node = tree.descend(origin, saved.anchor.path);
    this.open(node, saved.k);
    const generation = this.navigation;
    this.restoring = true;
    try {
      const placed = await this.walk(origin, node);
      if (tree !== this.tree || generation !== this.navigation) return;
      this.focus = null;
      if (placed) {
        this.cam = { origin, x: saved.x, k: saved.k, anchor: saved.anchor, a: saved.a };
      } else {
        this.setStatus("The saved path is only partly available.");
      }
      this.invalidate();
    } catch (error) {
      if (generation === this.navigation) this.setStatus(String(error));
    } finally {
      if (generation === this.navigation) this.restoring = false;
    }
  }

  /**
   * Past the origin's interval lies its parent, unknown until fetched: the
   * view may go there, and finds it empty. Fetch the parent's placement of
   * the origin; once it has it, make the parent the origin, a change of
   * coordinates that keeps the picture: x gains the origin's own width.
   */
  private reachOut(frame: Frame): void {
    const tree = this.tree!;
    const { origin } = this.cam;
    const parent = tree.parent(origin);
    if (parent === null || !(this.cam.x < 0 || frame.originTop > 0 || frame.originBottom < 1)) return;
    // A flight's or a restore's path comes first: one path at a time.
    if (this.flight !== null || this.restoring) return;
    const edge = tree.entry(parent, this.lens, tree.branch(origin));
    if (edge === undefined || !Number.isFinite(edge.natsQ)) {
      void this.place(parent, origin);
      return;
    }
    this.cam.x += edge.natsQ;
    this.cam.anchor = { path: [tree.branch(origin), ...this.cam.anchor.path], digits: this.cam.anchor.digits };
    this.cam.origin = parent;
    // The root's left edge is the start of everything.
    if (parent === ROOT) this.cam.x = Math.max(0, this.cam.x);
    this.invalidate();
  }

  private async place(parent: NodeId, origin: NodeId): Promise<void> {
    if (this.placing === origin) return;
    this.placing = origin;
    const tree = this.tree!;
    try {
      if (await this.walk(parent, origin) && tree === this.tree) this.invalidate();
    } catch (error) {
      if (tree === this.tree) this.setStatus(String(error));
    } finally {
      if (this.placing === origin) this.placing = null;
    }
  }

  /**
   * Fly to ``target`` as the walk down to it arrives: the camera follows the
   * deepest node placed so far, standing each as the origin does, and the
   * descent is the progress. A target below the origin is flown to from the
   * view as it is, from the deepest node on both the camera's path and the
   * target's; any other from the root. Where no source can place the path,
   * ``target`` opens as the origin instead.
   */
  private fly(target: NodeId): void {
    const tree = this.tree!;
    const above = tree.path(this.cam.origin);
    const path = tree.path(target);
    const below = above.length <= path.length && above.every((branch, i) => branch === path[i]);
    if (below) {
      this.cancelNavigation();
      this.focus = null;
    } else this.open(ROOT);
    const { origin } = this.cam;
    const chain = tree.chain(origin, target);
    const flight: Flight = { chain, reached: 0, lnMass: 0, x1: 0, listed: false, time: null };
    if (below) {
      // Down the camera's path while the target's goes the same way, and p
      // and q both place it.
      settle(tree, this.lens, this.cam);
      const { path: anchor } = this.cam.anchor;
      while (flight.reached < Math.min(anchor.length, chain.length - 1) && anchor[flight.reached] === tree.branch(chain[flight.reached + 1]!)) {
        const edge = tree.entry(chain[flight.reached]!, this.lens, anchor[flight.reached]!);
        if (edge === undefined || !Number.isFinite(edge.natsQ)) break;
        flight.lnMass += edge.lnP;
        flight.x1 += edge.natsQ;
        flight.reached++;
      }
      // The point moves to that node's top, where the view shows it now.
      this.cam.a = extent(tree, this.lens, this.cam, flight.reached).top;
      this.cam.anchor = top(anchor.slice(0, flight.reached));
      this.invalidate();
    }
    this.flight = flight;
    void this.walk(origin, target).then((placed) => {
      flight.listed = true;
      if (this.flight !== flight) return;
      if (placed === false) this.open(target);
      this.invalidate();
    }, (error: unknown) => {
      if (this.flight === flight) this.flight = null;
      if (tree === this.tree) this.setStatus(String(error));
    });
  }

  /** One frame of the flight: move the point down to the deepest placed node, then close in on it. */
  private stepFlight(time: number): void {
    const flight = this.flight!;
    const tree = this.tree!;
    const { chain } = flight;
    const from = flight.reached;
    while (flight.reached < chain.length - 1) {
      const node = chain[flight.reached]!;
      const edge = tree.entry(node, this.lens, tree.branch(chain[flight.reached + 1]!));
      if (edge === undefined || !Number.isFinite(edge.natsQ)) break;
      // The point moves from this node's top to its child's, keeping the picture.
      this.cam.a += Math.exp(flight.lnMass - logView(this.cam)) * edge.before;
      flight.lnMass += edge.lnP;
      flight.x1 += edge.natsQ;
      flight.reached++;
    }
    if (from < flight.reached) this.cam.anchor = top(chain.slice(1, flight.reached + 1).map((node) => tree.branch(node)));
    const dt = flight.time === null ? 0 : Math.min(MAX_FRAME_MS, time - flight.time);
    flight.time = time;
    const close = this.reducedMotion.matches ? 1 : 1 - Math.exp(-dt / FLIGHT_MS);
    const toX = rootLike(flight.lnMass, flight.x1);
    const height = Math.exp(flight.lnMass + toX);
    glide(this.cam, { x: toX, a: 0.5 - height / 2 }, close);
    const arrived = flight.reached === chain.length - 1;
    if (arrived && flight.listed && Math.abs(toX - this.cam.x) < 1e-3) {
      this.flight = null;
      this.showPath(chain.at(-1)!, null);
    }
  }

  /**
   * Make ``node`` the origin: it stands as the root does, its children filling
   * the view. Nothing above it is fetched, so this is immediate however deep
   * the node is; its children load while the previous view fades out.
   */
  private open(node: NodeId, k = this.cam.k): void {
    this.cancelNavigation();
    if (this.frame !== null && !this.reducedMotion.matches) {
      const image = this.transition?.image ?? document.createElement("canvas");
      image.width = this.canvas.width;
      image.height = this.canvas.height;
      image.getContext("2d")!.drawImage(this.canvas, 0, 0);
      this.transition = { image, start: performance.now() };
    }
    this.focus = null;
    this.cam = { origin: node, x: 0, k, anchor: top(), a: 0 };
    if (this.dataset?.searchable) this.showPath(node, null);
    this.invalidate();
  }

  private setLens(lens: Lens): void {
    this.cancelNavigation();
    this.focus = null;
    this.result.hidden = true;
    // Digits are fractions of p's mass: under another p only the path holds.
    if (lens.p !== this.lens.p) this.cam.anchor = top(this.cam.anchor.path);
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
    this.cam = { origin: ROOT, x: 0, k, anchor: top(), a: 0 };
    this.focus = null;
    this.transition = null;
    this.invalidate();
  }

  /** A monitor's scale can change without changing the canvas's CSS size. */
  private watchPixelRatio(): void {
    matchMedia(`(resolution: ${devicePixelRatio || 1}dppx)`).addEventListener("change", () => {
      this.watchPixelRatio();
      this.invalidate();
    }, { once: true });
  }

  private resize(): void {
    const ratio = devicePixelRatio || 1;
    this.pixelRatio = ratio;
    const { clientWidth, clientHeight } = this.canvas;
    // At least a pixel each way, even before the page is laid out: the home
    // scale is the width over a number of nats, and a scale of zero is no
    // scale at all.
    const view = { width: Math.max(1, clientWidth), height: Math.max(1, clientHeight - RULER_HEIGHT) };
    if (view.width !== this.view.width || view.height !== this.view.height || this.windowHeight !== window.innerHeight) this.whereHeight = null;
    this.windowHeight = window.innerHeight;
    this.view = view;
    const pixelWidth = Math.round(clientWidth * ratio);
    const pixelHeight = Math.round(clientHeight * ratio);
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
    if (this.flight) this.stepFlight(time);
    this.stepKeys(time);
    const previous = this.frame;
    const frame = this.currentFrame();
    this.frame = frame;
    if (previous !== null && previous !== frame) previous.dispose();
    // What the camera looks at, at the picture's finest: physical rows.
    this.link?.look(this.lens, view(tree, this.lens, this.cam, this.view, this.view.height * this.pixelRatio));
    this.reachOut(frame);
    this.updateHover();
    const ancestors = new Set<string>();
    if (this.hover) {
      const selected = this.hover.box;
      ancestors.add(childKey(selected.node, selected.rank));
      for (const node of tree.chain(this.cam.origin, selected.node).slice(1)) {
        const parent = tree.parent(node)!;
        const entry = tree.entry(parent, this.lens, tree.branch(node));
        if (entry !== undefined) ancestors.add(childKey(parent, entry.rank));
      }
    }
    const origin = this.originNote(tree);
    // The reading's time, from the page's pixels to the canvas's.
    const timeline = this.reading?.timeline?.() ?? null;
    const canvasLeft = timeline === null ? 0 : this.canvas.getBoundingClientRect().left;
    draw(this.ctx, frame, this.cam, this.view, this.lens, this.palette, {
      known: (node) => frame.known(tree, this.lens, node),
      timeline: timeline && {
        x: (time) => timeline.x(time) - canvasLeft,
        left: timeline.left - canvasLeft,
        right: timeline.right - canvasLeft,
      },
      hover: this.hover,
      ancestors,
      units: this.units,
      glyph: (node, branch) => this.glyph(node, branch),
      child: (node, branch) => tree.child(node, branch),
      note: origin && { text: origin.text, sweep: origin.loading && !this.reducedMotion.matches ? time : null },
    });
    this.fadeOut(time);
    this.drawProgress();
    this.updateReadout();
    this.saveView();
    if (this.focus || this.flight || this.held.size > 0 || this.transition || origin?.loading) this.invalidate();
  }

  /** How far down its path the flight has placed, along the top of the view. */
  private drawProgress(): void {
    const flight = this.flight;
    if (flight === null || flight.chain.length < 2) return;
    this.ctx.fillStyle = this.palette.scheme.inkMuted;
    this.ctx.fillRect(0, 0, (this.view.width * flight.reached) / (flight.chain.length - 1), 2);
  }

  /** What the view says while the origin has no children to draw. */
  private originNote(tree: IntervalTree): { text: string; loading: boolean } | null {
    const { origin } = this.cam;
    const measures = [...new Set([this.lens.p, this.lens.q])];
    if (tree.ended(origin, this.lens.p)) return { text: "The sequence ends here · press 0 for the root", loading: false };
    if (measures.some((measure) => this.link!.lacks(origin, measure))) {
      return { text: "This recording does not continue here · press 0 for the root", loading: false };
    }
    const waiting = measures.filter((measure) => !tree.has(origin, measure));
    if (waiting.length === 0) return null;
    return { text: `Loading ${waiting.map((measure) => this.dataset!.measures[measure]).join(" and ")} …`, loading: true };
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
    return `${this.tree!.revision}:${this.lens.p}:${this.lens.q}:${origin}:${x}:${k}:${anchor.path}:${anchor.digits}:${a}:${this.view.width}:${this.view.height}:${this.pixelRatio}`;
  }

  /** The frame for a camera, kept while nothing it depends on changes. */
  private currentFrame(camera = this.cam): Frame {
    const key = this.geometryKey(camera);
    if (this.geometry?.key === key) return this.geometry.frame;
    settle(this.tree!, this.lens, camera);
    clamp(this.tree!, this.lens, camera);
    const settled = this.geometryKey(camera);
    if (this.geometry?.key === settled) return this.geometry.frame;
    const frame = layout(this.tree!, this.lens, camera, this.view, { rowPx: 1 / this.pixelRatio });
    if (this.geometry?.frame !== this.frame) this.geometry?.frame.dispose();
    this.geometry = { frame, key: settled };
    return frame;
  }

  // Motions. The point follows them: a zoom's fixed point becomes the point
  // itself, so depth alone changes and zooming back returns exactly; a pan
  // keeps the point in view, where it tells positions exactly.

  private zoom(dx: number, fy: number): void {
    if (this.tree === null) return;
    aim(this.tree, this.lens, this.cam, fy);
    moveDepth(this.cam, dx, fy);
  }

  private pan(dy: number): void {
    if (this.tree === null) return;
    panVertical(this.cam, dy);
    aim(this.tree, this.lens, this.cam, Math.min(1, Math.max(0, this.cam.a)));
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
      this.dragging = { id: event.pointerId, x: event.offsetX, y: event.offsetY, moved: false };
      this.focus = null;
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      this.pointer = { x: event.clientX, y: event.clientY };
      const grab = this.dragging;
      if (grab && grab.id === event.pointerId) {
        const dx = event.offsetX - grab.x;
        const dy = event.offsetY - grab.y;
        if (dx !== 0 || dy !== 0) {
          // Content follows the pointer, zooming about its height.
          this.pan(dy / this.view.height);
          this.zoom(-dx / this.cam.k, event.offsetY / this.view.height);
          grab.moved = true;
        }
        grab.x = event.offsetX;
        grab.y = event.offsetY;
      }
      this.invalidate();
    });
    const release = (event: PointerEvent) => {
      const grab = this.dragging;
      if (grab?.id !== event.pointerId) return;
      this.dragging = null;
      this.pointer = event.type === "pointercancel" ? null : { x: event.clientX, y: event.clientY };
      canvas.classList.remove("dragging");
      if (event.type === "pointerup" && !grab.moved) {
        // A click, unlike a drag, lets go of the inspector's draft.
        this.draft = null;
        const x = pixelCorner(event.offsetX, this.pixelRatio);
        const y = pixelCorner(event.offsetY, this.pixelRatio);
        const selection = this.frame ? pick(this.tree!, this.lens, this.frame, this.cam, this.view, x, y, 1 / this.pixelRatio) : null;
        if (selection) this.startFocus(selection.box);
      }
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
    else if (event.shiftKey) this.pan(-(dy || dx) / this.view.height);
    else this.zoom((dx - dy) / this.cam.k, fy);
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
    if (depth !== 0) this.zoom((depth * KEY_DEPTH_SPEED * dt) / this.cam.k, 0.5);
    if (pan !== 0) this.pan(pan * KEY_PAN_SPEED * dt);
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
      const draft = this.draft;
      this.draft = null;
      this.where.textContent = draft ?? this.shownText;
      this.whereHeight = null;
      if (draft !== null) void this.search(false, ++this.searchGeneration);
      this.invalidate();
    });
    // Focus leaving the editor and its result puts the hovered path back,
    // and keeps what was edited for the next focus.
    const left = (event: FocusEvent) => {
      const to = event.relatedTarget;
      if (!this.editing || to === this.where || to === this.result) return;
      // textContent is exactly what focus set while nothing was edited.
      const edited = this.where.textContent !== this.shownText;
      const text = this.where.innerText;
      this.stopEditing();
      if (edited) this.draft = text;
    };
    this.where.addEventListener("blur", left);
    this.result.addEventListener("blur", left);
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
        this.stopEditing();
        this.where.blur();
        return;
      }
      if (event.key !== "Enter" || event.shiftKey) return;
      event.preventDefault();
      if (this.searchTimer !== null) clearTimeout(this.searchTimer);
      void this.search(true, ++this.searchGeneration);
    });
  }

  /** Show the hovered path again, leaving the editor's text behind. */
  private stopEditing(): void {
    if (!this.editing) return;
    this.editing = false;
    this.found = null;
    this.searchGeneration++;
    if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    this.result.hidden = true;
    this.result.classList.remove("busy");
    this.where.classList.remove("found");
    this.where.removeAttribute("title");
    this.shown = "";
    if (this.hover !== null) this.showSelectionPath(this.hover);
    else if (this.shownPath === null) this.where.replaceChildren();
    else this.showPath(this.shownPath.node, this.shownPath.branch, this.shownPath.more);
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
      this.found = null;
      this.result.hidden = true;
      this.where.classList.remove("found");
      this.where.removeAttribute("title");
      this.setStatus("");
      this.fly(this.tree!.descend(this.tree!.root, path));
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
    const tree = this.tree;
    if (tree === null) return;
    // The point becomes the box's top: a child's by name, a gap's or a
    // row's, which have none, by where it is drawn.
    if (box.branch === null) {
      aim(tree, this.lens, this.cam, box.y0);
      if (this.cam.a !== box.y0) return;
    } else {
      const path = tree.chain(this.cam.origin, box.node).slice(1).map((node) => tree.branch(node));
      this.cam.anchor = top([...path, box.branch]);
      this.cam.a = box.y0;
    }
    // Standing like the root: as tall as the view where its width allows,
    // centred.
    const x = rootLike(box.lnMass, box.x1);
    const height = Math.exp(box.lnMass + x);
    this.focus = { to: { x, a: 0.5 - height / 2 }, start: performance.now(), eased: 0 };
    this.invalidate();
  }

  private stepFocus(time: number): void {
    const focus = this.focus!;
    const t = Math.min(1, Math.max(0, (time - focus.start) / FOCUS_MS));
    const eased = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    // This frame's share of what was left.
    glide(this.cam, focus.to, focus.eased < 1 ? (eased - focus.eased) / (1 - focus.eased) : 1);
    focus.eased = eased;
    if (t === 1) this.focus = null;
  }

  // Selection and readouts.

  /** The corner of the physical pixel under the pointer, which selects what that pixel shows. */
  private pointerPosition(): { x: number; y: number } | null {
    if (this.pointer === null) return null;
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: pixelCorner(this.pointer.x - rect.left, this.pixelRatio),
      y: pixelCorner(this.pointer.y - rect.top, this.pixelRatio),
    };
  }

  private updateHover(): void {
    const pointer = this.pointerPosition();
    if (pointer && !this.dragging && this.frame) {
      if (this.lastPick?.frame !== this.frame || this.lastPick.x !== pointer.x || this.lastPick.y !== pointer.y) {
        this.lastPick = { frame: this.frame, x: pointer.x, y: pointer.y, selection: pick(this.tree!, this.lens, this.frame, this.cam, this.view, pointer.x, pointer.y, 1 / this.pixelRatio) };
      }
      this.hover = this.lastPick.selection;
    } else this.hover = null;
    this.avoidPointer();
    this.updateInspect();
    if (this.hover) this.showSelectionPath(this.hover);
  }

  private showSelectionPath({ box, exact }: Selection): void {
    const more = !exact || box.branch === null ? " …" : undefined;
    this.showPath(box.node, box.branch, more);
  }

  /**
   * Move the inspector to the other bottom corner while it covers the
   * pointer, unless it would cover it there too. Read before this frame
   * writes to the page, so it lays nothing out.
   */
  private avoidPointer(): void {
    if (this.pointer === null || this.inspect.hidden) return;
    const { x, y } = this.pointer;
    const { left, right, top, bottom } = this.inspect.getBoundingClientRect();
    if (!(top <= y && y < bottom)) return;
    // The corners mirror each other: the same margin from either side.
    const width = document.documentElement.clientWidth;
    const covers = (from: number, to: number) => from <= x && x < to;
    if (covers(left, right) && !covers(width - right, width - left)) this.inspect.classList.toggle("left");
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

    const key = `${box.node}:${box.rank}:${box.kind}:${box.branch}:${box.p}:${box.q}:${box.lnMass}:${exact}:${this.lens.p}:${this.lens.q}:${this.units.name}:${tree.revisionOf(box.node)}`;
    if (this.inspected === key) return;
    this.inspected = key;

    const last = box.branch === null ? "…" : this.glyph(box.node, box.branch).text;
    const { p, q } = this.lens;
    const names = this.dataset?.measures ?? [];
    const known = !Number.isNaN(box.q);
    const rows: [string, string][] = [
      [`p · ${names[p]}`, `${box.p.toPrecision(4)}  (${this.fmt(-Math.log(box.p))})`],
    ];
    if (p !== q || box.kind === "gap") {
      rows.push([`q · ${names[q]}`, known ? `${box.q.toPrecision(4)}  (${this.fmt(-Math.log(box.q))})` : "not yet known"]);
    }
    rows.push(["p · −log q", known ? this.fmt(-box.p * Math.log(box.q)) : "not yet known"]);
    rows.push([this.cam.origin === tree.root ? "−log P (path)" : "−log P (from origin)", this.fmt(-box.lnMass)]);
    const statisticsKey = `${box.node}:${p}:${q}:${tree.revisionOf(box.node)}`;
    if (this.nodeStatistics?.key !== statisticsKey) {
      this.nodeStatistics = { key: statisticsKey, value: tree.statistics(box.node, this.lens) };
    }
    const stats = this.nodeStatistics.value;
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
      ? `${box.kind === "node" ? "expanded" : box.kind === "leaf" ? "unexpanded" : "not yet listed"} · ${this.units.name}`
      : `known subtree · continuation unknown · ${this.units.name}`);
    this.details.hidden = false;
    this.inspect.hidden = false;
  }

  private updateLegend(): void {
    // With p = q, color says nothing: every box has one fill.
    const equal = this.lens.p === this.lens.q;
    this.legend.hidden = equal;
    if (equal) return;
    const gradient = `linear-gradient(to right, ${this.palette.scheme.diverging.join(", ")})`;
    const [low, high] = [`p/q ≤ e^−${DIVERGING_NATS}`, `p/q ≥ e^${DIVERGING_NATS}`];
    this.legend.innerHTML = "";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = gradient;
    const lowText = document.createElement("span");
    lowText.textContent = low;
    const highText = document.createElement("span");
    highText.textContent = high;
    this.legend.append(lowText, swatch, highText);
    this.legend.title = "Color is ln(p/q) per slot: red where q under-weights it. Area is cross-entropy.";
  }

  // Remember the camera as it is: its point is named by the deepest node
  // covering the view, which a reload opens first.

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

  private saveView(): void {
    if (!this.storage || this.restoring || this.flight || this.dataset === null || this.tree === null) return;
    if (this.saveTimer !== null) clearTimeout(this.saveTimer);
    const { origin, x, k, anchor, a } = this.cam;
    this.pendingView = {
      id: this.dataset.id,
      view: { origin: this.tree.path(origin), anchor, x, k, a, ...this.lens },
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
      localStorage.setItem(viewKey(pending.id), serializeView(pending.view));
    } catch (error) {
      this.storage = false;
      console.warn("Could not save the natlas view", error);
    }
  }

  private toggleUnits(): void {
    this.units = this.units === NATS ? BITS : NATS;
    this.unitsButton.textContent = rulerCaption(this.units);
    try {
      localStorage.setItem(UNITS_KEY, this.units.name);
    } catch {
      // Remembering the units is only a convenience.
    }
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

function storedUnits(): Units {
  try {
    return localStorage.getItem(UNITS_KEY) === BITS.name ? BITS : NATS;
  } catch {
    return NATS;
  }
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
