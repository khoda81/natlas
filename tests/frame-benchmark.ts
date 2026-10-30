// Browser benchmark of the real App.render callback, including DOM layout.
// Served by scripts/bench-frames.ts; synthetic sources never contact a model.
import { App } from "../src/ui/main";
import { TextLabels } from "../src/readings/labels";
import type { Camera, Frame, Viewport } from "../src/core/layout";
import type { IntervalTree, Lens } from "../src/core/tree";
import type { Fetched } from "../src/core/tree";
import type { Dataset } from "../src/data/source";
import type { Expander } from "../src/data/expand";

interface Driver {
  setDataset(dataset: Dataset): void;
  setLens(lens: Lens): void;
  render(time: number): void;
  invalidate(): void;
  resize(): void;
  tree: IntervalTree;
  cam: Camera;
  lens: Lens;
  view: Viewport;
  frame: Frame;
  pointer: { x: number; y: number } | null;
  storage: boolean;
  expander: Expander;
  focus: unknown;
}

const app = new App();
const driver = app as unknown as Driver;
driver.invalidate = () => {};
driver.storage = false;
driver.resize();
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const output = document.getElementById("benchmark-output")!;
const button = document.getElementById("benchmark-run") as HTMLButtonElement;
const params = new URLSearchParams(location.search);
const frames = Number(params.get("frames") ?? 240);
const label = params.get("label") ?? "current";
const phases: Record<string, number> = {};
if (params.has("trace")) {
  const trace = (object: object, names: string[], prefix = "") => {
    const methods = object as Record<string, (...args: unknown[]) => unknown>;
    for (const name of names) {
      const original = methods[name];
      if (typeof original !== "function") continue;
      methods[name] = function (...args: unknown[]) {
        const start = performance.now();
        try { return original.apply(this, args); }
        finally { phases[prefix + name] = (phases[prefix + name] ?? 0) + performance.now() - start; }
      };
    }
  };
  trace(app, ["resize", "currentFrame", "updateHover", "updateReadout", "saveView"]);
  trace(canvas.getContext("2d")!, ["measureText", "fillText", "fillRect", "drawImage"]);
}
const tick = () => new Promise<number>(requestAnimationFrame);
const encoder = new TextEncoder();
const labels = new TextLabels(Array.from({ length: 512 }, (_, i) => encoder.encode([" the", "tree", "سلام", "🙂", "\n", "and"][i % 6]!)));
let live: { listing: Fetched; labels: string[] } | null = null;
let liveLabels: TextLabels | null = null;

function setup(count: number, zipf: boolean, deep = false, real = false) {
  const dataset: Dataset = { id: "benchmark", name: "benchmark", title: `Frame benchmark · ${count.toLocaleString()} branches`,
    measures: zipf ? ["p", "q"] : ["p"], sources: Array.from({ length: zipf ? 2 : 1 }, () => ({
      expand: () => new Promise(() => {}), // Four pending requests, like a busy live model.
    })), labels: real ? liveLabels! : labels, searchable: true, search: async () => [0] };
  driver.setDataset(dataset);
  const weights = Array.from({ length: count }, (_, i) => zipf ? (i + 1) ** -1.15 : 1);
  const total = weights.reduce((a, b) => a + b, 0);
  const listing = { labels: weights.map((_, i) => i), lnWeights: weights.map(w => Math.log(w / total)), horizon: 4096, complete: 4096, leftover: 0 };
  driver.tree.setListing(0, 0, real ? live!.listing : listing);
  if (zipf) {
    driver.tree.setListing(0, 1, { ...listing, lnWeights: listing.lnWeights.toReversed() });
    driver.setLens({ p: 0, q: 1 });
  }
  if (deep) {
    let node = 0;
    for (let depth = 0; depth < 80; depth++) {
      node = driver.tree.child(node, depth % 3);
      driver.tree.setListing(node, 0, { labels: [0, 1, 2], lnWeights: [0.6, 0.3, 0.1].map(Math.log), horizon: 4096, complete: 4096, leftover: 0 });
    }
    driver.cam = { x: 125, k: 140, anchor: { node, slot: 1 }, a: 0.4 };
  } else driver.cam = { x: 0, k: 100, anchor: { node: 0, slot: 0 }, a: 0 };
  driver.resize();
  const rect = canvas.getBoundingClientRect();
  driver.pointer = { x: rect.left + driver.view.width * 0.7, y: rect.top + driver.view.height * 0.43 };
}

async function checksum() {
  const data = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
  const hash = await crypto.subtle.digest("SHA-256", data);
  return { pixels: Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, "0")).join(""),
    image: canvas.toDataURL(),
    inspector: document.getElementById("inspect-details")!.innerHTML,
    path: document.getElementById("where")!.innerHTML,
    readout: document.getElementById("readout")!.textContent };
}

button.addEventListener("click", async () => {
  button.disabled = true;
  try {
    if (params.has("live")) {
      live = await (await fetch("/fixture")).json();
      liveLabels = TextLabels.fromJson(live!.labels);
    }
    const results = [];
    for (const scenario of [
      { name: "uniform hover", count: 150000, zipf: false, moving: false },
      { name: "uniform camera", count: 150000, zipf: false, moving: true },
      { name: "long tail hover", count: 150000, zipf: true, moving: false },
      { name: "long tail camera", count: 150000, zipf: true, moving: true },
      { name: "deep camera", count: 73, zipf: false, moving: true, deep: true },
      ...(live === null ? [] : [
        { name: "Qwen hover", count: live.listing.labels.length, zipf: false, moving: false, real: true },
        { name: "Qwen camera", count: live.listing.labels.length, zipf: false, moving: true, real: true },
      ]),
    ].filter(s => params.has("live") ? "real" in s : !params.has("scenario") || s.name === params.get("scenario"))) {
      output.textContent = `Running ${scenario.name}…`;
      setup(scenario.count, scenario.zipf, "deep" in scenario && scenario.deep, "real" in scenario && scenario.real);
      const base = { ...driver.cam, anchor: { ...driver.cam.anchor } };
      const samples: number[] = [];
      const phaseSamples: Record<string, number>[] = [];
      for (let i = -15; i < frames; i++) {
        const time = await tick();
        if (scenario.moving) {
          driver.cam.x = base.x + (scenario.deep ? 0.1 : 0.6) + 0.1 * Math.sin(i / 13);
          driver.cam.k = base.k * (1.1 + 0.1 * Math.cos(i / 19));
        }
        const rect = canvas.getBoundingClientRect();
        driver.pointer = { x: rect.left + driver.view.width * (0.55 + (i % 80) / 200), y: rect.top + driver.view.height * (0.2 + (i % 120) / 200) };
        for (const key of Object.keys(phases)) phases[key] = 0;
        const start = performance.now();
        driver.render(time);
        // Include synchronous style/layout caused by inspector/path updates.
        void document.body.offsetHeight;
        if (i >= 0) {
          samples.push(performance.now() - start);
          if (params.has("trace")) phaseSamples.push({ ...phases });
        }
      }
      samples.sort((a, b) => a - b);
      results.push({ ...scenario, frames, min: samples[0], median: samples[Math.floor(frames / 2)], p95: samples[Math.floor(frames * 0.95)],
        p99: samples[Math.floor(frames * 0.99)], max: samples.at(-1), overBudget: samples.filter(ms => ms >= 2).length, samples, phaseSamples });
    }
    // Read back pixels only after timings: it can change the canvas backend.
    const parity = [];
    for (const [count, zipf, deep] of [[150000, false, false], [150000, true, false], [73, false, true]] as const) {
      setup(count, zipf, deep);
      // Let path-panel height changes settle before comparing the visible UI.
      for (let i = 0; i < 3; i++) driver.render(await tick());
      parity.push({ count, zipf, deep, ...await checksum() });
    }
    const report = { label, userAgent: navigator.userAgent, view: driver.view, dpr: devicePixelRatio, results, parity };
    await fetch("/result", { method: "POST", body: JSON.stringify(report) });
    output.textContent = `Done · ${label}\n` + results.map(r => `${r.name}: median ${r.median!.toFixed(3)} ms · p95 ${r.p95!.toFixed(3)} · max ${r.max!.toFixed(3)} · ≥2 ms ${r.overBudget}/${frames}`).join("\n");
  } catch (error) { output.textContent = `Failed: ${String(error)}`; }
  button.disabled = false;
});
