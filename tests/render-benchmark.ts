// Browser render-pipeline & data-transfer benchmark.
// Isolates each stage of a frame (without inspector/DOM noise):
//   1. Ingest (JS -> WASM tree.insert + approximate wire bytes)
//   2. WASM layout (Rust core layoutFrame)
//   3. WASM -> JS transfer (packed typed arrays + Box/Span hydration)
//   4. Canvas paint (shapes/spans only — no fillText/DOM)
//   5. Pick (WASM layout.pick)
//
// Served by `bun scripts/bench-render.ts`.

import {
  layout as wasmLayout,
  Session as WasmSession,
  settle as wasmSettle,
  view as wasmView,
} from "../src/core/wasm";
import { Frame, type Camera, type Viewport } from "../src/core/layout";
import { IntervalTree, RecordedTree, ROOT, type Lens } from "../src/core/tree";
import type { Update } from "../src/data/source";
import { digits, run } from "./build";

const VIEW: Viewport = { width: 1200, height: 900 };
const ROW_PX = 1;
const SPINE_DEPTH = 12;
const WIRE_BYTES_PER_BRANCH = 4 + 8;

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
canvas.width = VIEW.width;
canvas.height = VIEW.height;
const ctx = canvas.getContext("2d")!;
const output = document.getElementById("benchmark-output")!;
const button = document.getElementById("benchmark-run") as HTMLButtonElement;

const params = new URLSearchParams(location.search);
const frames = Number(params.get("frames") ?? 180);
const label = params.get("label") ?? "render";

interface StageStats {
  p50: number;
  p95: number;
  max: number;
}

function stats(samples: number[]): StageStats {
  const sorted = samples.toSorted((a, b) => a - b);
  const n = sorted.length;
  return {
    p50: sorted[Math.floor(n / 2)] ?? 0,
    p95: sorted[Math.floor(n * 0.95)] ?? 0,
    max: sorted[n - 1] ?? 0,
  };
}

function zipfWeights(count: number, alpha = 1.15): number[] {
  const raw = Array.from({ length: count }, (_, i) => (i + 1) ** -alpha);
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((w) => w / total);
}

function buildSourceTree<T extends IntervalTree | RecordedTree>(tree: T, count: number, withQ: boolean) {
  const p = tree.measure("p");
  const q = withQ ? tree.measure("q") : p;
  const lens: Lens = { p, q };
  const weights = zipfWeights(count);
  const rootRun = run(weights, undefined, 0, 0);
  const rootQ = withQ ? { ...rootRun, branches: rootRun.branches.toReversed() } : null;
  const midRun = run(zipfWeights(1024, 1.25), undefined, 0, 0);
  const midQ = withQ ? { ...midRun, branches: midRun.branches.toReversed() } : null;

  const t0 = performance.now();
  tree.insert(ROOT, p, { runs: [rootRun], count });
  if (rootQ) tree.insert(ROOT, q, { runs: [rootQ], count });

  // Expand a 12-level spine along branch 0 with 1024-branch mid nodes
  let node = ROOT;
  for (let d = 0; d < SPINE_DEPTH; d++) {
    node = tree.child(node, 0);
    tree.insert(node, p, { runs: [midRun], count: midRun.branches.length });
    if (midQ) tree.insert(node, q, { runs: [midQ], count: midQ.branches.length });
  }
  const ingestMs = performance.now() - t0;
  const wireBytes = (count + SPINE_DEPTH * midRun.branches.length)
    * (withQ ? 2 : 1) * WIRE_BYTES_PER_BRANCH;
  return { tree, lens, ingestMs, wireBytes };
}

/** Paint only geometry spans (no labels, no DOM) to isolate canvas rect fill cost. */
function paintSpansOnly(frame: Frame, cam: Camera): void {
  ctx.fillStyle = "#faf8f5";
  ctx.fillRect(0, 0, VIEW.width, VIEW.height);
  for (const span of frame.spans) {
    const y0 = Math.floor(span.y0 * VIEW.height);
    const y1 = Math.ceil(span.y1 * VIEW.height);
    const h = y1 - y0;
    if (h <= 0) continue;
    const x0 = Math.max(0, (span.x0 - cam.x) * cam.k);
    const fill = Math.min(VIEW.width, (span.fill - cam.x) * cam.k);
    const reach = Math.min(VIEW.width, (span.reach - cam.x) * cam.k);
    if (x0 < fill) {
      ctx.fillStyle = "#e7e1d8";
      ctx.fillRect(x0, y0, fill - x0, h);
    }
    const outline = Math.max(x0, fill);
    if (outline < reach) {
      ctx.fillStyle = "#5c5449";
      ctx.fillRect(outline, y0, reach - outline, h);
    }
  }
}

interface Scenario {
  name: string;
  count: number;
  withQ: boolean;
  streaming: boolean;
  stepCamera: (i: number, n: number, cam: Camera, client: IntervalTree, lens: Lens) => void;
}

const zoomDive: Scenario["stepCamera"] = (i, n, cam, client, lens) => {
  cam.x = (Math.max(0, i) / Math.max(1, n - 1)) * 16;
  if (0 <= i) cam.anchor = wasmSettle(client.raw, lens.p, lens.q, cam) as Camera["anchor"];
};

const tailPan: Scenario["stepCamera"] = (i, n, cam) => {
  const t = Math.max(0, i) / Math.max(1, n - 1);
  const cdf = 1 - (1 - 0.995 * t) ** 2;
  cam.x = 2.5;
  cam.anchor = { path: [], digits: digits(Math.min(0.9999, cdf)) };
};

const SCENARIOS: Scenario[] = [
  {
    name: "zoom dive (p==q, 150k)",
    count: 150_000,
    withQ: false,
    streaming: false,
    stepCamera: zoomDive,
  },
  {
    name: "tail pan (p==q, 150k)",
    count: 150_000,
    withQ: false,
    streaming: false,
    stepCamera: tailPan,
  },
  {
    name: "tail pan (p!=q, 150k)",
    count: 150_000,
    withQ: true,
    streaming: false,
    stepCamera: tailPan,
  },
  {
    name: "streaming zoom dive (p==q)",
    count: 150_000,
    withQ: false,
    streaming: true,
    stepCamera: zoomDive,
  },
];

const tick = () => new Promise<number>(requestAnimationFrame);

type SessionUpdate =
  | (Omit<Extract<Update, { kind: "answer" }>, "measure"> & { measure: number })
  | { kind: Exclude<Update["kind"], "answer"> };

async function runScenario(scenario: Scenario) {
  const recording = scenario.streaming ? new RecordedTree() : null;
  const client = new IntervalTree();
  const { lens, ingestMs: bulkIngestMs, wireBytes: bulkWireBytes } = buildSourceTree(
    recording ?? client,
    scenario.count,
    scenario.withQ,
  );
  const session = recording === null ? null : new WasmSession();

  const cam: Camera = {
    origin: ROOT,
    x: 0,
    k: 110,
    anchor: { path: [], digits: digits(0.12) },
    a: 0.35,
  };

  const tIngest: number[] = [];
  const tLayout: number[] = [];
  const tTransfer: number[] = [];
  const tPaint: number[] = [];
  const tPick: number[] = [];
  let streamWireBytes = 0;
  let totalBoxes = 0;
  let totalSpans = 0;

  for (let i = -10; i < frames; i++) {
    await tick();
    const frameIdx = Math.max(0, i);
    scenario.stepCamera(i, frames, cam, client, lens);

    // Stage 1: Streaming demand + transfer + client ingest (when streaming)
    let ingestThisFrame = 0;
    if (session && recording) {
      const t0 = performance.now();
      const v = wasmView(client.raw, lens.p, lens.q, cam, VIEW, VIEW.height / ROW_PX);
      session.demand(recording.raw, lens.p, lens.q, v, null);
      const updates = session.answer(recording.raw) as SessionUpdate[];
      for (const u of updates) {
        if (u.kind !== "answer") continue;
        if (u.parent !== null && u.branch !== null) {
          const named = client.child(u.parent, u.branch);
          if (named !== u.node) throw new Error(`the server named node ${u.node} as ${named}`);
        }
        for (const r of u.runs) {
          streamWireBytes += 8 + r.branches.length * WIRE_BYTES_PER_BRANCH + r.top.length;
        }
        client.insert(u.node, u.measure, {
          runs: u.runs,
          count: u.count ?? undefined,
        });
      }
      ingestThisFrame = performance.now() - t0;
    }

    // Stage 2: Pure WASM layout
    const tL0 = performance.now();
    const raw = wasmLayout(client.raw, lens.p, lens.q, cam, VIEW, ROW_PX);
    const dtLayout = performance.now() - tL0;

    // Stage 3: WASM -> JS transfer & hydration
    const tT0 = performance.now();
    const frame = new Frame(raw);
    const dtTransfer = performance.now() - tT0;

    // Stage 4: Canvas geometry paint (no text/DOM)
    const tP0 = performance.now();
    paintSpansOnly(frame, cam);
    const dtPaint = performance.now() - tP0;

    // Stage 5: Pick
    const px = VIEW.width * (0.2 + ((frameIdx * 37) % 60) / 100);
    const py = VIEW.height * (0.1 + ((frameIdx * 53) % 80) / 100);
    const tK0 = performance.now();
    frame.raw.pick(client.raw, lens.p, lens.q, cam, VIEW, px, py, 1);
    const dtPick = performance.now() - tK0;

    if (0 <= i) {
      tIngest.push(ingestThisFrame);
      tLayout.push(dtLayout);
      tTransfer.push(dtTransfer);
      tPaint.push(dtPaint);
      tPick.push(dtPick);
      totalBoxes += frame.boxes.length;
      totalSpans += frame.spans.length;
    }
    frame.dispose();
  }

  return {
    name: scenario.name,
    frames,
    bulkIngestMs,
    wireKB: (scenario.streaming ? streamWireBytes : bulkWireBytes) / 1024,
    ingest: stats(tIngest),
    layout: stats(tLayout),
    transfer: stats(tTransfer),
    paint: stats(tPaint),
    pick: stats(tPick),
    avgBoxes: totalBoxes / frames,
    avgSpans: totalSpans / frames,
  };
}

button.addEventListener("click", async () => {
  button.disabled = true;
  try {
    if (!Number.isSafeInteger(frames) || frames < 1) {
      throw new Error("frames must be a positive integer");
    }
    const results = [];
    for (const s of SCENARIOS) {
      output.textContent = `Running ${s.name}…`;
      results.push(await runScenario(s));
    }
    const report = { label, userAgent: navigator.userAgent, view: VIEW, dpr: devicePixelRatio, results };
    const saved = await fetch("/result", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(report),
    });
    if (!saved.ok) throw new Error(`saving report failed (${saved.status})`);
    output.textContent =
      `Done · ${label} (stage p50 and bulk ingest in ms; estimated wire KB)\n` +
      results
        .map(
          (r) =>
            `${r.name.padEnd(28)} | lay ${r.layout.p50.toFixed(3)} | xfer ${r.transfer.p50.toFixed(3)} | paint ${r.paint.p50.toFixed(3)} | pick ${r.pick.p50.toFixed(3)} | ing ${r.ingest.p50.toFixed(3)} | bulk ${r.bulkIngestMs.toFixed(2)} | wire ${r.wireKB.toFixed(0)} KB`,
        )
        .join("\n");
  } catch (err) {
    output.textContent = `Failed: ${String(err)}`;
  }
  button.disabled = false;
});

output.textContent = "Ready";
button.disabled = false;
