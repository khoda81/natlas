import type { Capture } from "./reference";
import type { FromWorker, Snapshot, ToWorker } from "./messages";
import type { Budgets, Meter } from "./simulation";

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const select = (id: string) => el<HTMLSelectElement>(id);
const input = (id: string) => el<HTMLInputElement>(id);
const put = (id: string, text: string) => { el(id).textContent = text; };
const fmt = (value: number, digits = 3) => Number.isFinite(value) ? value.toLocaleString("en", { maximumFractionDigits: digits }) : value === Infinity ? "∞" : "−∞";
const bytes = (value: number) => value < 1024 ? `${value} B` : value < 1048576 ? `${fmt(value / 1024, 1)} KiB` : `${fmt(value / 1048576, 2)} MiB`;
const hex = (value: number) => value.toString(16).padStart(8, "0");
const interval = (lo: number, hi: number, digits = 3) => `${fmt(lo, digits)} – ${fmt(hi, digits)}`;
const worker = new Worker("/playground-worker.js", { type: "module" });
const send = (message: ToWorker) => worker.postMessage(message);
let capture: Capture | undefined, snapshot: Snapshot | undefined, labels: string[] = [];
let sequence = 0, timer = 0, dirty = true, pointer: { x: number; y: number } | undefined;
const canvas = el<HTMLCanvasElement>("plot"), context = canvas.getContext("2d")!;
const params = new URLSearchParams(location.search);
const budget: Budgets = { p: 65536, q: 65536, unit: "bytes", order: "high", fanout: 4, clip: 12 };

function choose(id: string, parameter: string) {
  const value = params.get(parameter);
  if (value !== null && [...select(id).options].some((option) => option.value === value)) select(id).value = value;
}
for (const id of ["unit", "order", "fanout", "clip", "window"]) choose(id, id);
budget.unit = select("unit").value as Budgets["unit"];
budget.order = select("order").value as Budgets["order"];
budget.fanout = Number(select("fanout").value);
budget.clip = Number(select("clip").value);
const maximum = () => 2 ** (budget.unit === "bytes" ? 24 : 18);
const clamp = (value: number) => Math.max(0, Math.min(maximum(), Math.floor(Number.isFinite(value) ? value : 0)));
for (const side of ["p", "q"] as const) {
  const value = params.get(side);
  budget[side] = value !== null ? clamp(Number(value)) : budget.unit === "bytes" ? 65536 : 64;
}
input("reference").checked = params.get("reference") !== "0";

function syncCaps(preserveSlider?: string, preserveInput?: string) {
  for (const side of ["p", "q"] as const) {
    if (side !== preserveInput) input(`${side}-cap`).value = String(budget[side]);
    input(`${side}-cap`).max = String(maximum());
    // Several low slider positions round to the same integer budget. Keep
    // the thumb's position so keyboard steps can still leave zero.
    if (side !== preserveSlider) input(`${side}-slider`).value = String(Math.log2(budget[side] + 1) / Math.log2(maximum() + 1) * 240);
    input(`${side}-slider`).setAttribute("aria-valuetext", `${fmt(budget[side], 0)} ${budget.unit}`);
  }
  document.querySelectorAll(".budget-unit").forEach((node) => { node.textContent = budget.unit; });
  document.querySelectorAll(".range-max").forEach((node) => { node.textContent = budget.unit === "bytes" ? bytes(maximum()) : `${fmt(maximum(), 0)} requests`; });
}

function persist() {
  const values = { ...budget, pm: select("p-model").value, qm: select("q-model").value,
    window: select("window").value, reference: input("reference").checked ? 1 : 0 };
  const query = new URLSearchParams(Object.entries(values).map(([key, value]) => [key, String(value)]));
  history.replaceState(null, "", `${location.pathname}?${query}`);
}

function queue() {
  if (!capture) return;
  ++sequence; // Invalidate old results immediately, including during debounce.
  window.clearTimeout(timer);
  el("status").classList.remove("error");
  put("status", "Replaying from zero with the new limits…");
  document.body.classList.add("loading");
  el<HTMLButtonElement>("export").disabled = true;
  el("tooltip").hidden = true;
  persist();
  timer = window.setTimeout(() => send({ type: "run", id: sequence, budget: { ...budget },
    pIndex: Number(select("p-model").value), qIndex: Number(select("q-model").value),
    start: Number(select("window").value), end: 1 }), 55);
}

function error(message: string) {
  document.body.classList.remove("loading");
  el("status").classList.add("error");
  put("status", message);
}

async function loadLabels(file: string) {
  try {
    const response = await fetch(`/playground/qwen3-4b/${file}`);
    if (!response.ok) throw Error(String(response.status));
    const buffer = await response.arrayBuffer();
    labels = await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
  } catch { /* Labels are optional oracle annotations, never inference inputs. */ }
}

worker.onmessage = (event: MessageEvent<FromWorker>) => {
  const message = event.data;
  if (message.type === "error") {
    if (message.id === undefined || message.id === sequence) error(message.message);
  } else if (message.type === "ready") {
    capture = message.capture;
    put("prompt", capture.prompt);
    put("vocabulary", `${fmt(capture.vocabulary, 0)} token IDs`);
    for (const side of ["p", "q"] as const) {
      const node = select(`${side}-model`);
      capture.measures.forEach((measure, index) => node.add(new Option(measure.name, String(index))));
      node.value = side === "p" ? "0" : "1";
      choose(`${side}-model`, `${side}m`);
      node.disabled = false;
    }
    el<HTMLButtonElement>("swap").disabled = false;
    el<HTMLButtonElement>("same").disabled = false;
    put("baseline", `Full vector per endpoint: ${bytes(capture.vocabulary * 4)} at float32, ${bytes(capture.vocabulary * 8)} at float64. No token IDs needed in ID order.`);
    put("provenance", `${capture.note} Source: ${capture.source} @ ${capture.source_revision}. Vocabulary SHA-256: ${capture.vocabulary_sha256}. Capture format: ${capture.format}. Simulation bootstrap: ${bytes(message.bootstrapBytes)} for the compressed vectors and manifest, plus optional labels; excluded from the sliders.`);
    void loadLabels(capture.labels);
    queue();
  } else if (message.id === sequence && message.type === "progress") {
    put("status", `Replaying… P ${fmt(message.p, 0)} requests · Q ${fmt(message.q, 0)} requests`);
  } else if (message.id === sequence && message.type === "result") {
    snapshot = message;
    document.body.classList.remove("loading");
    el<HTMLButtonElement>("export").disabled = false;
    renderResult(message);
    dirty = true;
  }
};
worker.onerror = (event) => error(`Playground worker failed: ${event.message}`);

function renderMeter(side: "p" | "q", meter: Meter, result: Snapshot) {
  const total = meter.sent + meter.received, cap = result.budget[side];
  const used = result.budget.unit === "bytes" ? total : meter.requests;
  put(`${side}-used`, `${bytes(total)} · ${fmt(meter.requests, 0)} ${meter.requests === 1 ? "request" : "requests"}`);
  put(`${side}-direction`, `↑ ${fmt(meter.sent, 0)} B request · ↓ ${fmt(meter.received, 0)} B reply`);
  el<HTMLProgressElement>(`${side}-progress`).value = cap > 0 ? used / cap : 0;
  const reason = meter.stopped === "complete" ? side === "q" ? "No useful Q splits at this P partition and width." : "Every positive token mass resolved."
    : meter.nextBytes ? `Next complete exchange needs ${fmt(meter.nextBytes, 0)} B; ${fmt(cap - total, 0)} B left.`
    : "Request limit reached.";
  put(`${side}-state`, reason);
}

function row(values: string[]) {
  const tr = document.createElement("tr");
  for (const value of values) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
  return tr;
}

function renderResult(result: Snapshot) {
  renderMeter("p", result.p, result); renderMeter("q", result.q, result);
  put("cross", interval(result.stats.lower, result.stats.upper, 4));
  put("gap", `${fmt(result.stats.gap, 4)} nats`);
  put("resolved", `${fmt(100 * result.stats.singletonMass, 2)}%`);
  put("group-count", `${fmt(result.stats.pGroups, 0)} P · ${fmt(result.stats.qGroups, 0)} Q`);
  put("chain", `Common-partition chain-rule lower term: ${fmt(result.stats.chain, 6)} nats. The cross-entropy lower bound also uses individual-probability extrema.`);
  el("groups").replaceChildren(...result.groups.map((g) => row([
    `${hex(g.mask)} / ${hex(g.value)}`, fmt(g.count, 0), `${fmt(100 * g.lower, 6)}%`, interval(g.a, g.b, 5),
  ])));
  el("ledger").replaceChildren(...result.exchanges.map((e) => row([
    e.side, `${hex(e.mask)} / ${hex(e.value)}`, String(e.requestBytes), String(e.responseBytes),
  ])));
  const total = result.p.sent + result.p.received + result.q.sent + result.q.received;
  put("status", `${bytes(total)} admitted across both endpoints · ${fmt(result.milliseconds, 0)} ms local replay · simulated payload traffic, not network latency`);
  renderReference();
}

function renderReference() {
  const visible = input("reference").checked;
  el("reference-readout").hidden = !visible;
  if (snapshot) put("reference-readout", `Full-data reference only: H(P,Q) = ${fmt(snapshot.reference.cross, 6)} nats · H(P) = ${fmt(snapshot.reference.entropy, 6)} · KL(P∥Q) = ${fmt(snapshot.reference.kl, 6)}. Not used by the estimator.`);
  dirty = true;
}

for (const side of ["p", "q"] as const) {
  input(`${side}-slider`).addEventListener("input", () => {
    budget[side] = clamp(Math.round(2 ** (Number(input(`${side}-slider`).value) / 240 * Math.log2(maximum() + 1)) - 1));
    syncCaps(side); queue();
  });
  input(`${side}-cap`).addEventListener("input", () => {
    budget[side] = clamp(Number(input(`${side}-cap`).value)); syncCaps(undefined, side); queue();
  });
  input(`${side}-cap`).addEventListener("change", () => syncCaps());
  select(`${side}-model`).addEventListener("change", queue);
}
select("unit").addEventListener("change", () => {
  budget.unit = select("unit").value as Budgets["unit"];
  for (const side of ["p", "q"] as const) {
    const meter = snapshot?.[side];
    budget[side] = meter ? clamp(budget.unit === "requests" ? meter.requests : meter.sent + meter.received) : budget.unit === "bytes" ? 65536 : 64;
  }
  syncCaps(); queue();
});
select("order").addEventListener("change", () => { budget.order = select("order").value as Budgets["order"]; queue(); });
select("fanout").addEventListener("change", () => { budget.fanout = Number(select("fanout").value); queue(); });
select("clip").addEventListener("change", () => { budget.clip = Number(select("clip").value); queue(); });
select("window").addEventListener("change", queue);
input("reference").addEventListener("change", () => { renderReference(); persist(); });
el("swap").addEventListener("click", () => {
  const p = select("p-model").value; select("p-model").value = select("q-model").value; select("q-model").value = p; queue();
});
el("same").addEventListener("click", () => { select("q-model").value = select("p-model").value; queue(); });
el("zero").addEventListener("click", () => { budget.p = budget.q = 0; syncCaps(); queue(); });
el("example").addEventListener("click", () => { budget.p = budget.q = budget.unit === "bytes" ? 65536 : 64; syncCaps(); queue(); });
el("export").addEventListener("click", () => {
  if (!snapshot) return;
  const { lower, upper, reference, ...result } = snapshot;
  const payload = { capture, ...result, envelope: { lower: [...lower], upper: [...upper] },
    oracle: { crossEntropy: reference.cross, entropy: reference.entropy, kl: reference.kl },
    accounting: "Simulated uncompressed protobuf request + response bytes; excludes bootstrap, HTTP and TLS." };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, (_, value) => typeof value === "number" && !Number.isFinite(value) ? String(value) : value, 2)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = "natlas-partition-replay.json"; link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
});

const margins = { left: 57, right: 20, top: 18, bottom: 40 };
function draw() {
  const width = canvas.clientWidth, height = canvas.clientHeight, scale = window.devicePixelRatio || 1;
  const w = Math.round(width * scale), h = Math.round(height * scale);
  // Size changes clear the bitmap: resize and redraw together in this frame.
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = margins.left, top = margins.top, plotW = width - left - margins.right, plotH = height - top - margins.bottom;
  if (plotW <= 0 || plotH <= 0) return;
  const limit = snapshot?.budget.clip ?? budget.clip;
  const x = (value: number) => left + Math.max(0, Math.min(limit, value)) / limit * plotW;
  context.fillStyle = "#0d161b"; context.fillRect(left, top, plotW, plotH);
  context.font = "10px ui-monospace, monospace";
  context.textBaseline = "middle";
  for (let tick = 0; tick <= 6; tick++) {
    const value = limit * tick / 6, xx = x(value);
    context.strokeStyle = "#24343c"; context.lineWidth = 1; context.beginPath(); context.moveTo(xx, top); context.lineTo(xx, top + plotH); context.stroke();
    context.fillStyle = "#a4bbc3"; context.textAlign = "center"; context.fillText(fmt(value, 1), xx, top + plotH + 15);
  }
  const start = snapshot?.start ?? Number(select("window").value), end = snapshot?.end ?? 1;
  for (let tick = 0; tick <= 4; tick++) {
    const y = top + tick / 4 * plotH, mass = 100 * (start + tick / 4 * (end - start));
    context.strokeStyle = "#24343c"; context.beginPath(); context.moveTo(left, y); context.lineTo(left + plotW, y); context.stroke();
    context.fillStyle = "#a4bbc3"; context.textAlign = "right"; context.fillText(`${fmt(mass, start >= .999 ? 3 : start >= .99 ? 2 : 1)}%`, left - 9, y);
  }
  context.textAlign = "right"; context.fillStyle = "#819ba6"; context.fillText("−ln Q →  (nats)", left + plotW, height - 5);
  context.textAlign = "left"; context.fillText("P ↓", 4, 7);
  const result = snapshot;
  if (!result) return;
  const rows = result.lower.length;
  const path = (lo: Float64Array | undefined, hi: Float64Array) => {
    context.beginPath(); context.moveTo(lo ? x(lo[0]!) : left, top);
    for (let i = 0; i < rows; i++) { const y = top + i / rows * plotH; context.lineTo(x(hi[i]!), y); context.lineTo(x(hi[i]!), top + (i + 1) / rows * plotH); }
    for (let i = rows - 1; i >= 0; i--) { context.lineTo(lo ? x(lo[i]!) : left, top + (i + 1) / rows * plotH); context.lineTo(lo ? x(lo[i]!) : left, top + i / rows * plotH); }
    context.closePath();
  };
  path(undefined, result.lower); context.fillStyle = "#286a64b8"; context.fill();
  path(result.lower, result.upper); context.fillStyle = "#b58b3d23"; context.fill();
  context.save(); context.clip(); context.strokeStyle = "#d3b27142"; context.lineWidth = 1;
  context.beginPath();
  for (let step = -plotH; step < plotW + plotH; step += 7) { context.moveTo(left + step, top); context.lineTo(left + step - plotH, top + plotH); }
  context.stroke(); context.restore();
  const curve = (values: Float64Array, color: string, dash: number[] = []) => {
    context.strokeStyle = color; context.lineWidth = dash.length ? 1.5 : 1.2; context.setLineDash(dash); context.beginPath();
    for (let i = 0; i < rows; i++) {
      const xx = x(values[i]!), yy = top + i / rows * plotH;
      if (i === 0) context.moveTo(xx, yy); else context.lineTo(xx, yy);
      context.lineTo(xx, top + (i + 1) / rows * plotH);
    }
    context.stroke(); context.setLineDash([]);
  };
  curve(result.lower, "#6ccfba"); curve(result.upper, "#d4b272");
  if (input("reference").checked) curve(result.reference.widths, "#f0f7f7", [4, 3]);
  const clipped = result.upper[rows - 1]! > limit;
  if (clipped) { context.fillStyle = "#d4b272"; context.textAlign = "right"; context.fillText("upper bound continues →", left + plotW - 8, top + 12); }
  if (pointer && !document.body.classList.contains("loading")) {
    const py = Math.max(top, Math.min(top + plotH, pointer.y));
    const index = Math.min(rows - 1, Math.max(0, Math.floor((py - top) / plotH * rows)));
    context.strokeStyle = "#c2d5dc70"; context.setLineDash([2, 3]); context.beginPath(); context.moveTo(left, py); context.lineTo(left + plotW, py); context.stroke(); context.setLineDash([]);
    const tip = el("tooltip"), lines: HTMLElement[] = [];
    const line = (text: string, tag = "div") => { const node = document.createElement(tag); node.textContent = text; lines.push(node); };
    line(`P quantile ${fmt(100 * (start + (end - start) * (index + .5) / rows), 4)}%`, "b");
    line(`Width bound ${interval(result.lower[index]!, result.upper[index]!, 5)} nats`);
    if (input("reference").checked) {
      const id = result.reference.ids[index]!;
      line(`Reference only · token ${id}`, "span");
      line(labels[id] === undefined ? "Label loading…" : JSON.stringify(labels[id]), "b");
      line(`P ${fmt(result.reference.pAt[index]!, 8)} · Q ${fmt(result.reference.qAt[index]!, 8)}`);
      line(`Exact width ${fmt(result.reference.widths[index]!, 5)} nats`);
    }
    tip.replaceChildren(...lines); tip.hidden = false;
    tip.style.left = `${Math.max(12, Math.min(width - tip.offsetWidth - 4, pointer.x + 22))}px`;
    tip.style.top = `${Math.max(8, Math.min(height - tip.offsetHeight - 3, pointer.y + 12))}px`;
  }
}

canvas.addEventListener("pointermove", (event) => { const rect = canvas.getBoundingClientRect(); pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top }; dirty = true; });
canvas.addEventListener("pointerleave", () => { pointer = undefined; el("tooltip").hidden = true; dirty = true; });
canvas.addEventListener("keydown", (event) => {
  if (event.key === "Escape") { pointer = undefined; el("tooltip").hidden = true; dirty = true; return; }
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  event.preventDefault();
  pointer = { x: canvas.clientWidth / 2, y: Math.max(margins.top, Math.min(canvas.clientHeight - margins.bottom, (pointer?.y ?? margins.top) + (event.key === "ArrowDown" ? 6 : -6))) };
  dirty = true;
});
canvas.addEventListener("blur", () => { pointer = undefined; el("tooltip").hidden = true; dirty = true; });
new ResizeObserver(() => { dirty = true; }).observe(canvas);
function frame() { if (dirty) { dirty = false; draw(); } requestAnimationFrame(frame); }
syncCaps(); renderReference(); requestAnimationFrame(frame);
send({ type: "load", base: new URL("/playground/qwen3-4b/", location.href).href });
