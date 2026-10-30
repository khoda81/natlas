import { PartitionServer } from "./server";
import { groups, silhouette, Simulation, statistics } from "./simulation";
import { reference, type Capture } from "./reference";
import type { ToWorker, FromWorker, Run } from "./messages";

const scope = self as unknown as { onmessage: (event: MessageEvent<ToWorker>) => void; postMessage(message: FromWorker): void };
let vectors: Float64Array[] = [], servers: PartitionServer[] = [], generation = 0;
let lastOrder = "";
const send = (message: FromWorker) => scope.postMessage(message);

async function load(base: string) {
  const response = await fetch(new URL("manifest.json", base));
  if (!response.ok) throw Error(`capture manifest: ${response.status}`);
  const text = await response.text(), capture: Capture = JSON.parse(text);
  let bootstrapBytes = new TextEncoder().encode(text).length;
  vectors = await Promise.all(capture.measures.map(async (measure) => {
    const response = await fetch(new URL(measure.file, base));
    if (!response.ok) throw Error(`capture ${measure.file}: ${response.status}`);
    const compressed = await response.arrayBuffer();
    bootstrapBytes += compressed.byteLength;
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", compressed))]
      .map((b) => b.toString(16).padStart(2, "0")).join("");
    if (digest !== measure.sha256) throw Error(`capture checksum mismatch: ${measure.file}`);
    const buffer = await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    if (buffer.byteLength !== capture.vocabulary * 8) throw Error("capture vocabulary mismatch");
    // DataView specifies the fixture's endianness independently of the host.
    const view = new DataView(buffer);
    return Float64Array.from({ length: capture.vocabulary }, (_, i) => view.getFloat64(i * 8, true));
  }));
  servers = vectors.map((logs) => new PartitionServer(logs));
  send({ type: "ready", capture, bootstrapBytes });
}

async function run(request: Run, token: number) {
  if (!servers[request.pIndex] || !servers[request.qIndex]) throw Error("capture not loaded");
  // Different bit orders create different cache trees; bound retained memory
  // by keeping only the currently explored order/fanout's server caches.
  const order = `${request.budget.order}:${request.budget.fanout}`;
  if (lastOrder && lastOrder !== order) servers = vectors.map((logs) => new PartitionServer(logs));
  lastOrder = order;
  const started = performance.now();
  const sim = new Simulation(vectors[0]!.length, [servers[request.pIndex]!, servers[request.qIndex]!], request.budget);
  const finished = await sim.run(async () => {
    send({ type: "progress", id: request.id, p: sim.pm.requests, q: sim.qm.requests });
    await new Promise((resolve) => setTimeout(resolve, 0));
    return token === generation;
  });
  if (!finished || token !== generation) return;
  const all = groups(sim), shape = silhouette(all, 1536, request.start, request.end);
  const truth = reference(vectors[request.pIndex]!, vectors[request.qIndex]!, 1536, request.start, request.end);
  const top = [...all].sort((a, b) => b.upper - a.upper).slice(0, 14);
  send({ ...request, type: "result", p: sim.pm, q: sim.qm, groups: top, stats: statistics(sim, all),
    lower: shape.lower, upper: shape.upper, reference: truth, milliseconds: performance.now() - started,
    exchanges: [...sim.pm.last.map((e) => ({ ...e, side: "P" as const })), ...sim.qm.last.map((e) => ({ ...e, side: "Q" as const }))] });
}

scope.onmessage = (event) => {
  const request = event.data;
  const token = ++generation;
  const work = request.type === "load" ? load(request.base) : run(request, token);
  work.catch((error) => send({ type: "error", message: String(error), ...(request.type === "run" ? { id: request.id } : {}) }));
};
