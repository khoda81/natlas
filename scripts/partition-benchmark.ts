// Bounded CPU-only replay over the checked-in Qwen captures.
import { PartitionServer } from "../src/playground/server";
import { groups, silhouette, Simulation, statistics } from "../src/playground/simulation";
import { reference } from "../src/playground/reference";

const root = new URL("../public/playground/qwen3-4b/", import.meta.url);
const vectors = await Promise.all([0, 1].map(async (i) => {
  const bytes = Bun.gunzipSync(new Uint8Array(await Bun.file(new URL(`${i}.f64.gz`, root)).arrayBuffer()));
  return new Float64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}));
const truth = reference(vectors[0]!, vectors[1]!, 1024);
const results = [];
for (const order of ["high", "low"] as const) {
  const servers: [PartitionServer, PartitionServer] = [new PartitionServer(vectors[0]!), new PartitionServer(vectors[1]!)];
  for (const cap of [0, 8, 64, 512]) {
    const started = performance.now();
    const sim = new Simulation(vectors[0]!.length, servers, { p: cap, q: cap, unit: "requests", order, fanout: 4, clip: 24 });
    await sim.run();
    const cells = groups(sim), stats = statistics(sim, cells), shape = silhouette(cells, 1024);
    for (let i = 0; i < 1024; i++) {
      if (shape.lower[i]! > truth.widths[i]! + 1e-8 || shape.upper[i]! < truth.widths[i]! - 1e-8)
        throw Error(`oracle escaped ${order}/${cap} silhouette at row ${i}`);
    }
    if (stats.lower > truth.cross + 1e-8 || stats.upper < truth.cross - 1e-8) throw Error("cross-entropy escaped");
    results.push({ order, cap, pBytes: sim.pm.sent + sim.pm.received, qBytes: sim.qm.sent + sim.qm.received,
      pRequests: sim.pm.requests, qRequests: sim.qm.requests, groups: stats.pGroups,
      lower: stats.lower, upper: Number.isFinite(stats.upper) ? stats.upper : "infinity", chain: stats.chain,
      clippedGap: stats.gap, singletonMass: stats.singletonMass, milliseconds: Math.round(performance.now() - started) });
  }
}
console.log(JSON.stringify({ vocabulary: vectors[0]!.length, exactCrossEntropy: truth.cross, exactKL: truth.kl,
  note: "Protobuf request+response payload bytes; no HTTP or fixture bootstrap. Timings are local CPU replay, with server caches reused within each bit-order sweep.", results }, null, 2));
