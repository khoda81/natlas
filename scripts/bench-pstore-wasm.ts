// Run `bun run core && bun scripts/bench-pstore-wasm.ts` to compare
// BTreeMax, EytzingerMax, and TieredPStore inside WebAssembly.

import init, { PStoreBench } from "../src/core/generated/natlas_core";

await init({
  module_or_path: new URL("../src/core/generated/natlas_core_bg.wasm", import.meta.url),
});

const VOCAB = 150_000;
const QUERIES = 256;
const REPS = 20;

const DISTS = [
  [0, "Zipf-150k (Heaviest order)"],
  [1, "Zipf-150k (Shuffled / arbitrary order)"],
  [2, "Uniform-150k (All sub-pixel pathological)"],
] as const;

for (const [distKind, label] of DISTS) {
  console.log(`=== ${label} (WASM) ===`);
  for (const implId of [0, 1, 2]) {
    // The WASM constructor also generates the workload; the native build timer
    // starts after that setup.
    const tBuild = performance.now();
    const bench = new PStoreBench(implId, distKind, VOCAB, QUERIES);
    const buildMs = performance.now() - tBuild;

    // Warmup
    bench.run_query_tall(2);

    const tQt = performance.now();
    const hits = bench.run_query_tall(REPS);
    const qtUs = ((performance.now() - tQt) * 1e3) / (REPS * QUERIES);

    const tLoc = performance.now();
    bench.run_locate(REPS * 10);
    const locNs = ((performance.now() - tLoc) * 1e6) / (REPS * 10 * QUERIES);

    const tCb = performance.now();
    bench.run_contains_boundary(REPS * 10);
    const cbNs = ((performance.now() - tCb) * 1e6) / (REPS * 10 * QUERIES);

    const name = bench.name().padEnd(22);
    const ramKb = (bench.memory_bytes() / 1024).toFixed(1).padStart(6);
    const avgHits = (hits / (REPS * QUERIES)).toFixed(1).padStart(5);
    console.log(
      `${name} | build ${buildMs.toFixed(2).padStart(6)}ms | RAM ${ramKb} KB | query_tall ${qtUs.toFixed(1).padStart(7)}us (avg ${avgHits} hits) | locate ${locNs.toFixed(0).padStart(6)}ns | boundary ${cbNs.toFixed(0).padStart(5)}ns`,
    );
    bench.free();
  }
  console.log();
}
