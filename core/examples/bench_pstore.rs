//! Native x86 benchmark comparing `BTreeMax`, `EytzingerMax`, and `TieredPStore`.
//!
//! Run with:
//! ```bash
//! cargo run --release --manifest-path core/Cargo.toml --no-default-features --example bench_pstore
//! ```

use std::hint::black_box;
use std::time::Instant;

use natlas_core::pstore::{
    bench_logits, bench_queries, verify_equivalence, BTreeMax, CdfRange, EytzingerMax, PStore,
    QueryScratch, TieredPStore,
};

const VOCAB: usize = 150_000;
const QUERIES: usize = 256;
const REPS: usize = 20;

fn bench_impl<S: PStore>(logits: &[f32], queries: &[(CdfRange, u64)]) {
    let t_build = Instant::now();
    let store = S::from_logits(logits);
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

    let mut scratch = QueryScratch::default();
    let mut out = Vec::with_capacity(1024);

    // Warmup
    for &(range, min_mass) in queries.iter().take(16) {
        store.query_tall(range, min_mass, &mut scratch, &mut out);
        black_box(&out);
    }

    // 1. query_tall
    let t_qt = Instant::now();
    let mut total_hits = 0usize;
    for _ in 0..REPS {
        for &(range, min_mass) in queries {
            store.query_tall(range, min_mass, &mut scratch, &mut out);
            total_hits += out.len();
            black_box(&out);
        }
    }
    let qt_ns = t_qt.elapsed().as_nanos() as f64 / (REPS * queries.len()) as f64;

    // 2. locate (point query at midpoint of each range)
    let t_loc = Instant::now();
    let mut loc_acc = 0u64;
    for _ in 0..(REPS * 10) {
        for &(range, _) in queries {
            let mid = range.midpoint();
            if let Some(hit) = store.locate(mid) {
                loc_acc = loc_acc.wrapping_add(u64::from(hit.branch));
            }
        }
    }
    black_box(loc_acc);
    let loc_ns = t_loc.elapsed().as_nanos() as f64 / (REPS * 10 * queries.len()) as f64;

    // 3. contains_boundary (1px-tall slices)
    let t_cb = Instant::now();
    let mut cb_acc = 0u64;
    for _ in 0..(REPS * 10) {
        for &(range, min_mass) in queries {
            let px = range.first_pixel(min_mass);
            cb_acc += u64::from(store.contains_boundary(px));
        }
    }
    black_box(cb_acc);
    let cb_ns = t_cb.elapsed().as_nanos() as f64 / (REPS * 10 * queries.len()) as f64;

    println!(
        "{:<22} | build {:>6.2}ms | RAM {:>6.1} KB | query_tall {:>7.1}us (avg {:>5.1} hits) | locate {:>6.0}ns | boundary {:>5.0}ns",
        S::name(),
        build_ms,
        store.memory_bytes() as f64 / 1024.0,
        qt_ns / 1e3,
        total_hits as f64 / (REPS * queries.len()) as f64,
        loc_ns,
        cb_ns,
    );
}

fn main() {
    let queries = bench_queries(QUERIES);
    for (kind, label) in [
        (0u32, "Zipf-150k (Heaviest order)"),
        (1u32, "Zipf-150k (Shuffled / arbitrary order)"),
        (2u32, "Uniform-150k (All sub-pixel pathological)"),
    ] {
        let logits = bench_logits(kind, VOCAB);
        verify_equivalence(&logits).expect("all PStore designs must match");
        println!("=== {label} ===");
        bench_impl::<BTreeMax>(&logits, &queries);
        bench_impl::<EytzingerMax>(&logits, &queries);
        bench_impl::<TieredPStore>(&logits, &queries);
        println!();
    }
}
