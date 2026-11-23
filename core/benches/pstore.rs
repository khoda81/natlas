//! PStore construction and 256-query batches, timed with the same Rust workload
//! on native x86 and browser Wasm.

use std::hint::black_box;

#[cfg(not(target_arch = "wasm32"))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_bench, wasm_bindgen_test_configure, Criterion};

use natlas_core::pstore::{
    bench_logits, bench_queries, verify_equivalence, BTreeMax, CdfRange, EytzingerMax, PStore,
    QueryScratch, TieredPStore,
};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_dedicated_worker);

fn bench_store<S: PStore>(
    criterion: &mut Criterion,
    workload: &str,
    logits: &[f32],
    queries: &[(CdfRange, u64)],
) {
    let name = format!("{workload}/{}", S::name());
    criterion.bench_function(&format!("{name}/build_and_drop"), |b| {
        b.iter(|| drop(black_box(S::from_logits(black_box(logits)))));
    });

    let store = S::from_logits(logits);
    let mut scratch = QueryScratch::default();
    let mut hits = Vec::new();
    criterion.bench_function(&format!("{name}/query_tall_256"), |b| {
        b.iter(|| {
            for &(range, min_mass) in black_box(queries) {
                store.query_tall(range, min_mass, &mut scratch, &mut hits);
                black_box(&hits);
            }
        });
    });
    criterion.bench_function(&format!("{name}/locate_256"), |b| {
        b.iter(|| {
            for &(range, _) in black_box(queries) {
                black_box(store.locate(range.midpoint()));
            }
        });
    });
    criterion.bench_function(&format!("{name}/contains_boundary_256"), |b| {
        b.iter(|| {
            for &(range, min_mass) in black_box(queries) {
                black_box(store.contains_boundary(range.first_pixel(min_mass)));
            }
        });
    });
}

fn bench_workload(criterion: &mut Criterion, kind: u32, label: &str) {
    let queries = bench_queries(256);
    let logits = bench_logits(kind, 150_000);
    verify_equivalence(&logits).expect("all PStore designs must match");
    bench_store::<BTreeMax>(criterion, label, &logits, &queries);
    bench_store::<EytzingerMax>(criterion, label, &logits, &queries);
    bench_store::<TieredPStore>(criterion, label, &logits, &queries);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_bench)]
fn zipf_heaviest(criterion: &mut Criterion) {
    bench_workload(criterion, 0, "zipf-heaviest-150k");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_bench)]
fn zipf_shuffled(criterion: &mut Criterion) {
    bench_workload(criterion, 1, "zipf-shuffled-150k");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_bench)]
fn uniform(criterion: &mut Criterion) {
    bench_workload(criterion, 2, "uniform-150k");
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(benches, zipf_heaviest, zipf_shuffled, uniform);
#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);

// Cargo uses the custom benchmark harness on both targets. For Wasm,
// wasm-bindgen-test-runner calls the exported benchmark directly.
#[cfg(target_arch = "wasm32")]
fn main() {}
