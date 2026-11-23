//! Each Criterion iteration lays out the same 180-frame camera trajectory.
//! Use the PStore bench commands from the README with `--bench qlayout`.

use std::hint::black_box;

#[cfg(not(target_arch = "wasm32"))]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_bench, wasm_bindgen_test_configure, Criterion};

use natlas_core::qlayout::{layout_frame, QFrame};

#[path = "../examples/support/qlayout.rs"]
mod workload;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_dedicated_worker);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_bench)]
fn layout_trajectory(criterion: &mut Criterion) {
    let trees = [workload::build_tree(false), workload::build_tree(true)];
    for (motion, lens, shuffled) in workload::CASES {
        let tree = &trees[usize::from(shuffled)];
        let fixture = if shuffled { "shuffled" } else { "heaviest" };
        let relation = if lens.p == lens.q {
            "same_q"
        } else {
            "distinct_q"
        };
        let name = format!(
            "qlayout/{fixture}/{}/{relation}/FastTree/{}",
            motion.name(),
            workload::VOCAB,
        );
        let viewports = motion.viewports(workload::FRAMES);
        let mut frame = QFrame::default();
        criterion.bench_function(&name, |bencher| {
            bencher.iter(|| {
                for &view in black_box(&viewports) {
                    layout_frame(tree, lens, view, &mut frame);
                    black_box(&frame);
                }
            });
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(benches, layout_trajectory);
#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);

#[cfg(target_arch = "wasm32")]
fn main() {}
