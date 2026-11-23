//! Comparative layout timings on the packed benchmark's camera trajectories.

use std::hint::black_box;
use std::time::{Duration, Instant};

use natlas_core::client::ClientTree;
use natlas_core::layout::{layout_into, Camera, Frame};
use natlas_core::tree::{Lens, ROOT};

#[path = "support/qlayout.rs"]
mod workload;

fn bench_trajectory(name: &str, tree: &ClientTree, lens: Lens, viewports: &[Camera]) {
    let mut frame = Frame::default();
    for camera in viewports.iter().take(10) {
        frame = layout_into(
            tree,
            lens,
            camera,
            workload::VIEW,
            1.0,
            std::mem::take(&mut frame),
        )
        .unwrap();
        black_box(&frame);
    }

    let mut times = Vec::with_capacity(viewports.len());
    let mut total_boxes = 0;
    let mut total_spans = 0;
    for camera in viewports {
        let start = Instant::now();
        frame = layout_into(
            tree,
            lens,
            camera,
            workload::VIEW,
            1.0,
            std::mem::take(&mut frame),
        )
        .unwrap();
        times.push(start.elapsed());
        total_boxes += frame.boxes.len();
        total_spans += frame.spans.len();
        black_box(&frame);
    }

    times.sort_unstable();
    let n = times.len();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    println!(
        "{name:<32} | p50 {:>7.1}us | p95 {:>7.1}us | max {:>7.1}us | boxes {:>5.0} | spans {:>5.0}",
        us(times[n / 2]),
        us(times[n * 95 / 100]),
        us(times[n - 1]),
        total_boxes as f64 / n as f64,
        total_spans as f64 / n as f64,
    );
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let frames = match args.as_slice() {
        [] => workload::FRAMES,
        [flag, count] if flag == "--frames" => {
            count.parse().expect("--frames needs a positive integer")
        }
        _ => panic!("usage: bench_qlayout [--frames count]"),
    };
    assert!(0 < frames, "--frames needs a positive integer");

    println!("=== ClientTree + TieredPStore + QStore ({frames} frames per scenario) ===");
    let mut trees = Vec::new();
    for (label, shuffled) in [("Heaviest", false), ("Shuffled", true)] {
        let start = Instant::now();
        let tree = workload::build_tree(shuffled);
        let build_ms = start.elapsed().as_secs_f64() * 1e3;
        let mib = 1024.0 * 1024.0;
        let root_p_mib = tree.distribution(ROOT, 0).unwrap().bytes() as f64 / mib;
        println!(
            "{label:<8} {} nodes | build {build_ms:>6.1} ms | tree {:>5.2} MiB | root measure {root_p_mib:.2} MiB",
            tree.node_count(),
            tree.bytes() as f64 / mib,
        );
        trees.push(tree);
    }

    for (motion, lens, shuffled) in workload::CASES {
        let order = if shuffled { "Shuffled" } else { "Heaviest" };
        let relation = if lens.p == lens.q { "==" } else { "!=" };
        let name = format!("{} (p{relation}q, {order})", motion.name());
        bench_trajectory(
            &name,
            &trees[usize::from(shuffled)],
            lens,
            &motion.cameras(frames),
        );
    }
}
