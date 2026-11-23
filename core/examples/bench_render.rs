//! Headless render & data-transfer benchmark for `natlas-core`.
//!
//! Measures:
//! 1. Distribution construction (`from_weights`) & memory footprint
//! 2. Streaming from `Session` into `Tree::insert`, with an estimated payload size
//! 3. Frame layout (`layout::layout`) and picking (`layout::pick`) across zoom dives,
//!    tail pans, `p != q`, and branch-order distributions.
//!
//! Run with:
//! ```bash
//! cargo run --release --manifest-path core/Cargo.toml --no-default-features --example bench_render
//! ```
//! Optional flags: `--frames 200`

use std::hint::black_box;
use std::time::{Duration, Instant};

use natlas_core::distribution::{Distribution, Order, Run, Scale};
use natlas_core::layout::{self, Anchor, Camera, Viewport};
use natlas_core::point::Point;
use natlas_core::session::{Demand, Session, Update};
use natlas_core::tree::{Branch, Lens, NodeId, Tree, ROOT};

const VIEW: Viewport = Viewport {
    width: 1200.0,
    height: 900.0,
};
const ROW_PX: f64 = 1.0;

struct Stats {
    p50_us: f64,
    p95_us: f64,
    max_us: f64,
}

impl Stats {
    fn from_durations(samples: &mut [Duration]) -> Self {
        samples.sort_unstable();
        let n = samples.len();
        let us = |d: Duration| d.as_secs_f64() * 1e6;
        Self {
            p50_us: us(samples[n / 2]),
            p95_us: us(samples[(n * 95) / 100]),
            max_us: us(samples[n - 1]),
        }
    }
}

/// Deterministic xorshift64* PRNG (no external crate needed).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}

/// Zipf log-weights: w_i = (i + 1)^(-alpha).
fn zipf_logits(count: usize, alpha: f64) -> Vec<f32> {
    (0..count)
        .map(|i| (-alpha * ((i + 1) as f64).ln()) as f32)
        .collect()
}

fn uniform_logits(count: usize) -> Vec<f32> {
    vec![0.0f32; count]
}

/// Approximate wire payload size for an `Update::Answer` (branches + f64 weights + top digits + framing).
fn answer_wire_bytes(runs: &[Run]) -> usize {
    runs.iter()
        .map(|r| {
            // rank(4) + len(4) + branches(4*n) + ln_weights(8*n) + top_digits
            8 + r.branches().len() * 12 + r.top_digits().len()
        })
        .sum()
}

fn tree_bytes(tree: &Tree, nodes: &[NodeId], measures: &[usize]) -> usize {
    let mut total = 0usize;
    for &id in nodes {
        for &m in measures {
            if let Ok(Some(d)) = tree.distribution(id, m) {
                total += d.bytes();
            }
        }
    }
    total
}

/// Build a multi-level Zipf tree resembling an expanded LLM exploration tree:
/// - Depth 0 (root): `root_vocab` branches
/// - Depth 1..=2 (top children): `mid_vocab` branches
/// - Depth 3..=depth (along top-branch spine): `leaf_vocab` branches
fn build_multilevel_tree(
    root_vocab: usize,
    mid_vocab: usize,
    leaf_vocab: usize,
    depth: usize,
    order: Order,
    with_q: bool,
) -> (Tree, Vec<NodeId>, Duration) {
    let t0 = Instant::now();
    let mut tree = Tree::new();
    let mut nodes = vec![ROOT];
    let mut rng = Rng::new(0xdead_beef_cafe_babe);

    let root_p = zipf_logits(root_vocab, 1.15);
    tree.set(
        ROOT,
        0,
        Distribution::from_weights(&root_p, order, Scale::Relative).unwrap(),
    )
    .unwrap();
    if with_q {
        let mut root_q = root_p.clone();
        rng.shuffle(&mut root_q);
        tree.set(
            ROOT,
            1,
            Distribution::from_weights(&root_q, order, Scale::Relative).unwrap(),
        )
        .unwrap();
    }

    let mid_p = zipf_logits(mid_vocab, 1.2);
    let mut mid_q = mid_p.clone();
    rng.shuffle(&mut mid_q);

    for b1 in 0..6 as Branch {
        let n1 = tree.child(ROOT, b1).unwrap();
        nodes.push(n1);
        tree.set(
            n1,
            0,
            Distribution::from_weights(&mid_p, order, Scale::Relative).unwrap(),
        )
        .unwrap();
        if with_q {
            tree.set(
                n1,
                1,
                Distribution::from_weights(&mid_q, order, Scale::Relative).unwrap(),
            )
            .unwrap();
        }
        for b2 in 0..3 as Branch {
            let n2 = tree.child(n1, b2).unwrap();
            nodes.push(n2);
            tree.set(
                n2,
                0,
                Distribution::from_weights(&mid_p, order, Scale::Relative).unwrap(),
            )
            .unwrap();
            if with_q {
                tree.set(
                    n2,
                    1,
                    Distribution::from_weights(&mid_q, order, Scale::Relative).unwrap(),
                )
                .unwrap();
            }
        }
    }

    // Deep spine along branch 0 so zoom dives cross many levels.
    let leaf_p = zipf_logits(leaf_vocab, 1.35);
    let mut leaf_q = leaf_p.clone();
    rng.shuffle(&mut leaf_q);
    let mut cur = tree.child(ROOT, 0).unwrap();
    cur = tree.child(cur, 0).unwrap();
    for _ in 2..depth {
        cur = tree.child(cur, 0).unwrap();
        nodes.push(cur);
        tree.set(
            cur,
            0,
            Distribution::from_weights(&leaf_p, order, Scale::Relative).unwrap(),
        )
        .unwrap();
        if with_q {
            tree.set(
                cur,
                1,
                Distribution::from_weights(&leaf_q, order, Scale::Relative).unwrap(),
            )
            .unwrap();
        }
    }

    (tree, nodes, t0.elapsed())
}

fn run_camera_bench(name: &str, tree: &Tree, lens: Lens, cameras: &[Camera]) {
    // Warmup
    for cam in cameras.iter().take(10) {
        let frame = layout::layout(tree, lens, cam, VIEW, ROW_PX).unwrap();
        black_box(&frame.boxes);
    }

    let mut layout_times = Vec::with_capacity(cameras.len());
    let mut pick_times = Vec::with_capacity(cameras.len());
    let mut total_boxes = 0usize;
    let mut total_spans = 0usize;

    for (i, cam) in cameras.iter().enumerate() {
        let t0 = Instant::now();
        let frame = layout::layout(tree, lens, cam, VIEW, ROW_PX).unwrap();
        layout_times.push(t0.elapsed());
        total_boxes += frame.boxes.len();
        total_spans += frame.spans.len();

        let x_px = VIEW.width * (0.15 + ((i * 37) % 70) as f64 / 100.0);
        let y_px = VIEW.height * (0.05 + ((i * 53) % 90) as f64 / 100.0);
        let t1 = Instant::now();
        let picked = layout::pick(tree, lens, &frame, cam, VIEW, x_px, y_px, 1.0).unwrap();
        pick_times.push(t1.elapsed());
        black_box(picked);
    }

    let ls = Stats::from_durations(&mut layout_times);
    let ps = Stats::from_durations(&mut pick_times);
    let avg_boxes = total_boxes as f64 / cameras.len() as f64;
    let avg_spans = total_spans as f64 / cameras.len() as f64;

    println!(
        "{name:<28} | {:>8.1} {:>8.1} {:>8.1} | {:>7.1} {:>7.1} | {:>6.0} {:>6.0}",
        ls.p50_us, ls.p95_us, ls.max_us, ps.p50_us, ps.p95_us, avg_boxes, avg_spans
    );
}

/// Pass the session's updates directly to the client and estimate their wire payload.
fn run_streaming_bench(
    name: &str,
    server_tree: &mut Tree,
    lens: Lens,
    frames: usize,
    mut cam: Camera,
    mut step_camera: impl FnMut(usize, usize, &mut Camera),
) {
    let mut client = Tree::new();
    let mut client_nodes = vec![ROOT];
    let mut session = Session::new();

    let mut server_times = Vec::with_capacity(frames);
    let mut ingest_times = Vec::with_capacity(frames);
    let mut layout_times = Vec::with_capacity(frames);
    let mut total_wire_bytes = 0usize;
    let mut total_answers = 0usize;

    for i in 0..frames {
        step_camera(i, frames, &mut cam);
        cam.anchor = layout::settle(&client, lens, &cam).unwrap();

        // 1. Client computes view for current camera
        let v = layout::view(&client, lens, &cam, VIEW, VIEW.height / ROW_PX).unwrap();

        // 2. Server computes incremental runs for that view
        let ts = Instant::now();
        session
            .demand(
                server_tree,
                Demand {
                    lens,
                    view: Some(v),
                    path: None,
                },
            )
            .unwrap();
        let updates = session.step(server_tree).unwrap();
        server_times.push(ts.elapsed());

        // 3. Measure client ingestion and count estimated payload bytes.
        let ti = Instant::now();
        for update in updates {
            if let Update::Answer {
                measure,
                node,
                edge,
                runs,
                count,
            } = update
            {
                total_wire_bytes += answer_wire_bytes(&runs);
                total_answers += 1;
                if let Some((parent, branch)) = edge {
                    if !client.contains(node) {
                        let created = client.child(parent, branch).unwrap();
                        assert_eq!(created, node);
                        client_nodes.push(node);
                    }
                }
                client.insert(node, measure, runs, count).unwrap();
            }
        }
        ingest_times.push(ti.elapsed());

        // 4. Client renders frame from what it knows so far
        let tl = Instant::now();
        let frame = layout::layout(&client, lens, &cam, VIEW, ROW_PX).unwrap();
        layout_times.push(tl.elapsed());
        black_box(&frame.boxes);
    }

    let ss = Stats::from_durations(&mut server_times);
    let is = Stats::from_durations(&mut ingest_times);
    let ls = Stats::from_durations(&mut layout_times);
    let measures: Vec<usize> = if lens.p == lens.q {
        vec![lens.p]
    } else {
        vec![lens.p, lens.q]
    };
    let client_kb = tree_bytes(&client, &client_nodes, &measures) as f64 / 1024.0;

    println!(
        "{name:<28} | srv p50 {:>7.1}us | ing p50 {:>7.1}us | lay p50 {:>7.1}us (p95 {:>7.1}us) | wire {:>7.1} KB ({total_answers:>4} msgs) | RAM {:>7.1} KB",
        ss.p50_us,
        is.p50_us,
        ls.p50_us,
        ls.p95_us,
        total_wire_bytes as f64 / 1024.0,
        client_kb,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let frames: usize = match args.as_slice() {
        [] => 180,
        [flag, count] if flag == "--frames" => {
            count.parse().expect("--frames needs a positive integer")
        }
        _ => panic!("usage: bench_render [--frames count]"),
    };
    assert!(0 < frames, "--frames needs a positive integer");

    println!("=== natlas-core Render & Transfer Benchmark ({frames} frames/scenario) ===\n");

    // -------------------------------------------------------------------------
    // 1. Construction & Memory Footprint (150k vocab)
    // -------------------------------------------------------------------------
    println!("--- 1. Distribution Construction & Footprint (150,000 branches) ---");
    for (label, logits, order) in [
        (
            "zipf-150k Heaviest",
            zipf_logits(150_000, 1.15),
            Order::Heaviest,
        ),
        (
            "zipf-150k Branch",
            zipf_logits(150_000, 1.15),
            Order::Branch,
        ),
        (
            "uniform-150k Heaviest",
            uniform_logits(150_000),
            Order::Heaviest,
        ),
    ] {
        let t0 = Instant::now();
        let d = Distribution::from_weights(&logits, order, Scale::Relative).unwrap();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "{label:<24} | build {:>7.2} ms | RAM {:>8.1} KB ({:.2} B/branch)",
            elapsed_ms,
            d.bytes() as f64 / 1024.0,
            d.bytes() as f64 / 150_000.0
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // 2. Build Multi-Level Trees
    // -------------------------------------------------------------------------
    let (mut tree_heaviest, nodes_h, build_h) =
        build_multilevel_tree(150_000, 4_096, 256, 20, Order::Heaviest, true);
    let (tree_branch, nodes_b, build_b) =
        build_multilevel_tree(150_000, 4_096, 256, 20, Order::Branch, true);
    println!(
        "Multi-level trees ({} nodes): Heaviest built in {:.1} ms ({:.1} MB) | Branch built in {:.1} ms ({:.1} MB)\n",
        nodes_h.len(),
        build_h.as_secs_f64() * 1e3,
        tree_bytes(&tree_heaviest, &nodes_h, &[0, 1]) as f64 / (1024.0 * 1024.0),
        build_b.as_secs_f64() * 1e3,
        tree_bytes(&tree_branch, &nodes_b, &[0, 1]) as f64 / (1024.0 * 1024.0),
    );

    // -------------------------------------------------------------------------
    // 3. Camera Trajectories
    // -------------------------------------------------------------------------
    // Trajectory A: Smooth zoom dive (base-conversion stress test: x from 0 -> 18 nats)
    let zoom_dive: Vec<Camera> = (0..frames)
        .map(|i| {
            let t = i as f64 / (frames - 1).max(1) as f64;
            let mut cam = Camera {
                origin: ROOT,
                x: t * 18.0,
                k: 120.0,
                anchor: Anchor {
                    path: Vec::new(),
                    digits: Point::top().plus(0.12).digits(),
                },
                a: 0.35,
            };
            cam.anchor = layout::settle(&tree_heaviest, Lens { p: 0, q: 0 }, &cam).unwrap();
            cam
        })
        .collect();

    // Trajectory B: Vertical pan from head (cdf=0.0) deep into the 150k tail (cdf=0.995)
    let tail_pan: Vec<Camera> = (0..frames)
        .map(|i| {
            let t = i as f64 / (frames - 1).max(1) as f64;
            let cdf = 1.0 - (1.0 - 0.995 * t).powi(2);
            Camera {
                origin: ROOT,
                x: 2.5,
                k: 110.0,
                anchor: Anchor {
                    path: Vec::new(),
                    digits: Point::top().plus(cdf.min(0.9999)).digits(),
                },
                a: 0.5,
            }
        })
        .collect();

    // Trajectory C: Fixed zoom-out full view jitter (tests sub-pixel tail rasterization)
    let full_view_jitter: Vec<Camera> = (0..frames)
        .map(|i| Camera {
            origin: ROOT,
            x: 0.4 + 0.25 * (i as f64 / 13.0).sin(),
            k: 100.0 * (1.1 + 0.1 * (i as f64 / 19.0).cos()),
            anchor: Anchor::default(),
            a: 0.0,
        })
        .collect();

    println!("--- 2. Frame Layout & Pick Latency (complete Packed tree, in us) ---");
    println!(
        "{:<28} | {:>8} {:>8} {:>8} | {:>7} {:>7} | {:>6} {:>6}",
        "Scenario", "lay p50", "lay p95", "lay max", "pk p50", "pk p95", "boxes", "spans"
    );
    println!("{}", "-".repeat(94));

    let same = Lens { p: 0, q: 0 };
    let diff = Lens { p: 0, q: 1 };

    run_camera_bench(
        "zoom_dive (p==q, Heaviest)",
        &tree_heaviest,
        same,
        &zoom_dive,
    );
    run_camera_bench(
        "zoom_dive (p!=q, Heaviest)",
        &tree_heaviest,
        diff,
        &zoom_dive,
    );
    run_camera_bench(
        "tail_pan  (p==q, Heaviest)",
        &tree_heaviest,
        same,
        &tail_pan,
    );
    run_camera_bench(
        "tail_pan  (p!=q, Heaviest)",
        &tree_heaviest,
        diff,
        &tail_pan,
    );
    run_camera_bench(
        "full_view (p==q, Heaviest)",
        &tree_heaviest,
        same,
        &full_view_jitter,
    );
    run_camera_bench(
        "full_view (p!=q, Heaviest)",
        &tree_heaviest,
        diff,
        &full_view_jitter,
    );
    run_camera_bench("tail_pan  (p==q, Branch)", &tree_branch, same, &tail_pan);
    run_camera_bench(
        "full_view (p==q, Branch)",
        &tree_branch,
        same,
        &full_view_jitter,
    );
    println!();

    // -------------------------------------------------------------------------
    // 4. End-to-End Streaming Transfer + Client Render (Sparse Client Tree)
    // -------------------------------------------------------------------------
    println!("--- 3. Streaming (Session -> Client Insert -> Layout, estimated wire payload) ---");
    let dive_start = Camera {
        origin: ROOT,
        x: 0.0,
        k: 120.0,
        anchor: Anchor {
            path: Vec::new(),
            digits: Point::top().plus(0.12).digits(),
        },
        a: 0.35,
    };
    let pan_start = Camera {
        origin: ROOT,
        x: 2.5,
        k: 110.0,
        anchor: Anchor::default(),
        a: 0.5,
    };

    run_streaming_bench(
        "stream zoom_dive (p==q)",
        &mut tree_heaviest,
        same,
        frames,
        dive_start.clone(),
        |i, n, cam| {
            cam.x = (i as f64 / (n - 1).max(1) as f64) * 18.0;
        },
    );
    run_streaming_bench(
        "stream zoom_dive (p!=q)",
        &mut tree_heaviest,
        diff,
        frames,
        dive_start,
        |i, n, cam| {
            cam.x = (i as f64 / (n - 1).max(1) as f64) * 18.0;
        },
    );
    run_streaming_bench(
        "stream tail_pan  (p==q)",
        &mut tree_heaviest,
        same,
        frames,
        pan_start.clone(),
        |i, n, cam| {
            let t = i as f64 / (n - 1).max(1) as f64;
            let cdf = 1.0 - (1.0 - 0.995 * t).powi(2);
            cam.anchor = Anchor {
                path: Vec::new(),
                digits: Point::top().plus(cdf.min(0.9999)).digits(),
            };
        },
    );
    run_streaming_bench(
        "stream tail_pan  (p!=q)",
        &mut tree_heaviest,
        diff,
        frames,
        pan_start,
        |i, n, cam| {
            let t = i as f64 / (n - 1).max(1) as f64;
            let cdf = 1.0 - (1.0 - 0.995 * t).powi(2);
            cam.anchor = Anchor {
                path: Vec::new(),
                digits: Point::top().plus(cdf.min(0.9999)).digits(),
            };
        },
    );
}
