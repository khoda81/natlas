use natlas_core::client::ClientTree;
use natlas_core::distribution::{log_softmax, Distribution};
use natlas_core::layout::{Anchor, Camera, Viewport};
use natlas_core::point::Point;
use natlas_core::tree::{Lens, NodeId, ROOT};

pub const FRAMES: usize = 180;
pub const VOCAB: usize = 150_000;
pub const VIEW: Viewport = Viewport {
    width: 1200.0,
    height: 900.0,
};

struct Rng(u64);

impl Rng {
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let next = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
            items.swap(i, (next % (i as u64 + 1)) as usize);
        }
    }
}

fn logits(count: usize, alpha: f64) -> Vec<f32> {
    (0..count)
        .map(|i| (-alpha * ((i + 1) as f64).ln()) as f32)
        .collect()
}

fn insert_logits(tree: &mut ClientTree, node: NodeId, measure: usize, logits: &[f32]) {
    let mut children: Vec<_> = (0..).zip(logits.iter().copied().map(f64::from)).collect();
    log_softmax(&mut children);
    let d = Distribution::from_ordered(children).unwrap();
    let count = d.count().unwrap();
    let children: Vec<_> = (0..count as usize).map(|at| (0, at)).collect();
    tree.insert(node, measure, d.answer(&children, 0), Some(count))
        .unwrap();
}

/// The same topology, logits and q permutation as the packed benchmark for
/// Heaviest. Shuffled gives the root and intermediate p measures arbitrary order.
pub fn build_tree(shuffled: bool) -> ClientTree {
    let mut rng = Rng(0xdead_beef_cafe_babe);
    let mut pair = |count, alpha, shuffle_p| {
        let mut p = logits(count, alpha);
        if shuffle_p {
            rng.shuffle(&mut p);
        }
        let mut q = p.clone();
        rng.shuffle(&mut q);
        (p, q)
    };
    let (root_p, root_q) = pair(VOCAB, 1.15, shuffled);
    let (mid_p, mid_q) = pair(4096, 1.2, shuffled);
    let (leaf_p, leaf_q) = pair(256, 1.35, false);

    let mut tree = ClientTree::new();
    insert_logits(&mut tree, ROOT, 0, &root_p);
    insert_logits(&mut tree, ROOT, 1, &root_q);
    for b1 in 0..6 {
        let n1 = tree.child(ROOT, b1).unwrap();
        insert_logits(&mut tree, n1, 0, &mid_p);
        insert_logits(&mut tree, n1, 1, &mid_q);
        for b2 in 0..3 {
            let n2 = tree.child(n1, b2).unwrap();
            insert_logits(&mut tree, n2, 0, &mid_p);
            insert_logits(&mut tree, n2, 1, &mid_q);
        }
    }
    let mut spine = tree.child(ROOT, 0).unwrap();
    spine = tree.child(spine, 0).unwrap();
    for _ in 2..20 {
        spine = tree.child(spine, 0).unwrap();
        insert_logits(&mut tree, spine, 0, &leaf_p);
        insert_logits(&mut tree, spine, 1, &leaf_q);
    }
    tree
}

#[derive(Clone, Copy)]
pub enum Motion {
    FullView,
    TailPan,
    ZoomDive,
}

impl Motion {
    pub fn name(self) -> &'static str {
        match self {
            Self::FullView => "full_view",
            Self::TailPan => "tail_pan",
            Self::ZoomDive => "zoom_dive",
        }
    }

    pub fn cameras(self, frames: usize) -> Vec<Camera> {
        (0..frames)
            .map(|i| {
                let t = i as f64 / (frames - 1).max(1) as f64;
                // Match the root-relative camera trajectory in bench_render.
                let (x, k, anchor, a) = match self {
                    Self::FullView => (
                        0.4 + 0.25 * (i as f64 / 13.0).sin(),
                        100.0 * (1.1 + 0.1 * (i as f64 / 19.0).cos()),
                        0.0,
                        0.0,
                    ),
                    Self::TailPan => (
                        2.5,
                        110.0,
                        (1.0 - (1.0 - 0.995 * t).powi(2)).min(0.9999),
                        0.5,
                    ),
                    Self::ZoomDive => (t * 18.0, 120.0, 0.12, 0.35),
                };
                Camera {
                    origin: ROOT,
                    x,
                    k,
                    anchor: Anchor {
                        path: Vec::new(),
                        digits: Point::top().plus(anchor).digits(),
                    },
                    a,
                }
            })
            .collect()
    }
}

const SAME: Lens = Lens { p: 0, q: 0 };
const DIFF: Lens = Lens { p: 0, q: 1 };

pub const CASES: [(Motion, Lens, bool); 8] = [
    (Motion::FullView, SAME, false),
    (Motion::FullView, DIFF, false),
    (Motion::TailPan, SAME, false),
    (Motion::TailPan, DIFF, false),
    (Motion::ZoomDive, SAME, false),
    (Motion::ZoomDive, DIFF, false),
    (Motion::FullView, SAME, true),
    (Motion::TailPan, DIFF, true),
];
