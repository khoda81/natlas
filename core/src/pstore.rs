//! Experimental order-agnostic spatial-priority `PStore` implementations.
//!
//! All coordinates use the full `0..=u64::MAX` range as inclusive intervals `[start, end]`:
//! - `start` represents `start / 2^64` (so `0` is `0.0`)
//! - `end` represents `(end + 1) / 2^64` (so `u64::MAX` is `1.0` exactly)
//! - `[0, u64::MAX]` represents `[0.0, 1.0]` exactly
//! - True mass of `[start, end]` is `((end - start) + 1) / 2^64`
//! - `span_m1 = end - start` (in `0..=u64::MAX`) is strictly order-preserving with mass
//!   and never overflows `u64`. For `0 < min_mass`, a branch meets that minimum
//!   exactly when `min_mass - 1 <= end - start`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::tree::Branch;

/// Inclusive CDF interval `[start, end]` over `0..=u64::MAX`,
/// representing `[start / 2^64, (end + 1) / 2^64]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CdfRange {
    start: u64,
    end: u64,
}

impl CdfRange {
    pub const FULL: Self = Self {
        start: 0,
        end: u64::MAX,
    };

    pub fn new(start: u64, end: u64) -> Self {
        assert!(start <= end, "a CDF range must start at or before its end");
        Self { start, end }
    }

    pub fn midpoint(self) -> u64 {
        self.start + (self.end - self.start) / 2
    }

    /// A pixel starting at this range's top, clipped to the range.
    pub fn first_pixel(self, mass: u64) -> Self {
        Self {
            start: self.start,
            end: self
                .start
                .saturating_add(mass.saturating_sub(1))
                .min(self.end),
        }
    }
}

/// A branch with its inclusive CDF interval `[start, end]` and log-probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BranchHit {
    pub branch: Branch,
    pub start: u64,
    pub end: u64,
    pub ln_p: f32,
}

impl BranchHit {
    /// `end - start`, representing `mass = (span_m1 + 1) / 2^64`.
    #[inline]
    pub fn span_m1(&self) -> u64 {
        self.end - self.start
    }
}

/// Heap storage reused across calls to `query_tall`.
#[derive(Default)]
pub struct QueryScratch {
    heap: BinaryHeap<HeapItem>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct HeapItem {
    /// `span_m1 = end - start` (strictly monotonic with true mass).
    span_m1: u64,
    /// High bit (1 << 31) = 1 => leaf index (`idx & !LEAF_BIT`), 0 => internal node index.
    tag: u32,
}

const LEAF_BIT: u32 = 1u32 << 31;

impl Ord for HeapItem {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.span_m1
            .cmp(&other.span_m1)
            .then_with(|| other.tag.cmp(&self.tag))
    }
}

impl PartialOrd for HeapItem {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Order-agnostic per-node CDF + spatial-priority store.
pub trait PStore: Sized {
    fn name() -> &'static str;

    /// Construct from finite logits or negative infinities in display order.
    /// Index `i` is `Branch` `i`. Negative infinities have no mass.
    fn from_logits(logits: &[f32]) -> Self;

    /// Replace `out` with all branches overlapping `range` that meet `min_mass`,
    /// ordered by descending mass (ties broken by ascending `start`).
    /// `min_mass` is in units of `2^-64` (so `min_mass <= 1` matches all branches).
    fn query_tall(
        &self,
        range: CdfRange,
        min_mass: u64,
        scratch: &mut QueryScratch,
        out: &mut Vec<BranchHit>,
    );

    /// Point location: branch containing `cdf` (for any `cdf` in `0..=u64::MAX`).
    fn locate(&self, cdf: u64) -> Option<BranchHit>;

    /// True if `[range.start, range.end]` straddles at least one boundary between adjacent branches.
    fn contains_boundary(&self, range: CdfRange) -> bool;

    /// Approximate memory footprint in bytes (struct + heap allocations).
    fn memory_bytes(&self) -> usize;
}

/// Shared normalization helper: converts raw logits (in any order) into
/// contiguous `(starts, ends, branches, ln_p)` where each branch `i` owns
/// the non-empty inclusive interval `[starts[i], ends[i]]`, `starts[0] == 0`,
/// `ends[n - 1] == u64::MAX`, and `starts[i + 1] == ends[i] + 1`.
/// Cumulative cuts are computed in f64 and rounded to the integer grid.
/// A branch whose rounded cuts coincide has no interval.
fn normalize_logits(logits: &[f32]) -> (Vec<u64>, Vec<u64>, Vec<Branch>, Vec<f32>) {
    assert!(
        logits.len() <= LEAF_BIT as usize,
        "too many branches for a PStore"
    );
    let max_l = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, logit| {
            assert!(
                !logit.is_nan() && logit != f32::INFINITY,
                "invalid PStore logit"
            );
            max.max(logit)
        });
    if max_l == f32::NEG_INFINITY {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut raw_branches = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        if f32::NEG_INFINITY < l {
            let e = (f64::from(l) - f64::from(max_l)).exp();
            if 0.0 < e {
                exps.push(e);
                raw_branches.push(i as Branch);
                sum += e;
            }
        }
    }
    let m = raw_branches.len();

    let ln_sum = sum.ln();
    let mut starts = Vec::with_capacity(m);
    let mut ends = Vec::with_capacity(m);
    let mut branches = Vec::with_capacity(m);
    let mut ln_p = Vec::with_capacity(m);

    const CDF_SIZE: u128 = 1 << 64;
    let scale = CDF_SIZE as f64 / sum;
    let mut acc = 0.0f64;
    let mut cur_start = 0;

    for (k, &e) in exps.iter().enumerate() {
        if cur_start == CDF_SIZE {
            break;
        }
        acc += e;
        let is_last = k + 1 == m;
        let next_start = if is_last {
            CDF_SIZE
        } else {
            ((acc * scale).round() as u128).min(CDF_SIZE)
        };
        if next_start <= cur_start {
            continue;
        }
        let orig = logits[raw_branches[k] as usize];
        starts.push(cur_start as u64);
        ends.push((next_start - 1) as u64);
        branches.push(raw_branches[k]);
        ln_p.push((f64::from(orig) - f64::from(max_l) - ln_sum) as f32);
        cur_start = next_start;
    }
    (starts, ends, branches, ln_p)
}

#[inline]
fn locate_in_slice(
    starts: &[u64],
    ends: &[u64],
    branches: &[Branch],
    ln_p: &[f32],
    cdf: u64,
) -> Option<BranchHit> {
    if branches.is_empty() {
        return None;
    }
    let idx = ends.partition_point(|&e| e < cdf);
    Some(BranchHit {
        branch: branches[idx],
        start: starts[idx],
        end: ends[idx],
        ln_p: ln_p[idx],
    })
}

#[inline]
fn contains_boundary_in_slice(ends: &[u64], range: CdfRange) -> bool {
    if range.start == range.end || ends.len() <= 1 {
        return false;
    }
    // Boundary between branch `i` and `i + 1` sits between `ends[i]` and `ends[i] + 1`.
    // Range `[start, end]` straddles it iff `start <= ends[i] && ends[i] < end`.
    let idx = ends.partition_point(|&e| e < range.start);
    ends[idx] < range.end
}

// =============================================================================
// Design 1: BTreeMax — Flat 8-ary max-tree over 16-element leaf blocks
// =============================================================================

const BTREE_B: usize = 8;
const LEAF_BLOCK: usize = 16;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
struct BNode {
    /// Inclusive upper CDF bound of each child slot (64 bytes = 1 cache line).
    ends: [u64; BTREE_B],
    /// Maximum leaf `span_m1` (`end - start`) in each child subtree (64 bytes = 1 cache line).
    max_span_m1: [u64; BTREE_B],
    /// Inclusive lower CDF bound of slot 0.
    start: u64,
    /// First child index (in `nodes` if internal, or leaf-block index if `leaf_parent`).
    child_base: u32,
    len: u8,
    leaf_parent: bool,
    _pad: [u8; 2],
}

/// Flat 8-ary spatial-priority tree with 16-leaf blocks.
pub struct BTreeMax {
    starts: Vec<u64>,
    ends: Vec<u64>,
    branches: Vec<Branch>,
    ln_p: Vec<f32>,
    /// Internal nodes, root is `nodes.last()`.
    nodes: Vec<BNode>,
    root: u32,
}

impl PStore for BTreeMax {
    fn name() -> &'static str {
        "BTreeMax(B=8,L=16)"
    }

    fn from_logits(logits: &[f32]) -> Self {
        let (starts, ends, branches, ln_p) = normalize_logits(logits);
        let n = branches.len();
        if n == 0 {
            return Self {
                starts,
                ends,
                branches,
                ln_p,
                nodes: Vec::new(),
                root: 0,
            };
        }

        let num_blocks = n.div_ceil(LEAF_BLOCK);
        let mut nodes = Vec::new();

        // Level 0: parents of leaf blocks
        let mut cur_start = 0usize;
        let mut level_len = 0usize;
        let mut b = 0usize;
        while b < num_blocks {
            let count = BTREE_B.min(num_blocks - b);
            let mut node = BNode {
                ends: [u64::MAX; BTREE_B],
                max_span_m1: [0; BTREE_B],
                start: starts[b * LEAF_BLOCK],
                child_base: b as u32,
                len: count as u8,
                leaf_parent: true,
                _pad: [0; 2],
            };
            for s in 0..count {
                let l0 = (b + s) * LEAF_BLOCK;
                let l1 = (l0 + LEAF_BLOCK).min(n);
                let mut m = 0u64;
                for i in l0..l1 {
                    m = m.max(ends[i] - starts[i]);
                }
                node.ends[s] = ends[l1 - 1];
                node.max_span_m1[s] = m;
            }
            nodes.push(node);
            level_len += 1;
            b += count;
        }

        // Upper levels until single root
        while 1 < level_len {
            let next_start = nodes.len();
            let mut next_len = 0usize;
            let mut i = 0usize;
            while i < level_len {
                let count = BTREE_B.min(level_len - i);
                let first_child = cur_start + i;
                let mut node = BNode {
                    ends: [u64::MAX; BTREE_B],
                    max_span_m1: [0; BTREE_B],
                    start: nodes[first_child].start,
                    child_base: first_child as u32,
                    len: count as u8,
                    leaf_parent: false,
                    _pad: [0; 2],
                };
                for s in 0..count {
                    let child = &nodes[first_child + s];
                    let c_len = child.len as usize;
                    node.ends[s] = child.ends[c_len - 1];
                    let mut m = 0u64;
                    for k in 0..c_len {
                        m = m.max(child.max_span_m1[k]);
                    }
                    node.max_span_m1[s] = m;
                }
                nodes.push(node);
                next_len += 1;
                i += count;
            }
            cur_start = next_start;
            level_len = next_len;
        }

        let root = (nodes.len() - 1) as u32;
        Self {
            starts,
            ends,
            branches,
            ln_p,
            nodes,
            root,
        }
    }

    fn query_tall(
        &self,
        range: CdfRange,
        min_mass: u64,
        scratch: &mut QueryScratch,
        out: &mut Vec<BranchHit>,
    ) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let min_span_m1 = min_mass.saturating_sub(1);
        scratch.heap.clear();
        let root_node = &self.nodes[self.root as usize];
        let root_max = root_node.max_span_m1[..root_node.len as usize]
            .iter()
            .copied()
            .max()
            .expect("the root has children");
        if root_max < min_span_m1 {
            return;
        }
        scratch.heap.push(HeapItem {
            span_m1: root_max,
            tag: self.root,
        });

        let n = self.branches.len();
        while let Some(item) = scratch.heap.pop() {
            if item.span_m1 < min_span_m1 {
                break;
            }
            if item.tag & LEAF_BIT != 0 {
                let idx = (item.tag & !LEAF_BIT) as usize;
                out.push(BranchHit {
                    branch: self.branches[idx],
                    start: self.starts[idx],
                    end: self.ends[idx],
                    ln_p: self.ln_p[idx],
                });
                continue;
            }
            let node = &self.nodes[item.tag as usize];
            let mut slot_start = node.start;
            for s in 0..node.len as usize {
                let slot_end = node.ends[s];
                if range.start <= slot_end
                    && slot_start <= range.end
                    && min_span_m1 <= node.max_span_m1[s]
                {
                    if node.leaf_parent {
                        let l0 = (node.child_base as usize + s) * LEAF_BLOCK;
                        let l1 = (l0 + LEAF_BLOCK).min(n);
                        for idx in l0..l1 {
                            let st = self.starts[idx];
                            let en = self.ends[idx];
                            let m = en - st;
                            if range.start <= en && st <= range.end && min_span_m1 <= m {
                                scratch.heap.push(HeapItem {
                                    span_m1: m,
                                    tag: (idx as u32) | LEAF_BIT,
                                });
                            }
                        }
                    } else {
                        scratch.heap.push(HeapItem {
                            span_m1: node.max_span_m1[s],
                            tag: node.child_base + s as u32,
                        });
                    }
                }
                if range.end <= slot_end {
                    break;
                }
                slot_start = slot_end + 1;
            }
        }
    }

    fn locate(&self, cdf: u64) -> Option<BranchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut idx = self.root as usize;
        loop {
            let node = &self.nodes[idx];
            let mut slot = 0usize;
            let len = node.len as usize;
            while slot + 1 < len && node.ends[slot] < cdf {
                slot += 1;
            }
            if node.leaf_parent {
                let l0 = (node.child_base as usize + slot) * LEAF_BLOCK;
                let l1 = (l0 + LEAF_BLOCK).min(self.branches.len());
                for i in l0..l1 {
                    if cdf <= self.ends[i] {
                        return Some(BranchHit {
                            branch: self.branches[i],
                            start: self.starts[i],
                            end: self.ends[i],
                            ln_p: self.ln_p[i],
                        });
                    }
                }
                unreachable!("the leaf block contains its CDF coordinate");
            }
            idx = node.child_base as usize + slot;
        }
    }

    fn contains_boundary(&self, range: CdfRange) -> bool {
        contains_boundary_in_slice(&self.ends, range)
    }

    fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + (self.starts.capacity() + self.ends.capacity()) * size_of::<u64>()
            + self.branches.capacity() * size_of::<Branch>()
            + self.ln_p.capacity() * size_of::<f32>()
            + self.nodes.capacity() * size_of::<BNode>()
    }
}

// =============================================================================
// Design 2: EytzingerMax — 16-byte implicit binary max-tree (1-based BFS)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct EytzNode {
    /// Inclusive upper CDF bound of this subtree.
    /// Lower bound is `0` if `idx.is_power_of_two()` else `nodes[idx - 1].end + 1`.
    end: u64,
    /// Maximum leaf `span_m1` in this subtree (`0` for empty padding nodes).
    max_span_m1: u64,
}

/// Power-of-two implicit binary max-tree over 16-element leaf blocks.
pub struct EytzingerMax {
    starts: Vec<u64>,
    ends: Vec<u64>,
    branches: Vec<Branch>,
    ln_p: Vec<f32>,
    /// 1-based complete binary tree; leaves live at `base .. base + num_blocks`.
    nodes: Vec<EytzNode>,
    base: usize,
    num_blocks: usize,
}

impl EytzingerMax {
    #[inline]
    fn node_start(&self, idx: usize) -> Option<u64> {
        if idx.is_power_of_two() {
            Some(0)
        } else {
            self.nodes[idx - 1].end.checked_add(1)
        }
    }
}

impl PStore for EytzingerMax {
    fn name() -> &'static str {
        "EytzingerMax(L=16)"
    }

    fn from_logits(logits: &[f32]) -> Self {
        let (starts, ends, branches, ln_p) = normalize_logits(logits);
        let n = branches.len();
        if n == 0 {
            return Self {
                starts,
                ends,
                branches,
                ln_p,
                nodes: Vec::new(),
                base: 0,
                num_blocks: 0,
            };
        }
        let num_blocks = n.div_ceil(LEAF_BLOCK);
        let base = num_blocks.next_power_of_two();
        let mut nodes = vec![
            EytzNode {
                end: u64::MAX,
                max_span_m1: 0,
            };
            2 * base
        ];

        for b in 0..num_blocks {
            let l0 = b * LEAF_BLOCK;
            let l1 = (l0 + LEAF_BLOCK).min(n);
            let mut m = 0u64;
            for i in l0..l1 {
                m = m.max(ends[i] - starts[i]);
            }
            nodes[base + b] = EytzNode {
                end: ends[l1 - 1],
                max_span_m1: m,
            };
        }
        for i in (1..base).rev() {
            let left = nodes[2 * i];
            let right = nodes[2 * i + 1];
            nodes[i] = EytzNode {
                end: right.end,
                max_span_m1: left.max_span_m1.max(right.max_span_m1),
            };
        }

        Self {
            starts,
            ends,
            branches,
            ln_p,
            nodes,
            base,
            num_blocks,
        }
    }

    fn query_tall(
        &self,
        range: CdfRange,
        min_mass: u64,
        scratch: &mut QueryScratch,
        out: &mut Vec<BranchHit>,
    ) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let min_span_m1 = min_mass.saturating_sub(1);
        scratch.heap.clear();
        let root_max = self.nodes[1].max_span_m1;
        if root_max < min_span_m1 {
            return;
        }
        scratch.heap.push(HeapItem {
            span_m1: root_max,
            tag: 1,
        });

        let n = self.branches.len();
        while let Some(item) = scratch.heap.pop() {
            if item.span_m1 < min_span_m1 {
                break;
            }
            if item.tag & LEAF_BIT != 0 {
                let idx = (item.tag & !LEAF_BIT) as usize;
                out.push(BranchHit {
                    branch: self.branches[idx],
                    start: self.starts[idx],
                    end: self.ends[idx],
                    ln_p: self.ln_p[idx],
                });
                continue;
            }
            let idx = item.tag as usize;
            if self.base <= idx {
                let b = idx - self.base;
                if self.num_blocks <= b {
                    continue;
                }
                let l0 = b * LEAF_BLOCK;
                let l1 = (l0 + LEAF_BLOCK).min(n);
                for leaf in l0..l1 {
                    let st = self.starts[leaf];
                    let en = self.ends[leaf];
                    let m = en - st;
                    if range.start <= en && st <= range.end && min_span_m1 <= m {
                        scratch.heap.push(HeapItem {
                            span_m1: m,
                            tag: (leaf as u32) | LEAF_BIT,
                        });
                    }
                }
            } else {
                let Some(left_start) = self.node_start(idx) else {
                    continue;
                };
                let left = 2 * idx;
                let right = left + 1;
                let left_end = self.nodes[left].end;
                if range.start <= left_end
                    && left_start <= range.end
                    && min_span_m1 <= self.nodes[left].max_span_m1
                {
                    scratch.heap.push(HeapItem {
                        span_m1: self.nodes[left].max_span_m1,
                        tag: left as u32,
                    });
                }
                if let Some(right_start) = left_end.checked_add(1) {
                    let right_end = self.nodes[right].end;
                    if range.start <= right_end
                        && right_start <= range.end
                        && min_span_m1 <= self.nodes[right].max_span_m1
                    {
                        scratch.heap.push(HeapItem {
                            span_m1: self.nodes[right].max_span_m1,
                            tag: right as u32,
                        });
                    }
                }
            }
        }
    }

    fn locate(&self, cdf: u64) -> Option<BranchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut idx = 1usize;
        while idx < self.base {
            let left = 2 * idx;
            idx = if cdf <= self.nodes[left].end {
                left
            } else {
                left + 1
            };
        }
        let b = idx - self.base;
        let l0 = b * LEAF_BLOCK;
        let l1 = (l0 + LEAF_BLOCK).min(self.branches.len());
        for i in l0..l1 {
            if cdf <= self.ends[i] {
                return Some(BranchHit {
                    branch: self.branches[i],
                    start: self.starts[i],
                    end: self.ends[i],
                    ln_p: self.ln_p[i],
                });
            }
        }
        unreachable!("the leaf block contains its CDF coordinate");
    }

    fn contains_boundary(&self, range: CdfRange) -> bool {
        contains_boundary_in_slice(&self.ends, range)
    }

    fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + (self.starts.capacity() + self.ends.capacity()) * size_of::<u64>()
            + self.branches.capacity() * size_of::<Branch>()
            + self.ln_p.capacity() * size_of::<f32>()
            + self.nodes.capacity() * size_of::<EytzNode>()
    }
}

// =============================================================================
// Design 3: TieredPStore — Scale-stratified base-2 tiers (heavy branches in L1)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TierEntry {
    start: u64,
    end: u64,
    branch: Branch,
    ln_p: f32,
}

/// Branches stratified by `span_m1.leading_zeros()` (65 power-of-two tiers `0..=64`),
/// sorted by `start` within each tier. Zero heap pops during `query_tall`.
pub struct TieredPStore {
    starts: Vec<u64>,
    ends: Vec<u64>,
    branches: Vec<Branch>,
    ln_p: Vec<f32>,
    /// All branches packed contiguously by tier `0..=64`, sorted by `start` inside each tier.
    entries: Vec<TierEntry>,
    tier_starts: [u32; 66],
}

impl PStore for TieredPStore {
    fn name() -> &'static str {
        "TieredPStore(base2)"
    }

    fn from_logits(logits: &[f32]) -> Self {
        let (starts, ends, branches, ln_p) = normalize_logits(logits);
        let n = branches.len();
        let mut counts = [0u32; 65];
        for i in 0..n {
            let t = (ends[i] - starts[i]).leading_zeros() as usize;
            counts[t] += 1;
        }
        let mut tier_starts = [0u32; 66];
        for t in 0..65 {
            tier_starts[t + 1] = tier_starts[t] + counts[t];
        }
        let mut entries = vec![
            TierEntry {
                start: 0,
                end: 0,
                branch: 0,
                ln_p: 0.0,
            };
            n
        ];
        let mut cursors = [0u32; 65];
        cursors.copy_from_slice(&tier_starts[..65]);
        // Input is already ordered by increasing `start`, so each tier stays sorted by `start`!
        for i in 0..n {
            let st = starts[i];
            let en = ends[i];
            let t = (en - st).leading_zeros() as usize;
            let pos = cursors[t] as usize;
            cursors[t] += 1;
            entries[pos] = TierEntry {
                start: st,
                end: en,
                branch: branches[i],
                ln_p: ln_p[i],
            };
        }
        Self {
            starts,
            ends,
            branches,
            ln_p,
            entries,
            tier_starts,
        }
    }

    fn query_tall(
        &self,
        range: CdfRange,
        min_mass: u64,
        _scratch: &mut QueryScratch,
        out: &mut Vec<BranchHit>,
    ) {
        out.clear();
        if self.entries.is_empty() {
            return;
        }
        let min_span_m1 = min_mass.saturating_sub(1);
        let max_tier = min_span_m1.leading_zeros() as usize;
        for t in 0..=max_tier {
            let s = self.tier_starts[t] as usize;
            let e = self.tier_starts[t + 1] as usize;
            if s == e {
                continue;
            }
            let slice = &self.entries[s..e];
            // First entry whose end meets or passes the range's start.
            let first = slice.partition_point(|x| x.end < range.start);
            for entry in &slice[first..] {
                if range.end < entry.start {
                    break;
                }
                if min_span_m1 <= entry.end - entry.start {
                    out.push(BranchHit {
                        branch: entry.branch,
                        start: entry.start,
                        end: entry.end,
                        ln_p: entry.ln_p,
                    });
                }
            }
        }
        out.sort_unstable_by(|a, b| {
            b.span_m1()
                .cmp(&a.span_m1())
                .then_with(|| a.start.cmp(&b.start))
        });
    }

    fn locate(&self, cdf: u64) -> Option<BranchHit> {
        locate_in_slice(&self.starts, &self.ends, &self.branches, &self.ln_p, cdf)
    }

    fn contains_boundary(&self, range: CdfRange) -> bool {
        contains_boundary_in_slice(&self.ends, range)
    }

    fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + (self.starts.capacity() + self.ends.capacity()) * size_of::<u64>()
            + self.branches.capacity() * size_of::<Branch>()
            + self.ln_p.capacity() * size_of::<f32>()
            + self.entries.capacity() * size_of::<TierEntry>()
    }
}

// =============================================================================
// Deterministic Benchmark Workloads (shared by x86 & WASM)
// =============================================================================

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

pub fn bench_logits(kind: u32, count: usize) -> Vec<f32> {
    match kind {
        // 0: Zipf heaviest-first (LLM-like)
        0 => (0..count)
            .map(|i| (-1.15 * ((i + 1) as f64).ln()) as f32)
            .collect(),
        // 1: Zipf shuffled (order-agnostic stress test: heavy branches scattered)
        1 => {
            let mut v: Vec<f32> = (0..count)
                .map(|i| (-1.15 * ((i + 1) as f64).ln()) as f32)
                .collect();
            Rng::new(0x1234_5678_9abc_def0).shuffle(&mut v);
            v
        }
        // 2: Uniform (pathological all-subpixel case)
        2 => vec![0.0f32; count],
        _ => panic!("unknown PStore benchmark distribution {kind}"),
    }
}

/// Generate `count` realistic `(CdfRange, min_mass)` viewport queries over `0..=u64::MAX`.
pub fn bench_queries(count: usize) -> Vec<(CdfRange, u64)> {
    let mut rng = Rng::new(0xfeed_face_dead_beef);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mode = i % 4;
        let (start, end) = match mode {
            // Full view [0.0, 1.0] exactly
            0 => (0u64, u64::MAX),
            // Medium zoom (1/16 of node, panning across [0, 1])
            1 => {
                let span_m1 = u64::MAX >> 4;
                let start = (rng.next_u64() % (u64::MAX - span_m1)) & !7;
                (start, start + span_m1)
            }
            // Deep zoom (1/1024 of node, panning across [0, 1])
            2 => {
                let span_m1 = u64::MAX >> 10;
                let start = (rng.next_u64() % (u64::MAX - span_m1)) & !7;
                (start, start + span_m1)
            }
            // Tail pan (top of tail [0.95, 1.0])
            _ => {
                let span_m1 = u64::MAX >> 6;
                let base = (u64::MAX / 20) * 19;
                let start = base + (rng.next_u64() % (u64::MAX - base - span_m1));
                (start, start + span_m1)
            }
        };
        let min_mass = (((u128::from(end - start) + 1) / 900) as u64).max(1);
        out.push((CdfRange::new(start, end), min_mass));
    }
    out
}

/// Verify all three implementations produce identical results on every query.
pub fn verify_equivalence(logits: &[f32]) -> Result<(), String> {
    let a = BTreeMax::from_logits(logits);
    let b = EytzingerMax::from_logits(logits);
    let c = TieredPStore::from_logits(logits);
    let queries = bench_queries(64);
    let mut scratch = QueryScratch::default();
    let (mut oa, mut ob, mut oc) = (Vec::new(), Vec::new(), Vec::new());

    for &(range, min_mass) in &queries {
        a.query_tall(range, min_mass, &mut scratch, &mut oa);
        b.query_tall(range, min_mass, &mut scratch, &mut ob);
        c.query_tall(range, min_mass, &mut scratch, &mut oc);
        if oa != ob || oa != oc {
            return Err(format!(
                "query_tall mismatch for {range:?} min_mass={min_mass}: lens {}, {}, {}",
                oa.len(),
                ob.len(),
                oc.len()
            ));
        }
        let mid = range.midpoint();
        if a.locate(mid) != b.locate(mid) || a.locate(mid) != c.locate(mid) {
            return Err(format!("locate mismatch at {mid}"));
        }
        if a.contains_boundary(range) != b.contains_boundary(range)
            || a.contains_boundary(range) != c.contains_boundary(range)
        {
            return Err(format!("contains_boundary mismatch for {range:?}"));
        }
    }
    // Explicitly check extreme endpoints 0 and u64::MAX
    for pt in [0u64, u64::MAX] {
        if a.locate(pt) != b.locate(pt) || a.locate(pt) != c.locate(pt) || a.locate(pt).is_none() {
            return Err(format!("endpoint locate mismatch at {pt}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_pstores_agree_on_zipf_shuffled_and_uniform() {
        for kind in 0..3 {
            let logits = bench_logits(kind, 4_096);
            verify_equivalence(&logits).unwrap();
        }
    }

    #[test]
    fn inclusive_cdf_endpoints_thresholds_and_ties() {
        fn check<S: PStore>() {
            let mut scratch = QueryScratch::default();
            let mut hits = Vec::new();
            let half = 1u64 << 63;
            let adjacent = CdfRange::new(half - 1, half);

            let pairs = S::from_logits(&[0.0, f32::NEG_INFINITY, 0.0]);
            pairs.query_tall(adjacent, half, &mut scratch, &mut hits);
            assert_eq!(hits.len(), 2);
            assert_eq!(
                (hits[0].branch, hits[0].start, hits[0].end),
                (0, 0, half - 1)
            );
            assert_eq!(
                (hits[1].branch, hits[1].start, hits[1].end),
                (2, half, u64::MAX)
            );
            assert_eq!(pairs.locate(half - 1), Some(hits[0]));
            assert_eq!(pairs.locate(half), Some(hits[1]));
            assert!(pairs.contains_boundary(adjacent));
            assert!(!pairs.contains_boundary(CdfRange::new(0, half - 1)));
            pairs.query_tall(CdfRange::FULL, half + 1, &mut scratch, &mut hits);
            assert!(hits.is_empty());

            let singleton = S::from_logits(&[0.0]);
            singleton.query_tall(CdfRange::FULL, u64::MAX, &mut scratch, &mut hits);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].span_m1(), u64::MAX);
            assert_eq!(singleton.locate(u64::MAX), Some(hits[0]));
            assert!(!singleton.contains_boundary(CdfRange::FULL));

            let empty = S::from_logits(&[]);
            empty.query_tall(CdfRange::FULL, 0, &mut scratch, &mut hits);
            assert!(hits.is_empty());
            assert_eq!(empty.locate(0), None);
            assert!(!empty.contains_boundary(CdfRange::FULL));
        }

        check::<BTreeMax>();
        check::<EytzingerMax>();
        check::<TieredPStore>();
    }
}
