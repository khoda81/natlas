//! Quantile-oriented layout prototype. Fixed-point CDF cuts place children.
//! Indexed surprisal widths size them. Subpixel rows use a midpoint estimate.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::size_of;

use crate::pstore::{max_logit, normalize_logits, BranchHit, CdfRange};
use crate::tree::{child_hash, derive_branch, Branch, Lens, NodeId, ROOT};
use crate::Result;

const CDF_SIZE: u128 = 1 << 64;
const TIERS: usize = u64::BITS as usize + 1;
const GAP: u32 = u32::MAX;

/// Branches are stored once, in descending mass tiers and in CDF order within
/// each tier. Compact sorted ends and entry indices serve point queries.
pub struct TieredPStore {
    entries: Vec<BranchHit>,
    tier_starts: [u32; TIERS + 1],
    ends: Vec<u64>,
    pos_to_entry: Vec<u32>,
}

impl TieredPStore {
    pub fn from_logits(logits: &[f32]) -> Self {
        let (starts, ends, branches, weights) = normalize_logits(logits);
        let mut entries = Vec::with_capacity(starts.len());
        for (((start, end), branch), ln_p) in
            starts.into_iter().zip(ends).zip(branches).zip(weights)
        {
            entries.push(BranchHit {
                start,
                end,
                branch,
                ln_p,
            });
        }
        Self::from_entries(&entries)
    }

    /// Disjoint known intervals in ascending CDF order, one per branch.
    /// The uncovered coordinates remain unknown.
    pub fn from_sorted_entries(entries: &[BranchHit]) -> Self {
        let mut branches = HashSet::with_capacity(entries.len());
        for entry in entries {
            assert!(
                branches.insert(entry.branch),
                "a branch may have only one CDF interval"
            );
        }
        Self::from_entries(entries)
    }

    fn from_entries(raw: &[BranchHit]) -> Self {
        u32::try_from(raw.len()).expect("the tier index must fit in u32");
        let mut counts = [0u32; TIERS];
        let mut next = 0;
        let mut positions = raw.len();
        for entry in raw {
            let start = u128::from(entry.start);
            assert!(
                next <= start && entry.start <= entry.end,
                "invalid CDF interval order"
            );
            assert!(
                entry.ln_p.is_finite() && entry.ln_p <= 0.0,
                "invalid log probability"
            );
            positions += usize::from(next < start);
            next = u128::from(entry.end) + 1;
            counts[entry.span_m1().leading_zeros() as usize] += 1;
        }
        positions += usize::from(!raw.is_empty() && next < CDF_SIZE);

        let mut tier_starts = [0u32; TIERS + 1];
        for (tier, count) in counts.into_iter().enumerate() {
            tier_starts[tier + 1] = tier_starts[tier] + count;
        }
        let mut entries = raw.to_vec();
        let mut cursors = tier_starts;
        let mut ends = Vec::with_capacity(positions);
        let mut pos_to_entry = Vec::with_capacity(positions);
        next = 0;
        for &entry in raw {
            if next < u128::from(entry.start) {
                ends.push(entry.start - 1);
                pos_to_entry.push(GAP);
            }
            let tier = entry.span_m1().leading_zeros() as usize;
            let index = cursors[tier];
            cursors[tier] += 1;
            entries[index as usize] = entry;
            ends.push(entry.end);
            pos_to_entry.push(index);
            next = u128::from(entry.end) + 1;
        }
        if !raw.is_empty() && next < CDF_SIZE {
            ends.push(u64::MAX);
            pos_to_entry.push(GAP);
        }
        Self {
            entries,
            tier_starts,
            ends,
            pos_to_entry,
        }
    }

    /// Replace `out` with known overlapping branches at least `min_mass`
    /// CDF units tall, heaviest first, with ties in CDF order.
    pub fn query_tall(&self, range: CdfRange, min_mass: u64, out: &mut Vec<BranchHit>) {
        self.query_spans(range, min_mass.saturating_sub(1), out);
    }

    fn query_spans(&self, range: CdfRange, min_span_m1: u64, out: &mut Vec<BranchHit>) {
        out.clear();
        let (start, end) = range.bounds();
        let last_tier = min_span_m1.leading_zeros() as usize;
        for t in 0..=last_tier {
            let tier =
                &self.entries[self.tier_starts[t] as usize..self.tier_starts[t + 1] as usize];
            let first = tier.partition_point(|entry| entry.end < start);
            for &entry in &tier[first..] {
                if end < entry.start {
                    break;
                }
                if min_span_m1 <= entry.span_m1() {
                    out.push(entry);
                }
            }
        }
        out.sort_unstable_by(|a, b| {
            b.span_m1()
                .cmp(&a.span_m1())
                .then_with(|| a.start.cmp(&b.start))
        });
    }

    /// Find the branch containing a coordinate, if its interval is known.
    pub fn locate(&self, at: u64) -> Option<BranchHit> {
        let pos = self.ends.partition_point(|&end| end < at);
        let &index = self.pos_to_entry.get(pos)?;
        (index != GAP).then(|| self.entries[index as usize])
    }

    /// Whether the range straddles a branch or gap boundary.
    pub fn contains_boundary(&self, range: CdfRange) -> bool {
        let (start, end) = range.bounds();
        let pos = self.ends.partition_point(|&end| end < start);
        self.ends.get(pos).is_some_and(|&cut| cut < end)
    }

    pub fn memory_bytes(&self) -> usize {
        size_of::<Self>() + self.heap_bytes()
    }

    fn heap_bytes(&self) -> usize {
        self.entries.capacity() * size_of::<BranchHit>()
            + self.ends.capacity() * size_of::<u64>()
            + self.pos_to_entry.capacity() * size_of::<u32>()
    }
}

/// Lookup of conditional surprisal by branch. Missing or weightless branches
/// have infinite surprisal.
pub struct QStore(Widths);

enum Widths {
    Dense(Vec<f32>),
    Sparse(Vec<(Branch, f32)>),
}

impl QStore {
    pub fn from_logits(logits: &[f32]) -> Self {
        let max = f64::from(max_logit(logits));
        if max == f64::NEG_INFINITY {
            return Self(Widths::Dense(vec![f32::INFINITY; logits.len()]));
        }
        let shifted_sum: f64 = logits
            .iter()
            .map(|&logit| (f64::from(logit) - max).exp())
            .sum();
        let ln_sum = shifted_sum.ln();
        let widths = logits
            .iter()
            .map(|&logit| (ln_sum + (max - f64::from(logit))) as f32)
            .collect();
        Self(Widths::Dense(widths))
    }

    pub fn from_sparse(mut widths: Vec<(Branch, f32)>) -> Self {
        assert!(
            widths.iter().all(|&(_, nats)| 0.0 <= nats),
            "invalid surprisal"
        );
        widths.sort_unstable_by_key(|&(branch, _)| branch);
        assert!(
            widths.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "duplicate width for branch"
        );
        Self(Widths::Sparse(widths))
    }

    pub fn nats(&self, branch: Branch) -> f32 {
        match &self.0 {
            Widths::Dense(widths) => widths
                .get(branch as usize)
                .copied()
                .unwrap_or(f32::INFINITY),
            Widths::Sparse(widths) => widths
                .binary_search_by_key(&branch, |&(b, _)| b)
                .map_or(f32::INFINITY, |index| widths[index].1),
        }
    }

    fn heap_bytes(&self) -> usize {
        match &self.0 {
            Widths::Dense(widths) => widths.capacity() * size_of::<f32>(),
            Widths::Sparse(widths) => widths.capacity() * size_of::<(Branch, f32)>(),
        }
    }
}

#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("only NodeId keys are hashed")
    }

    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIdx(u32);

pub const ROOT_IDX: NodeIdx = NodeIdx(0);

struct Node {
    id: NodeId,
    parent: Option<NodeIdx>,
    /// Expanded edges are sorted by branch and searched within this node.
    children: Vec<(Branch, NodeIdx)>,
    measures: Vec<Measure>,
}

impl Node {
    fn new(id: NodeId, parent: Option<NodeIdx>) -> Self {
        Self {
            id,
            parent,
            children: Vec::new(),
            measures: Vec::new(),
        }
    }

    fn slot(&mut self, measure: usize) -> &mut Measure {
        if self.measures.len() <= measure {
            self.measures.resize_with(measure + 1, Measure::default);
        }
        &mut self.measures[measure]
    }
}

#[derive(Default)]
struct Measure {
    p: Option<TieredPStore>,
    q: Option<QStore>,
}

/// Arena-backed client tree. The node-ID map is used for naming and collision
/// checks. Layout walks the arena and each node's expanded edges.
pub struct FastTree {
    by_id: HashMap<NodeId, NodeIdx, BuildHasherDefault<IdHasher>>,
    nodes: Vec<Node>,
}

impl Default for FastTree {
    fn default() -> Self {
        Self::new()
    }
}

impl FastTree {
    pub fn new() -> Self {
        let mut by_id = HashMap::default();
        by_id.insert(ROOT, ROOT_IDX);
        Self {
            by_id,
            nodes: vec![Node::new(ROOT, None)],
        }
    }

    pub fn idx(&self, id: NodeId) -> Option<NodeIdx> {
        self.by_id.get(&id).copied()
    }

    pub fn id(&self, node: NodeIdx) -> NodeId {
        self.nodes[node.0 as usize].id
    }

    pub fn parent(&self, node: NodeIdx) -> Option<NodeIdx> {
        self.nodes[node.0 as usize].parent
    }

    pub fn branch(&self, node: NodeIdx) -> Option<Branch> {
        self.parent(node).map(|parent| {
            derive_branch(self.id(parent), self.id(node)).expect("a stored edge has a branch")
        })
    }

    pub fn get_child(&self, parent: NodeIdx, branch: Branch) -> Option<NodeIdx> {
        let edges = &self.nodes[parent.0 as usize].children;
        edges
            .binary_search_by_key(&branch, |&(b, _)| b)
            .ok()
            .map(|index| edges[index].1)
    }

    pub fn child(&mut self, parent: NodeIdx, branch: Branch) -> Result<NodeIdx> {
        let edges = &self.nodes[parent.0 as usize].children;
        let slot = match edges.binary_search_by_key(&branch, |&(b, _)| b) {
            Ok(index) => return Ok(edges[index].1),
            Err(slot) => slot,
        };
        let id = child_hash(self.id(parent), branch);
        if self.by_id.contains_key(&id) {
            return Err(format!("node id {id:#018x} collides"));
        }
        let idx = NodeIdx(u32::try_from(self.nodes.len()).map_err(|_| "node index exhausted")?);
        self.nodes.push(Node::new(id, Some(parent)));
        self.nodes[parent.0 as usize]
            .children
            .insert(slot, (branch, idx));
        self.by_id.insert(id, idx);
        Ok(idx)
    }

    pub fn set_measure_logits(&mut self, node: NodeIdx, measure: usize, logits: &[f32]) {
        let p = TieredPStore::from_logits(logits);
        let q = QStore::from_logits(logits);
        *self.nodes[node.0 as usize].slot(measure) = Measure {
            p: Some(p),
            q: Some(q),
        };
    }

    pub fn set_p(&mut self, node: NodeIdx, measure: usize, store: TieredPStore) {
        self.nodes[node.0 as usize].slot(measure).p = Some(store);
    }

    pub fn set_q(&mut self, node: NodeIdx, measure: usize, store: QStore) {
        self.nodes[node.0 as usize].slot(measure).q = Some(store);
    }

    pub fn p_store(&self, node: NodeIdx, measure: usize) -> Option<&TieredPStore> {
        self.nodes[node.0 as usize]
            .measures
            .get(measure)?
            .p
            .as_ref()
    }

    pub fn q_store(&self, node: NodeIdx, measure: usize) -> Option<&QStore> {
        self.nodes[node.0 as usize]
            .measures
            .get(measure)?
            .q
            .as_ref()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Estimated footprint of the arena, indices, and measure stores.
    pub fn memory_bytes(&self) -> usize {
        let mut total = size_of::<Self>()
            + self.nodes.capacity() * size_of::<Node>()
            + self.by_id.capacity() * size_of::<(NodeId, NodeIdx)>();
        for node in &self.nodes {
            total += node.children.capacity() * size_of::<(Branch, NodeIdx)>()
                + node.measures.capacity() * size_of::<Measure>();
            for measure in &node.measures {
                total += measure.p.as_ref().map_or(0, TieredPStore::heap_bytes);
                total += measure.q.as_ref().map_or(0, QStore::heap_bytes);
            }
        }
        total
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QBox {
    pub node: NodeIdx,
    pub branch: Branch,
    pub expanded: bool,
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
    pub ln_p: f32,
    pub nats_q: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct QSpan {
    pub node: NodeIdx,
    pub row_start: u32,
    pub row_end: u32,
    pub x0: f32,
    pub fill: f32,
    pub reach: f32,
}

/// Reuse the frame and work buffers across calls to `layout_frame`.
#[derive(Default)]
pub struct QFrame {
    pub boxes: Vec<QBox>,
    pub spans: Vec<QSpan>,
    pending: Vec<PlacedNode>,
    hits: Vec<BranchHit>,
    rows: Vec<Option<Row>>,
}

#[derive(Clone, Copy, PartialEq)]
struct Row {
    fill: f32,
    reach: f32,
}

#[derive(Clone, Copy)]
struct PlacedNode {
    node: NodeIdx,
    range: CdfRange,
    x: f32,
    top: f64,
    bottom: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QViewport {
    pub root: NodeIdx,
    /// Visible coordinates within the root.
    pub range: CdfRange,
    /// Pixel interval occupied by that range in the viewport.
    pub visible_rows: [f64; 2],
    pub x_left: f32,
    pub x_right: f32,
    pub rows: u32,
}

pub fn layout_frame(tree: &FastTree, lens: Lens, vp: QViewport, frame: &mut QFrame) {
    let [top, bottom] = vp.visible_rows;
    assert!(
        0.0 <= top && top < bottom && bottom <= f64::from(vp.rows),
        "invalid visible rows"
    );
    assert!(
        vp.x_left.is_finite() && vp.x_right.is_finite() && vp.x_left < vp.x_right,
        "invalid horizontal window"
    );
    frame.boxes.clear();
    frame.spans.clear();
    frame.rows.resize(vp.rows as usize, None);
    frame.pending.push(PlacedNode {
        node: vp.root,
        range: vp.range,
        x: 0.0,
        top,
        bottom,
    });
    while let Some(place) = frame.pending.pop() {
        lay_node(tree, lens, vp, place, frame);
    }
}

fn lay_node(tree: &FastTree, lens: Lens, vp: QViewport, place: PlacedNode, frame: &mut QFrame) {
    if vp.x_right <= place.x {
        return;
    }
    let Some(p) = tree.p_store(place.node, lens.p) else {
        return;
    };
    let q = tree.q_store(place.node, lens.q);
    let width = |hit: BranchHit| {
        if lens.p == lens.q {
            -hit.ln_p
        } else {
            q.map_or(f32::INFINITY, |q| q.nats(hit.branch))
        }
    };

    let (start, end) = place.range.bounds();
    let first_cut = u128::from(start);
    let last_cut = u128::from(end) + 1;
    let unit_count = last_cut - first_cut;
    let height = place.bottom - place.top;
    let units_per_row = unit_count as f64 / height;
    let min_span_m1 = (units_per_row.ceil() as u128).saturating_sub(1);
    if let Ok(span) = u64::try_from(min_span_m1) {
        p.query_spans(place.range, span, &mut frame.hits);
    } else {
        frame.hits.clear();
    }

    let first_row = place.top.floor() as usize;
    let end_row = place.bottom.ceil() as usize;
    frame.rows[first_row..end_row].fill(None);

    // Subtract the cut in integers before converting to pixels. Coordinates
    // above the visible slice must remain negative relative to its top.
    let row_at = |cut: u128| place.top + (cut as i128 - first_cut as i128) as f64 / units_per_row;
    for &hit in &frame.hits {
        let hit_start = u128::from(hit.start);
        let hit_end = u128::from(hit.end) + 1;
        let visible_start = hit_start.max(first_cut);
        let visible_end = hit_end.min(last_cut);
        let visible_top = row_at(visible_start).max(place.top);
        let visible_bottom = row_at(visible_end).min(place.bottom);
        if visible_bottom <= visible_top {
            continue;
        }
        let nats_q = width(hit);
        let x1 = place.x + nats_q;
        let expanded = tree
            .get_child(place.node, hit.branch)
            .filter(|&child| tree.p_store(child, lens.p).is_some());
        if vp.x_left < x1 {
            frame.boxes.push(QBox {
                node: place.node,
                branch: hit.branch,
                expanded: expanded.is_some(),
                x0: place.x,
                x1,
                y0: row_at(hit_start) as f32,
                y1: row_at(hit_end) as f32,
                ln_p: hit.ln_p,
                nats_q,
            });
        }

        let inside_start = visible_top.ceil() as usize;
        let inside_end = visible_bottom.floor() as usize;
        for row in visible_top.floor() as usize..visible_bottom.ceil() as usize {
            let painted = frame.rows[row].get_or_insert(Row {
                fill: 0.0,
                reach: nats_q,
            });
            painted.reach = painted.reach.max(nats_q);
            if inside_start <= row && row < inside_end && nats_q.is_finite() {
                painted.fill = nats_q;
            }
        }

        // A parent's box can end left of the screen while its children still
        // reach into view. Schedule traversal independently of box emission.
        if let Some(child) = expanded.filter(|_| x1 < vp.x_right) {
            let mass = hit_end - hit_start;
            let start = (visible_start - hit_start) * CDF_SIZE / mass;
            let end = if visible_end == hit_end {
                CDF_SIZE
            } else {
                ((visible_end - hit_start) * CDF_SIZE).div_ceil(mass)
            };
            frame.pending.push(PlacedNode {
                node: child,
                range: CdfRange::new(start as u64, (end - 1) as u64),
                x: x1,
                top: visible_top,
                bottom: visible_bottom,
            });
        }
    }

    // Rows untouched by tall branches use one sample in the visible part of
    // the row. Their estimated reach can miss a wider branch off the midpoint.
    for row in first_row..end_row {
        let paint = frame.rows[row].unwrap_or_else(|| {
            let top = place.top.max(row as f64);
            let bottom = place.bottom.min((row + 1) as f64);
            let offset = (((top + bottom) * 0.5 - place.top) * units_per_row) as u128;
            let at = (first_cut + offset.min(unit_count - 1)) as u64;
            let reach = p.locate(at).map_or(f32::INFINITY, width);
            Row { fill: 0.0, reach }
        });
        let reach = place.x + paint.reach;
        if reach <= vp.x_left {
            continue;
        }
        let fill = place.x + paint.fill;
        let row_start = row as u32;
        let row_end = row_start + 1;
        if let Some(last) = frame.spans.last_mut().filter(|last| {
            last.node == place.node
                && last.row_end == row_start
                && last.fill == fill
                && last.reach == reach
        }) {
            last.row_end = row_end;
        } else {
            frame.spans.push(QSpan {
                node: place.node,
                row_start,
                row_end,
                x0: place.x,
                fill,
                reach,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pstore::{bench_logits, bench_queries, PStore, QueryScratch};
    use crate::tree::derive_parent;

    #[test]
    fn shared_node_ids_are_invertible() {
        for (parent, branch) in [
            (ROOT, 0),
            (ROOT, 151_935),
            (0xdead_beef_cafe_babe, 42),
            (u64::MAX, u32::MAX),
        ] {
            let child = child_hash(parent, branch);
            assert_eq!(derive_branch(parent, child), Some(branch));
            assert_eq!(derive_parent(child, branch), parent);
        }
        let mut tree = FastTree::new();
        let child = tree.child(ROOT_IDX, Branch::MAX).unwrap();
        assert_eq!(tree.child(ROOT_IDX, Branch::MAX), Ok(child));
        assert_eq!(tree.idx(child_hash(ROOT, Branch::MAX)), Some(child));
        assert_eq!(tree.parent(child), Some(ROOT_IDX));
        assert_eq!(tree.branch(child), Some(Branch::MAX));
    }

    #[test]
    fn compact_tiers_agree_with_the_previous_store() {
        let mut scratch = QueryScratch::default();
        let mut reference = Vec::new();
        let mut hits = Vec::new();
        for kind in 0..3 {
            let logits = bench_logits(kind, 4096);
            let old = crate::pstore::TieredPStore::from_logits(&logits);
            let new = TieredPStore::from_logits(&logits);
            for (range, min_mass) in bench_queries(64) {
                old.query_tall(range, min_mass, &mut scratch, &mut reference);
                new.query_tall(range, min_mass, &mut hits);
                assert_eq!(hits, reference);
                assert_eq!(new.locate(range.midpoint()), old.locate(range.midpoint()));
                assert_eq!(new.contains_boundary(range), old.contains_boundary(range));
            }
            assert_eq!(new.locate(u64::MAX), old.locate(u64::MAX));
        }
    }

    #[test]
    fn sparse_tiers_keep_gaps_at_both_ends_and_between_entries() {
        let sparse = TieredPStore::from_sorted_entries(&[
            BranchHit {
                start: 10,
                end: 100,
                branch: 7,
                ln_p: -1.0,
            },
            BranchHit {
                start: 500,
                end: 600,
                branch: 9,
                ln_p: -0.5,
            },
        ]);
        for gap in [0, 9, 101, 499, 601, u64::MAX] {
            assert!(sparse.locate(gap).is_none());
        }
        assert_eq!(sparse.locate(10).unwrap().branch, 7);
        assert_eq!(sparse.locate(600).unwrap().branch, 9);
        assert!(sparse.contains_boundary(CdfRange::new(9, 10)));
        assert!(!sparse.contains_boundary(CdfRange::new(101, 499)));
        let mut hits = Vec::new();
        sparse.query_tall(CdfRange::FULL, 92, &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].branch, 9);
    }

    #[test]
    fn clipped_windows_reach_children_to_the_right_of_hidden_parents() {
        let mut tree = FastTree::new();
        tree.set_measure_logits(ROOT_IDX, 0, &[0.0, 0.0]);
        let child = tree.child(ROOT_IDX, 0).unwrap();
        tree.set_measure_logits(child, 0, &[0.0, 0.0]);
        let quarter = 1u64 << 62;
        let mut frame = QFrame::default();
        layout_frame(
            &tree,
            Lens { p: 0, q: 0 },
            QViewport {
                root: ROOT_IDX,
                range: CdfRange::new(quarter, 3 * quarter - 1),
                visible_rows: [0.0, 4.0],
                x_left: 1.0,
                x_right: 4.0,
                rows: 4,
            },
            &mut frame,
        );
        assert_eq!(frame.boxes.len(), 1);
        let b = frame.boxes[0];
        assert_eq!((b.node, b.branch, b.y0, b.y1), (child, 1, 0.0, 2.0));
        assert_eq!(frame.spans.len(), 1);
        assert_eq!((frame.spans[0].row_start, frame.spans[0].row_end), (0, 2));
        assert_eq!(frame.spans[0].fill, b.x1);
    }

    #[test]
    fn a_missing_q_stays_unknown_and_sparse_q_uses_branch_ids() {
        let mut tree = FastTree::new();
        tree.set_measure_logits(ROOT_IDX, 0, &[0.0, 0.0]);
        let mut frame = QFrame::default();
        let vp = QViewport {
            root: ROOT_IDX,
            range: CdfRange::FULL,
            visible_rows: [0.0, 2.0],
            x_left: 0.0,
            x_right: 4.0,
            rows: 2,
        };
        let lens = Lens { p: 0, q: 1 };
        layout_frame(&tree, lens, vp, &mut frame);
        assert_eq!(frame.boxes.len(), 2);
        assert!(frame.boxes.iter().all(|b| b.nats_q == f32::INFINITY));
        tree.set_q(ROOT_IDX, 1, QStore::from_sparse(vec![(1, 3.0), (0, 2.0)]));
        layout_frame(&tree, lens, vp, &mut frame);
        assert_eq!((frame.boxes[0].nats_q, frame.boxes[1].nats_q), (2.0, 3.0));
        assert_eq!(
            QStore::from_logits(&[f32::MAX; 2]).nats(0),
            std::f32::consts::LN_2
        );
    }
}
