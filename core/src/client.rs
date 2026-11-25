//! The browser's incremental tree. Rank intervals own the wire precision;
//! the quantile index is a read-time view of those intervals.

use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use crate::distribution::{
    same_branches, sort_runs, BranchHasher, Item, Rank, RankView, Run, RunRef, RunSlice,
};
use crate::point::Point;
use crate::pstore::{TieredPStore, CDF_SIZE};
use crate::qlayout::WidthIndex;
use crate::tree::{child_hash, derive_branch, Branch, Entry, Lens, NodeId, Statistics, ROOT};
use crate::Result;

#[derive(Clone, Copy)]
struct Width {
    rank: Rank,
    nats: f64,
}

/// Branch-indexed ranks and conditional surprisal. As a dense prefix forms,
/// each entry moves from the sparse table to flat arrays at most once.
#[derive(Default)]
pub(crate) struct QStore {
    ranks: Vec<Rank>,
    nats: Vec<f64>,
    sparse: HashMap<Branch, Width, BuildHasherDefault<BranchHasher>>,
}

impl QStore {
    #[inline]
    fn get(&self, branch: Branch) -> Option<Width> {
        let at = branch as usize;
        self.ranks.get(at).map_or_else(
            || self.sparse.get(&branch).copied(),
            |&rank| {
                Some(Width {
                    rank,
                    nats: self.nats[at],
                })
            },
        )
    }

    #[inline]
    pub(crate) fn nats(&self, branch: Branch) -> f64 {
        self.nats
            .get(branch as usize)
            .copied()
            .or_else(|| self.sparse.get(&branch).map(|width| width.nats))
            .unwrap_or(f64::INFINITY)
    }

    fn insert(&mut self, branch: Branch, rank: Rank, nats: f64) {
        let at = branch as usize;
        let width = Width { rank, nats };
        if at < self.ranks.len() {
            self.ranks[at] = rank;
            self.nats[at] = nats;
        } else if self.ranks.len() < at {
            self.sparse.insert(branch, width);
        } else {
            self.ranks.push(rank);
            self.nats.push(nats);
            while let Some(next) = self.sparse.remove(&(self.ranks.len() as Branch)) {
                self.ranks.push(next.rank);
                self.nats.push(next.nats);
            }
        }
    }

    fn from_segments(segments: &[Segment], len: usize) -> Result<Self> {
        let mut result = Self::default();
        if segments
            .iter()
            .all(|s| s.view().branches().iter().all(|&b| (b as usize) < len))
        {
            // A batch can give every dense branch in any rank order. Rank::MAX
            // is an exclusive end, so it also marks an empty slot during validation.
            result.ranks = vec![Rank::MAX; len];
            result.nats = vec![0.0; len];
            for (branch, rank, nats) in segments.iter().flat_map(Segment::children) {
                let at = branch as usize;
                if result.ranks[at] != Rank::MAX {
                    return Err(format!("branch {branch} is placed twice"));
                }
                result.ranks[at] = rank;
                result.nats[at] = nats;
            }
        } else {
            result.sparse.reserve(len);
            for (branch, rank, nats) in segments.iter().flat_map(Segment::children) {
                if result.get(branch).is_some() {
                    return Err(format!("branch {branch} is placed twice"));
                }
                result.insert(branch, rank, nats);
            }
        }
        Ok(result)
    }

    fn len(&self) -> usize {
        self.ranks.len() + self.sparse.len()
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.ranks.capacity() * size_of::<Rank>()
            + self.nats.capacity() * size_of::<f64>()
            + self.sparse.capacity() * (size_of::<(Branch, Width)>() + 1)
    }
}

#[derive(Default)]
pub(crate) struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("only mixed node ids are hashed")
    }

    fn write_u64(&mut self, id: u64) {
        self.0 = id;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct NodeIdx(usize);

struct Node {
    parent: NodeId,
    measures: Vec<Option<ClientMeasure>>,
}

pub struct ClientTree {
    indices: HashMap<NodeId, NodeIdx, BuildHasherDefault<IdHasher>>,
    nodes: Vec<Node>,
}

impl Default for ClientTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientTree {
    pub fn new() -> Self {
        Self {
            indices: HashMap::from_iter([(ROOT, NodeIdx(0))]),
            nodes: vec![Node {
                parent: ROOT,
                measures: Vec::new(),
            }],
        }
    }

    pub fn contains(&self, node: NodeId) -> bool {
        self.indices.contains_key(&node)
    }

    pub fn child(&mut self, parent: NodeId, branch: Branch) -> Result<NodeId> {
        self.index(parent)?;
        let id = child_hash(parent, branch);
        if let Some(index) = self.indices.get(&id) {
            if id != ROOT && self.nodes[index.0].parent == parent {
                return Ok(id);
            }
            return Err(format!("node id {id:#018x} collides"));
        }
        self.indices.insert(id, NodeIdx(self.nodes.len()));
        self.nodes.push(Node {
            parent,
            measures: Vec::new(),
        });
        Ok(id)
    }

    pub fn descend(&mut self, mut node: NodeId, path: &[Branch]) -> Result<NodeId> {
        self.index(node)?;
        for &branch in path {
            node = self.child(node, branch)?;
        }
        Ok(node)
    }

    pub fn parent(&self, id: NodeId) -> Result<Option<NodeId>> {
        let index = self.index(id)?;
        Ok((id != ROOT).then_some(self.nodes[index.0].parent))
    }

    pub fn branch(&self, id: NodeId) -> Result<Option<Branch>> {
        Ok(self
            .parent(id)?
            .map(|parent| derive_branch(parent, id).expect("a node id names its branch")))
    }

    pub fn chain(&self, origin: NodeId, id: NodeId) -> Result<Vec<NodeId>> {
        let mut chain = vec![id];
        let mut current = id;
        while current != origin {
            current = self.parent(current)?.ok_or_else(|| {
                format!("node {origin:#018x} is not an ancestor of node {id:#018x}")
            })?;
            chain.push(current);
        }
        chain.reverse();
        Ok(chain)
    }

    pub fn path(&self, id: NodeId) -> Result<Vec<Branch>> {
        Ok(self
            .chain(ROOT, id)?
            .windows(2)
            .map(|edge| derive_branch(edge[0], edge[1]).expect("a node id names its branch"))
            .collect())
    }

    fn index(&self, id: NodeId) -> Result<NodeIdx> {
        self.indices
            .get(&id)
            .copied()
            .ok_or_else(|| format!("unknown node {id:#018x}"))
    }

    pub fn distribution(&self, id: NodeId, measure: usize) -> Option<&ClientMeasure> {
        let index = self.indices.get(&id)?;
        self.nodes[index.0].measures.get(measure)?.as_ref()
    }

    pub fn insert(
        &mut self,
        id: NodeId,
        measure: usize,
        runs: Vec<Run>,
        count: Option<Rank>,
    ) -> Result<bool> {
        let index = self.index(id)?;
        if runs.is_empty() && count.is_none() {
            return Ok(false);
        }
        let measures = &mut self.nodes[index.0].measures;
        if let Some(Some(existing)) = measures.get_mut(measure) {
            return existing.insert(runs, count);
        }
        let mut distribution = ClientMeasure::default();
        distribution.insert(runs, count)?;
        if measures.len() <= measure {
            measures.resize_with(measure + 1, || None);
        }
        measures[measure] = Some(distribution);
        Ok(true)
    }

    pub(crate) fn nats_q(&self, node: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> f64 {
        if lens.p == lens.q {
            return -ln_p;
        }
        self.distribution(node, lens.q)
            .map_or(f64::INFINITY, |q| q.nats(branch))
    }

    pub fn entry(&self, node: NodeId, lens: Lens, branch: Branch) -> Option<Entry> {
        let (rank, top, ln_p) = self.distribution(node, lens.p)?.entry(branch)?;
        Some(Entry {
            rank,
            before: Point::ln_between(&Point::top(), &top).exp(),
            ln_p,
            nats_q: self.nats_q(node, lens, branch, ln_p),
        })
    }

    pub fn statistics(&self, node: NodeId, lens: Lens) -> Statistics {
        let mut stats = Statistics {
            entropy: 0.0,
            cross: 0.0,
            bounded: true,
        };
        let Some(p) = self.distribution(node, lens.p) else {
            return stats;
        };
        let q = self.distribution(node, lens.q);
        stats.bounded = !p.complete();
        for segment in &p.segments {
            for (&branch, &ln_p) in segment
                .view()
                .branches()
                .iter()
                .zip(segment.view().ln_weights())
            {
                if f64::NEG_INFINITY < ln_p && ln_p < 0.0 {
                    stats.entropy += -ln_p * ln_p.exp();
                }
                if lens.p != lens.q {
                    let nats = q.map_or(f64::INFINITY, |q| q.nats(branch));
                    if nats.is_finite() {
                        stats.cross += ln_p.exp() * nats;
                    } else {
                        stats.bounded = true;
                    }
                }
            }
        }
        if lens.p == lens.q {
            stats.cross = stats.entropy;
        }
        stats
    }

    pub fn bytes(&self) -> usize {
        self.nodes
            .iter()
            .flat_map(|node| node.measures.iter().flatten())
            .map(ClientMeasure::bytes)
            .sum()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

struct Segment {
    source: Arc<Run>,
    ranks: Range<Rank>,
    cuts: [u128; 2],
}

impl Segment {
    fn new(run: Run) -> Self {
        let ranks = run.rank()..run.end();
        Self::from_source(Arc::new(run), ranks)
    }

    fn from_source(source: Arc<Run>, ranks: Range<Rank>) -> Self {
        let base = source.q64_base();
        let offsets = source.window(ranks.start, ranks.end).offsets();
        let cuts = [
            fixed_cut(base, offsets[0]),
            fixed_cut(base, offsets[offsets.len() - 1]),
        ];
        Self {
            source,
            ranks,
            cuts,
        }
    }

    fn view(&self) -> RunSlice<'_> {
        self.source.window(self.ranks.start, self.ranks.end)
    }

    fn slice(&self, start: Rank, end: Rank) -> Self {
        Self::from_source(Arc::clone(&self.source), start..end)
    }

    fn children(&self) -> impl Iterator<Item = (Branch, Rank, f64)> + '_ {
        let run = self.view();
        run.branches()
            .iter()
            .copied()
            .zip(run.ln_weights().iter().copied())
            .zip(self.ranks.clone())
            .map(|((branch, ln_p), rank)| (branch, rank, -ln_p))
    }
}

#[inline]
fn fixed_cut(base: u128, offset: f64) -> u128 {
    let offset = offset * CDF_SIZE as f64;
    if offset < CDF_SIZE as f64 {
        base + u128::from(offset as u64)
    } else if offset == CDF_SIZE as f64 {
        base + CDF_SIZE
    } else {
        CDF_SIZE + 1
    }
}

/// The ordered items and their CDF partition. Coincident items keep their
/// rank and boundary here. The mass tiers only index positive intervals.
pub(crate) struct FixedIndex {
    pub(crate) p: TieredPStore,
    pub(crate) items: Vec<Item>,
    pub(crate) widths: RefCell<Option<((usize, u64), WidthIndex)>>,
}

#[doc(hidden)]
#[derive(Default)]
pub struct ClientMeasure {
    /// Disjoint rank segments sorted by ascending `ranks.start`.
    segments: Vec<Segment>,
    q: QStore,
    count: Option<Rank>,
    revision: u64,
    fixed: OnceCell<Option<FixedIndex>>,
}

impl ClientMeasure {
    pub(crate) fn complete(&self) -> bool {
        self.count == Some(self.q.len() as Rank)
    }

    #[cfg(feature = "wasm")]
    pub(crate) fn ended(&self) -> bool {
        self.count == Some(0)
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    #[inline]
    fn segment_index(&self, rank: Rank) -> Option<usize> {
        self.segments
            .partition_point(|s| s.ranks.start <= rank)
            .checked_sub(1)
    }

    fn segment(&self, rank: Rank) -> &Segment {
        &self.segments[self.segment_index(rank).expect("a known rank has a run")]
    }

    pub(crate) fn nats(&self, branch: Branch) -> f64 {
        self.q.nats(branch)
    }

    pub(crate) fn entry(&self, branch: Branch) -> Option<(Rank, Point, f64)> {
        let rank = self.rank(branch)?;
        let run = self.segment(rank).view();
        let at = (rank - run.rank()) as usize;
        Some((rank, run.edge(at), run.ln_weights()[at]))
    }

    fn overlapping(&self, run: RunSlice<'_>) -> &[Segment] {
        let first = self.segments.partition_point(|s| s.ranks.end <= run.rank());
        let last = self.segments.partition_point(|s| s.ranks.start < run.end());
        &self.segments[first..last]
    }

    fn contradicts(&self, fresh: &[Segment], count: Option<Rank>, old_end: Rank) -> bool {
        if let Some(count) = count {
            if count < old_end || fresh.last().is_some_and(|last| count < last.ranks.end) {
                return true;
            }
        }
        for fresh in fresh {
            let run = fresh.view();
            if fresh
                .children()
                .any(|(branch, rank, _)| self.q.get(branch).is_some_and(|old| old.rank != rank))
            {
                return true;
            }
            if run.rank() < old_end {
                for old in self.overlapping(run) {
                    let start = old.ranks.start.max(run.rank());
                    let end = old.ranks.end.min(run.end());
                    if !same_branches(old.view(), run, start, end) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn insert(&mut self, mut runs: Vec<Run>, count: Option<Rank>) -> Result<bool> {
        sort_runs(&mut runs, count)?;
        let fresh: Vec<_> = runs.into_iter().map(Segment::new).collect();
        let len: usize = fresh.iter().map(|segment| segment.view().len()).sum();
        let merged_count = count.or(self.count);
        let old_end = self.segments.last().map_or(0, |s| s.ranks.end);
        if (self.segments.is_empty() && self.count.is_none())
            || self.contradicts(&fresh, merged_count, old_end)
        {
            self.q = QStore::from_segments(&fresh, len)?;
            self.segments = fresh;
            self.count = count;
            self.invalidate();
            return Ok(true);
        }
        if len > 1 {
            if len <= 16 {
                let mut small = [0 as Branch; 16];
                let mut count_seen = 0;
                for &branch in fresh.iter().flat_map(|segment| segment.view().branches()) {
                    if small[..count_seen].contains(&branch) {
                        return Err(format!("branch {branch} is placed twice"));
                    }
                    small[count_seen] = branch;
                    count_seen += 1;
                }
            } else {
                let mut seen = HashSet::with_capacity_and_hasher(
                    len,
                    BuildHasherDefault::<BranchHasher>::default(),
                );
                for &branch in fresh.iter().flat_map(|segment| segment.view().branches()) {
                    if !seen.insert(branch) {
                        return Err(format!("branch {branch} is placed twice"));
                    }
                }
            }
        }
        let mut changed = self.count != merged_count;
        self.count = merged_count;

        // Fast path: all fresh runs sit strictly after existing segments.
        if fresh
            .first()
            .is_none_or(|first| old_end <= first.ranks.start)
        {
            for seg in fresh {
                for (branch, rank, nats) in seg.children() {
                    self.q.insert(branch, rank, nats);
                }
                self.segments.push(seg);
                changed = true;
            }
            if changed {
                self.invalidate();
            }
            return Ok(changed);
        }

        // Linear two-pointer merge of sorted disjoint segments.
        let mut old = VecDeque::from(std::mem::take(&mut self.segments));
        let mut merged = Vec::with_capacity(old.len() + fresh.len());
        for f in fresh {
            let f_start = f.ranks.start;
            let f_end = f.ranks.end;
            let f_prec = f.view().precision();
            while old.front().is_some_and(|o| o.ranks.end <= f_start) {
                merged.push(old.pop_front().unwrap());
            }
            let mut from = f_start;
            while old.front().is_some_and(|o| o.ranks.start < f_end) {
                let o = old.pop_front().unwrap();
                let (o_start, o_end) = (o.ranks.start, o.ranks.end);
                if o.view() == f.view() {
                    merged.push(o);
                    from = f_end;
                    break;
                }
                if f_prec < o.view().precision() {
                    if from < o_start {
                        let part = f.slice(from, o_start);
                        for (branch, rank, nats) in part.children() {
                            self.q.insert(branch, rank, nats);
                        }
                        merged.push(part);
                        changed = true;
                    }
                    from = o_end.min(f_end);
                    if o_end <= f_end {
                        merged.push(o);
                    } else {
                        old.push_front(o);
                        break;
                    }
                } else {
                    if o_start < from {
                        merged.push(o.slice(o_start, from));
                    }
                    if f_end < o_end {
                        let tail = o.slice(f_end, o_end);
                        old.push_front(tail);
                        break;
                    }
                }
            }
            if from < f_end {
                let part = if from == f_start {
                    f
                } else {
                    f.slice(from, f_end)
                };
                for (branch, rank, nats) in part.children() {
                    self.q.insert(branch, rank, nats);
                }
                merged.push(part);
                changed = true;
            }
        }
        merged.extend(old);
        self.segments = merged;

        if changed {
            self.invalidate();
        }
        Ok(changed)
    }

    fn invalidate(&mut self) {
        self.revision += 1;
        self.fixed.take();
    }

    pub(crate) fn fixed(&self) -> Option<&FixedIndex> {
        self.fixed.get_or_init(|| self.build_fixed()).as_ref()
    }

    fn build_fixed(&self) -> Option<FixedIndex> {
        let mut items = Vec::with_capacity(self.q.len() + self.segments.len() + 1);
        let mut ends = Vec::with_capacity(items.capacity());
        let mut cut = 0;
        let mut push = |item, end| {
            if end == 0 || end < cut || CDF_SIZE < end {
                return None;
            }
            ends.push((end - 1) as u64);
            items.push(item);
            cut = end;
            Some(())
        };
        let mut rank = 0;
        for (run, segment) in self.segments.iter().enumerate() {
            if rank < segment.ranks.start {
                push(Item::Gap { before: run }, segment.cuts[0])?;
            }
            let end = segment.ranks.end;
            let last = match self.segments.get(run + 1) {
                Some(next) if next.ranks.start == end => next.cuts[0],
                None if self.count == Some(end) => CDF_SIZE,
                _ => segment.cuts[1],
            };
            let len = (end - segment.ranks.start) as usize;
            if 1 < len {
                let offsets = segment.view().offsets();
                let base = segment.source.q64_base();
                for (at, &offset) in offsets[1..len].iter().enumerate() {
                    push(Item::Child { run, at }, fixed_cut(base, offset))?;
                }
            }
            push(Item::Child { run, at: len - 1 }, last)?;
            rank = end;
        }
        if self.count.is_none_or(|count| rank < count) {
            push(
                Item::Gap {
                    before: self.segments.len(),
                },
                CDF_SIZE,
            )?;
        }
        Some(FixedIndex {
            p: TieredPStore::from_partition(ends),
            items,
            widths: RefCell::default(),
        })
    }

    pub(crate) fn fixed_widths(
        &self,
        fixed: &FixedIndex,
        q: Option<&Self>,
        same: bool,
    ) -> WidthIndex {
        let width = |run: RunSlice<'_>, at: usize| {
            if same {
                -run.ln_weights()[at]
            } else {
                q.map_or(f64::INFINITY, |q| q.nats(run.branches()[at]))
            }
        };
        let mut values = Vec::with_capacity(fixed.items.len());
        for &item in &fixed.items {
            let next = match item {
                Item::Child { run, at } => width(self.segments[run].view(), at),
                Item::Gap { .. } => f64::INFINITY,
            };
            values.push(next);
        }
        WidthIndex::new(0, values)
    }

    pub fn bytes(&self) -> usize {
        let mut data = HashSet::new();
        let runs: usize = self
            .segments
            .iter()
            .map(|segment| {
                size_of::<Segment>()
                    + if data.insert(Arc::as_ptr(&segment.source)) {
                        segment.source.bytes()
                    } else {
                        0
                    }
            })
            .sum();
        runs + self.q.heap_bytes()
            + self
                .fixed
                .get()
                .and_then(Option::as_ref)
                .map_or(0, |fixed| {
                    fixed.p.memory_bytes()
                        + fixed.items.capacity() * size_of::<Item>()
                        + fixed
                            .widths
                            .borrow()
                            .as_ref()
                            .map_or(0, |(_, widths)| widths.heap_bytes())
                })
    }
}

impl RankView for ClientMeasure {
    fn run_count(&self) -> usize {
        self.segments.len()
    }

    fn run_at(&self, index: usize) -> Option<RunRef<'_>> {
        self.segments.get(index).map(|s| RunRef::slice(s.view()))
    }

    fn rank(&self, branch: Branch) -> Option<Rank> {
        self.q.get(branch).map(|width| width.rank)
    }

    fn locate_rank(&self, rank: Rank) -> Option<(usize, usize)> {
        let index = self.segment_index(rank)?;
        let seg = &self.segments[index];
        (rank < seg.ranks.end).then(|| (index, (rank - seg.ranks.start) as usize))
    }

    fn locate(&self, point: &Point) -> Option<Item> {
        if let Some(fixed) = self.fixed.get().and_then(Option::as_ref) {
            let Ok(at) = u64::try_from(point.q64()) else {
                return None;
            };
            let position = fixed.p.position(at);
            if let Some(&item) = fixed.items.get(position) {
                let begin = crate::qlayout::QLayout::start(&fixed.p, position);
                // Run endpoints combine two individually rounded operands.
                // Check the original edges at a boundary where a bit can matter.
                if begin < u128::from(at) && at < fixed.p.ends()[position] {
                    return Some(item);
                }
            }
        }
        self.locate_exact(point)
    }

    fn count(&self) -> Option<Rank> {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    #[test]
    fn overlapping_and_replacing_samples_keep_the_source_edges_and_branch_widths() {
        let mut client = ClientTree::new();
        let mut source = Tree::new();
        let run = |rank, branches: &[Branch], weights: &[f64], top: &[u8]| {
            Run::new(
                rank,
                branches.to_vec(),
                weights.iter().map(|w| w.ln()).collect(),
                top,
            )
            .unwrap()
        };
        let answers = [
            (vec![run(0, &[9, 7, 4], &[0.5, 0.25, 0.125], &[0])], Some(4)),
            (vec![run(1, &[7], &[0.2], &[0x80, 0])], None),
            (vec![run(1, &[7], &[0.25], &[0x80])], None),
            (vec![run(3, &[2], &[0.125], &[0xe0])], None),
            // The same branch at a new rank replaces the entire prior answer.
            (vec![run(1, &[4, 3], &[0.25, 0.25], &[0x40])], Some(3)),
        ];
        for (runs, count) in answers {
            assert_eq!(
                client.insert(ROOT, 0, runs.clone(), count),
                source.insert(ROOT, 0, runs, count),
            );
            let actual = client.distribution(ROOT, 0).unwrap();
            let expected = source.distribution(ROOT, 0).unwrap().unwrap();
            assert_eq!(actual.count(), expected.count());
            let edges = |d: &dyn RankView| {
                std::iter::successors(d.first(), |&item| d.next(item))
                    .map(|item| (d.item_rank(item), d.item_end(item), d.item_edges(item)))
                    .collect::<Vec<_>>()
            };
            assert_eq!(edges(actual), edges(expected));
            for branch in [2, 3, 4, 7, 9] {
                assert_eq!(actual.rank(branch), expected.rank(branch));
                assert_eq!(
                    actual.nats(branch),
                    -expected.ln_p(branch).unwrap_or(f64::NEG_INFINITY)
                );
            }
        }

        let revision = client.distribution(ROOT, 0).unwrap().revision();
        let duplicate = run(0, &[2, 2], &[0.5, 0.5], &[0]);
        assert!(client.insert(ROOT, 0, vec![duplicate], Some(2)).is_err());
        assert_eq!(client.distribution(ROOT, 0).unwrap().revision(), revision);
    }
}
