//! A node's children under one measure, as far as they are known: runs of
//! consecutive ranks in the measure's own order, heaviest first.
//!
//! Positions are ln of the mass at and below an edge (ln of 1 − cdf). Ranks
//! sort the children by weight, so the mass below rank r is at most the
//! number of children below it times r's own weight: ln-below edges keep
//! every child's extent precise relative to its own weight, however light it
//! is or deep in its parent's tail.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::OnceLock;

use crate::point::Point;
use crate::tree::Branch;
use crate::Result;

/// A child's position in its measure's order: rank 0 is the heaviest.
pub type Rank = u32;

/// How far a window descends through children that hold it before a source
/// lists one: each step costs a server one forward pass.
pub const MAX_DESCENT: usize = 64;

/// ln(e^a + e^b), exact for tiny terms.
pub fn ln_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        high
    } else {
        high + (a.min(b) - high).exp().ln_1p()
    }
}

/// ln(e^top − e^bottom) for top ≥ bottom: the mass between two edges, exact
/// however close they are.
pub fn ln_between(top: f64, bottom: f64) -> f64 {
    if bottom == f64::NEG_INFINITY {
        top
    } else {
        top + (-(bottom - top).exp_m1()).ln()
    }
}

/// Consecutive ranks from `rank`: their branches and every edge between them.
#[derive(Clone, Debug)]
pub struct Run {
    rank: Rank,
    branches: Vec<Branch>,
    /// ln of the mass at and below each edge: edge i tops rank `rank + i`,
    /// and the last one bottoms the run.
    edges: Edges,
    /// ln Σ p (−ln p) over the run's children from each edge to the run's
    /// end, accumulated from the light end: a merged band's area-preserving
    /// width without visiting its children. Derived when first read: only
    /// layout reads it, and a server, which builds a run of its whole
    /// vocabulary per node, never does.
    ln_surprisal_below: OnceLock<Vec<f64>>,
}

/// Runs are what they place: the derived sums are not compared.
impl PartialEq for Run {
    fn eq(&self, other: &Self) -> bool {
        (self.rank, &self.branches, &self.edges) == (other.rank, &other.branches, &other.edges)
    }
}

/// A measure's run keeps every `KEPT`th edge and sums any other from the
/// nearest kept one below it. Measured on a 152k vocabulary, a read of 1600
/// points takes 2.4 ms with 8 (1.5 ms keeping every edge), 3.5 ms with 16 and
/// 5.9 ms with 32, for 13, 12.5 and 12.25 bytes per child.
const KEPT: usize = 8;

/// A run's edges.
#[derive(Clone, Debug, PartialEq)]
enum Edges {
    /// Every edge, as a source sent them.
    Exact(Vec<f64>),
    /// A measure's own children: each one's ln weight as an f32, the
    /// precision a model computes its logits in (about 1e-15 nats of KL
    /// divergence from f64 on Qwen3-4B), and the sums of those weights from
    /// the light end up to ranks 0, `KEPT`, 2 `KEPT`, …; the first is the
    /// total. Edge i is the sum up to i less the total, summed from the
    /// kept one below it just as the kept ones were: every edge is exact
    /// for these weights, and the top is 0.
    Weights {
        ln_weights: Vec<f32>,
        kept: Vec<f64>,
    },
}

impl Edges {
    /// ln of the weight from `at` to the end, before the total is taken off.
    fn sum_below(ln_weights: &[f32], kept: &[f64], at: usize) -> f64 {
        let block = at.div_ceil(KEPT);
        let (mut sum, from) = match kept.get(block) {
            Some(&sum) => (sum, block * KEPT),
            None => (f64::NEG_INFINITY, ln_weights.len()),
        };
        for weight in ln_weights[at..from].iter().rev() {
            sum = ln_add(sum, f64::from(*weight));
        }
        sum
    }

    fn edge(&self, at: usize) -> f64 {
        match self {
            Self::Exact(edges) => edges[at],
            Self::Weights { ln_weights, kept } => Self::sum_below(ln_weights, kept, at) - kept[0],
        }
    }

    /// Edges `from` through `to`, summed once.
    fn range(&self, from: usize, to: usize) -> Cow<'_, [f64]> {
        match self {
            Self::Exact(edges) => Cow::Borrowed(&edges[from..=to]),
            Self::Weights { ln_weights, kept } => {
                let mut sum = Self::sum_below(ln_weights, kept, to);
                let mut edges = vec![sum - kept[0]; to - from + 1];
                for at in (from..to).rev() {
                    sum = ln_add(sum, f64::from(ln_weights[at]));
                    edges[at - from] = sum - kept[0];
                }
                Cow::Owned(edges)
            }
        }
    }

    fn bytes(&self) -> usize {
        use std::mem::size_of;
        match self {
            Self::Exact(edges) => edges.capacity() * size_of::<f64>(),
            Self::Weights { ln_weights, kept } => {
                ln_weights.capacity() * size_of::<f32>() + kept.capacity() * size_of::<f64>()
            }
        }
    }
}

impl Run {
    /// Validates what a source sent: one edge more than branches, never
    /// increasing, never above the node's whole mass. A branch placed twice
    /// is refused where the run joins a distribution.
    pub fn new(rank: Rank, branches: Vec<Branch>, ln_below: Vec<f64>) -> Result<Self> {
        if ln_below.len() != branches.len() + 1 {
            return Err(format!(
                "a run of {} branches has {} edges",
                branches.len(),
                ln_below.len()
            ));
        }
        if u64::from(rank) + branches.len() as u64 > u64::from(Rank::MAX) {
            return Err("a run reaches past the last rank".into());
        }
        if ln_below[0].is_nan() || ln_below[0] > 0.0 {
            return Err("a run's edges must be log masses".into());
        }
        if ln_below
            .windows(2)
            .any(|pair| pair[1].is_nan() || pair[1] > pair[0])
        {
            return Err("a run's edges must never increase".into());
        }
        Ok(Self {
            rank,
            branches,
            edges: Edges::Exact(ln_below),
            ln_surprisal_below: OnceLock::new(),
        })
    }

    /// A measure's children from rank 0, at least one: their branches and
    /// ln weights, heaviest first.
    fn weights(branches: Vec<Branch>, ln_weights: Vec<f32>) -> Self {
        assert!(!branches.is_empty() && branches.len() == ln_weights.len());
        let mut kept = vec![0.0; ln_weights.len().div_ceil(KEPT)];
        let mut sum = f64::NEG_INFINITY;
        for (at, weight) in ln_weights.iter().enumerate().rev() {
            sum = ln_add(sum, f64::from(*weight));
            if at % KEPT == 0 {
                kept[at / KEPT] = sum;
            }
        }
        Self {
            rank: 0,
            branches,
            edges: Edges::Weights { ln_weights, kept },
            ln_surprisal_below: OnceLock::new(),
        }
    }

    fn bytes(&self) -> usize {
        use std::mem::size_of;
        let sums = self.ln_surprisal_below.get().map_or(0, Vec::capacity);
        size_of::<Self>()
            + self.branches.capacity() * size_of::<Branch>()
            + self.edges.bytes()
            + sums * size_of::<f64>()
    }

    fn ln_surprisal_below(&self) -> &[f64] {
        self.ln_surprisal_below.get_or_init(|| {
            let mut sums = vec![f64::NEG_INFINITY; self.len() + 1];
            for i in (0..self.len()).rev() {
                let ln_p = self.ln_p(i);
                // p (−ln p) is zero both at p = 0 and at p = 1.
                let ln_surprisal = if ln_p < 0.0 && ln_p > f64::NEG_INFINITY {
                    ln_p + (-ln_p).ln()
                } else {
                    f64::NEG_INFINITY
                };
                sums[i] = ln_add(sums[i + 1], ln_surprisal);
            }
            sums
        })
    }

    /// The node's children end at `rank`: nothing is below it. A node
    /// without children is the end at rank 0.
    pub fn end_at(rank: Rank) -> Run {
        Self::new(rank, Vec::new(), vec![f64::NEG_INFINITY]).expect("an end is a valid run")
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// The rank just past the run: its last edge's.
    pub fn end(&self) -> Rank {
        self.rank + self.branches.len() as Rank
    }

    pub fn len(&self) -> usize {
        self.branches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }

    pub fn branches(&self) -> &[Branch] {
        &self.branches
    }

    /// Every edge, top to bottom.
    pub fn ln_below(&self) -> Cow<'_, [f64]> {
        self.edges.range(0, self.len())
    }

    pub fn branch(&self, at: usize) -> Branch {
        self.branches[at]
    }

    /// The top edge of the child at `at`; `at == len` is the run's bottom.
    pub fn edge(&self, at: usize) -> f64 {
        self.edges.edge(at)
    }

    pub fn ln_p(&self, at: usize) -> f64 {
        match &self.edges {
            Edges::Exact(edges) => ln_between(edges[at], edges[at + 1]),
            Edges::Weights { ln_weights, kept } => f64::from(ln_weights[at]) - kept[0],
        }
    }

    /// ln Σ p (−ln p) over children [from, to).
    pub fn ln_surprisal(&self, from: usize, to: usize) -> f64 {
        let sums = self.ln_surprisal_below();
        ln_between(sums[from], sums[to])
    }

    /// The child holding `ln_below`: the first whose bottom edge is under it.
    pub fn holding(&self, ln_below: f64) -> usize {
        self.holding_by(|edge| edge >= ln_below)
    }

    /// The child holding a point, told by which edges it is `below`, itself
    /// included: the first whose bottom edge it is not.
    fn holding_by(&self, below: impl Fn(f64) -> bool) -> usize {
        match &self.edges {
            Edges::Exact(edges) => edges[1..].partition_point(|&edge| below(edge)),
            Edges::Weights { .. } => {
                // The kept edges under the top, and the bottom: the first
                // the point is not below, then the edges summed above it.
                let len = self.len();
                let blocks = len.div_ceil(KEPT);
                let at = |block: usize| (block * KEPT).min(len);
                let (mut low, mut high) = (0, blocks);
                while low < high {
                    let middle = (low + high) / 2;
                    if below(self.edge(at(middle + 1))) {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                if low == blocks {
                    return len;
                }
                let (from, to) = (at(low), at(low + 1));
                let edges = self.edges.range(from, to);
                from + edges[1..].partition_point(|&edge| below(edge))
            }
        }
    }

    /// Children [from, to) as a run of their own.
    fn slice(&self, from: usize, to: usize) -> Run {
        Self {
            rank: self.rank + from as Rank,
            branches: self.branches[from..to].to_vec(),
            edges: Edges::Exact(self.edges.range(from, to).into_owned()),
            ln_surprisal_below: OnceLock::new(),
        }
    }
}

/// One of a node's parts, in order: a known child, or a gap of unknown ones
/// before the run at `before` (`before == runs.len()` for the node's bottom).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Child { run: usize, at: usize },
    Gap { before: usize },
}

/// What is known of a node's children under one measure.
#[derive(Clone, Debug)]
pub struct Distribution {
    /// Disjoint, by rank; two neighbours always have a gap between them.
    runs: Vec<Run>,
    rank: Ranks,
}

/// Each known child's rank by its branch.
#[derive(Clone, Debug)]
enum Ranks {
    /// Indexed by branch, `NO_RANK` for a weightless one: a whole vocabulary
    /// as a measure computes it, a quarter of a hash table's size.
    Dense(Vec<Rank>),
    /// What a source's answers placed, of any branches.
    Sparse(HashMap<Branch, Rank, BuildHasherDefault<BranchHasher>>),
}

/// No child has it: a run ends at `Rank::MAX` at the latest.
const NO_RANK: Rank = Rank::MAX;

impl Ranks {
    fn get(&self, branch: Branch) -> Option<Rank> {
        match self {
            Self::Dense(ranks) => ranks
                .get(branch as usize)
                .copied()
                .filter(|&rank| rank != NO_RANK),
            Self::Sparse(ranks) => ranks.get(&branch).copied(),
        }
    }

    fn bytes(&self) -> usize {
        use std::mem::size_of;
        match self {
            Self::Dense(ranks) => ranks.capacity() * size_of::<Rank>(),
            // Slots of a branch, a rank and a control byte, one in eight
            // left empty.
            Self::Sparse(ranks) => ranks.capacity() * 8 / 7 * (size_of::<(Branch, Rank)>() + 1),
        }
    }
}

/// Branches come from the node's own source, mostly small and dense: one
/// multiply spreads them across the table's bits at a fraction of SipHash's
/// cost.
#[derive(Default)]
struct BranchHasher(u64);

impl Hasher for BranchHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("only branches are hashed")
    }

    fn write_u32(&mut self, branch: u32) {
        self.0 = u64::from(branch).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl Distribution {
    pub fn new(run: Run) -> Self {
        Self::from_runs(vec![run]).expect("one valid run is a valid distribution")
    }

    /// Every branch's ln weight, indexed by branch, as a measure computes it:
    /// complete, normalized, ties ranked by increasing branch. Weightless
    /// branches have no rank.
    pub fn from_weights(ln_weights: &[f64]) -> Result<Self> {
        if ln_weights.iter().any(|weight| weight.is_nan()) {
            return Err("a weight is not a number".into());
        }
        // Kept as f32, and so ranked as f32: ties among them by branch.
        // Each weight beside its branch: sorting them together reads memory
        // in order rather than looking every weight up.
        let mut order: Vec<(f32, Branch)> = ln_weights
            .iter()
            .zip(0..)
            .map(|(&weight, branch)| (weight as f32, branch))
            .filter(|&(weight, _)| weight > f32::NEG_INFINITY)
            .collect();
        if order.iter().any(|(weight, _)| weight.is_infinite()) {
            return Err("a weight is past what an f32 holds".into());
        }
        order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        if order.is_empty() {
            return Ok(Self::new(Run::end_at(0)));
        }
        let mut ranks = vec![NO_RANK; ln_weights.len()];
        for (rank, &(_, branch)) in (0..).zip(&order) {
            ranks[branch as usize] = rank;
        }
        let (weights, branches) = order.into_iter().unzip();
        // One run of distinct branches: nothing for `from_runs` to check.
        Ok(Self {
            runs: vec![Run::weights(branches, weights)],
            rank: Ranks::Dense(ranks),
        })
    }

    /// Weights by branch, exact in f64 where a measure's keep f32: tests of
    /// geometry rather than of what a server keeps.
    #[cfg(test)]
    pub(crate) fn exact(weights: &[f64]) -> Self {
        let mut order: Vec<Branch> = (0..weights.len() as Branch)
            .filter(|&branch| 0.0 < weights[branch as usize])
            .collect();
        order.sort_by(|&a, &b| {
            weights[b as usize]
                .total_cmp(&weights[a as usize])
                .then(a.cmp(&b))
        });
        let mut ln_below = vec![f64::NEG_INFINITY; order.len() + 1];
        for i in (0..order.len()).rev() {
            ln_below[i] = ln_add(ln_below[i + 1], weights[order[i] as usize].ln());
        }
        let total = ln_below[0];
        for edge in &mut ln_below {
            *edge = (*edge - total).min(0.0);
        }
        Self::new(Run::new(0, order, ln_below).unwrap())
    }

    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// Add what a source said of the node in one answer, and whether
    /// anything changed: its runs, in any order, none overlapping. Where it
    /// contradicts what is known, as a source that changed or recomputed
    /// differently would, it replaces it.
    pub fn insert(&mut self, runs: Vec<Run>) -> Result<bool> {
        let fresh = coalesce(runs)?;
        let merged =
            merge(&self.runs, &fresh).map_or_else(|| Self::from_runs(fresh.clone()), Ok)?;
        let changed = merged.runs != self.runs;
        *self = merged;
        Ok(changed)
    }

    /// What one answer said of a node, as all that is known of it.
    pub fn from_answer(runs: Vec<Run>) -> Result<Self> {
        Self::from_runs(coalesce(runs)?)
    }

    /// Runs disjoint by rank, two neighbours with a gap between them.
    fn from_runs(runs: Vec<Run>) -> Result<Self> {
        let mut rank =
            HashMap::with_capacity_and_hasher(runs.iter().map(Run::len).sum(), Default::default());
        for run in &runs {
            for (at, &branch) in (run.rank..).zip(&run.branches) {
                if rank.insert(branch, at).is_some() {
                    return Err(format!("branch {branch} is placed twice"));
                }
            }
        }
        // A gap holds what its neighbours leave: never a negative mass.
        if runs
            .windows(2)
            .any(|pair| pair[0].edge(pair[0].len()) < pair[1].edge(0))
        {
            return Err("runs overlap in mass".into());
        }
        Ok(Self {
            runs,
            rank: Ranks::Sparse(rank),
        })
    }
    pub fn rank(&self, branch: Branch) -> Option<Rank> {
        self.rank.get(branch)
    }

    /// The run and offset of a known child.
    pub fn locate_rank(&self, rank: Rank) -> Option<(usize, usize)> {
        let run = self.runs.partition_point(|run| run.end() <= rank);
        let found = self.runs.get(run)?;
        (found.rank <= rank).then(|| (run, (rank - found.rank) as usize))
    }

    /// ln of a branch's weight, where known: one a complete distribution
    /// does not rank weighs nothing.
    pub fn ln_p(&self, branch: Branch) -> Option<f64> {
        match self.rank(branch) {
            Some(rank) => {
                let (run, at) = self.locate_rank(rank)?;
                Some(self.runs[run].ln_p(at))
            }
            None => self.complete().then_some(f64::NEG_INFINITY),
        }
    }

    /// How many children have weight, where the node's end is known: the
    /// rank its last child ends at.
    pub fn count(&self) -> Option<Rank> {
        let last = self.runs.last().expect("a distribution has a run");
        (last.edge(last.len()) == f64::NEG_INFINITY).then(|| last.end())
    }

    /// Whether every child is known.
    pub fn complete(&self) -> bool {
        (0..=self.runs.len()).all(|before| !self.gap_exists(before))
    }

    /// About the bytes of memory this holds: a server keeps distributions
    /// within a budget of them.
    pub fn bytes(&self) -> usize {
        use std::mem::size_of;
        let runs: usize = self.runs.iter().map(Run::bytes).sum();
        size_of::<Self>() + runs + self.rank.bytes()
    }

    /// A known edge: rank 0's is the node's top, the whole mass, even where
    /// its first child starts a rounding error below; any other is a run's.
    pub fn edge(&self, rank: Rank) -> Option<f64> {
        if rank == 0 {
            return Some(0.0);
        }
        let run = self.runs.partition_point(|run| run.end() < rank);
        let found = self.runs.get(run)?;
        (found.rank <= rank).then(|| found.edge((rank - found.rank) as usize))
    }

    /// Whether the node has no children at all.
    pub fn ended(&self) -> bool {
        self.complete() && self.runs[0].is_empty()
    }

    pub fn gap_exists(&self, before: usize) -> bool {
        match before {
            0 => self.runs.first().is_none_or(|run| run.rank > 0),
            _ if before == self.runs.len() => {
                let run = &self.runs[before - 1];
                run.edge(run.len()) != f64::NEG_INFINITY
            }
            _ => self.runs[before - 1].end() < self.runs[before].rank,
        }
    }

    pub fn item_top(&self, item: Item) -> f64 {
        match item {
            Item::Child { run, at } => self.runs[run].edge(at),
            Item::Gap { before: 0 } => 0.0,
            Item::Gap { before } => {
                let run = &self.runs[before - 1];
                run.edge(run.len())
            }
        }
    }

    pub fn item_bottom(&self, item: Item) -> f64 {
        match item {
            Item::Child { run, at } => self.runs[run].edge(at + 1),
            Item::Gap { before } => self
                .runs
                .get(before)
                .map_or(f64::NEG_INFINITY, |run| run.edge(0)),
        }
    }

    /// The rank an item starts at: a gap's is its first unknown one.
    pub fn item_rank(&self, item: Item) -> Rank {
        match item {
            Item::Child { run, at } => self.runs[run].rank + at as Rank,
            Item::Gap { before: 0 } => 0,
            Item::Gap { before } => self.runs[before - 1].end(),
        }
    }

    /// The bottom edge's rank, where one is known.
    pub fn item_end(&self, item: Item) -> Option<Rank> {
        match item {
            Item::Child { run, at } => Some(self.runs[run].rank + at as Rank + 1),
            Item::Gap { before } => self.runs.get(before).map(|run| run.rank),
        }
    }

    /// The item after `item`, if any. Two runs always have a gap between them.
    pub fn next(&self, item: Item) -> Option<Item> {
        let before = match item {
            Item::Child { run, at } if at + 1 < self.runs[run].len() => {
                return Some(Item::Child { run, at: at + 1 });
            }
            Item::Child { run, .. } => run + 1,
            Item::Gap { before } => {
                return self
                    .runs
                    .get(before)
                    .filter(|run| !run.is_empty())
                    .map(|_| Item::Child { run: before, at: 0 });
            }
        };
        self.gap_exists(before).then_some(Item::Gap { before })
    }

    /// The node's first item, if it has any.
    pub fn first(&self) -> Option<Item> {
        if self.gap_exists(0) {
            Some(Item::Gap { before: 0 })
        } else {
            (!self.runs[0].is_empty()).then_some(Item::Child { run: 0, at: 0 })
        }
    }

    /// The item holding `ln_below`: the first whose bottom edge is under it.
    /// Above every item, it is the first; below every one, there is none.
    pub fn locate(&self, ln_below: f64) -> Option<Item> {
        self.locate_by(|edge| edge >= ln_below)
    }

    /// The item holding a point, told by which edges it is `below`, itself
    /// included: a point a float cannot hold finds its item as exactly.
    pub fn locate_by(&self, below: impl Fn(f64) -> bool) -> Option<Item> {
        let run = self.runs.partition_point(|run| below(run.edge(run.len())));
        let Some(found) = self.runs.get(run) else {
            return self.gap_exists(run).then_some(Item::Gap { before: run });
        };
        if below(found.edge(0)) && !found.is_empty() {
            Some(Item::Child {
                run,
                at: found.holding_by(below),
            })
        } else if self.gap_exists(run) {
            Some(Item::Gap { before: run })
        } else {
            // Between the node's top and a first run a float short of it.
            self.first()
        }
    }

    /// `branch` alone, as a source places a path through it; `None` where it
    /// weighs nothing or is unknown.
    pub fn place(&self, branch: Branch) -> Option<Run> {
        let (run, at) = self.locate_rank(self.rank(branch)?)?;
        Some(self.runs[run].slice(at, at + 1))
    }

    /// An item's edges as a point is measured from them: the node's top is
    /// its whole mass, even where its first child starts a rounding error
    /// below.
    pub fn item_edges(&self, item: Item) -> (f64, f64) {
        let top = match self.item_rank(item) {
            0 => 0.0,
            _ => self.item_top(item),
        };
        (top, self.item_bottom(item))
    }

    /// The known child holding a point, however deep the point is told.
    pub fn child_at(&self, point: &Point) -> Option<(usize, usize)> {
        let holds = |item: Item| {
            let (top, bottom) = self.item_edges(item);
            point.within(top, bottom)
        };
        // Where a float of the point finds it, most of the time; exactly
        // where the point is too near an edge for that.
        let item = self
            .locate(point.ln_below())
            .filter(|&item| holds(item))
            .or_else(|| self.locate_by(|edge| point.below(edge)))?;
        match item {
            Item::Child { run, at } if holds(item) => Some((run, at)),
            _ => None,
        }
    }

    /// The known children holding `points`, once each and by rank, and
    /// whether every point found one: a point in a gap holds no known child.
    pub fn children_at(&self, points: &[Point]) -> (Vec<(usize, usize)>, bool) {
        let mut found: Vec<_> = points
            .iter()
            .filter_map(|point| self.child_at(point))
            .collect();
        let all = found.len() == points.len();
        found.sort_unstable();
        found.dedup();
        (found, all)
    }

    /// Known children as runs: those consecutive in one run joined.
    pub fn runs_of(&self, children: &[(usize, usize)]) -> Vec<Run> {
        let mut runs = Vec::new();
        let mut start = 0;
        for i in 1..=children.len() {
            let joined = children.get(i).is_some_and(|&(run, at)| {
                let (last_run, last_at) = children[i - 1];
                run == last_run && at == last_at + 1
            });
            if !joined {
                let (run, at) = children[start];
                runs.push(self.runs[run].slice(at, children[i - 1].1 + 1));
                start = i;
            }
        }
        runs
    }
}

/// One answer's runs as disjoint runs by rank, those that touch joined.
fn coalesce(mut runs: Vec<Run>) -> Result<Vec<Run>> {
    runs.sort_by_key(|run| run.rank);
    let mut joined: Vec<Builder> = Vec::with_capacity(runs.len());
    for run in &runs {
        match joined.last_mut() {
            Some(last) if run.rank < last.end() => {
                return Err("an answer places a rank twice".into());
            }
            Some(last) if run.rank == last.end() => {
                last.add(run, true).ok_or("an answer contradicts itself")?
            }
            _ => joined.push(Builder::new(run, true)),
        }
    }
    joined.into_iter().map(Builder::finish).collect()
}

/// `fresh` runs over `old` ones, those that overlap or touch joined, the
/// fresh edges kept where both have one; `None` where they contradict.
fn merge(old: &[Run], fresh: &[Run]) -> Option<Distribution> {
    let mut old = old.iter().peekable();
    let mut fresh = fresh.iter().peekable();
    let mut joined: Vec<Builder> = Vec::new();
    loop {
        let take_old = match (old.peek(), fresh.peek()) {
            (Some(a), Some(b)) => a.rank <= b.rank,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let run = if take_old { old.next() } else { fresh.next() }.expect("peeked");
        match joined.last_mut() {
            Some(last) if run.rank <= last.end() => last.add(run, !take_old)?,
            _ => joined.push(Builder::new(run, !take_old)),
        }
    }
    let runs = joined
        .into_iter()
        .map(Builder::finish)
        .collect::<Result<_>>()
        .ok()?;
    Distribution::from_runs(runs).ok()
}

/// A run being joined from others: each edge remembers whether a fresh
/// answer gave it, so an older one never overwrites it.
struct Builder {
    rank: Rank,
    branches: Vec<Branch>,
    ln_below: Vec<f64>,
    fresh: Vec<bool>,
}

impl Builder {
    fn new(run: &Run, fresh: bool) -> Self {
        Self {
            rank: run.rank,
            branches: run.branches.clone(),
            ln_below: run.ln_below().into_owned(),
            fresh: vec![fresh; run.len() + 1],
        }
    }

    fn end(&self) -> Rank {
        self.rank + self.branches.len() as Rank
    }

    /// Join a run starting at or before this one's end; `None` where a rank
    /// names another branch.
    fn add(&mut self, run: &Run, fresh: bool) -> Option<()> {
        for (i, &edge) in run.ln_below().iter().enumerate() {
            let at = (run.rank - self.rank) as usize + i;
            if at < self.ln_below.len() {
                if fresh || !self.fresh[at] {
                    self.ln_below[at] = edge;
                    self.fresh[at] |= fresh;
                }
            } else {
                self.ln_below.push(edge);
                self.fresh.push(fresh);
            }
            if let Some(&branch) = run.branches.get(i) {
                match self.branches.get(at) {
                    Some(&known) if known != branch => return None,
                    Some(_) => {}
                    None => self.branches.push(branch),
                }
            }
        }
        Some(())
    }

    fn finish(self) -> Result<Run> {
        Run::new(self.rank, self.branches, self.ln_below)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run from weights heaviest first, with `rest` below it.
    fn run(rank: Rank, branches: &[Branch], weights: &[f64], rest: f64) -> Run {
        let mut ln_below = vec![rest.ln(); weights.len() + 1];
        for i in (0..weights.len()).rev() {
            ln_below[i] = ln_add(ln_below[i + 1], weights[i].ln()).min(0.0);
        }
        Run::new(rank, branches.to_vec(), ln_below).unwrap()
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} is not {expected}"
        );
    }

    #[test]
    fn runs_are_validated_where_they_arrive() {
        assert!(Run::new(0, vec![1], vec![0.0]).is_err());
        // A branch twice is refused where the run joins the node.
        let twice = Run::new(0, vec![1, 1], vec![0.0, -1.0, -2.0]).unwrap();
        assert!(Distribution::from_answer(vec![twice]).is_err());
        assert!(Run::new(0, vec![1, 2], vec![0.0, -2.0, -1.0]).is_err());
        assert!(Run::new(0, vec![1], vec![0.5, -1.0]).is_err());
        assert!(Run::new(0, vec![1], vec![f64::NAN, -1.0]).is_err());
        assert!(Run::new(Rank::MAX, vec![1], vec![0.0, -1.0]).is_err());
        assert!(Run::new(0, vec![], vec![f64::NEG_INFINITY]).is_ok());
    }

    #[test]
    fn weights_rank_heaviest_first_and_keep_tiny_children_as_f32s() {
        let tiny = 1e-300f64;
        let d = Distribution::from_weights(&[
            0.2f64.ln(),
            0.5f64.ln(),
            f64::NEG_INFINITY,
            0.3f64.ln() + tiny.ln_1p(),
            tiny.ln(),
            0.2f64.ln(),
        ])
        .unwrap();
        let run = &d.runs()[0];
        assert_eq!(run.branches(), [1, 3, 0, 5, 4]);
        // Every branch's rank, none for a weightless one or one past them.
        let ranks: Vec<_> = (0..7).map(|branch| d.rank(branch)).collect();
        assert_eq!(
            ranks,
            [Some(2), Some(0), None, Some(1), Some(4), Some(3), None]
        );
        // Each child's ln weight to an f32's precision.
        let near = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 1e-7 * expected.abs(),
                "{actual} is not {expected}"
            );
        };
        near(run.ln_p(0), (0.5f64 / 1.2).ln());
        // The lightest child keeps its own weight, far below any float of 1 − cdf.
        near(run.ln_p(4), tiny.ln() - 1.2f64.ln());
        assert_eq!(*run.ln_below().last().unwrap(), f64::NEG_INFINITY);
        assert!(Distribution::from_weights(&[]).unwrap().ended());
        assert!(Distribution::from_weights(&[f64::NAN]).is_err());
    }

    #[test]
    fn kept_edges_sum_to_the_same_edges_as_summing_them_all() {
        // Edges summed from the kept ones are those of one pass over every
        // weight, and points find the same children among them.
        let weights: Vec<f64> = (0..1000)
            .map(|i| -(f64::from(i % 97) * 0.37 + f64::from(i) * 0.01))
            .collect();
        let d = Distribution::from_weights(&weights).unwrap();
        let run = &d.runs()[0];
        let mut sorted: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let mut all = vec![f64::NEG_INFINITY; sorted.len() + 1];
        for i in (0..sorted.len()).rev() {
            all[i] = ln_add(all[i + 1], f64::from(sorted[i]));
        }
        let total = all[0];
        let all: Vec<f64> = all.iter().map(|edge| edge - total).collect();
        assert_eq!(&run.ln_below()[..], &all[..]);
        assert_eq!((0..=1000).map(|at| run.edge(at)).collect::<Vec<_>>(), all);
        for i in 0..4000u32 {
            let ln_below = all[(i / 4) as usize] - 1e-9 * f64::from(i % 3);
            assert_eq!(
                run.holding(ln_below),
                all[1..].partition_point(|&edge| edge >= ln_below)
            );
        }
        assert_eq!(run.holding(1.0), 0);
        assert_eq!(run.holding(f64::NEG_INFINITY), 1000);
    }

    #[test]
    fn inserts_merge_neighbours_and_leave_gaps_between() {
        let all = [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05];
        let whole = run(0, &[10, 11, 12, 13, 14, 15, 16], &all, 0.0);
        let piece = |from: usize, to: usize| {
            let rest: f64 = all[to..].iter().sum();
            run(
                from as Rank,
                &whole.branches()[from..to],
                &all[from..to],
                rest,
            )
        };
        let mut d = Distribution::new(piece(0, 2));
        d.insert(vec![piece(4, 5)]).unwrap();
        assert_eq!(d.runs().len(), 2);
        assert!(d.gap_exists(1) && d.gap_exists(2) && !d.gap_exists(0));
        assert_eq!(d.rank(14), Some(4));
        // Touching on both sides joins all three.
        d.insert(vec![piece(2, 4)]).unwrap();
        assert_eq!(d.runs().len(), 1);
        assert_eq!(d.runs()[0].branches(), &whole.branches()[..5]);
        d.insert(vec![piece(5, 7)]).unwrap();
        assert_eq!(d.runs().len(), 1);
        assert!(!d.gap_exists(1));
        close(d.runs()[0].ln_p(6), 0.05f64.ln());
        // A rank that names another branch: the source changed, start over.
        d.insert(vec![run(3, &[99], &[0.1], 0.2)]).unwrap();
        assert_eq!(d.runs().len(), 1);
        assert_eq!(d.rank(10), None);
        assert_eq!(d.rank(99), Some(3));
        // So does a branch at another rank.
        let mut d = Distribution::new(piece(0, 2));
        d.insert(vec![run(5, &[10], &[0.05], 0.05)]).unwrap();
        assert_eq!(d.rank(10), Some(5));
        assert_eq!(d.runs().len(), 1);
    }

    #[test]
    fn items_walk_children_and_gaps_in_order() {
        let mut d = Distribution::new(run(0, &[7, 8], &[0.5, 0.25], 0.25));
        d.insert(vec![run(5, &[9], &[0.01], 0.02)]).unwrap();
        let mut items = vec![];
        let mut item = d.locate(0.0);
        while let Some(found) = item {
            items.push(found);
            item = d.next(found);
        }
        assert_eq!(
            items,
            [
                Item::Child { run: 0, at: 0 },
                Item::Child { run: 0, at: 1 },
                Item::Gap { before: 1 },
                Item::Child { run: 1, at: 0 },
                Item::Gap { before: 2 },
            ]
        );
        assert_eq!(d.item_rank(Item::Gap { before: 1 }), 2);
        assert_eq!(d.item_end(Item::Gap { before: 1 }), Some(5));
        assert_eq!(d.item_end(Item::Gap { before: 2 }), None);
        assert_eq!(d.locate(0.6f64.ln()), Some(Item::Child { run: 0, at: 0 }));
        assert_eq!(d.locate(0.4f64.ln()), Some(Item::Child { run: 0, at: 1 }));
        assert_eq!(d.locate(0.1f64.ln()), Some(Item::Gap { before: 1 }));
        assert_eq!(d.locate(0.025f64.ln()), Some(Item::Child { run: 1, at: 0 }));
        assert_eq!(d.locate(0.001f64.ln()), Some(Item::Gap { before: 2 }));
        close(d.edge(5).unwrap(), 0.03f64.ln());
        assert_eq!(d.edge(3), None);
        // The node's end, as an empty run, bounds the bottom gap and counts it.
        let mut ending = Distribution::new(run(0, &[7, 8], &[0.5, 0.25], 0.25));
        assert_eq!(ending.count(), None);
        ending.insert(vec![Run::end_at(9)]).unwrap();
        assert_eq!(ending.count(), Some(9));
        assert_eq!(ending.item_end(Item::Gap { before: 1 }), Some(9));
        assert_eq!(ending.next(Item::Gap { before: 1 }), None);
        assert_eq!(ending.edge(9), Some(f64::NEG_INFINITY));
        assert!(Distribution::new(Run::end_at(0)).ended());
        let late = Distribution::new(run(3, &[1], &[0.1], 0.1));
        assert_eq!(late.locate(0.0), Some(Item::Gap { before: 0 }));
        assert_eq!(late.edge(0), Some(0.0));
    }

    #[test]
    fn points_find_the_children_holding_them_once_each() {
        let mut d = Distribution::new(run(0, &[3, 4], &[0.5, 0.3], 0.2));
        d.insert(vec![run(4, &[8], &[0.05], 0.05)]).unwrap();
        // 0.9 and 0.6 lie in branch 3, 0.4 in branch 4, 0.15 in a gap, 0.07 in branch 8.
        let points = [0.9f64, 0.6, 0.4, 0.15, 0.07]
            .map(|below| Point::off(below.ln(), true, 0.0, 0.0).unwrap());
        let (children, all) = d.children_at(&points);
        assert_eq!(children, [(0, 0), (0, 1), (1, 0)]);
        assert!(!all);
        let runs = d.runs_of(&children);
        assert_eq!(
            runs.iter()
                .map(|run| (run.rank(), run.branches().to_vec()))
                .collect::<Vec<_>>(),
            [(0, vec![3, 4]), (4, vec![8])]
        );
        // A point 2^-800 above branch 4's top is branch 3's, and the edge
        // itself branch 4's: no float of a mass tells them apart.
        let edge = 0.5f64.ln();
        let over = Point::off(edge, false, 1.0, -800.0 * std::f64::consts::LN_2).unwrap();
        assert_eq!(d.child_at(&over), Some((0, 0)));
        assert_eq!(
            d.child_at(&Point::off(edge, true, 0.0, 0.0).unwrap()),
            Some((0, 1))
        );
        // An answer that places one rank twice is refused, and one that
        // contradicts what is known replaces it.
        assert!(d
            .insert(vec![run(0, &[3], &[0.5], 0.5), run(0, &[3], &[0.5], 0.5)])
            .is_err());
        assert!(d.insert(vec![run(1, &[9], &[0.3], 0.2)]).unwrap());
        assert_eq!((d.rank(3), d.rank(9)), (None, Some(1)));
    }
}
