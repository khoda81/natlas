//! A node's children under one measure, as far as they are known: runs of
//! consecutive ranks in the measure's own order, whatever order that is.
//!
//! An edge between two children is a point of the node, exact: a run's top,
//! as precise as the answer that sent it, plus the f64 sum of the weights of
//! the run's children above it. Sums from a top are precise to 2^-53 of the
//! sum, so a source ends a run, and starts the next at an exact top of its
//! own, wherever the sum would carry an edge past the last digit of the top.
//! Each child's own weight is kept beside its edges, at relative precision
//! however light it is: its extent places it, its weight sizes it.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

use crate::packed::Packed;
use crate::point::Point;
use crate::tree::Branch;
use crate::Result;

/// A child's position in its measure's order, from 0.
pub type Rank = u32;

/// ln(e^a + e^b), exact for tiny terms.
pub fn ln_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        high
    } else {
        high + (a.min(b) - high).exp().ln_1p()
    }
}

/// ln(e^top − e^bottom) for top ≥ bottom, exact however close they are.
pub fn ln_between(top: f64, bottom: f64) -> f64 {
    if bottom == f64::NEG_INFINITY {
        top
    } else {
        top + (-(bottom - top).exp_m1()).ln()
    }
}

/// What a measure's ln weights are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// Up to a constant, as a model's logits: each node's are normalized.
    Relative,
    /// Each child's ln weight conditional on the node, as given: a measure
    /// that sets widths, its −ln q whatever it sums to.
    Absolute,
}

/// The order a measure ranks a node's children in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Heaviest first, ties by increasing branch.
    Heaviest,
    /// By increasing branch.
    Branch,
}

/// Consecutive ranks from `rank`: their branches, weights and edges.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    rank: Rank,
    children: Children,
}

#[derive(Clone, Debug, PartialEq)]
enum Children {
    /// As a source sent them. Edge i is `top` plus `offsets[i]`, the f64 sum
    /// of the weights from the top the source sent down to the edge: a part
    /// of a run sent whole keeps its top and sums, so its edges never move.
    Listed {
        branches: Vec<Branch>,
        ln_p: Vec<f64>,
        top: Point,
        /// The bits the top was sent to.
        precision: u64,
        offsets: Vec<f64>,
    },
    /// A measure's own, its whole vocabulary from rank 0, packed.
    Packed(Packed),
    /// The node's children end at the run's rank: its one edge is the
    /// node's bottom.
    End,
}

/// The f64 sums of `ln_p`'s weights from a top: the sums every edge of a
/// listed run is measured by, on the source that anchors runs and on the
/// client that reads them alike.
fn offsets(ln_p: &[f64]) -> Vec<f64> {
    let mut offsets = Vec::with_capacity(ln_p.len() + 1);
    let mut sum = 0.0;
    offsets.push(sum);
    for &ln in ln_p {
        sum += ln.exp();
        offsets.push(sum);
    }
    offsets
}

impl Run {
    /// Validates what a source sent: as many weights as branches, each a
    /// log mass, and a top as base-256 digits; rank 0's is the node's top.
    /// A branch placed twice is refused where the run joins a distribution.
    pub fn new(
        rank: Rank,
        branches: Vec<Branch>,
        ln_weights: Vec<f64>,
        top: &[u8],
    ) -> Result<Self> {
        if ln_weights.len() != branches.len() {
            return Err(format!(
                "a run of {} branches has {} weights",
                branches.len(),
                ln_weights.len()
            ));
        }
        if branches.is_empty() {
            return Err("a run places no children".into());
        }
        if u64::from(rank) + branches.len() as u64 > u64::from(Rank::MAX) {
            return Err("a run reaches past the last rank".into());
        }
        if ln_weights
            .iter()
            .any(|ln| ln.is_nan() || *ln == f64::INFINITY)
        {
            return Err("a run's weights must be log masses".into());
        }
        if rank == 0 && top.iter().any(|&digit| digit != 0) {
            return Err("the first child's top is the node's".into());
        }
        Ok(Self::listed(
            rank,
            branches,
            ln_weights,
            Point::new(top),
            8 * top.len() as u64,
        ))
    }

    /// Children of these weights from `rank`, `above` of the node above
    /// them, told to `digits` digits: a run as a source sends it.
    #[cfg(test)]
    pub(crate) fn weighed(
        rank: Rank,
        branches: &[Branch],
        weights: &[f64],
        above: &Point,
        digits: usize,
    ) -> Run {
        let ln = weights.iter().map(|weight| weight.ln()).collect();
        Run::new(rank, branches.to_vec(), ln, &above.digits_at(digits)).unwrap()
    }

    fn listed(
        rank: Rank,
        branches: Vec<Branch>,
        ln_p: Vec<f64>,
        top: Point,
        precision: u64,
    ) -> Self {
        let offsets = offsets(&ln_p);
        Self::of(
            rank,
            Children::Listed {
                branches,
                ln_p,
                top,
                precision,
                offsets,
            },
        )
    }

    fn of(rank: Rank, children: Children) -> Self {
        Self { rank, children }
    }

    /// The node's children end at `rank`: nothing is below it. A node
    /// without children is the end at rank 0.
    pub fn end_at(rank: Rank) -> Run {
        Self::of(rank, Children::End)
    }

    fn bytes(&self) -> usize {
        use std::mem::size_of;
        let children = match &self.children {
            Children::Listed {
                branches,
                ln_p,
                offsets,
                ..
            } => {
                branches.capacity() * size_of::<Branch>()
                    + (ln_p.capacity() + offsets.capacity()) * size_of::<f64>()
            }
            Children::Packed(packed) => packed.bytes(),
            Children::End => 0,
        };
        size_of::<Self>() + children
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// The rank just past the run: its last edge's.
    pub fn end(&self) -> Rank {
        self.rank + self.len() as Rank
    }

    pub fn len(&self) -> usize {
        match &self.children {
            Children::Listed { branches, .. } => branches.len(),
            Children::Packed(packed) => packed.len(),
            Children::End => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the node's children end with this run.
    fn ends(&self) -> bool {
        matches!(self.children, Children::Packed(_) | Children::End)
    }

    /// The bits its top is told to: a measure's own and the end are exact.
    pub fn precision(&self) -> u64 {
        match &self.children {
            Children::Listed { precision, .. } => *precision,
            Children::Packed(_) | Children::End => u64::MAX,
        }
    }

    pub fn branches(&self) -> Cow<'_, [Branch]> {
        match &self.children {
            Children::Listed { branches, .. } => Cow::Borrowed(branches),
            Children::Packed(packed) => Cow::Owned(packed.branches(0, packed.len())),
            Children::End => Cow::Borrowed(&[]),
        }
    }

    /// Each child's ln weight, conditional on the node.
    pub fn ln_weights(&self) -> Cow<'_, [f64]> {
        match &self.children {
            Children::Listed { ln_p, .. } => Cow::Borrowed(ln_p),
            _ => Cow::Owned((0..self.len()).map(|at| self.ln_p(at)).collect()),
        }
    }

    /// The top as a source sends it: digits to the run's precision.
    pub fn top_digits(&self) -> Vec<u8> {
        let precision = self.precision();
        assert!(precision < u64::MAX, "only listed runs are sent");
        self.edge(0).digits_at((precision / 8) as usize)
    }

    pub fn branch(&self, at: usize) -> Branch {
        match &self.children {
            Children::Listed { branches, .. } => branches[at],
            Children::Packed(packed) => packed.branch(at),
            Children::End => unreachable!("the end has no children"),
        }
    }

    /// The top edge of the child at `at`; `at == len` is the run's bottom.
    pub fn edge(&self, at: usize) -> Point {
        match &self.children {
            Children::Listed { top, offsets, .. } => top.plus(offsets[at]),
            Children::Packed(packed) => packed.edge(at),
            Children::End => Point::bottom(),
        }
    }

    /// Its edges as one point and the f64 offsets of each from it, where
    /// it is a source's: a view places the point once and the rest by
    /// floats, as precise as the run is.
    pub fn offsets(&self) -> Option<(&Point, &[f64])> {
        match &self.children {
            Children::Listed { top, offsets, .. } => Some((top, offsets)),
            Children::Packed(_) | Children::End => None,
        }
    }

    /// The top of the child at `at`, rounded down to `bits` bits.
    fn top_at(&self, at: usize, bits: u64) -> Point {
        match &self.children {
            Children::Packed(packed) => packed.top(at, bits),
            _ => self.edge(at).floor(bits),
        }
    }

    pub fn ln_p(&self, at: usize) -> f64 {
        match &self.children {
            Children::Listed { ln_p, .. } => ln_p[at],
            Children::Packed(packed) => packed.ln_p(at),
            Children::End => unreachable!("the end has no children"),
        }
    }

    /// The child holding a point the run holds: the last whose top it is at
    /// or below.
    pub fn holding(&self, point: &Point) -> usize {
        let at = match &self.children {
            Children::Listed { .. } => {
                let (mut low, mut high) = (1, self.len() + 1);
                while low < high {
                    let middle = (low + high) / 2;
                    if point.below(&self.edge(middle)) {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                low - 1
            }
            Children::Packed(packed) => packed.holding(point),
            Children::End => unreachable!("the end has no children"),
        };
        at.min(self.len() - 1)
    }

    /// Children [from, to) as a run of their own, its edges unmoved.
    fn slice(&self, from: usize, to: usize) -> Run {
        let rank = self.rank + from as Rank;
        match &self.children {
            Children::Listed {
                branches,
                ln_p,
                top,
                precision,
                offsets,
            } => Self::of(
                rank,
                Children::Listed {
                    branches: branches[from..to].to_vec(),
                    ln_p: ln_p[from..to].to_vec(),
                    top: top.clone(),
                    precision: *precision,
                    offsets: offsets[from..=to].to_vec(),
                },
            ),
            Children::Packed(packed) => Self::listed(
                rank,
                packed.branches(from, to),
                (from..to).map(|at| packed.ln_p(at)).collect(),
                packed.edge(from),
                u64::MAX,
            ),
            Children::End => self.clone(),
        }
    }
}

/// An edge of a node as what is known gives it: the node's own top or
/// bottom, or edge `at` of a run, `at == len` its bottom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Top,
    Bottom,
    Of { run: usize, at: usize },
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
    /// Disjoint by rank, in order; neighbours may touch. The end, if known,
    /// is last.
    runs: Vec<Run>,
    rank: Ranks,
}

/// Each known child's rank by its branch.
#[derive(Clone, Debug)]
enum Ranks {
    /// What a source's answers placed, of any branches.
    Listed(HashMap<Branch, Rank, BuildHasherDefault<BranchHasher>>),
    /// Found in the one packed run: a pass through its branches, which a
    /// path asks once a node, rather than a table as large as the rest.
    Packed,
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
    /// complete, as an f32 like a model's logits, on its `scale`, ranked in
    /// `order`. Weightless branches have no rank.
    pub fn from_weights(ln_weights: &[f32], order: Order, scale: Scale) -> Result<Self> {
        if ln_weights.iter().any(|weight| weight.is_nan()) {
            return Err("a weight is not a number".into());
        }
        // Each weight beside its branch: sorting them together reads memory
        // in order rather than looking every weight up.
        let mut children: Vec<(f32, Branch)> = ln_weights
            .iter()
            .zip(0..)
            .map(|(&weight, branch)| (weight, branch))
            .filter(|&(weight, _)| weight > f32::NEG_INFINITY)
            .collect();
        if children.iter().any(|(weight, _)| weight.is_infinite()) {
            return Err("a weight is infinite".into());
        }
        if order == Order::Heaviest {
            children.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        }
        if children.is_empty() {
            return Ok(Self::new(Run::end_at(0)));
        }
        // One run of distinct branches: nothing for `from_runs` to check.
        Ok(Self {
            runs: vec![Run::of(0, Children::Packed(Packed::new(&children, scale)))],
            rank: Ranks::Packed,
        })
    }

    /// Weights by branch, heaviest first, exact in f64 where a measure's
    /// keep f32: tests of geometry rather than of what a server keeps.
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
        let total: f64 = weights.iter().sum();
        let ln_p = order
            .iter()
            .map(|&branch| (weights[branch as usize] / total).ln())
            .collect();
        let end = order.len() as Rank;
        let run = Run::listed(0, order, ln_p, Point::top(), u64::MAX);
        Self::from_runs(vec![run, Run::end_at(end)]).unwrap()
    }

    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// Add what a source said of the node in one answer, and whether
    /// anything changed: its runs, in any order, none overlapping. Each rank
    /// keeps the run with the more precise top, the fresh one where they are
    /// alike. Where the answer contradicts what is known, as a source that
    /// changed or recomputed differently would, it replaces it.
    pub fn insert(&mut self, runs: Vec<Run>) -> Result<bool> {
        let fresh = Self::from_runs(runs)?;
        let merged = merge(&self.runs, &fresh.runs).unwrap_or(fresh);
        let changed = merged.runs != self.runs;
        *self = merged;
        Ok(changed)
    }

    /// What one answer said of a node, as all that is known of it.
    pub fn from_answer(runs: Vec<Run>) -> Result<Self> {
        Self::from_runs(runs)
    }

    /// Runs disjoint by rank, the end last, each branch placed once.
    fn from_runs(mut runs: Vec<Run>) -> Result<Self> {
        runs.sort_by_key(|run| (run.rank, run.ends()));
        if runs.windows(2).any(|pair| pair[1].rank < pair[0].end())
            || runs.iter().rev().skip(1).any(Run::ends)
            || runs.windows(2).any(|pair| pair[0].rank == pair[1].rank)
        {
            return Err("an answer places a rank twice".into());
        }
        let mut rank =
            HashMap::with_capacity_and_hasher(runs.iter().map(Run::len).sum(), Default::default());
        for run in &runs {
            for (at, &branch) in (run.rank..).zip(run.branches().iter()) {
                if rank.insert(branch, at).is_some() {
                    return Err(format!("branch {branch} is placed twice"));
                }
            }
        }
        Ok(Self {
            runs,
            rank: Ranks::Listed(rank),
        })
    }

    pub fn rank(&self, branch: Branch) -> Option<Rank> {
        match &self.rank {
            Ranks::Listed(ranks) => ranks.get(&branch).copied(),
            Ranks::Packed => self.ranks(&[branch])[0],
        }
    }

    /// The rank of each of `branches` the node has, in one pass where the
    /// node is packed.
    pub fn ranks(&self, branches: &[Branch]) -> Vec<Option<Rank>> {
        match (&self.rank, &self.runs[0].children) {
            (Ranks::Listed(ranks), _) => branches
                .iter()
                .map(|branch| ranks.get(branch).copied())
                .collect(),
            (Ranks::Packed, Children::Packed(packed)) => packed
                .positions(branches)
                .into_iter()
                .map(|at| at.map(|at| at as Rank))
                .collect(),
            (Ranks::Packed, _) => unreachable!("a packed node's run is packed"),
        }
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
        last.ends().then(|| last.end())
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
        let ranks = match &self.rank {
            // Slots of a branch, a rank and a control byte, one in eight
            // left empty.
            Ranks::Listed(ranks) => ranks.capacity() * 8 / 7 * (size_of::<(Branch, Rank)>() + 1),
            Ranks::Packed => 0,
        };
        size_of::<Self>() + runs + ranks
    }

    /// A known edge: the top of rank `rank`, from the run that starts it or
    /// holds it, or else the bottom of the run that ends there.
    pub fn edge(&self, rank: Rank) -> Option<Point> {
        if rank == 0 {
            return Some(Point::top());
        }
        let run = self.runs.partition_point(|run| run.end() <= rank);
        if let Some(found) = self.runs.get(run).filter(|found| found.rank <= rank) {
            return Some(found.edge((rank - found.rank) as usize));
        }
        self.runs[..run]
            .iter()
            .rev()
            .find(|found| found.end() == rank)
            .map(|found| found.edge(found.len()))
    }

    /// Whether the node has no children at all.
    pub fn ended(&self) -> bool {
        self.complete() && self.runs[0].is_empty()
    }

    pub fn gap_exists(&self, before: usize) -> bool {
        match before {
            0 => self.runs.first().is_none_or(|run| run.rank > 0),
            _ if before == self.runs.len() => !self.runs[before - 1].ends(),
            _ => self.runs[before - 1].end() < self.runs[before].rank,
        }
    }

    pub fn point(&self, edge: Edge) -> Point {
        match edge {
            Edge::Top => Point::top(),
            Edge::Bottom => Point::bottom(),
            Edge::Of { run, at } => self.runs[run].edge(at),
        }
    }

    /// The bottom of the run at `run` as its items see it: the top of the
    /// run after it, where that touches it.
    fn run_bottom(&self, run: usize) -> Edge {
        match self.runs.get(run + 1) {
            Some(next) if next.rank == self.runs[run].end() => Edge::Of {
                run: run + 1,
                at: 0,
            },
            _ => Edge::Of {
                run,
                at: self.runs[run].len(),
            },
        }
    }

    /// Where an item's edges come from.
    pub fn item_bounds(&self, item: Item) -> (Edge, Edge) {
        let top = match item {
            Item::Child { run, at } => Edge::Of { run, at },
            Item::Gap { before: 0 } => Edge::Top,
            Item::Gap { before } => self.run_bottom(before - 1),
        };
        let bottom = match item {
            Item::Child { run, at } if at + 1 < self.runs[run].len() => {
                Edge::Of { run, at: at + 1 }
            }
            Item::Child { run, .. } => self.run_bottom(run),
            Item::Gap { before } if before < self.runs.len() => Edge::Of { run: before, at: 0 },
            Item::Gap { .. } => Edge::Bottom,
        };
        (top, bottom)
    }

    /// Both of an item's edges.
    pub fn item_edges(&self, item: Item) -> (Point, Point) {
        let (top, bottom) = self.item_bounds(item);
        (self.point(top), self.point(bottom))
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

    /// The item after `item`, if any.
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
        if self.gap_exists(before) {
            Some(Item::Gap { before })
        } else {
            self.runs
                .get(before)
                .filter(|run| !run.is_empty())
                .map(|_| Item::Child { run: before, at: 0 })
        }
    }

    /// The node's first item, if it has any.
    pub fn first(&self) -> Option<Item> {
        if self.gap_exists(0) {
            Some(Item::Gap { before: 0 })
        } else {
            (!self.runs[0].is_empty()).then_some(Item::Child { run: 0, at: 0 })
        }
    }

    /// The item holding a point: the first whose bottom edge is under it.
    /// At or below the node's bottom, there is none.
    pub fn locate(&self, point: &Point) -> Option<Item> {
        if Point::bottom() <= *point {
            return None;
        }
        let (mut run, mut high) = (0, self.runs.len());
        while run < high {
            let middle = (run + high) / 2;
            if point.below(&self.point(self.run_bottom(middle))) {
                run = middle + 1;
            } else {
                high = middle;
            }
        }
        let Some(found) = self.runs.get(run) else {
            return self.gap_exists(run).then_some(Item::Gap { before: run });
        };
        if point.below(&found.edge(0)) && !found.is_empty() {
            Some(Item::Child {
                run,
                at: found.holding(point),
            })
        } else {
            self.gap_exists(run).then_some(Item::Gap { before: run })
        }
    }

    /// Known children, in order, as runs to send a client: each run's top
    /// rounded down to `bits` bits, or the source's own where that is
    /// coarser, and a new run started wherever an f64 sum from the top would
    /// carry an edge past its last bit.
    pub fn answer(&self, children: &[(usize, usize)], bits: u64) -> Vec<Run> {
        let mut runs = Vec::new();
        let mut i = 0;
        while let Some(&(run, from)) = children.get(i) {
            let source = &self.runs[run];
            let bits = bits.next_multiple_of(8).min(source.precision());
            // Sums of n weights are precise to n · 2^-53 of the sum.
            let limit = (53.0 - bits as f64).exp2();
            let (mut branches, mut ln_p, mut sum) = (Vec::new(), Vec::new(), 0.0);
            let mut at = from;
            loop {
                branches.push(source.branch(at));
                ln_p.push(source.ln_p(at));
                sum += source.ln_p(at).exp();
                i += 1;
                let joins = children.get(i) == Some(&(run, at + 1))
                    && ((branches.len() + 1) as f64) * (sum + source.ln_p(at + 1).exp()) < limit;
                if !joins {
                    break;
                }
                at += 1;
            }
            let rank = source.rank + from as Rank;
            runs.push(Run::listed(
                rank,
                branches,
                ln_p,
                source.top_at(from, bits),
                bits,
            ));
        }
        runs
    }
}

/// `fresh` runs over `old` ones: each rank from the run with the more
/// precise top, the fresh one where they are alike; `None` where they
/// contradict, naming one rank by two branches or one branch at two ranks.
fn merge(old: &[Run], fresh: &[Run]) -> Option<Distribution> {
    let is_end = |run: &&Run| matches!(run.children, Children::End);
    let mut runs: Vec<Run> = fresh
        .iter()
        .find(is_end)
        .or(old.iter().find(is_end))
        .cloned()
        .into_iter()
        .collect();
    for (side, over, fresh_side) in [(old, fresh, false), (fresh, old, true)] {
        for run in side.iter().filter(|run| !is_end(run)) {
            #[allow(clippy::single_range_in_vec_init)]
            let mut left = vec![run.rank..run.end()];
            for other in over.iter().filter(|other| !is_end(other)) {
                let shared = run.rank.max(other.rank)..run.end().min(other.end());
                for rank in shared.clone() {
                    let (a, b) = (rank - run.rank, rank - other.rank);
                    if run.branch(a as usize) != other.branch(b as usize) {
                        return None;
                    }
                }
                // Each rank from the more precise top, the fresh where alike.
                let wins = if fresh_side {
                    run.precision() < other.precision()
                } else {
                    run.precision() <= other.precision()
                };
                if shared.start < shared.end && wins {
                    left = left
                        .into_iter()
                        .flat_map(|range| minus(range, &shared))
                        .collect();
                }
            }
            for range in left {
                let (from, to) = (range.start - run.rank, range.end - run.rank);
                runs.push(run.slice(from as usize, to as usize));
            }
        }
    }
    Distribution::from_runs(runs).ok()
}

/// The ranks of `range` outside `cut`.
fn minus(range: Range<Rank>, cut: &Range<Rank>) -> Vec<Range<Rank>> {
    [
        range.start..cut.start.min(range.end),
        cut.end.max(range.start)..range.end,
    ]
    .into_iter()
    .filter(|part| part.start < part.end)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run from weights, its top where `above` of the node is above it,
    /// to `digits` digits.
    fn run(rank: Rank, branches: &[Branch], weights: &[f64], above: f64, digits: usize) -> Run {
        Run::weighed(rank, branches, weights, &Point::top().plus(above), digits)
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} is not {expected}"
        );
    }

    #[test]
    fn runs_are_validated_where_they_arrive() {
        assert!(Run::new(0, vec![1], vec![], &[]).is_err());
        assert!(Run::new(0, vec![], vec![], &[]).is_err());
        // A branch twice is refused where the run joins the node.
        let twice = Run::new(0, vec![1, 1], vec![-1.0, -2.0], &[]).unwrap();
        assert!(Distribution::new(Run::end_at(5))
            .insert(vec![twice])
            .is_err());
        assert!(Run::new(0, vec![1], vec![f64::NAN], &[]).is_err());
        assert!(Run::new(0, vec![1], vec![f64::INFINITY], &[]).is_err());
        assert!(Run::new(Rank::MAX, vec![1], vec![-1.0], &[]).is_err());
        assert!(Run::new(0, vec![1], vec![-1.0], &[0x80]).is_err());
        assert!(Run::new(3, vec![1], vec![f64::NEG_INFINITY], &[0x80]).is_ok());
    }

    #[test]
    fn edges_are_a_top_plus_the_weights_above_them() {
        let r = run(4, &[7, 8, 9], &[0.25, 2f64.powi(-80), 0.125], 0.5, 1);
        assert_eq!(r.edge(0), Point::new(&[0x80]));
        assert_eq!(r.edge(1), Point::new(&[0xc0]));
        // The light child keeps its weight, and its extent as the float sum
        // from the top tells it.
        close(r.ln_p(1), -80.0 * std::f64::consts::LN_2);
        assert_eq!(r.edge(3), Point::new(&[0xe0]));
        assert_eq!(r.top_digits(), [0x80]);
        assert_eq!(r.holding(&Point::new(&[0xc8])), 2);
        assert_eq!(r.holding(&Point::new(&[0x80])), 0);
    }

    #[test]
    fn weights_rank_in_their_order_and_keep_tiny_children() {
        let tiny = 1e-300f64;
        let weights = [
            0.2f64.ln(),
            0.5f64.ln(),
            f64::NEG_INFINITY,
            0.3f64.ln() + tiny.ln_1p(),
            tiny.ln(),
            0.2f64.ln(),
        ]
        .map(|weight| weight as f32);
        let d = Distribution::from_weights(&weights, Order::Heaviest, Scale::Relative).unwrap();
        let run = &d.runs()[0];
        assert_eq!(run.branches().as_ref(), [1, 3, 0, 5, 4]);
        let ranks: Vec<_> = (0..7).map(|branch| d.rank(branch)).collect();
        assert_eq!(
            ranks,
            [Some(2), Some(0), None, Some(1), Some(4), Some(3), None]
        );
        let near = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 1e-7 * expected.abs(),
                "{actual} is not {expected}"
            );
        };
        near(run.ln_p(0), (0.5f64 / 1.2).ln());
        near(run.ln_p(4), tiny.ln() - 1.2f64.ln());
        assert_eq!(d.count(), Some(5));
        assert!(d.complete());
        // By branch, the tiny child sits between heavy ones, with an extent
        // of its own.
        let d = Distribution::from_weights(&weights, Order::Branch, Scale::Relative).unwrap();
        assert_eq!(d.runs()[0].branches().as_ref(), [0, 1, 3, 4, 5]);
        let (top, bottom) = d.item_edges(Item::Child { run: 0, at: 3 });
        assert!(top < bottom);
        assert_eq!(d.locate(&top), Some(Item::Child { run: 0, at: 3 }));
        assert!(
            Distribution::from_weights(&[], Order::Branch, Scale::Relative)
                .unwrap()
                .ended()
        );
        assert!(Distribution::from_weights(&[f32::NAN], Order::Branch, Scale::Relative).is_err());
        // Absolute weights are each child's own, whatever they sum to: a
        // measure of widths.
        let widths =
            Distribution::from_weights(&[-1.0, -1.0, -3.0], Order::Branch, Scale::Absolute)
                .unwrap();
        assert_eq!(
            (0..3)
                .map(|at| widths.runs()[0].ln_p(at))
                .collect::<Vec<_>>(),
            [-1.0, -1.0, -3.0]
        );
        assert!(
            Distribution::from_weights(&[f32::INFINITY], Order::Branch, Scale::Relative).is_err()
        );
    }

    #[test]
    fn answers_anchor_runs_as_precisely_as_asked() {
        // Heavy children, then one far lighter than any float of a sum of
        // them tells apart, in branch order.
        let mut weights = vec![-1.0f32; 10];
        weights[5] = -90.0;
        let d = Distribution::from_weights(&weights, Order::Branch, Scale::Relative).unwrap();
        let all: Vec<_> = (0..10).map(|at| (0, at)).collect();
        // Coarse: one run, its top the node's.
        let coarse = d.answer(&all, 16);
        assert_eq!(coarse.len(), 1);
        assert_eq!(coarse[0].top_digits(), [0, 0]);
        // Fine enough for the light child: runs break before sums drift.
        let fine = d.answer(&all, 8 * 20);
        assert!(1 < fine.len());
        let mut client = Distribution::new(Run::end_at(10));
        client.insert(fine).unwrap();
        let (top, bottom) = client.item_edges(Item::Child { run: 5, at: 0 });
        let (exact_top, exact_bottom) = d.item_edges(Item::Child { run: 0, at: 5 });
        let ln_p = d.runs()[0].ln_p(5);
        // Its extent as precise as asked: 2^-160 of the node, 2^-30 of it.
        assert!((Point::ln_between(&top, &bottom) - ln_p).abs() < 1e-8);
        assert!(
            Point::ln_between(&top, &exact_top).max(Point::ln_between(&exact_top, &top))
                < ln_p - 10.0
        );
        assert!(
            Point::ln_between(&bottom, &exact_bottom)
                .max(Point::ln_between(&exact_bottom, &bottom))
                < ln_p - 10.0
        );
        assert_eq!(client.count(), Some(10));
        assert!(client.complete());
    }

    #[test]
    fn inserts_keep_the_more_precise_run_for_each_rank() {
        let mut d = Distribution::new(run(0, &[10, 11, 12], &[0.5, 0.25, 0.125], 0.0, 1));
        assert!(d.gap_exists(1) && !d.gap_exists(0));
        // A finer answer for rank 1 replaces it; the coarse ranks around it
        // keep their edges.
        assert!(d.insert(vec![run(1, &[11], &[0.25], 0.5, 4)]).unwrap());
        assert_eq!(d.runs().len(), 3);
        assert_eq!(d.runs()[1].precision(), 32);
        assert_eq!(d.runs()[2].edge(0), Point::new(&[0xc0]));
        // A coarser one does not.
        assert!(!d.insert(vec![run(1, &[11], &[0.25], 0.5, 1)]).unwrap());
        // At one precision, the latest wins.
        assert!(d.insert(vec![run(1, &[11], &[0.2], 0.5, 4)]).unwrap());
        close(d.runs()[1].ln_p(0), 0.2f64.ln());
        // Touching runs: the next one's top bottoms the child before it.
        let bottom = d.item_edges(Item::Child { run: 0, at: 0 }).1;
        assert_eq!(bottom, d.runs()[1].edge(0));
        assert_eq!(
            d.next(Item::Child { run: 0, at: 0 }),
            Some(Item::Child { run: 1, at: 0 })
        );
        // A rank that names another branch: the source changed, start over.
        d.insert(vec![run(2, &[99], &[0.1], 0.75, 1)]).unwrap();
        assert_eq!(d.runs().len(), 1);
        assert_eq!((d.rank(10), d.rank(99)), (None, Some(2)));
        // So does a branch at another rank.
        let mut d = Distribution::new(run(0, &[10, 11], &[0.5, 0.25], 0.0, 1));
        d.insert(vec![run(5, &[10], &[0.05], 0.9, 1)]).unwrap();
        assert_eq!(d.rank(10), Some(5));
        assert_eq!(d.runs().len(), 1);
        // An answer that places one rank twice is refused.
        assert!(d
            .insert(vec![
                run(0, &[3], &[0.5], 0.0, 1),
                run(0, &[3], &[0.5], 0.0, 1)
            ])
            .is_err());
    }

    #[test]
    fn items_walk_children_and_gaps_in_order() {
        let mut d = Distribution::new(run(0, &[7, 8], &[0.5, 0.25], 0.0, 1));
        d.insert(vec![run(5, &[9], &[0.01], 0.97, 2)]).unwrap();
        let mut items = vec![];
        let mut item = d.locate(&Point::top());
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
        let at = |above: f64| d.locate(&Point::top().plus(above));
        assert_eq!(at(0.4), Some(Item::Child { run: 0, at: 0 }));
        assert_eq!(at(0.6), Some(Item::Child { run: 0, at: 1 }));
        assert_eq!(at(0.9), Some(Item::Gap { before: 1 }));
        assert_eq!(at(0.975), Some(Item::Child { run: 1, at: 0 }));
        assert_eq!(at(0.999), Some(Item::Gap { before: 2 }));
        assert_eq!(d.locate(&Point::bottom()), None);
        assert_eq!(d.edge(3), None);
        // The node's end bounds the bottom gap and counts it.
        let mut ending = Distribution::new(run(0, &[7, 8], &[0.5, 0.25], 0.0, 1));
        assert_eq!(ending.count(), None);
        ending.insert(vec![Run::end_at(9)]).unwrap();
        assert_eq!(ending.count(), Some(9));
        assert_eq!(ending.item_end(Item::Gap { before: 1 }), Some(9));
        assert_eq!(ending.next(Item::Gap { before: 1 }), None);
        assert_eq!(ending.edge(9), Some(Point::bottom()));
        assert!(Distribution::new(Run::end_at(0)).ended());
        let late = Distribution::new(run(3, &[1], &[0.1], 0.5, 1));
        assert_eq!(late.locate(&Point::top()), Some(Item::Gap { before: 0 }));
        assert_eq!(late.edge(0), Some(Point::top()));
    }
}
