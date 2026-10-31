//! A node's children under one measure, as far as they are known: runs of
//! consecutive ranks in the measure's own order, heaviest first.
//!
//! Positions are ln of the mass at and below an edge (ln of 1 − cdf). Ranks
//! sort the children by weight, so the mass below rank r is at most the
//! number of children below it times r's own weight: ln-below edges keep
//! every child's extent precise relative to its own weight, however light it
//! is or deep in its parent's tail.

use std::collections::HashMap;
use std::f64::consts::LN_2;

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
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    rank: Rank,
    branches: Vec<Branch>,
    /// ln of the mass at and below each edge: edge i tops rank `rank + i`,
    /// and the last one bottoms the run.
    ln_below: Vec<f64>,
    /// ln Σ p (−ln p) over the run's children from each edge to the run's
    /// end, accumulated from the light end: a merged band's area-preserving
    /// width without visiting its children.
    ln_surprisal_below: Vec<f64>,
}

impl Run {
    /// Validates what a source sent: one edge more than branches, never
    /// increasing, never above the node's whole mass, no branch twice.
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
        let mut seen = std::collections::HashSet::with_capacity(branches.len());
        if let Some(branch) = branches.iter().find(|&&branch| !seen.insert(branch)) {
            return Err(format!("branch {branch} is listed twice"));
        }
        let mut ln_surprisal_below = vec![f64::NEG_INFINITY; ln_below.len()];
        for i in (0..branches.len()).rev() {
            let ln_p = ln_between(ln_below[i], ln_below[i + 1]);
            // p (−ln p) is zero both at p = 0 and at p = 1.
            let ln_surprisal = if ln_p < 0.0 && ln_p > f64::NEG_INFINITY {
                ln_p + (-ln_p).ln()
            } else {
                f64::NEG_INFINITY
            };
            ln_surprisal_below[i] = ln_add(ln_surprisal_below[i + 1], ln_surprisal);
        }
        Ok(Self {
            rank,
            branches,
            ln_below,
            ln_surprisal_below,
        })
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

    pub fn ln_below(&self) -> &[f64] {
        &self.ln_below
    }

    pub fn branch(&self, at: usize) -> Branch {
        self.branches[at]
    }

    /// The top edge of the child at `at`; `at == len` is the run's bottom.
    pub fn edge(&self, at: usize) -> f64 {
        self.ln_below[at]
    }

    pub fn ln_p(&self, at: usize) -> f64 {
        ln_between(self.ln_below[at], self.ln_below[at + 1])
    }

    /// ln Σ p (−ln p) over children [from, to).
    pub fn ln_surprisal(&self, from: usize, to: usize) -> f64 {
        ln_between(self.ln_surprisal_below[from], self.ln_surprisal_below[to])
    }

    /// The child holding `ln_below`: the first whose bottom edge is under it.
    pub fn holding(&self, ln_below: f64) -> usize {
        self.ln_below[1..].partition_point(|&edge| edge >= ln_below)
    }

    /// Children [from, to) as a run of their own.
    fn slice(&self, from: usize, to: usize) -> Run {
        let end = self.ln_surprisal_below[to];
        Self {
            rank: self.rank + from as Rank,
            branches: self.branches[from..to].to_vec(),
            ln_below: self.ln_below[from..=to].to_vec(),
            ln_surprisal_below: self.ln_surprisal_below[from..=to]
                .iter()
                .map(|&below| ln_between(below, end))
                .collect(),
        }
    }
}

/// A part of a node, in ln of the mass below its edges: (bottom, top].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    top: f64,
    bottom: f64,
}

impl Window {
    pub const WHOLE: Window = Window {
        top: 0.0,
        bottom: f64::NEG_INFINITY,
    };

    pub fn new(top: f64, bottom: f64) -> Result<Self> {
        if top <= 0.0 && bottom < top {
            Ok(Self { top, bottom })
        } else {
            Err(format!("({bottom}, {top}] is no window of a node"))
        }
    }

    pub fn top(&self) -> f64 {
        self.top
    }

    pub fn bottom(&self) -> f64 {
        self.bottom
    }
}

/// What a source answers for one node of a window's descent.
#[derive(Debug, PartialEq)]
pub enum Step {
    /// One child holds the window: its run, and the window within it.
    Descend {
        run: Run,
        branch: Branch,
        window: Window,
    },
    /// The children across the window.
    List(Run),
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
    rank: HashMap<Branch, Rank>,
}

impl Distribution {
    pub fn new(run: Run) -> Self {
        let rank = (run.rank..)
            .zip(&run.branches)
            .map(|(rank, &branch)| (branch, rank))
            .collect();
        Self {
            runs: vec![run],
            rank,
        }
    }

    /// Every branch's ln weight, indexed by branch, as a measure computes it:
    /// complete, normalized, ties ranked by increasing branch. Weightless
    /// branches have no rank.
    pub fn from_weights(ln_weights: &[f64]) -> Result<Self> {
        if ln_weights.iter().any(|weight| weight.is_nan()) {
            return Err("a weight is not a number".into());
        }
        let mut order: Vec<Branch> = (0..ln_weights.len() as Branch)
            .filter(|&branch| ln_weights[branch as usize] > f64::NEG_INFINITY)
            .collect();
        order.sort_unstable_by(|&a, &b| {
            ln_weights[b as usize]
                .total_cmp(&ln_weights[a as usize])
                .then(a.cmp(&b))
        });
        let mut ln_below = vec![f64::NEG_INFINITY; order.len() + 1];
        for i in (0..order.len()).rev() {
            ln_below[i] = ln_add(ln_below[i + 1], ln_weights[order[i] as usize]);
        }
        let total = ln_below[0];
        if total.is_finite() {
            for edge in &mut ln_below {
                *edge = (*edge - total).min(0.0);
            }
        }
        Ok(Self::new(Run::new(0, order, ln_below)?))
    }

    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// Add what a source said, and whether anything changed. Where it
    /// contradicts what is known, as a source that changed or recomputed
    /// differently would, it replaces it.
    pub fn insert(&mut self, run: Run) -> bool {
        let Some((first, last, merged)) = self.merged(&run) else {
            let changed = self.runs != std::slice::from_ref(&run);
            *self = Self::new(run);
            return changed;
        };
        if self.runs[first..last] == *std::slice::from_ref(&merged) {
            return false;
        }
        for (at, &branch) in (merged.rank..).zip(&merged.branches) {
            self.rank.insert(branch, at);
        }
        self.runs.splice(first..last, [merged]);
        true
    }

    /// The runs [first, last) the new one overlaps or touches, as one run
    /// with it; `None` where they contradict it.
    fn merged(&self, run: &Run) -> Option<(usize, usize, Run)> {
        // Runs overlapping or touching the new one become one run with it.
        let first = self.runs.partition_point(|old| old.end() < run.rank);
        let last = self.runs.partition_point(|old| old.rank <= run.end());
        let near = &self.runs[first..last];
        let start = near.first().map_or(run.rank, |old| old.rank.min(run.rank));
        let end = near
            .last()
            .map_or(run.end(), |old| old.end().max(run.end()));
        let mut branches = Vec::with_capacity((end - start) as usize);
        let mut ln_below = Vec::with_capacity((end - start) as usize + 1);
        let mut old = near.iter().peekable();
        for rank in start..=end {
            while old.next_if(|old| old.end() < rank).is_some() {}
            let known = old.peek().filter(|old| old.rank <= rank);
            let new = (run.rank..=run.end()).contains(&rank);
            ln_below.push(if new {
                run.ln_below[(rank - run.rank) as usize]
            } else {
                known?.ln_below[(rank - known?.rank) as usize]
            });
            if rank == end {
                break;
            }
            let was = known
                .filter(|old| rank < old.end())
                .map(|old| old.branches[(rank - old.rank) as usize]);
            let branch = if rank < run.end() && rank >= run.rank {
                let branch = run.branches[(rank - run.rank) as usize];
                // A branch keeps its rank, and a rank its branch.
                if was.is_some_and(|was| was != branch)
                    || self.rank.get(&branch).is_some_and(|&at| at != rank)
                {
                    return None;
                }
                branch
            } else {
                was?
            };
            branches.push(branch);
        }
        let merged = Run::new(start, branches, ln_below).ok()?;
        // A gap holds what its neighbours leave: never a negative mass.
        let above = first.checked_sub(1).is_some_and(|before| {
            self.runs[before].ln_below[self.runs[before].len()] < merged.ln_below[0]
        });
        let below = self
            .runs
            .get(last)
            .is_some_and(|after| merged.ln_below[merged.len()] < after.ln_below[0]);
        (!above && !below).then_some((first, last, merged))
    }

    pub fn rank(&self, branch: Branch) -> Option<Rank> {
        self.rank.get(&branch).copied()
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

    /// Whether every child is known.
    pub fn complete(&self) -> bool {
        (0..=self.runs.len()).all(|before| !self.gap_exists(before))
    }

    /// A known edge: rank 0's is the node's top, the whole mass, even where
    /// its first child starts a rounding error below; any other is a run's.
    pub fn edge(&self, rank: Rank) -> Option<f64> {
        if rank == 0 {
            return Some(0.0);
        }
        let run = self.runs.partition_point(|run| run.end() < rank);
        let found = self.runs.get(run)?;
        (found.rank <= rank).then(|| found.ln_below[(rank - found.rank) as usize])
    }

    /// Whether the node has no children at all.
    pub fn ended(&self) -> bool {
        self.complete() && self.runs[0].is_empty()
    }

    /// Whether the run from rank 0 lists every child at least `e^ln_weight` heavy.
    pub fn covers(&self, ln_weight: f64) -> bool {
        match self.runs.first() {
            Some(run) if run.rank == 0 => {
                run.ln_below[run.len()] == f64::NEG_INFINITY
                    || (!run.is_empty() && run.ln_p(run.len() - 1) <= ln_weight)
            }
            _ => false,
        }
    }

    pub fn gap_exists(&self, before: usize) -> bool {
        match before {
            0 => self.runs.first().is_none_or(|run| run.rank > 0),
            _ if before == self.runs.len() => {
                self.runs[before - 1].ln_below.last() != Some(&f64::NEG_INFINITY)
            }
            _ => self.runs[before - 1].end() < self.runs[before].rank,
        }
    }

    pub fn item_top(&self, item: Item) -> f64 {
        match item {
            Item::Child { run, at } => self.runs[run].ln_below[at],
            Item::Gap { before: 0 } => 0.0,
            Item::Gap { before } => *self.runs[before - 1].ln_below.last().expect("edges"),
        }
    }

    pub fn item_bottom(&self, item: Item) -> f64 {
        match item {
            Item::Child { run, at } => self.runs[run].ln_below[at + 1],
            Item::Gap { before } => self
                .runs
                .get(before)
                .map_or(f64::NEG_INFINITY, |run| run.ln_below[0]),
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

    /// The heaviest a child in a gap can be: as heavy as the child above it.
    pub fn gap_ceiling(&self, before: usize) -> f64 {
        before
            .checked_sub(1)
            .map(|run| &self.runs[run])
            .filter(|run| !run.is_empty())
            .map_or(0.0, |run| run.ln_p(run.len() - 1))
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
        let run = self
            .runs
            .partition_point(|run| run.ln_below[run.len()] >= ln_below);
        let Some(found) = self.runs.get(run) else {
            return self.gap_exists(run).then_some(Item::Gap { before: run });
        };
        if found.ln_below[0] >= ln_below && !found.is_empty() {
            Some(Item::Child {
                run,
                at: found.holding(ln_below),
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

    /// The known children across `window`, from the one holding its top,
    /// down while each weighs at least 2^-horizon of the window (the first
    /// always), and only as far as they are known without a gap.
    pub fn list(&self, window: Window, horizon: u32) -> Option<Run> {
        let (run, at) = self.first_in(window)?;
        let found = &self.runs[run];
        let floor = ln_between(window.top, window.bottom) - f64::from(horizon) * LN_2;
        let mut end = at + 1;
        while end < found.len() && found.ln_below[end] > window.bottom && found.ln_p(end) >= floor {
            end += 1;
        }
        Some(found.slice(at, end))
    }

    /// The known child holding the window's top, or the first known one
    /// below it within the window.
    fn first_in(&self, window: Window) -> Option<(usize, usize)> {
        let run = self
            .runs
            .partition_point(|run| run.ln_below[run.len()] >= window.top);
        let found = self.runs.get(run)?;
        if found.is_empty() || found.ln_below[0] <= window.bottom {
            return None;
        }
        Some((run, found.holding(window.top)))
    }

    /// One node of a window's descent: into the child holding all of a
    /// window narrower than the node, or else the children across it.
    pub fn step(&self, window: Window, horizon: u32) -> Option<Step> {
        let (run, at) = self.first_in(window)?;
        let found = &self.runs[run];
        let (top, bottom) = (found.ln_below[at], found.ln_below[at + 1]);
        let whole = window.top == 0.0 && window.bottom == f64::NEG_INFINITY;
        if whole || top < window.top || bottom > window.bottom {
            return self.list(window, horizon).map(Step::List);
        }
        // Rescale the window into the child: mass below an edge, relative to
        // the child's own.
        let ln_p = found.ln_p(at);
        let within = |edge: f64| {
            if edge >= top {
                0.0
            } else if edge <= bottom {
                f64::NEG_INFINITY
            } else {
                (ln_between(edge, bottom) - ln_p).min(0.0)
            }
        };
        let (child_top, child_bottom) = (within(window.top), within(window.bottom));
        let window = if child_bottom < child_top {
            Window {
                top: child_top,
                bottom: child_bottom,
            }
        } else {
            // Narrower than a float can tell apart within the child.
            return self.list(window, horizon).map(Step::List);
        };
        Some(Step::Descend {
            run: found.slice(at, at + 1),
            branch: found.branches[at],
            window,
        })
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
        assert!(Run::new(0, vec![1, 1], vec![0.0, -1.0, -2.0]).is_err());
        assert!(Run::new(0, vec![1, 2], vec![0.0, -2.0, -1.0]).is_err());
        assert!(Run::new(0, vec![1], vec![0.5, -1.0]).is_err());
        assert!(Run::new(0, vec![1], vec![f64::NAN, -1.0]).is_err());
        assert!(Run::new(Rank::MAX, vec![1], vec![0.0, -1.0]).is_err());
        assert!(Run::new(0, vec![], vec![f64::NEG_INFINITY]).is_ok());
    }

    #[test]
    fn weights_rank_heaviest_first_and_keep_tiny_children_exact() {
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
        assert_eq!(d.rank(2), None);
        close(run.ln_p(0), (0.5f64 / 1.2).ln());
        // The lightest child keeps its own weight, far below any float of 1 − cdf.
        close(run.ln_p(4), tiny.ln() - 1.2f64.ln());
        assert_eq!(*run.ln_below().last().unwrap(), f64::NEG_INFINITY);
        assert!(d.covers(-1000.0));
        assert!(Distribution::from_weights(&[]).unwrap().ended());
        assert!(Distribution::from_weights(&[f64::NAN]).is_err());
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
        d.insert(piece(4, 5));
        assert_eq!(d.runs().len(), 2);
        assert!(d.gap_exists(1) && d.gap_exists(2) && !d.gap_exists(0));
        assert_eq!(d.rank(14), Some(4));
        close(d.gap_ceiling(1), 0.2f64.ln());
        // Touching on both sides joins all three.
        d.insert(piece(2, 4));
        assert_eq!(d.runs().len(), 1);
        assert_eq!(d.runs()[0].branches(), &whole.branches()[..5]);
        d.insert(piece(5, 7));
        assert_eq!(d.runs().len(), 1);
        assert!(!d.gap_exists(1));
        close(d.runs()[0].ln_p(6), 0.05f64.ln());
        // A rank that names another branch: the source changed, start over.
        d.insert(run(3, &[99], &[0.1], 0.2));
        assert_eq!(d.runs().len(), 1);
        assert_eq!(d.rank(10), None);
        assert_eq!(d.rank(99), Some(3));
        // So does a branch at another rank.
        let mut d = Distribution::new(piece(0, 2));
        d.insert(run(5, &[10], &[0.05], 0.05));
        assert_eq!(d.rank(10), Some(5));
        assert_eq!(d.runs().len(), 1);
    }

    #[test]
    fn items_walk_children_and_gaps_in_order() {
        let mut d = Distribution::new(run(0, &[7, 8], &[0.5, 0.25], 0.25));
        d.insert(run(5, &[9], &[0.01], 0.02));
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
        let late = Distribution::new(run(3, &[1], &[0.1], 0.1));
        assert_eq!(late.locate(0.0), Some(Item::Gap { before: 0 }));
        assert_eq!(late.edge(0), Some(0.0));
    }

    #[test]
    fn windows_list_down_to_the_horizon_or_descend_into_one_child() {
        let d = Distribution::from_weights(&[0.5f64.ln(), 0.3f64.ln(), 0.15f64.ln(), 0.05f64.ln()])
            .unwrap();
        let listed = d.list(Window::WHOLE, 2).unwrap();
        assert_eq!(listed.branches(), [0, 1]);
        assert_eq!(d.list(Window::WHOLE, 0).unwrap().branches(), [0]);
        // Within the window only: (0.05, 0.4] starts inside branch 1.
        let window = Window::new(0.4f64.ln(), 0.05f64.ln()).unwrap();
        assert_eq!(d.list(window, 30).unwrap().branches(), [1, 2]);
        assert!(matches!(d.step(window, 30), Some(Step::List(_))));
        // (0.3, 0.45] lies in branch 1's (0.2, 0.5]: half of it, the upper half.
        let inside = Window::new(0.45f64.ln(), 0.3f64.ln()).unwrap();
        let Some(Step::Descend {
            run,
            branch,
            window,
        }) = d.step(inside, 30)
        else {
            panic!("a window inside one child descends");
        };
        assert_eq!((branch, run.rank(), run.branches()), (1, 1, &[1][..]));
        close(window.top(), (0.25f64 / 0.3).ln());
        close(window.bottom(), (0.1f64 / 0.3).ln());
        assert!(matches!(d.step(Window::WHOLE, 30), Some(Step::List(_))));
        assert_eq!(d.place(2).unwrap().rank(), 2);
        assert_eq!(d.place(9), None);
    }
}
