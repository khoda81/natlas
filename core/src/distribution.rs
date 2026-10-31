//! A node's children under one measure, as far as they are known: runs of
//! consecutive ranks in the measure's own order, heaviest first.
//!
//! Positions are ln of the mass at and below an edge (ln of 1 − cdf). Ranks
//! sort the children by weight, so the mass below rank r is at most the
//! number of children below it times r's own weight: ln-below edges keep
//! every child's extent precise relative to its own weight, however light it
//! is or deep in its parent's tail.

use std::collections::HashMap;

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
        Self::from_runs(vec![run]).expect("one valid run is a valid distribution")
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
        let mut rank = HashMap::new();
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
            .any(|pair| pair[0].ln_below[pair[0].len()] < pair[1].ln_below[0])
        {
            return Err("runs overlap in mass".into());
        }
        Ok(Self { runs, rank })
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

    /// How many children have weight, where the node's end is known: the
    /// rank its last child ends at.
    pub fn count(&self) -> Option<Rank> {
        let last = self.runs.last().expect("a distribution has a run");
        (last.ln_below[last.len()] == f64::NEG_INFINITY).then(|| last.end())
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

    /// The known children holding `points`, each ln of the mass below it,
    /// once each and by rank, and whether every point found one: a point in
    /// a gap holds no known child.
    pub fn children_at(&self, points: &[f64]) -> (Vec<(usize, usize)>, bool) {
        let mut found = Vec::with_capacity(points.len());
        let mut all = true;
        for &point in points {
            match self.locate(point) {
                Some(Item::Child { run, at })
                    if point <= self.runs[run].ln_below[at]
                        && self.runs[run].ln_below[at + 1] < point =>
                {
                    found.push((run, at));
                }
                _ => all = false,
            }
        }
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

    /// A point of the node, as ln of the mass below it, within the child at
    /// `(run, at)` holding it.
    pub fn within(&self, (run, at): (usize, usize), point: f64) -> f64 {
        let run = &self.runs[run];
        let (top, bottom) = (run.ln_below[at], run.ln_below[at + 1]);
        if top <= point {
            0.0
        } else if point <= bottom {
            f64::NEG_INFINITY
        } else {
            (ln_between(point, bottom) - run.ln_p(at)).min(0.0)
        }
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
            ln_below: run.ln_below.clone(),
            fresh: vec![fresh; run.ln_below.len()],
        }
    }

    fn end(&self) -> Rank {
        self.rank + self.branches.len() as Rank
    }

    /// Join a run starting at or before this one's end; `None` where a rank
    /// names another branch.
    fn add(&mut self, run: &Run, fresh: bool) -> Option<()> {
        for (i, &edge) in run.ln_below.iter().enumerate() {
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
        let points = [0.9f64, 0.6, 0.4, 0.15, 0.07].map(f64::ln);
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
        // 0.6 lies 0.1 into branch 3's (0.5, 1]: a fifth of it from its bottom.
        close(d.within((0, 0), 0.6f64.ln()), 0.2f64.ln());
        assert_eq!(d.within((0, 0), 0.0), 0.0);
        assert_eq!(d.within((0, 0), 0.5f64.ln()), f64::NEG_INFINITY);
        // An answer that places one rank twice is refused, and one that
        // contradicts what is known replaces it.
        assert!(d
            .insert(vec![run(0, &[3], &[0.5], 0.5), run(0, &[3], &[0.5], 0.5)])
            .is_err());
        assert!(d.insert(vec![run(1, &[9], &[0.3], 0.2)]).unwrap());
        assert_eq!((d.rank(3), d.rank(9)), (None, Some(1)));
    }
}
