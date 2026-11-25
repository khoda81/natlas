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

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::OnceLock;

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

/// Normalize logits before ingestion, preserving the supplied branch order.
pub fn log_softmax(children: &mut [(Branch, f64)]) {
    let (max, log_sum) = log_softmax_offsets(children.iter().map(|&(_, weight)| weight));
    for (_, weight) in children {
        *weight = (*weight - max) - log_sum;
    }
}

/// The peak and shifted log sum to subtract in turn, over finite logits
/// and masked weights of negative infinity.
pub(crate) fn log_softmax_offsets(weights: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let mut max = f64::NEG_INFINITY;
    let mut peak = 0;
    for (at, weight) in weights.clone().enumerate() {
        if max < weight {
            max = weight;
            peak = at;
        }
    }
    if max == f64::NEG_INFINITY {
        return (0.0, 0.0);
    }
    // The peak contributes 1. Sum the other mass with compensation
    // and use ln_1p to preserve widths of almost certain branches.
    let mut sum = 0.0;
    let mut correction = 0.0;
    for (at, weight) in weights.enumerate() {
        if at != peak {
            let mass = (weight - max).exp() - correction;
            let next = sum + mass;
            correction = (next - sum) - mass;
            sum = next;
        }
    }
    (max, sum.ln_1p())
}

/// Consecutive ranks from `rank`: their branches, weights and edges.
#[derive(Clone, Debug)]
pub struct Run {
    rank: Rank,
    top: RunTop,
    /// The bits the top was sent to.
    precision: u64,
    contents: RunContents,
}

#[derive(Clone, Debug)]
enum RunTop {
    // Most wire anchors fit in one word. Materialize the exact point only
    // when a camera, recorder, or deep layout asks for it.
    Wire(u64, OnceLock<Point>),
    Point(Point),
}

impl RunTop {
    fn from_digits(digits: &[u8]) -> Self {
        if digits.len() <= 8 {
            let mut word = [0; 8];
            word[..digits.len()].copy_from_slice(digits);
            Self::Wire(u64::from_be_bytes(word), OnceLock::new())
        } else {
            Self::Point(Point::new(digits))
        }
    }

    fn point(&self) -> &Point {
        match self {
            Self::Wire(word, point) => point.get_or_init(|| {
                let bytes = word.to_be_bytes();
                let used = bytes.len() - (word.trailing_zeros() / u8::BITS) as usize;
                Point::new(&bytes[..used])
            }),
            Self::Point(point) => point,
        }
    }

    fn q64(&self) -> u128 {
        match self {
            Self::Wire(word, _) => u128::from(*word),
            Self::Point(point) => point.q64(),
        }
    }
}

#[derive(Clone, Debug)]
enum RunContents {
    Single {
        branch: Branch,
        ln_p: f64,
        offsets: [f64; 2],
    },
    Many {
        branches: Vec<Branch>,
        ln_p: Vec<f64>,
        // A slice keeps offsets from the original top, so its edges stay put.
        offsets: Vec<f64>,
    },
}

impl PartialEq for Run {
    fn eq(&self, other: &Self) -> bool {
        self.whole() == other.whole()
    }
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
        Self::validate(rank, branches.len(), &ln_weights, top)?;
        Ok(Self::with_top(
            rank,
            branches,
            ln_weights,
            RunTop::from_digits(top),
            8 * top.len() as u64,
        ))
    }

    #[cfg(feature = "wasm")]
    pub(crate) fn from_slices(
        rank: Rank,
        branches: &[Branch],
        ln_p: &[f64],
        top: &[u8],
    ) -> Result<Self> {
        Self::validate(rank, branches.len(), ln_p, top)?;
        let point = RunTop::from_digits(top);
        let precision = 8 * top.len() as u64;
        Ok(if let [branch] = branches {
            Self::single(rank, *branch, ln_p[0], point, precision)
        } else {
            Self::with_top(rank, branches.to_vec(), ln_p.to_vec(), point, precision)
        })
    }

    fn validate(rank: Rank, len: usize, ln_weights: &[f64], top: &[u8]) -> Result<()> {
        if ln_weights.len() != len {
            return Err(format!(
                "a run of {} branches has {} weights",
                len,
                ln_weights.len()
            ));
        }
        if len == 0 {
            return Err("a run places no children".into());
        }
        if u64::from(Rank::MAX) < u64::from(rank) + len as u64 {
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
        Ok(())
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
        Self::with_top(rank, branches, ln_p, RunTop::Point(top), precision)
    }

    fn with_top(
        rank: Rank,
        branches: Vec<Branch>,
        ln_p: Vec<f64>,
        top: RunTop,
        precision: u64,
    ) -> Self {
        if let [branch] = *branches {
            return Self::single(rank, branch, ln_p[0], top, precision);
        }
        let offsets = offsets(&ln_p);
        let contents = RunContents::Many {
            branches,
            ln_p,
            offsets,
        };
        Self {
            rank,
            top,
            precision,
            contents,
        }
    }

    fn single(rank: Rank, branch: Branch, ln_p: f64, top: RunTop, precision: u64) -> Self {
        let contents = RunContents::Single {
            branch,
            ln_p,
            offsets: [0.0, ln_p.exp()],
        };
        Self {
            rank,
            top,
            precision,
            contents,
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + match &self.contents {
                RunContents::Single { .. } => 0,
                RunContents::Many {
                    branches,
                    ln_p,
                    offsets,
                } => {
                    branches.capacity() * size_of::<Branch>()
                        + (ln_p.capacity() + offsets.capacity()) * size_of::<f64>()
                }
            }
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// The rank just past the run: its last edge's.
    pub fn end(&self) -> Rank {
        self.rank + self.len() as Rank
    }

    pub(crate) fn len(&self) -> usize {
        self.branches().len()
    }

    pub fn branches(&self) -> &[Branch] {
        match &self.contents {
            RunContents::Single { branch, .. } => std::slice::from_ref(branch),
            RunContents::Many { branches, .. } => branches,
        }
    }

    /// Each child's ln weight, conditional on the node.
    pub fn ln_weights(&self) -> &[f64] {
        match &self.contents {
            RunContents::Single { ln_p, .. } => std::slice::from_ref(ln_p),
            RunContents::Many { ln_p, .. } => ln_p,
        }
    }

    /// The top as a source sends it: digits to the run's precision.
    pub fn top_digits(&self) -> Vec<u8> {
        self.edge(0).digits_at((self.precision() / 8) as usize)
    }

    pub(crate) fn base(&self) -> &Point {
        self.top.point()
    }

    pub(crate) fn q64_base(&self) -> u128 {
        self.top.q64()
    }

    pub(crate) fn offsets(&self) -> &[f64] {
        match &self.contents {
            RunContents::Single { offsets, .. } => offsets,
            RunContents::Many { offsets, .. } => offsets,
        }
    }

    pub(crate) fn precision(&self) -> u64 {
        self.precision
    }

    /// The top edge of the child at `at`; `at == len` is the run's bottom.
    fn edge(&self, at: usize) -> Point {
        self.base().plus(self.offsets()[at])
    }

    fn whole(&self) -> RunSlice<'_> {
        self.window(self.rank(), self.end())
    }

    pub(crate) fn window(&self, from: Rank, to: Rank) -> RunSlice<'_> {
        RunSlice {
            run: self,
            from: (from - self.rank) as usize,
            to: (to - self.rank) as usize,
        }
    }

    fn slice_ranks(&self, from: Rank, to: Rank) -> Self {
        let part = self.window(from, to);
        let contents = if let [branch] = part.branches() {
            RunContents::Single {
                branch: *branch,
                ln_p: part.ln_weights()[0],
                offsets: part.offsets().try_into().expect("two edges of one child"),
            }
        } else {
            RunContents::Many {
                branches: part.branches().to_vec(),
                ln_p: part.ln_weights().to_vec(),
                offsets: part.offsets().to_vec(),
            }
        };
        Self {
            rank: from,
            top: self.top.clone(),
            precision: self.precision,
            contents,
        }
    }
}

/// A view of consecutive ranks within one owned Run. Browser segments share
/// the owned run, so splitting an overlap only narrows this range.
#[derive(Clone, Copy)]
pub(crate) struct RunSlice<'a> {
    run: &'a Run,
    from: usize,
    to: usize,
}

impl<'a> RunSlice<'a> {
    pub(crate) fn rank(self) -> Rank {
        self.run.rank() + self.from as Rank
    }
    pub(crate) fn end(self) -> Rank {
        self.run.rank() + self.to as Rank
    }
    pub(crate) fn len(self) -> usize {
        self.to - self.from
    }
    pub(crate) fn branches(self) -> &'a [Branch] {
        &self.run.branches()[self.from..self.to]
    }
    pub(crate) fn ln_weights(self) -> &'a [f64] {
        &self.run.ln_weights()[self.from..self.to]
    }
    pub(crate) fn offsets(self) -> &'a [f64] {
        &self.run.offsets()[self.from..=self.to]
    }
    pub(crate) fn base(self) -> &'a Point {
        self.run.base()
    }
    pub(crate) fn q64_base(self) -> u128 {
        self.run.q64_base()
    }
    pub(crate) fn precision(self) -> u64 {
        self.run.precision()
    }
    pub(crate) fn edge(self, at: usize) -> Point {
        self.run.edge(self.from + at)
    }
}

impl PartialEq for RunSlice<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.rank() == other.rank()
            && self.len() == other.len()
            && (std::ptr::eq(self.run, other.run)
                || self.precision() == other.precision()
                    && self.run.base() == other.run.base()
                    && self.branches() == other.branches()
                    && self.ln_weights() == other.ln_weights()
                    && self.offsets() == other.offsets())
    }
}

/// A contiguous range of known ranks. A packed distribution holds one range
/// starting at rank zero, which can be read without unpacking it into a run.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct RunRef<'a>(RunStorage<'a>);

#[derive(Clone, Copy)]
enum RunStorage<'a> {
    Listed(RunSlice<'a>),
    Packed(&'a Packed),
}

impl<'a> RunRef<'a> {
    pub(crate) fn listed(run: &'a Run) -> Self {
        Self::slice(run.whole())
    }

    pub(crate) fn slice(run: RunSlice<'a>) -> Self {
        Self(RunStorage::Listed(run))
    }

    pub(crate) fn rank(self) -> Rank {
        match self.0 {
            RunStorage::Listed(run) => run.rank(),
            RunStorage::Packed(_) => 0,
        }
    }

    fn end(self) -> Rank {
        self.rank() + self.len() as Rank
    }

    pub(crate) fn len(self) -> usize {
        match self.0 {
            RunStorage::Listed(run) => run.len(),
            RunStorage::Packed(packed) => packed.len(),
        }
    }

    pub(crate) fn branch(self, at: usize) -> Branch {
        match self.0 {
            RunStorage::Listed(run) => run.branches()[at],
            RunStorage::Packed(packed) => packed.branch(at),
        }
    }

    pub(crate) fn ln_p(self, at: usize) -> f64 {
        match self.0 {
            RunStorage::Listed(run) => run.ln_weights()[at],
            RunStorage::Packed(packed) => packed.ln_p(at),
        }
    }

    fn edge(self, at: usize) -> Point {
        match self.0 {
            RunStorage::Listed(run) => run.edge(at),
            RunStorage::Packed(packed) => packed.edge(at),
        }
    }

    pub(crate) fn listed_slice(self) -> Option<RunSlice<'a>> {
        match self.0 {
            RunStorage::Listed(run) => Some(run),
            RunStorage::Packed(_) => None,
        }
    }

    fn holding(self, point: &Point) -> usize {
        if let RunStorage::Packed(packed) = self.0 {
            return packed.holding(point).min(packed.len() - 1);
        }
        let (mut low, mut high) = (1, self.len() + 1);
        while low < high {
            let middle = (low + high) / 2;
            if point.below(&self.edge(middle)) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        (low - 1).min(self.len() - 1)
    }

    fn precision(self) -> u64 {
        match self.0 {
            RunStorage::Listed(run) => run.precision(),
            RunStorage::Packed(_) => u64::MAX,
        }
    }

    fn top_at(self, at: usize, bits: u64) -> Point {
        match self.0 {
            RunStorage::Listed(run) => run.edge(at).floor(bits),
            RunStorage::Packed(packed) => packed.top(at, bits),
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
    storage: Storage,
}

#[derive(Clone, Debug)]
enum Storage {
    Listed(Listed),
    Packed(Box<Packed>),
}

#[derive(Clone, Debug)]
struct Listed {
    /// Disjoint by rank, in order. Neighbours may touch.
    runs: Box<[Run]>,
    rank: HashMap<Branch, Rank, BuildHasherDefault<BranchHasher>>,
    count: Option<Rank>,
}

impl Listed {
    fn new(mut runs: Vec<Run>, count: Option<Rank>) -> Result<Self> {
        sort_runs(&mut runs, count)?;
        let mut rank =
            HashMap::with_capacity_and_hasher(runs.iter().map(Run::len).sum(), Default::default());
        for run in &runs {
            for (at, &branch) in (run.rank..).zip(run.branches()) {
                if rank.insert(branch, at).is_some() {
                    return Err(format!("branch {branch} is placed twice"));
                }
            }
        }
        Ok(Self {
            runs: runs.into_boxed_slice(),
            rank,
            count,
        })
    }
}

pub(crate) fn sort_runs(runs: &mut [Run], count: Option<Rank>) -> Result<()> {
    if runs.is_empty() && count.is_none() {
        return Err("an answer gives neither runs nor a child count".into());
    }
    runs.sort_unstable_by_key(Run::rank);
    if runs.windows(2).any(|pair| pair[1].rank() < pair[0].end()) {
        return Err("an answer places a rank twice".into());
    }
    if let (Some(count), Some(last)) = (count, runs.last()) {
        if count < last.end() {
            return Err("an answer places children past its count".into());
        }
    }
    Ok(())
}

/// Branches come from the node's own source, mostly small and dense: one
/// multiply spreads them across the table's bits at a fraction of SipHash's
/// cost.
#[derive(Default)]
pub(crate) struct BranchHasher(u64);

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
    /// Complete, ordered children with their final log weights. A
    /// weightless branch has no rank.
    pub fn from_ordered(mut children: Vec<(Branch, f64)>) -> Result<Self> {
        if u64::from(Rank::MAX) < children.len() as u64 {
            return Err("too many children to rank".into());
        }
        if children
            .iter()
            .any(|&(_, weight)| weight.is_nan() || weight == f64::INFINITY)
        {
            return Err("a weight must be a log mass".into());
        }
        children.retain(|&(_, weight)| f64::NEG_INFINITY < weight);
        let mut seen = HashSet::with_capacity_and_hasher(
            children.len(),
            BuildHasherDefault::<BranchHasher>::default(),
        );
        for &(branch, _) in &children {
            if !seen.insert(branch) {
                return Err(format!("branch {branch} is placed twice"));
            }
        }
        Self::pack_unique(children.iter().copied())
    }

    /// Pack already validated unique branches and their finite final log weights.
    pub(crate) fn pack_unique(
        children: impl ExactSizeIterator<Item = (Branch, f64)> + Clone,
    ) -> Result<Self> {
        if children.len() == 0 {
            return Self::from_answer(Vec::new(), Some(0));
        }
        Ok(Self {
            storage: Storage::Packed(Box::new(Packed::new(children))),
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
        let runs = if order.is_empty() {
            Vec::new()
        } else {
            vec![Run::listed(0, order, ln_p, Point::top(), u64::MAX)]
        };
        Self::from_answer(runs, Some(end)).unwrap()
    }

    pub(crate) fn run_count(&self) -> usize {
        match &self.storage {
            Storage::Listed(listed) => listed.runs.len(),
            Storage::Packed(_) => 1,
        }
    }

    pub(crate) fn run_at(&self, index: usize) -> Option<RunRef<'_>> {
        match &self.storage {
            Storage::Listed(listed) => listed.runs.get(index).map(RunRef::listed),
            Storage::Packed(packed) => (index == 0).then_some(RunRef(RunStorage::Packed(packed))),
        }
    }

    /// Add what a source said of the node in one answer, and whether
    /// anything changed: its runs, in any order, none overlapping. Each rank
    /// keeps the run with the more precise top, the fresh one where they are
    /// alike. Where the answer contradicts what is known, as a source that
    /// changed or recomputed differently would, it replaces it.
    pub fn insert(&mut self, runs: Vec<Run>, count: Option<Rank>) -> Result<bool> {
        if runs.is_empty() && count.is_none() {
            return Ok(false);
        }
        let fresh = Listed::new(runs, count)?;
        let listed = match &self.storage {
            Storage::Listed(old) => {
                let merged = merge(old, &fresh).unwrap_or(fresh);
                if merged.runs == old.runs && merged.count == old.count {
                    return Ok(false);
                }
                merged
            }
            Storage::Packed(packed) => {
                // Packed data already knows every rank and its edges. An
                // answer that disagrees with its order or count replaces it.
                if fresh
                    .count
                    .is_none_or(|count| count as usize == packed.len())
                    && fresh.runs.iter().all(|run| {
                        let start = run.rank as usize;
                        run.end() as usize <= packed.len()
                            && run
                                .branches()
                                .iter()
                                .enumerate()
                                .all(|(at, &branch)| packed.branch(start + at) == branch)
                    })
                {
                    return Ok(false);
                }
                fresh
            }
        };
        self.storage = Storage::Listed(listed);
        Ok(true)
    }

    /// What one answer said of a node, as all that is known of it.
    pub fn from_answer(runs: Vec<Run>, count: Option<Rank>) -> Result<Self> {
        Ok(Self {
            storage: Storage::Listed(Listed::new(runs, count)?),
        })
    }

    pub fn rank(&self, branch: Branch) -> Option<Rank> {
        match &self.storage {
            Storage::Listed(listed) => listed.rank.get(&branch).copied(),
            Storage::Packed(_) => self.ranks(&[branch])[0],
        }
    }

    pub(crate) fn is_packed(&self) -> bool {
        matches!(self.storage, Storage::Packed(_))
    }

    /// The rank of each of `branches` the node has, in one pass where the
    /// node is packed.
    pub fn ranks(&self, branches: &[Branch]) -> Vec<Option<Rank>> {
        match &self.storage {
            Storage::Listed(listed) => branches
                .iter()
                .map(|branch| listed.rank.get(branch).copied())
                .collect(),
            Storage::Packed(packed) => packed
                .positions(branches)
                .into_iter()
                .map(|at| at.map(|at| at as Rank))
                .collect(),
        }
    }

    /// The run and offset of a known child.
    pub fn locate_rank(&self, rank: Rank) -> Option<(usize, usize)> {
        match &self.storage {
            Storage::Listed(listed) => {
                let run = listed.runs.partition_point(|run| run.end() <= rank);
                let found = listed.runs.get(run)?;
                (found.rank <= rank).then(|| (run, (rank - found.rank) as usize))
            }
            Storage::Packed(packed) => {
                ((rank as usize) < packed.len()).then_some((0, rank as usize))
            }
        }
    }

    /// ln of a branch's weight, where known: one a complete distribution
    /// does not rank weighs nothing.
    pub fn ln_p(&self, branch: Branch) -> Option<f64> {
        match self.rank(branch) {
            Some(rank) => {
                let (run, at) = self.locate_rank(rank)?;
                Some(self.run(run).ln_p(at))
            }
            None => self.complete().then_some(f64::NEG_INFINITY),
        }
    }

    /// How many children have weight, where the node's end is known: the
    /// rank its last child ends at.
    pub fn count(&self) -> Option<Rank> {
        match &self.storage {
            Storage::Listed(listed) => listed.count,
            Storage::Packed(packed) => Some(packed.len() as Rank),
        }
    }

    /// Whether every child is known.
    pub fn complete(&self) -> bool {
        match &self.storage {
            // Known ranks are distinct and all within the count.
            Storage::Listed(listed) => listed.count == Some(listed.rank.len() as Rank),
            Storage::Packed(_) => true,
        }
    }

    /// About the bytes of memory this holds: a server keeps distributions
    /// within a budget of them.
    pub fn bytes(&self) -> usize {
        use std::mem::size_of;
        let heap = match &self.storage {
            Storage::Listed(listed) => {
                let runs: usize = listed.runs.iter().map(Run::bytes).sum();
                // Slots of a branch, a rank and a control byte, one in eight left empty.
                runs + listed.rank.capacity() * 8 / 7 * (size_of::<(Branch, Rank)>() + 1)
            }
            Storage::Packed(packed) => packed.bytes(),
        };
        size_of::<Self>() + heap
    }

    /// A known edge: the top of rank `rank`, from the run that starts it or
    /// holds it, or else the bottom of the run that ends there.
    pub fn edge(&self, rank: Rank) -> Option<Point> {
        if rank == 0 {
            return Some(Point::top());
        }
        if self.count() == Some(rank) {
            return Some(Point::bottom());
        }
        match &self.storage {
            Storage::Listed(listed) => {
                let run = listed.runs.partition_point(|run| run.end() <= rank);
                if let Some(found) = listed.runs.get(run).filter(|found| found.rank <= rank) {
                    return Some(found.edge((rank - found.rank) as usize));
                }
                listed.runs[..run]
                    .last()
                    .filter(|found| found.end() == rank)
                    .map(|found| found.edge(found.len()))
            }
            Storage::Packed(packed) => {
                ((rank as usize) < packed.len()).then(|| packed.edge(rank as usize))
            }
        }
    }

    /// Whether the node has no children at all.
    pub fn ended(&self) -> bool {
        self.count() == Some(0)
    }

    /// Known children, in order, as runs to send a client: each run's top
    /// rounded down to `bits` bits, or the source's own where that is
    /// coarser, and a new run started wherever an f64 sum from the top would
    /// carry an edge past its last bit.
    pub fn answer(&self, children: &[(usize, usize)], bits: u64) -> Vec<Run> {
        let mut runs = Vec::new();
        let mut i = 0;
        while let Some(&(run, from)) = children.get(i) {
            let source = self.run(run);
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
            let rank = source.rank() + from as Rank;
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

/// Ordered ranks and exact edges shared by recorded and client geometry.
#[doc(hidden)]
pub trait RankView {
    fn run_count(&self) -> usize;
    fn run_at(&self, index: usize) -> Option<RunRef<'_>>;
    fn rank(&self, branch: Branch) -> Option<Rank>;
    fn locate_rank(&self, rank: Rank) -> Option<(usize, usize)>;
    fn count(&self) -> Option<Rank>;

    fn run(&self, index: usize) -> RunRef<'_> {
        self.run_at(index).expect("a known run index")
    }

    fn gap_exists(&self, before: usize) -> bool {
        let gap = Item::Gap { before };
        self.item_end(gap)
            .is_none_or(|end| self.item_rank(gap) < end)
    }

    fn point(&self, edge: Edge) -> Point {
        match edge {
            Edge::Top => Point::top(),
            Edge::Bottom => Point::bottom(),
            Edge::Of { run, at } => self.run(run).edge(at),
        }
    }

    /// The next run's top or the node's bottom when it bounds this run.
    fn run_bottom(&self, run: usize) -> Edge {
        let current = self.run(run);
        match self.run_at(run + 1) {
            Some(next) if next.rank() == current.end() => Edge::Of {
                run: run + 1,
                at: 0,
            },
            None if self.count() == Some(current.end()) => Edge::Bottom,
            _ => Edge::Of {
                run,
                at: current.len(),
            },
        }
    }

    /// Where an item's edges come from.
    fn item_bounds(&self, item: Item) -> (Edge, Edge) {
        let top = match item {
            Item::Child { run, at } => Edge::Of { run, at },
            Item::Gap { before: 0 } => Edge::Top,
            Item::Gap { before } => self.run_bottom(before - 1),
        };
        let bottom = match item {
            Item::Child { run, at } if at + 1 < self.run(run).len() => Edge::Of { run, at: at + 1 },
            Item::Child { run, .. } => self.run_bottom(run),
            Item::Gap { before } if before < self.run_count() => Edge::Of { run: before, at: 0 },
            Item::Gap { .. } => Edge::Bottom,
        };
        (top, bottom)
    }

    /// Both of an item's edges.
    fn item_edges(&self, item: Item) -> (Point, Point) {
        let (top, bottom) = self.item_bounds(item);
        (self.point(top), self.point(bottom))
    }

    /// The rank an item starts at: a gap's is its first unknown one.
    fn item_rank(&self, item: Item) -> Rank {
        match item {
            Item::Child { run, at } => self.run(run).rank() + at as Rank,
            Item::Gap { before: 0 } => 0,
            Item::Gap { before } => self.run(before - 1).end(),
        }
    }

    /// The bottom edge's rank, where one is known.
    fn item_end(&self, item: Item) -> Option<Rank> {
        match item {
            Item::Child { run, at } => Some(self.run(run).rank() + at as Rank + 1),
            Item::Gap { before } => self
                .run_at(before)
                .map(RunRef::rank)
                .or_else(|| self.count()),
        }
    }

    /// The item after `item`, if any.
    fn next(&self, item: Item) -> Option<Item> {
        let before = match item {
            Item::Child { run, at } if at + 1 < self.run(run).len() => {
                return Some(Item::Child { run, at: at + 1 });
            }
            Item::Child { run, .. } => run + 1,
            Item::Gap { before } => {
                return (before < self.run_count()).then_some(Item::Child { run: before, at: 0 });
            }
        };
        if self.gap_exists(before) {
            Some(Item::Gap { before })
        } else {
            (before < self.run_count()).then_some(Item::Child { run: before, at: 0 })
        }
    }

    /// The node's first item, if it has any.
    fn first(&self) -> Option<Item> {
        if self.gap_exists(0) {
            Some(Item::Gap { before: 0 })
        } else {
            (0 < self.run_count()).then_some(Item::Child { run: 0, at: 0 })
        }
    }

    /// The item holding a point: the first whose bottom edge is under it.
    /// At or below the node's bottom, there is none.
    fn locate(&self, point: &Point) -> Option<Item> {
        self.locate_exact(point)
    }

    fn locate_exact(&self, point: &Point) -> Option<Item> {
        if Point::bottom() <= *point {
            return None;
        }
        let (mut run, mut high) = (0, self.run_count());
        while run < high {
            let middle = (run + high) / 2;
            if point.below(&self.point(self.run_bottom(middle))) {
                run = middle + 1;
            } else {
                high = middle;
            }
        }
        if let Some(found) = self.run_at(run).filter(|found| point.below(&found.edge(0))) {
            return Some(Item::Child {
                run,
                at: found.holding(point),
            });
        }
        self.gap_exists(run).then_some(Item::Gap { before: run })
    }
}

impl RankView for Distribution {
    fn run_count(&self) -> usize {
        self.run_count()
    }

    fn run_at(&self, index: usize) -> Option<RunRef<'_>> {
        self.run_at(index)
    }

    fn rank(&self, branch: Branch) -> Option<Rank> {
        self.rank(branch)
    }

    fn locate_rank(&self, rank: Rank) -> Option<(usize, usize)> {
        self.locate_rank(rank)
    }

    fn count(&self) -> Option<Rank> {
        self.count()
    }
}

/// `fresh` runs over `old` ones: each rank from the run with the more
/// precise top, the fresh one where they are alike. A contradiction replaces
/// the old answer, including a branch named by two ranks.
fn merge(old: &Listed, fresh: &Listed) -> Option<Listed> {
    let mut runs = Vec::new();
    for (side, over, fresh_side) in [
        (&old.runs, &fresh.runs, false),
        (&fresh.runs, &old.runs, true),
    ] {
        for run in side {
            let first = over.partition_point(|other| other.end() <= run.rank());
            let mut from = run.rank();
            for other in &over[first..] {
                if run.end() <= other.rank() {
                    break;
                }
                let start = run.rank().max(other.rank());
                let end = run.end().min(other.end());
                if !fresh_side && !same_branches(run.whole(), other.whole(), start, end) {
                    return None;
                }
                let other_wins = if fresh_side {
                    run.precision() < other.precision()
                } else {
                    run.precision() <= other.precision()
                };
                if other_wins {
                    if from < start {
                        runs.push(run.slice_ranks(from, start));
                    }
                    from = end;
                }
            }
            if from < run.end() {
                runs.push(run.slice_ranks(from, run.end()));
            }
        }
    }
    Listed::new(runs, fresh.count.or(old.count)).ok()
}

pub(crate) fn same_branches(a: RunSlice<'_>, b: RunSlice<'_>, start: Rank, end: Rank) -> bool {
    a.branches()[(start - a.rank()) as usize..(end - a.rank()) as usize]
        == b.branches()[(start - b.rank()) as usize..(end - b.rank()) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run from weights, its top where `above` of the node is above it,
    /// to `digits` digits.
    fn run(rank: Rank, branches: &[Branch], weights: &[f64], above: f64, digits: usize) -> Run {
        Run::weighed(rank, branches, weights, &Point::top().plus(above), digits)
    }

    fn distribution(run: Run) -> Distribution {
        Distribution::from_answer(vec![run], None).unwrap()
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
        assert!(Distribution::from_answer(Vec::new(), Some(5))
            .unwrap()
            .insert(vec![twice], None)
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
        close(r.ln_weights()[1], -80.0 * std::f64::consts::LN_2);
        assert_eq!(r.edge(3), Point::new(&[0xe0]));
        assert_eq!(r.top_digits(), [0x80]);
        assert_eq!(RunRef::listed(&r).holding(&Point::new(&[0xc8])), 2);
        assert_eq!(RunRef::listed(&r).holding(&Point::new(&[0x80])), 0);
    }

    fn ordered(weights: &[f32], heaviest: bool) -> Distribution {
        let mut children: Vec<_> = (0..).zip(weights.iter().copied().map(f64::from)).collect();
        if heaviest {
            children.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        }
        log_softmax(&mut children);
        Distribution::from_ordered(children).unwrap()
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
        let d = ordered(&weights, true);
        let run = d.run(0);
        assert_eq!(
            (0..run.len()).map(|at| run.branch(at)).collect::<Vec<_>>(),
            [1, 3, 0, 5, 4]
        );
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
        let d = ordered(&weights, false);
        let run = d.run(0);
        assert_eq!(
            (0..run.len()).map(|at| run.branch(at)).collect::<Vec<_>>(),
            [0, 1, 3, 4, 5]
        );
        let (top, bottom) = d.item_edges(Item::Child { run: 0, at: 3 });
        assert!(top < bottom);
        assert_eq!(d.locate(&top), Some(Item::Child { run: 0, at: 3 }));
        assert!(Distribution::from_ordered(Vec::new()).unwrap().ended());
        assert!(Distribution::from_ordered(vec![(0, f64::NAN)]).is_err());
        // Absolute weights are each child's own, whatever they sum to: a
        // measure of widths.
        let widths = Distribution::from_ordered(vec![(0, -1.0), (1, -1.0), (2, -3.0)]).unwrap();
        assert_eq!(
            (0..3).map(|at| widths.run(0).ln_p(at)).collect::<Vec<_>>(),
            [-1.0, -1.0, -3.0]
        );
        assert!(Distribution::from_ordered(vec![(0, f64::INFINITY)]).is_err());
    }

    #[test]
    fn answers_anchor_runs_as_precisely_as_asked() {
        // Heavy children, then one far lighter than any float of a sum of
        // them tells apart, in branch order.
        let mut weights = vec![-1.0f32; 10];
        weights[5] = -90.0;
        let d = ordered(&weights, false);
        let all: Vec<_> = (0..10).map(|at| (0, at)).collect();
        // Coarse: one run, its top the node's.
        let coarse = d.answer(&all, 16);
        assert_eq!(coarse.len(), 1);
        assert_eq!(coarse[0].top_digits(), [0, 0]);
        // Fine enough for the light child: runs break before sums drift.
        let fine = d.answer(&all, 8 * 20);
        assert!(1 < fine.len());
        let mut client = Distribution::from_answer(Vec::new(), Some(10)).unwrap();
        client.insert(fine, None).unwrap();
        let (top, bottom) = client.item_edges(Item::Child { run: 5, at: 0 });
        let (exact_top, exact_bottom) = d.item_edges(Item::Child { run: 0, at: 5 });
        let ln_p = d.run(0).ln_p(5);
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
        let mut d = distribution(run(0, &[10, 11, 12], &[0.5, 0.25, 0.125], 0.0, 1));
        assert!(d.gap_exists(1) && !d.gap_exists(0));
        // A finer answer for rank 1 replaces it; the coarse ranks around it
        // keep their edges.
        assert!(d
            .insert(vec![run(1, &[11], &[0.25], 0.5, 4)], None)
            .unwrap());
        assert_eq!(d.run_count(), 3);
        assert_eq!(d.run(1).precision(), 32);
        assert_eq!(d.run(2).edge(0), Point::new(&[0xc0]));
        // A coarser one does not.
        assert!(!d
            .insert(vec![run(1, &[11], &[0.25], 0.5, 1)], None)
            .unwrap());
        // At one precision, the latest wins.
        assert!(d.insert(vec![run(1, &[11], &[0.2], 0.5, 4)], None).unwrap());
        close(d.run(1).ln_p(0), 0.2f64.ln());
        // Touching runs: the next one's top bottoms the child before it.
        let bottom = d.item_edges(Item::Child { run: 0, at: 0 }).1;
        assert_eq!(bottom, d.run(1).edge(0));
        assert_eq!(
            d.next(Item::Child { run: 0, at: 0 }),
            Some(Item::Child { run: 1, at: 0 })
        );
        // A rank that names another branch: the source changed, start over.
        d.insert(vec![run(2, &[99], &[0.1], 0.75, 1)], None)
            .unwrap();
        assert_eq!(d.run_count(), 1);
        assert_eq!((d.rank(10), d.rank(99)), (None, Some(2)));
        // So does a branch at another rank.
        let mut d = distribution(run(0, &[10, 11], &[0.5, 0.25], 0.0, 1));
        d.insert(vec![run(5, &[10], &[0.05], 0.9, 1)], None)
            .unwrap();
        assert_eq!(d.rank(10), Some(5));
        assert_eq!(d.run_count(), 1);
        // An answer that places one rank twice is refused.
        assert!(d
            .insert(
                vec![run(0, &[3], &[0.5], 0.0, 1), run(0, &[3], &[0.5], 0.0, 1)],
                None,
            )
            .is_err());
    }

    #[test]
    fn items_walk_children_and_gaps_in_order() {
        let mut d = distribution(run(0, &[7, 8], &[0.5, 0.25], 0.0, 1));
        d.insert(vec![run(5, &[9], &[0.01], 0.97, 2)], None)
            .unwrap();
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
        let mut ending = distribution(run(0, &[7, 8], &[0.5, 0.25], 0.0, 1));
        assert_eq!(ending.count(), None);
        ending.insert(Vec::new(), Some(9)).unwrap();
        assert_eq!(ending.count(), Some(9));
        assert_eq!(ending.item_end(Item::Gap { before: 1 }), Some(9));
        assert_eq!(ending.next(Item::Gap { before: 1 }), None);
        assert_eq!(ending.edge(9), Some(Point::bottom()));
        assert!(Distribution::from_answer(Vec::new(), Some(0))
            .unwrap()
            .ended());
        let late = distribution(run(3, &[1], &[0.1], 0.5, 1));
        assert_eq!(late.locate(&Point::top()), Some(Item::Gap { before: 0 }));
        assert_eq!(late.edge(0), Some(Point::top()));
    }

    #[test]
    fn a_count_alone_bounds_a_gap_until_its_children_arrive() {
        let mut d = Distribution::from_answer(Vec::new(), Some(2)).unwrap();
        let gap = Item::Gap { before: 0 };
        assert_eq!(d.first(), Some(gap));
        assert_eq!(d.locate(&Point::top()), Some(gap));
        assert_eq!(d.item_end(gap), Some(2));
        assert_eq!(d.item_edges(gap), (Point::top(), Point::bottom()));
        assert_eq!(d.ln_p(99), None);
        assert!(!d.ended());

        let tail = run(1, &[8], &[0.25], 0.75, 1);
        assert!(d.insert(vec![tail.clone()], Some(1)).is_err());
        assert_eq!(d.first(), Some(gap));
        assert!(d.insert(vec![tail], None).unwrap());
        assert!(d.insert(vec![run(0, &[7], &[0.75], 0.0, 1)], None).unwrap());
        assert!(d.complete());
        assert_eq!(d.ln_p(99), Some(f64::NEG_INFINITY));
        assert!(!d.insert(Vec::new(), Some(2)).unwrap());
    }

    #[test]
    fn a_listed_answer_replaces_packed_data_only_when_it_disagrees() {
        let mut d = ordered(&[-1.0; 2], false);
        assert!(!d
            .insert(vec![run(1, &[1], &[0.5], 0.5, 1)], Some(2))
            .unwrap());
        assert!(d.complete());
        assert_eq!(d.rank(0), Some(0));

        assert!(d
            .insert(vec![run(1, &[2], &[0.5], 0.5, 1)], Some(2))
            .unwrap());
        assert!(!d.complete());
        assert_eq!(d.rank(0), None);
        assert_eq!(d.rank(2), Some(1));
    }
}
