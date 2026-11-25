//! Order-agnostic, scale-stratified Q0.64 CDF store (`TieredPStore`).
//!
//! All coordinates use the full `0..=u64::MAX` range as inclusive intervals `[start, end]`:
//! - `start` represents `start / 2^64` (so `0` is `0.0`)
//! - `end` represents `(end + 1) / 2^64` (so `u64::MAX` is `1.0` exactly)
//! - `[0, u64::MAX]` represents `[0.0, 1.0]` exactly
//! - True mass of `[start, end]` is `((end - start) + 1) / 2^64`
//! - `span_m1 = end - start` (in `0..=u64::MAX`) is strictly order-preserving with mass
//!   and never overflows `u64`. For `0 < min_mass`, a branch meets that minimum
//!   exactly when `min_mass - 1 <= end - start`.

use std::collections::HashSet;
use std::mem::size_of;

use crate::tree::Branch;

pub(crate) const CDF_SIZE: u128 = 1 << 64;
const TIERS: usize = u64::BITS as usize + 1;
const GAP: u32 = u32::MAX;

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

    pub(crate) fn bounds(self) -> (u64, u64) {
        (self.start, self.end)
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

pub(crate) fn max_logit(logits: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &logit in logits {
        assert!(!logit.is_nan(), "a logit is not a number");
        assert!(logit != f32::INFINITY, "a logit is infinite");
        if max < logit {
            max = logit;
        }
    }
    max
}

pub(crate) fn normalize_logits(logits: &[f32]) -> (Vec<u64>, Vec<u64>, Vec<Branch>, Vec<f32>) {
    let max_l = max_logit(logits);
    if !max_l.is_finite() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut raw_branches = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        if l > f32::NEG_INFINITY {
            let e = (f64::from(l) - f64::from(max_l)).exp();
            if e > 0.0 {
                exps.push(e);
                raw_branches.push(i as Branch);
                sum += e;
            }
        }
    }
    let m = raw_branches.len();
    if m == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let ln_sum = sum.ln();
    let mut starts = Vec::with_capacity(m);
    let mut ends = Vec::with_capacity(m);
    let mut branches = Vec::with_capacity(m);
    let mut ln_p = Vec::with_capacity(m);

    const TWO_64: f64 = CDF_SIZE as f64;
    let scale = TWO_64 / sum;
    let mut acc = 0.0f64;
    let mut cur_start = 0u64;
    let mut exhausted = false;

    for (k, &e) in exps.iter().enumerate() {
        if exhausted {
            break;
        }
        acc += e;
        let next_start_u128 = if k + 1 == m {
            CDF_SIZE
        } else {
            ((acc * scale).round() as u128).min(CDF_SIZE)
        };
        if next_start_u128 <= u128::from(cur_start) {
            continue;
        }
        let end = (next_start_u128 - 1) as u64;
        let orig = logits[raw_branches[k] as usize];
        starts.push(cur_start);
        ends.push(end);
        branches.push(raw_branches[k]);
        ln_p.push((f64::from(orig) - f64::from(max_l) - ln_sum) as f32);
        if end == u64::MAX {
            exhausted = true;
        } else {
            cur_start = end + 1;
        }
    }
    if let Some(last_end) = ends.last_mut() {
        *last_end = u64::MAX;
    }
    (starts, ends, branches, ln_p)
}

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
        let mut next = 0;
        let mut positions = raw.len();
        for entry in raw {
            let start = u128::from(entry.start);
            assert!(
                next <= start && entry.start <= entry.end,
                "invalid CDF interval order"
            );
            assert!(entry.ln_p <= 0.0, "invalid log probability");
            positions += usize::from(next < start);
            next = u128::from(entry.end) + 1;
        }
        positions += usize::from(!raw.is_empty() && next < CDF_SIZE);

        let mut ends = Vec::with_capacity(positions);
        let mut pos_to_entry = Vec::with_capacity(positions);
        next = 0;
        let (entries, tier_starts) = Self::pack_tiers(raw.iter().copied(), |entry, index| {
            if next < u128::from(entry.start) {
                ends.push(entry.start - 1);
                pos_to_entry.push(GAP);
            }
            ends.push(entry.end);
            pos_to_entry.push(index);
            next = u128::from(entry.end) + 1;
        });
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

    /// A contiguous partition, with the position as payload. A repeated
    /// end is a coincident boundary and has no positive interval to index.
    pub(crate) fn from_partition(ends: Vec<u64>) -> Self {
        u32::try_from(ends.len()).expect("the tier index must fit in u32");
        let raw = ends.iter().copied().enumerate().filter_map(|(pos, end)| {
            let start = match pos.checked_sub(1) {
                Some(prior) => ends[prior].checked_add(1)?,
                None => 0,
            };
            (start <= end).then_some(BranchHit {
                branch: pos as u32,
                start,
                end,
                ln_p: 0.0,
            })
        });
        let mut pos_to_entry = vec![GAP; ends.len()];
        let (entries, tier_starts) = Self::pack_tiers(raw, |entry, index| {
            pos_to_entry[entry.branch as usize] = index;
        });
        Self {
            entries,
            tier_starts,
            ends,
            pos_to_entry,
        }
    }

    fn pack_tiers(
        raw: impl Iterator<Item = BranchHit> + Clone,
        mut position: impl FnMut(BranchHit, u32),
    ) -> (Vec<BranchHit>, [u32; TIERS + 1]) {
        let tier = |entry: BranchHit| entry.span_m1().leading_zeros() as usize;
        let mut starts = [0u32; TIERS + 1];
        for entry in raw.clone() {
            starts[tier(entry) + 1] += 1;
        }
        for tier in 0..TIERS {
            starts[tier + 1] += starts[tier];
        }
        let mut entries = vec![
            BranchHit {
                branch: 0,
                start: 0,
                end: 0,
                ln_p: 0.0
            };
            starts[TIERS] as usize
        ];
        let mut cursors = starts;
        for entry in raw {
            let at = &mut cursors[tier(entry)];
            entries[*at as usize] = entry;
            position(entry, *at);
            *at += 1;
        }
        (entries, starts)
    }

    /// Replace `out` with known overlapping branches at least `min_mass`
    /// CDF units tall, heaviest first, with ties in CDF order.
    pub fn tall(&self, range: CdfRange, min_mass: u64, out: &mut Vec<BranchHit>) {
        self.query_spans(range, min_mass.saturating_sub(1), out);
    }

    pub(crate) fn query_spans_unsorted(
        &self,
        range: CdfRange,
        min_span_m1: u64,
        out: &mut Vec<BranchHit>,
    ) {
        out.clear();
        let (start, end) = range.bounds();
        let last_tier = min_span_m1.leading_zeros() as usize;
        for t in 0..=last_tier {
            let tier =
                &self.entries[self.tier_starts[t] as usize..self.tier_starts[t + 1] as usize];
            if tier.is_empty() {
                continue;
            }
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
    }

    pub(crate) fn query_spans(&self, range: CdfRange, min_span_m1: u64, out: &mut Vec<BranchHit>) {
        self.query_spans_unsorted(range, min_span_m1, out);
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

    pub(crate) fn ends(&self) -> &[u64] {
        &self.ends
    }

    pub(crate) fn position(&self, at: u64) -> usize {
        self.ends.partition_point(|&end| end < at)
    }

    /// The first interval at or past a cursor, with a galloping search
    /// so a sequence of nearby row boundaries does not revisit the root.
    #[inline]
    pub(crate) fn seek(&self, at: u64, from: usize) -> usize {
        if self.ends.get(from).is_none_or(|&end| at <= end) {
            return from;
        }
        let mut skip = 1;
        let rest = &self.ends[from..];
        while rest.get(skip).is_some_and(|&end| end < at) {
            skip *= 2;
        }
        let first = skip / 2 + 1;
        from + first + rest[first..(skip + 1).min(rest.len())].partition_point(|&end| end < at)
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

    pub(crate) fn heap_bytes(&self) -> usize {
        self.entries.capacity() * size_of::<BranchHit>()
            + self.ends.capacity() * size_of::<u64>()
            + self.pos_to_entry.capacity() * size_of::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiered_store_matches_linear_scan_on_zipf_and_uniform() {
        let mut shuffled: Vec<f32> = (0..1024)
            .map(|i| (-1.15 * ((i + 1) as f64).ln()) as f32)
            .collect();
        shuffled.reverse();
        for logits in [
            (0..1024)
                .map(|i| (-1.15 * ((i + 1) as f64).ln()) as f32)
                .collect::<Vec<_>>(),
            shuffled,
            vec![0.0f32; 1024],
        ] {
            let store = TieredPStore::from_logits(&logits);
            let (starts, ends, branches, weights) = normalize_logits(&logits);
            let all: Vec<BranchHit> = starts
                .into_iter()
                .zip(ends)
                .zip(branches)
                .zip(weights)
                .map(|(((start, end), branch), ln_p)| BranchHit {
                    branch,
                    start,
                    end,
                    ln_p,
                })
                .collect();

            let mut out = Vec::new();
            for range in [
                CdfRange::FULL,
                CdfRange::new(0, u64::MAX >> 4),
                CdfRange::new(u64::MAX / 3, (u64::MAX / 3) * 2),
                CdfRange::new(u64::MAX - (u64::MAX >> 5), u64::MAX),
            ] {
                let (lo, hi) = range.bounds();
                let min_mass = (((u128::from(hi - lo) + 1) / 200) as u64).max(1);
                let min_span_m1 = min_mass.saturating_sub(1);
                let mut expected: Vec<BranchHit> = all
                    .iter()
                    .copied()
                    .filter(|h| h.end >= lo && h.start <= hi && h.span_m1() >= min_span_m1)
                    .collect();
                expected.sort_unstable_by(|a, b| {
                    b.span_m1()
                        .cmp(&a.span_m1())
                        .then_with(|| a.start.cmp(&b.start))
                });
                store.tall(range, min_mass, &mut out);
                assert_eq!(out, expected);

                let mid = range.midpoint();
                let exp_loc = all.iter().copied().find(|h| h.start <= mid && mid <= h.end);
                assert_eq!(store.locate(mid), exp_loc);
            }
            assert_eq!(store.locate(0), all.first().copied());
            assert_eq!(store.locate(u64::MAX), all.last().copied());
        }
    }
}
