//! Fixed-point CDF geometry and exact per-row width queries for the browser.

use std::mem::size_of;

use crate::point::Point;
use crate::pstore::{BranchHit, CdfRange, TieredPStore, CDF_SIZE};

/// A node placed in view heights, measured from exact camera anchors.
/// Subtracting the CDF coordinates before converting to f64 preserves the
/// visible differences even when the node starts many view heights away.
pub(crate) struct QLayout {
    anchor: (u64, f64),
    units: f64,
    height: f64,
}

impl QLayout {
    pub(crate) fn new(ln_scale: f64, top: (&Point, f64), bottom: (&Point, f64)) -> Option<Self> {
        if !Self::supports(ln_scale) {
            return None;
        }
        let edge = if (top.1 - 0.5).abs() <= (bottom.1 - 0.5).abs() {
            top
        } else {
            bottom
        };
        let height = ln_scale.exp() / CDF_SIZE as f64;
        let anchor = match u64::try_from(edge.0.q64()) {
            Ok(cut) => (cut, edge.1),
            // Use the last cell's top to represent the node's bottom anchor.
            Err(_) => (u64::MAX, edge.1 - height),
        };
        Some(Self {
            anchor,
            units: 1.0 / height,
            height,
        })
    }

    pub(crate) fn supports(ln_scale: f64) -> bool {
        ln_scale < 44.0 * std::f64::consts::LN_2
    }

    pub(crate) fn in_rows(&self, row: f64) -> Self {
        let scale = 1.0 / row;
        Self {
            anchor: (self.anchor.0, self.anchor.1 * scale),
            units: self.units * row,
            height: self.height * scale,
        }
    }

    #[inline]
    fn y_of(&self, cut: u64) -> f64 {
        let distance = cut.abs_diff(self.anchor.0) as f64 * self.height;
        if cut < self.anchor.0 {
            self.anchor.1 - distance
        } else {
            self.anchor.1 + distance
        }
    }

    pub(crate) fn y(&self, cut: u128) -> f64 {
        match u64::try_from(cut) {
            Ok(cut) => self.y_of(cut),
            Err(_) => self.y_of(u64::MAX) + self.height,
        }
    }

    #[inline]
    pub(crate) fn end_y(&self, end: u64) -> f64 {
        match end.checked_add(1) {
            Some(cut) => self.y_of(cut),
            None => self.y_of(end) + self.height,
        }
    }

    // Round the offset before adding it to the anchor, since a float
    // cannot represent every cut near the bottom of the CDF.
    fn cut(&self, y: f64, round: fn(f64) -> f64) -> i128 {
        let delta = round((y - self.anchor.1) * self.units);
        let magnitude = delta.abs();
        let offset = if magnitude < CDF_SIZE as f64 {
            i128::from(magnitude as u64)
        } else if magnitude == CDF_SIZE as f64 {
            CDF_SIZE as i128
        } else {
            CDF_SIZE as i128 + 1
        };
        let anchor = self.anchor.0 as i128;
        if delta < 0.0 {
            anchor - offset
        } else {
            anchor + offset
        }
    }

    pub(crate) fn position(&self, p: &TieredPStore, y: f64) -> usize {
        let cut = self.cut(y, f64::floor).max(0);
        u64::try_from(cut).map_or_else(|_| p.ends().len(), |at| p.position(at))
    }

    /// The inclusive cell before a rounded-up cut. Rows use this same
    /// rounding on both sides of a grid line.
    #[inline]
    fn before(&self, y: f64) -> BeforeCut {
        let delta = ((y - self.anchor.1) * self.units).ceil();
        let anchor = self.anchor.0;
        if delta <= 0.0 {
            let step = -delta as u64;
            if step < anchor {
                BeforeCut::Inside(anchor - step - 1)
            } else {
                BeforeCut::Above
            }
        } else if delta < CDF_SIZE as f64 {
            anchor
                .checked_add(delta as u64 - 1)
                .map_or(BeforeCut::Below, BeforeCut::Inside)
        } else if anchor == 0 && delta == CDF_SIZE as f64 {
            BeforeCut::Inside(u64::MAX)
        } else {
            BeforeCut::Below
        }
    }

    /// First interval with a bottom at or beyond y.
    #[inline]
    pub(crate) fn bottom_at(&self, p: &TieredPStore, y: f64, from: usize) -> usize {
        match self.before(y) {
            BeforeCut::Above => from,
            BeforeCut::Inside(at) => p.seek(at, from),
            BeforeCut::Below => p.ends().len(),
        }
    }

    /// Past the last interval with a top strictly above y.
    #[inline]
    pub(crate) fn tops_before(&self, p: &TieredPStore, y: f64, from: usize) -> usize {
        match self.before(y) {
            BeforeCut::Above => 0,
            BeforeCut::Inside(at) => (p.seek(at, from) + 1).min(p.ends().len()),
            BeforeCut::Below => p.ends().len(),
        }
    }

    pub(crate) fn span(&self, p: &TieredPStore, position: usize) -> (f64, f64) {
        (
            self.y(Self::start(p, position)),
            self.y(u128::from(p.ends()[position]) + 1),
        )
    }

    pub(crate) fn start(p: &TieredPStore, position: usize) -> u128 {
        position
            .checked_sub(1)
            .map_or(0, |at| u128::from(p.ends()[at]) + 1)
    }

    pub(crate) fn tall_in_order(
        &self,
        p: &TieredPStore,
        y: [f64; 2],
        row: f64,
        out: &mut Vec<BranchHit>,
    ) {
        let start = self.cut(y[0], f64::floor).max(0);
        let end = self.cut(y[1], f64::ceil).min(CDF_SIZE as i128);
        // A conservative threshold leaves the final pixel-height decision to
        // the projected edges. A full interval can hold all 2^64 units.
        let span = (row * self.units) as u128;
        if start < end {
            if let Ok(span_m1) = u64::try_from(span.saturating_sub(2)) {
                p.query_spans_unsorted(CdfRange::new(start as u64, (end - 1) as u64), span_m1, out);
                out.sort_unstable_by_key(|hit| hit.start);
                return;
            }
        }
        out.clear();
    }
}

enum BeforeCut {
    Above,
    Inside(u64),
    Below,
}

/// The rightmost boundary in a run of p-ordered children. Ordered widths
/// need one endpoint lookup. A block tree handles arbitrary q order.
pub(crate) struct WidthIndex {
    start: usize,
    values: Vec<f64>,
    order: WidthOrder,
}

enum WidthOrder {
    Increasing,
    Decreasing,
    Blocks { maxima: Vec<f64>, leaves: usize },
}

impl WidthIndex {
    // One cache line of widths per leaf.
    const BLOCK: usize = 8;

    pub(crate) fn new(start: usize, mut values: Vec<f64>) -> Self {
        for value in &mut values {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        let order = if values.first() <= values.last() {
            values.is_sorted().then_some(WidthOrder::Increasing)
        } else {
            values
                .is_sorted_by(|a, b| b <= a)
                .then_some(WidthOrder::Decreasing)
        }
        .unwrap_or_else(|| {
            let leaves = values.len().div_ceil(Self::BLOCK).next_power_of_two();
            let mut maxima = vec![f64::NEG_INFINITY; 2 * leaves];
            for (at, chunk) in values.chunks(Self::BLOCK).enumerate() {
                maxima[leaves + at] = chunk.iter().copied().reduce(f64::max).unwrap();
            }
            for at in (1..leaves).rev() {
                maxima[at] = maxima[2 * at].max(maxima[2 * at + 1]);
            }
            WidthOrder::Blocks { maxima, leaves }
        });
        Self {
            start,
            values,
            order,
        }
    }

    #[inline]
    pub(crate) fn at(&self, index: usize) -> f64 {
        self.values[index - self.start]
    }

    #[inline]
    pub(crate) fn maximum(&self, start: usize, end: usize) -> f64 {
        let (start, end) = (start - self.start, end - self.start);
        match &self.order {
            WidthOrder::Increasing => self.values[end - 1],
            WidthOrder::Decreasing => self.values[start],
            WidthOrder::Blocks { maxima, leaves } => {
                self.maximum_in_blocks(start, end, maxima, *leaves)
            }
        }
    }

    fn maximum_in_blocks(&self, start: usize, end: usize, maxima: &[f64], leaves: usize) -> f64 {
        let first = start.div_ceil(Self::BLOCK);
        let last = end / Self::BLOCK;
        if last <= first {
            return self.values[start..end]
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap();
        }
        let (mut left, mut right) = (first + leaves, last + leaves);
        let mut reach = f64::NEG_INFINITY;
        while left < right {
            if left % 2 != 0 {
                reach = reach.max(maxima[left]);
                left += 1;
            }
            if right % 2 != 0 {
                right -= 1;
                reach = reach.max(maxima[right]);
            }
            left /= 2;
            right /= 2;
        }
        self.values[start..first * Self::BLOCK]
            .iter()
            .chain(&self.values[last * Self::BLOCK..end])
            .copied()
            .fold(reach, f64::max)
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        let blocks = match &self.order {
            WidthOrder::Blocks { maxima, .. } => maxima.capacity(),
            _ => 0,
        };
        (self.values.capacity() + blocks) * size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ClientTree;
    use crate::pstore::{bench_logits, bench_queries, PStore, QueryScratch};
    use crate::tree::{child_hash, derive_branch, derive_parent, Branch, ROOT};

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
        let mut tree = ClientTree::new();
        let child = tree.child(ROOT, Branch::MAX).unwrap();
        assert_eq!(tree.child(ROOT, Branch::MAX), Ok(child));
        assert_eq!(child_hash(ROOT, Branch::MAX), child);
        assert!(tree.contains(child));
        assert_eq!(tree.parent(child), Ok(Some(ROOT)));
        assert_eq!(tree.branch(child), Ok(Some(Branch::MAX)));
    }

    #[test]
    fn compact_tiers_agree_with_the_previous_store() {
        let mut scratch = QueryScratch::default();
        let mut reference = Vec::new();
        let mut hits = Vec::new();
        for kind in 0..3 {
            let logits = bench_logits(kind, 4096);
            let old = crate::pstore::BTreeMax::from_logits(&logits);
            let new = TieredPStore::from_logits(&logits);
            for (range, min_mass) in bench_queries(64) {
                old.query_tall(range, min_mass, &mut scratch, &mut reference);
                new.tall(range, min_mass, &mut hits);
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
        sparse.tall(CdfRange::FULL, 92, &mut hits);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].branch, 9);
    }
}
