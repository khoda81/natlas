//! A whole vocabulary in its supplied rank order. The compact store keeps
//! branch ids and final f64 log weights losslessly. Nearby weights commonly
//! share low bits, so each block drops the zeros common to its deltas.

//! Each child's weight is fixed once as a float, `e^(ln w − max)`, and the
//! sums of those floats are exact: fixed-point over the node's range, kept
//! before every `SUMMED`th child. An edge's fraction above, Σ_{i<r} w_i / W,
//! is an exact rational a few hundred fixed-point additions from a kept sum,
//! however light the children above it or below it.

use num_bigint::BigUint;

use crate::point::Point;
use crate::tree::Branch;

/// The block header holds an f64 key. A 32-child block amortizes it while
/// a scalar width read decodes only up to the requested position.
const BLOCK: usize = 32;
/// A kept sum every 256 children balances cached bytes against edge time.
const SUMMED: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Packed {
    len: usize,
    /// Each child's branch, by rank, in `width` bits.
    branches: Bits,
    width: u32,
    /// Each block's first weight, as its key.
    keys: Vec<u64>,
    shifts: Vec<u8>,
    /// Where each block's steps start in `steps`, in bits, and where the
    /// last one's end.
    starts: Vec<u32>,
    /// Each weight after its block's first, as how far its key is from the
    /// one before, zigzagged.
    steps: Bits,
    /// The heaviest ln weight: weights are fixed relative to it.
    max: f64,
    /// The exact sum of the weights above every `SUMMED`th child, in units
    /// of 2^`lo`, `limbs` words each, least significant first.
    sums: Vec<u64>,
    limbs: usize,
    lo: i64,
    /// The sum of every weight, W, in the same units.
    total: BigUint,
}

impl Packed {
    /// Children in the measure's order: at least one, each weight finite.
    pub(crate) fn new(children: &[(Branch, f64)]) -> Self {
        assert!(!children.is_empty() && children.iter().all(|(_, weight)| weight.is_finite()));
        let len = children.len();
        let max_branch = children
            .iter()
            .map(|&(branch, _)| branch)
            .max()
            .expect("children");
        let width = u32::BITS - max_branch.leading_zeros();
        let mut branches = Bits::default();
        for &(branch, _) in children {
            branches.push(u64::from(branch), width);
        }
        let blocks = len.div_ceil(BLOCK);
        let mut keys = Vec::with_capacity(blocks);
        let mut shifts = Vec::with_capacity(blocks);
        let mut starts = Vec::with_capacity(blocks + 1);
        let mut steps = Bits::default();
        for block in children.chunks(BLOCK) {
            let mut words = [0; BLOCK];
            for (slot, &(_, weight)) in words.iter_mut().zip(block) {
                *slot = key(weight);
            }
            let words = &words[..block.len()];
            let zeroes = words
                .windows(2)
                .map(|pair| pair[1].wrapping_sub(pair[0]))
                .fold(0, |bits, delta| bits | delta)
                .trailing_zeros()
                .min(u64::BITS - 1);
            let encoded = |pair: &[u64]| step(pair[0], pair[1], zeroes);
            let width = words
                .windows(2)
                .map(encoded)
                .max()
                .map_or(0, |n| u64::BITS - n.leading_zeros());
            keys.push(words[0]);
            shifts.push(zeroes as u8);
            starts.push(u32::try_from(steps.len).expect("steps fit in u32 bits"));
            for pair in words.windows(2) {
                steps.push(encoded(pair), width);
            }
        }
        starts.push(u32::try_from(steps.len).expect("steps fit in u32 bits"));
        // Kept for as long as the node is: no room to grow.
        for bits in [&mut branches, &mut steps] {
            bits.words.shrink_to_fit();
        }
        keys.shrink_to_fit();
        starts.shrink_to_fit();
        let max = children
            .iter()
            .map(|&(_, weight)| weight)
            .fold(f64::NEG_INFINITY, f64::max);
        // Units below the lightest weight's last bit, and words enough for
        // the sum of every weight, each at most 1.
        let lo = children
            .iter()
            .map(|&(_, weight)| ulp_exponent(fixed(weight, max)))
            .min()
            .expect("children");
        let limbs = ((1 - lo) as u64 + u64::from(usize::BITS - len.leading_zeros())).div_ceil(64)
            as usize
            + 1;
        let mut sums = Vec::with_capacity(len.div_ceil(SUMMED) * limbs);
        let mut sum = vec![0; limbs];
        for (at, &(_, weight)) in children.iter().enumerate() {
            if at % SUMMED == 0 {
                sums.extend_from_slice(&sum);
            }
            add(&mut sum, fixed(weight, max), lo);
        }
        let total = whole(&sum);
        Self {
            len,
            branches,
            width,
            keys,
            shifts,
            starts,
            steps,
            max,
            sums,
            limbs,
            lo,
            total,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn branch(&self, at: usize) -> Branch {
        self.branches
            .get(at as u64 * u64::from(self.width), self.width) as Branch
    }

    /// The rank of each of `wanted` this node has, in one pass.
    pub(crate) fn positions(&self, wanted: &[Branch]) -> Vec<Option<usize>> {
        let mut sorted: Vec<(Branch, usize)> = wanted.iter().copied().zip(0..).collect();
        sorted.sort_unstable();
        let mut found = vec![None; wanted.len()];
        let mut left = wanted.len();
        for at in 0..self.len {
            if left == 0 {
                break;
            }
            let branch = self.branch(at);
            let first = sorted.partition_point(|&(want, _)| want < branch);
            for &(_, slot) in sorted[first..]
                .iter()
                .take_while(|&&(want, _)| want == branch)
            {
                found[slot] = Some(at);
                left -= 1;
            }
        }
        found
    }

    /// Decode only the part of the block the caller consumes.
    fn block(&self, block: usize) -> impl Iterator<Item = f64> + '_ {
        let count = BLOCK.min(self.len - block * BLOCK);
        let start = u64::from(self.starts[block]);
        let width = if count == 1 {
            0
        } else {
            (self.starts[block + 1] - self.starts[block]) / (count as u32 - 1)
        };
        let mut key = self.keys[block];
        let shift = self.shifts[block];
        std::iter::once(weight(key)).chain((0..count - 1).map(move |at| {
            let encoded = self.steps.get(start + at as u64 * u64::from(width), width);
            key = key.wrapping_add(unzigzag(encoded) << shift);
            weight(key)
        }))
    }

    pub(crate) fn ln_p(&self, at: usize) -> f64 {
        self.block(at / BLOCK)
            .nth(at % BLOCK)
            .expect("a known rank")
    }

    /// The exact sum of the weights above rank `at`, from the kept sum
    /// before it.
    fn sum(&self, at: usize) -> Vec<u64> {
        let kept = at / SUMMED;
        let mut sum = self.sums[kept * self.limbs..(kept + 1) * self.limbs].to_vec();
        let mut rank = kept * SUMMED;
        while rank < at {
            for weight in self.block(rank / BLOCK).take(at - rank) {
                add(&mut sum, fixed(weight, self.max), self.lo);
            }
            rank += BLOCK;
        }
        sum
    }

    /// The top of rank `at`, Σ_{i<at} w_i / W, rounded down to `bits` bits:
    /// rank `len` is the node's bottom.
    pub(crate) fn top(&self, at: usize, bits: u64) -> Point {
        if at == self.len {
            return Point::bottom();
        }
        Point::ratio(&whole(&self.sum(at)), &self.total, bits)
    }

    /// Bits that tell every edge as finely as any child's extent, and 64
    /// bits more: an edge as a point is its top this precise.
    fn bits(&self) -> u64 {
        64 * self.limbs as u64 + 64
    }

    pub(crate) fn edge(&self, at: usize) -> Point {
        self.top(at, self.bits())
    }

    /// The child holding a point: how many edges from rank 1 down it is at
    /// or below.
    pub(crate) fn holding(&self, point: &Point) -> usize {
        // An edge S/W, as a point to `bits` bits, is at or above the point
        // t when S · 2^bits < (⌊t · 2^bits⌋ + 1) · W, so when S is less
        // than the bound, that divided by 2^bits and rounded up.
        let bits = self.bits();
        let scaled = (point.numerator(bits) + 1u32) * &self.total;
        let bound = (&scaled >> bits) + u32::from(scaled.trailing_zeros() < Some(bits));
        let Some(bound) = limbs(&bound, self.limbs) else {
            // Beyond every sum: at or below the node's bottom.
            return self.len;
        };
        let under = |sum: &[u64]| less(sum, &bound);
        let kept = self.sums.len() / self.limbs;
        let at = |k: usize| &self.sums[k * self.limbs..(k + 1) * self.limbs];
        // The last kept sum under the bound: rank 0's always is.
        let (mut low, mut high) = (0, kept);
        while low + 1 < high {
            let middle = (low + high) / 2;
            if under(at(middle)) {
                low = middle;
            } else {
                high = middle;
            }
        }
        let mut sum = at(low).to_vec();
        let mut rank = low * SUMMED;
        while rank < self.len {
            for weight in self.block(rank / BLOCK) {
                add(&mut sum, fixed(weight, self.max), self.lo);
                if !under(&sum) {
                    return rank;
                }
                rank += 1;
            }
        }
        self.len
    }

    pub(crate) fn bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + self.branches.bytes()
            + self.steps.bytes()
            + self.sums.capacity() * size_of::<u64>()
            + self.keys.capacity() * size_of::<u64>()
            + self.starts.capacity() * size_of::<u32>()
            + self.shifts.capacity()
            + self.total.to_u64_digits().len() * size_of::<u64>()
    }
}

/// A child's weight as the node fixes it: e^(ln w − max), at most 1.
fn fixed(ln_weight: f64, max: f64) -> f64 {
    (ln_weight - max).exp()
}

/// The exponent of a float's last bit: of 2^-1074 for zero and subnormals.
fn ulp_exponent(value: f64) -> i64 {
    let exponent = ((value.to_bits() >> 52) & 0x7ff) as i64;
    if exponent == 0 {
        -1074
    } else {
        exponent - 1075
    }
}

/// Add a float that is not negative to a fixed-point sum in units of 2^lo,
/// exactly: its last bit is no finer than the unit.
fn add(sum: &mut [u64], value: f64, lo: i64) {
    let raw = value.to_bits();
    let fraction = raw & ((1 << 52) - 1);
    let mantissa = if (raw >> 52) & 0x7ff == 0 {
        fraction
    } else {
        fraction | 1 << 52
    };
    if mantissa == 0 {
        return;
    }
    let shift = (ulp_exponent(value) - lo) as usize;
    let (limb, offset) = (shift / 64, shift % 64);
    let wide = u128::from(mantissa) << offset;
    let mut carry = 0u128;
    for (i, part) in [wide as u64, (wide >> 64) as u64].into_iter().enumerate() {
        let next = u128::from(sum[limb + i]) + u128::from(part) + carry;
        sum[limb + i] = next as u64;
        carry = next >> 64;
    }
    let mut i = limb + 2;
    while carry != 0 {
        let next = u128::from(sum[i]) + carry;
        sum[i] = next as u64;
        carry = next >> 64;
        i += 1;
    }
}

fn whole(sum: &[u64]) -> BigUint {
    BigUint::from_slice(
        &sum.iter()
            .flat_map(|&limb| [limb as u32, (limb >> 32) as u32])
            .collect::<Vec<_>>(),
    )
}

/// A whole number as `count` words, least significant first, where it fits.
fn limbs(n: &BigUint, count: usize) -> Option<Vec<u64>> {
    let mut words = n.to_u64_digits();
    if count < words.len() {
        return None;
    }
    words.resize(count, 0);
    Some(words)
}

fn less(a: &[u64], b: &[u64]) -> bool {
    a.iter().rev().cmp(b.iter().rev()).is_lt()
}

/// An f64's bits ordered like its value, including signed zeros.
fn key(weight: f64) -> u64 {
    let bits = weight.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

fn weight(key: u64) -> f64 {
    f64::from_bits(if key >> 63 == 1 {
        key & !(1 << 63)
    } else {
        !key
    })
}

/// A wrapping delta without its known low zeros, zigzagged to be small either way.
fn step(from: u64, to: u64, shift: u32) -> u64 {
    let delta = (to.wrapping_sub(from) as i64) >> shift;
    ((delta << 1) ^ (delta >> 63)) as u64
}

fn unzigzag(zigzagged: u64) -> u64 {
    (zigzagged >> 1) ^ (zigzagged & 1).wrapping_neg()
}

/// Values of up to 64 bits each, packed end to end.
#[derive(Clone, Debug, Default, PartialEq)]
struct Bits {
    words: Vec<u64>,
    len: u64,
}

impl Bits {
    fn push(&mut self, value: u64, width: u32) {
        debug_assert!(width == 64 || (width < 64 && value >> width == 0));
        if width == 0 {
            return;
        }
        let offset = (self.len % 64) as u32;
        if offset == 0 {
            self.words.push(0);
        }
        *self.words.last_mut().expect("a word") |= value << offset;
        if 64 < offset + width {
            self.words.push(value >> (64 - offset));
        }
        self.len += u64::from(width);
    }

    fn get(&self, at: u64, width: u32) -> u64 {
        if width == 0 {
            return 0;
        }
        let (word, offset) = ((at / 64) as usize, (at % 64) as u32);
        let mut value = self.words[word] >> offset;
        if 64 < offset + width {
            value |= self.words[word + 1] << (64 - offset);
        }
        value & (u64::MAX >> (64 - width))
    }

    fn bytes(&self) -> usize {
        self.words.capacity() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_order_like_their_weights_and_steps_go_either_way() {
        let weights = [3.5f64, 0.0, -0.0, -1e-30, -1.0, -700.25, f64::MIN];
        for pair in weights.windows(2) {
            assert!(key(pair[1]) <= key(pair[0]), "{pair:?}");
            let (a, b) = (key(pair[0]), key(pair[1]));
            assert_eq!(a.wrapping_add(unzigzag(step(a, b, 0))), b);
            assert_eq!(b.wrapping_add(unzigzag(step(b, a, 0))), a);
        }
        for weight_ in weights {
            assert_eq!(weight(key(weight_)).to_bits(), weight_.to_bits());
        }
        assert_eq!(step(5, 5, 0), 0);
        assert!(step(5, 4, 0) < 4 && step(5, 6, 0) < 4);
    }

    #[test]
    fn packs_branches_and_weights_in_any_order_and_reads_them_back() {
        // Ties, single-child last block, and wide and zero-width steps, in
        // an order that is not by weight.
        let children: Vec<(Branch, f64)> = (0..37)
            .map(|i: u32| {
                (
                    151_935 - i * 4000,
                    f64::from((i * 7) % 11) * -0.37 - if i == 20 { 1e3 } else { 0.0 },
                )
            })
            .collect();
        let packed = Packed::new(&children);
        for (at, &(branch, weight_)) in children.iter().enumerate() {
            assert_eq!((packed.branch(at), packed.ln_p(at)), (branch, weight_));
        }
        let wanted = [children[36].0, 7, children[0].0];
        assert_eq!(packed.positions(&wanted), [Some(36), None, Some(0)]);
    }

    #[test]
    fn edges_are_exact_sums_and_points_find_the_child_between_them() {
        // A light child between heavy ones, in branch order, across many
        // kept sums: its extent is far below any float of the edges.
        let mut children: Vec<(Branch, f64)> = (0..1000)
            .map(|i| (i, f64::from(-(f64::from(i % 97) * 0.37) as f32)))
            .collect();
        children[600].1 = -60.0;
        let total: f64 = children.iter().map(|&(_, weight)| weight.exp()).sum();
        let mut normalized = children.clone();
        crate::distribution::log_softmax(&mut normalized);
        let packed = Packed::new(&normalized);
        assert_eq!(packed.edge(0), Point::top());
        assert_eq!(packed.top(1000, 64), Point::bottom());
        let mut above = 0.0;
        for (at, &(_, weight)) in children.iter().enumerate() {
            let edge = packed.edge(at);
            assert!((edge.ln_below().exp() - (1.0 - above / total)).abs() < 1e-12);
            assert!((packed.ln_p(at) - (weight - total.ln())).abs() < 1e-12);
            // The edge itself, and the points just either side of it.
            assert_eq!(packed.holding(&edge), at);
            let next = packed.edge(at + 1);
            let inside = next.moved(false, 1.0, packed.ln_p(at) - 20.0).unwrap();
            assert_eq!(packed.holding(&inside), at);
            if 0 < at {
                let before = edge.moved(false, 1.0, -200.0).unwrap();
                assert_eq!(packed.holding(&before), at - 1);
            }
            above += weight.exp();
        }
        let light = Point::ln_between(&packed.edge(600), &packed.edge(601));
        assert!((light - packed.ln_p(600)).abs() < 1e-9);
        assert_eq!(packed.holding(&Point::bottom()), 1000);
    }
}
