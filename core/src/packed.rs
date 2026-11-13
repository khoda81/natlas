//! A measure's children as a server keeps them, a whole vocabulary per node:
//! heaviest first, each one's branch at the vocabulary's width, and its ln
//! weight as an f32, the precision a model computes its logits in, told by
//! how far it falls below the one before.
//!
//! Sorted f32s lie close together: on Qwen3-4B's 152k children the steps
//! between their bit patterns take about 5 bits. Children are kept in blocks
//! of `BLOCK`; a block holds its first weight, where its steps start and the
//! sum of the weights from its top down, and packs its other steps at the
//! width of its largest. An edge inside a block is summed from the next
//! block's top, just as the tops were summed: every edge is exact for these
//! weights.

use crate::distribution::ln_add;
use crate::tree::Branch;

/// Children per block. Measured on Qwen3-4B, a read of 1600 points takes
/// about 2.4 ms with blocks of 8, 3.5 ms with 16 and 5.9 ms with 32; their
/// weights and edges take about 1.9, 1.4 and 1.2 bytes per child.
const BLOCK: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Packed {
    len: usize,
    /// Each child's branch, by rank, in `width` bits.
    branches: Bits,
    width: u32,
    /// ln of the weight from each block's top to the end: the first is the
    /// total.
    tops: Vec<f64>,
    /// Each block's first weight, as its key.
    keys: Vec<u32>,
    /// Where each block's steps start in `steps`, in bits, and where the
    /// last one's end.
    starts: Vec<u32>,
    /// Each weight after its block's first, as how far its key falls below
    /// the one before.
    steps: Bits,
}

impl Packed {
    /// Children heaviest first, ties by increasing branch: at least one,
    /// each weight finite.
    pub(crate) fn new(children: &[(f32, Branch)]) -> Self {
        assert!(!children.is_empty() && children.iter().all(|(weight, _)| weight.is_finite()));
        let len = children.len();
        let max = children
            .iter()
            .map(|&(_, branch)| branch)
            .max()
            .expect("children");
        let width = u32::BITS - max.leading_zeros();
        let mut branches = Bits::default();
        for &(_, branch) in children {
            branches.push(u64::from(branch), width);
        }
        let (mut keys, mut starts, mut steps) = (Vec::new(), Vec::new(), Bits::default());
        for block in children.chunks(BLOCK) {
            let block: Vec<u32> = block.iter().map(|&(weight, _)| key(weight)).collect();
            let width = block
                .windows(2)
                .map(|pair| u32::BITS - (pair[0] - pair[1]).leading_zeros())
                .max()
                .unwrap_or(0);
            keys.push(block[0]);
            starts.push(u32::try_from(steps.len).expect("steps fit in u32 bits"));
            for pair in block.windows(2) {
                steps.push(u64::from(pair[0] - pair[1]), width);
            }
        }
        starts.push(u32::try_from(steps.len).expect("steps fit in u32 bits"));
        // Kept for as long as the node is: no room to grow.
        for bits in [&mut branches, &mut steps] {
            bits.words.shrink_to_fit();
        }
        keys.shrink_to_fit();
        starts.shrink_to_fit();
        let mut packed = Self {
            len,
            branches,
            width,
            tops: Vec::new(),
            keys,
            starts,
            steps,
        };
        let mut tops = vec![0.0; len.div_ceil(BLOCK)];
        let mut sum = f64::NEG_INFINITY;
        for block in (0..tops.len()).rev() {
            let (weights, count) = packed.block(block);
            for &weight in weights[..count].iter().rev() {
                sum = ln_add(sum, f64::from(weight));
            }
            tops[block] = sum;
        }
        packed.tops = tops;
        packed
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn branch(&self, at: usize) -> Branch {
        self.branches
            .get(at as u64 * u64::from(self.width), self.width) as Branch
    }

    pub(crate) fn branches(&self, from: usize, to: usize) -> Vec<Branch> {
        (from..to).map(|at| self.branch(at)).collect()
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

    /// The weights of a block, its first first, and how many it has.
    fn block(&self, block: usize) -> ([f32; BLOCK], usize) {
        let count = BLOCK.min(self.len - block * BLOCK);
        let (start, end) = (self.starts[block], self.starts[block + 1]);
        let width = if count == 1 {
            0
        } else {
            (end - start) / (count as u32 - 1)
        };
        let mut key = self.keys[block];
        let mut weights = [0.0; BLOCK];
        weights[0] = weight(key);
        for (step, slot) in (0..).zip(&mut weights[1..count]) {
            key -= self
                .steps
                .get(u64::from(start) + step * u64::from(width), width) as u32;
            *slot = weight(key);
        }
        (weights, count)
    }

    pub(crate) fn ln_weight(&self, at: usize) -> f32 {
        self.block(at / BLOCK).0[at % BLOCK]
    }

    /// ln of the weight of every child from `at` to the end, the total
    /// left on.
    fn sum_below(&self, at: usize) -> f64 {
        if at == self.len {
            return f64::NEG_INFINITY;
        }
        let block = at / BLOCK;
        if at.is_multiple_of(BLOCK) {
            return self.tops[block];
        }
        let mut sum = self
            .tops
            .get(block + 1)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let (weights, count) = self.block(block);
        for &weight in weights[at % BLOCK..count].iter().rev() {
            sum = ln_add(sum, f64::from(weight));
        }
        sum
    }

    pub(crate) fn total(&self) -> f64 {
        self.tops[0]
    }

    /// The edge at the top of rank `at`: ln of the mass at and below it.
    pub(crate) fn edge(&self, at: usize) -> f64 {
        self.sum_below(at) - self.total()
    }

    /// Edges `from` through `to`, summed up from `to` once.
    pub(crate) fn edges(&self, from: usize, to: usize) -> Vec<f64> {
        let mut sum = self.sum_below(to);
        let mut edges = vec![sum - self.total(); to - from + 1];
        let mut at = to;
        while from < at {
            let block = (at - 1) / BLOCK;
            let (weights, _) = self.block(block);
            let first = (block * BLOCK).max(from);
            for rank in (first..at).rev() {
                sum = ln_add(sum, f64::from(weights[rank - block * BLOCK]));
                edges[rank - from] = sum - self.total();
            }
            at = first;
        }
        edges
    }

    /// How many edges from rank 1 down a point is `below`: the rank of the
    /// child holding it.
    pub(crate) fn holding_by(&self, below: impl Fn(f64) -> bool) -> usize {
        // The blocks whose bottom edges it is below, then edges among the
        // next block's.
        let blocks = self.tops.len();
        let bottom = |block: usize| ((block + 1) * BLOCK).min(self.len);
        let (mut low, mut high) = (0, blocks);
        while low < high {
            let middle = (low + high) / 2;
            if below(self.edge(bottom(middle))) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        if low == blocks {
            return self.len;
        }
        let from = low * BLOCK;
        let edges = self.edges(from, bottom(low));
        from + edges[1..].partition_point(|&edge| below(edge))
    }

    pub(crate) fn bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + self.branches.bytes()
            + self.steps.bytes()
            + self.tops.capacity() * size_of::<f64>()
            + (self.keys.capacity() + self.starts.capacity()) * size_of::<u32>()
    }
}

/// An f32's bits as a key ordered like the f32: larger for a larger value.
fn key(weight: f32) -> u32 {
    let bits = weight.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}

fn weight(key: u32) -> f32 {
    f32::from_bits(if key >> 31 == 1 {
        key & !(1 << 31)
    } else {
        !key
    })
}

/// Values of up to 32 bits each, packed end to end.
#[derive(Clone, Debug, Default, PartialEq)]
struct Bits {
    words: Vec<u64>,
    len: u64,
}

impl Bits {
    fn push(&mut self, value: u64, width: u32) {
        debug_assert!(width <= 32 && value >> width == 0);
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
        value & ((1 << width) - 1)
    }

    fn bytes(&self) -> usize {
        self.words.capacity() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_order_like_their_weights_and_return_them() {
        let weights = [3.5f32, 0.0, -0.0, -1e-30, -1.0, -700.25, f32::MIN];
        for pair in weights.windows(2) {
            assert!(key(pair[1]) <= key(pair[0]), "{pair:?}");
        }
        for weight_ in weights {
            assert_eq!(weight(key(weight_)).to_bits(), weight_.to_bits());
        }
    }

    #[test]
    fn packs_branches_and_weights_and_reads_them_back() {
        // Ties, single-child last block, and wide and zero-width steps.
        let mut children: Vec<(f32, Branch)> = (0..37)
            .map(|i: u32| {
                (
                    (i / 3) as f32 * -0.37 - if i == 20 { 1e3 } else { 0.0 },
                    151_935 - i * 4000,
                )
            })
            .collect();
        children.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let packed = Packed::new(&children);
        for (at, &(weight_, branch)) in children.iter().enumerate() {
            assert_eq!((packed.branch(at), packed.ln_weight(at)), (branch, weight_));
        }
        let wanted = [children[36].1, 7, children[0].1];
        assert_eq!(packed.positions(&wanted), [Some(36), None, Some(0)]);
    }
}
