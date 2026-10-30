use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::Result;

pub type NodeId = u32;
/// One of a node's 2^32 branches. Every branch exists; one a measure never
/// lists has weight zero. What a branch denotes is up to the reader.
pub type Branch = u32;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Lens {
    pub p: usize,
    pub q: usize,
}

/// What a measure said about a node's children: the heaviest ones, in order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Listing {
    /// The horizon asked for, in bits, so it is not asked for again.
    pub horizon: f64,
    pub branches: Vec<Branch>,
    pub ln_weights: Vec<f64>,
    /// Weight of every branch not listed.
    pub leftover: f64,
    /// The horizon the source vouches for: no unlisted branch is heavier than
    /// 2^-complete. A recording can vouch for less than was asked.
    pub complete: f64,
}

impl Listing {
    fn validate(&self) -> Result<()> {
        if self.branches.len() != self.ln_weights.len() {
            return Err(format!(
                "listing has {} branches but {} log weights",
                self.branches.len(),
                self.ln_weights.len()
            ));
        }
        if !(0.0..=1.0).contains(&self.leftover) || self.horizon.is_nan() || self.complete.is_nan()
        {
            return Err("listing has an invalid leftover or horizon".into());
        }
        let mut previous = 0.0;
        for &weight in &self.ln_weights {
            if weight.is_nan() || weight > previous {
                return Err("listing weights must be log probabilities, heaviest first".into());
            }
            previous = weight;
        }
        Ok(())
    }

    /// The branches at least 2^-horizon heavy in batches of `size`, each with
    /// the log weight of everything after it. Remainders are summed from the
    /// light end so that tiny ones are not lost to cancellation against 1.
    pub fn batches(&self, horizon: u32, size: usize) -> Vec<(std::ops::Range<usize>, f64)> {
        let threshold = -f64::from(horizon) * std::f64::consts::LN_2;
        let end = if horizon == 0 {
            0
        } else {
            self.ln_weights
                .partition_point(|&weight| weight >= threshold)
        };
        if end == 0 {
            // Nothing listed leaves the whole interval.
            return vec![(0..0, 0.0)];
        }
        let mut remaining = self.leftover.ln();
        let mut summed = self.ln_weights.len();
        let batches = end.div_ceil(size);
        let mut result = Vec::with_capacity(batches);
        for i in (0..batches).rev() {
            let stop = ((i + 1) * size).min(end);
            while summed > stop {
                summed -= 1;
                remaining = ln_add(remaining, self.ln_weights[summed]);
            }
            result.push((i * size..stop, remaining));
        }
        result.reverse();
        result
    }
}

fn ln_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        high
    } else {
        high + (a.min(b) - high).exp().ln_1p()
    }
}

/// Upper bound on the weight of any child `listing` leaves out.
fn unlisted_bound(listing: &Listing) -> f64 {
    let last = listing.ln_weights.last().map_or(1.0, |weight| weight.exp());
    listing.leftover.min(last).min(2f64.powf(-listing.complete))
}

/// Slot of each listed branch: direct for compact vocabularies, hashed otherwise.
#[derive(Debug)]
enum Index {
    Dense(Vec<usize>),
    Sparse(HashMap<Branch, usize>),
}

impl Index {
    fn new(branches: &[Branch]) -> Result<Self> {
        let duplicate = |branch| format!("branch {branch} is listed twice");
        let length = branches
            .iter()
            .max()
            .map_or(0, |&branch| u64::from(branch) + 1);
        if length <= 1_048_576 && length <= (branches.len() as u64).saturating_mul(4) {
            let mut slots = vec![usize::MAX; length as usize];
            for (slot, &branch) in branches.iter().enumerate() {
                if slots[branch as usize] != usize::MAX {
                    return Err(duplicate(branch));
                }
                slots[branch as usize] = slot;
            }
            Ok(Self::Dense(slots))
        } else {
            let mut slots = HashMap::with_capacity(branches.len());
            for (slot, &branch) in branches.iter().enumerate() {
                if slots.insert(branch, slot).is_some() {
                    return Err(duplicate(branch));
                }
            }
            Ok(Self::Sparse(slots))
        }
    }

    fn get(&self, branch: Branch) -> Option<usize> {
        match self {
            Self::Dense(slots) => slots
                .get(branch as usize)
                .copied()
                .filter(|&slot| slot != usize::MAX),
            Self::Sparse(slots) => slots.get(&branch).copied(),
        }
    }
}

/// A validated listing and the index every lens of it shares.
#[derive(Debug)]
struct Known {
    listing: Listing,
    index: Index,
}

/// A node's slots under one lens: p's listed branches, then p's leftover.
#[derive(Debug)]
pub struct Slots {
    p_known: Option<Arc<Known>>,
    /// Conditional p weight per slot; the tail holds p's leftover.
    p: Vec<f64>,
    /// Conditional q weight per slot, or an upper bound where `q_bound`.
    q: Vec<f64>,
    ln_p: Vec<f64>,
    nats_q: Vec<f64>,
    q_bound: Vec<bool>,
    prefixes: Vec<f64>,
    suffixes: Vec<f64>,
    entropy: f64,
    equal_entropy: f64,
    cross: f64,
    bounded: bool,
}

impl Slots {
    fn new(p_known: Option<&Arc<Known>>, q_known: Option<&Arc<Known>>) -> Self {
        let p = p_known.map(|known| &known.listing);
        let q = q_known.map(|known| &known.listing);
        let listed = p.map_or(0, |listing| listing.branches.len());
        let tail = p.is_some_and(|listing| listing.leftover > 0.0);
        let count = listed + usize::from(tail);
        let mut p_weights = vec![0.0; count];

        if let Some(listing) = p {
            for (weight, ln_weight) in p_weights.iter_mut().zip(&listing.ln_weights) {
                *weight = ln_weight.exp();
            }
            if tail {
                p_weights[listed] = listing.leftover;
            }
        }

        // Whatever q leaves out weighs at most this; a child it lists only under
        // p's tail can be heavier.
        let unlisted = q.map_or(1.0, unlisted_bound);
        let mut tail_bound = unlisted;
        let equal = matches!((p_known, q_known), (Some(p), Some(q)) if Arc::ptr_eq(p, q));
        let mut q_weights = if equal {
            p_weights.clone()
        } else {
            vec![unlisted; count]
        };
        let mut q_bound = vec![!equal; count];
        if let (Some(listing), false) = (q, equal) {
            for (rank, &branch) in listing.branches.iter().enumerate() {
                let weight = listing.ln_weights[rank].exp();
                match p_known.and_then(|known| known.index.get(branch)) {
                    None => tail_bound = tail_bound.max(weight),
                    Some(slot) => {
                        q_weights[slot] = weight;
                        q_bound[slot] = false;
                    }
                }
            }
        }
        if tail {
            q_weights[listed] = tail_bound;
            q_bound[listed] = true;
        }

        // Forward and backward sums avoid cancellation at slot boundaries.
        let mut prefixes = vec![0.0; count + 1];
        let mut suffixes = vec![0.0; count + 1];
        for slot in 0..count {
            prefixes[slot + 1] = prefixes[slot] + p_weights[slot];
        }
        for slot in (0..count).rev() {
            suffixes[slot] = suffixes[slot + 1] + p_weights[slot];
        }

        // Log weights come straight from the listings, not back through exp.
        let mut ln_p: Vec<_> = p_weights.iter().map(|p| p.ln()).collect();
        if let Some(listing) = p {
            ln_p[..listed].copy_from_slice(&listing.ln_weights);
        }
        let mut nats_q: Vec<_> = q_weights.iter().map(|q| -q.ln()).collect();
        if equal {
            for slot in 0..listed {
                nats_q[slot] = -ln_p[slot];
            }
        } else if let Some(listing) = q {
            for (&branch, &ln) in listing.branches.iter().zip(&listing.ln_weights) {
                if let Some(slot) = p_known.and_then(|known| known.index.get(branch)) {
                    nats_q[slot] = -ln;
                }
            }
        }
        let mut entropy = 0.0;
        let mut equal_entropy = 0.0;
        let mut cross = 0.0;
        for slot in 0..count {
            let p = p_weights[slot];
            if p == 0.0 {
                continue;
            }
            entropy -= p * ln_p[slot];
            equal_entropy -= p * if slot == listed {
                -nats_q[slot]
            } else {
                ln_p[slot]
            };
            cross += p * nats_q[slot];
        }
        let bounded = tail || q_bound.iter().any(|&bound| bound);

        Self {
            p_known: p_known.cloned(),
            p: p_weights,
            q: q_weights,
            ln_p,
            nats_q,
            q_bound,
            prefixes,
            suffixes,
            entropy,
            equal_entropy,
            cross,
            bounded,
        }
    }

    fn branches(&self) -> &[Branch] {
        self.p_known
            .as_ref()
            .map_or(&[], |known| &known.listing.branches)
    }

    pub fn listed(&self) -> usize {
        self.branches().len()
    }

    pub fn count(&self) -> usize {
        self.p.len()
    }

    pub fn p(&self, slot: usize) -> f64 {
        self.p[slot]
    }

    pub fn q(&self, slot: usize) -> f64 {
        self.q[slot]
    }

    pub fn ln_p(&self, slot: usize) -> f64 {
        self.ln_p[slot]
    }

    pub fn nats_q(&self, slot: usize) -> f64 {
        self.nats_q[slot]
    }

    /// The slot's branch; `None` for the tail.
    pub fn branch(&self, slot: usize) -> Option<Branch> {
        self.branches().get(slot).copied()
    }

    pub fn slot_of(&self, branch: Branch) -> Option<usize> {
        self.p_known.as_ref()?.index.get(branch)
    }

    pub fn statistics(&self, equal: bool) -> SlotStatistics {
        SlotStatistics {
            entropy: if equal {
                self.equal_entropy
            } else {
                self.entropy
            },
            cross: self.cross,
            bounded: self.bounded,
        }
    }

    pub fn q_bound(&self, slot: usize) -> bool {
        self.q_bound.get(slot).copied().unwrap_or(false)
    }

    /// Σ p over slots [0, slot): accumulated forward, so free of cancellation.
    pub fn prefix(&self, slot: usize) -> f64 {
        self.prefixes[slot.min(self.count())]
    }

    /// Σ p over slots [slot, count): accumulated backward, so free of cancellation.
    pub fn suffix(&self, slot: usize) -> f64 {
        self.suffixes[slot.min(self.count())]
    }

    pub fn view(&self) -> SlotsView<'_> {
        SlotsView {
            branches: self.branches(),
            p: &self.p,
            q: &self.q,
            q_bound: &self.q_bound,
            listed: self.listed(),
            count: self.count(),
        }
    }
}

#[derive(Serialize)]
pub struct SlotStatistics {
    pub entropy: f64,
    pub cross: f64,
    pub bounded: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotsView<'a> {
    branches: &'a [Branch],
    p: &'a [f64],
    q: &'a [f64],
    q_bound: &'a [bool],
    listed: usize,
    count: usize,
}

/// A node's id shared by every client and server: hashed from its path, so
/// naming a node never needs the path itself. SplitMix64's finalizer is a
/// bijection, so siblings never collide; other collisions are left to chance
/// at 64 bits and refused when they happen.
pub fn child_hash(parent: u64, branch: Branch) -> u64 {
    let mut z = parent.wrapping_add(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(u64::from(branch) + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Debug)]
struct Node {
    /// Parent and the branch to this node; `None` only for the root.
    edge: Option<(NodeId, Branch)>,
    hash: u64,
    depth: usize,
    children: HashMap<Branch, NodeId>,
    listings: Vec<Option<Arc<Known>>>,
    views: RwLock<HashMap<Lens, Arc<Slots>>>,
}

impl Node {
    fn new(edge: Option<(NodeId, Branch)>, hash: u64, depth: usize, measures: usize) -> Self {
        Self {
            edge,
            hash,
            depth,
            children: HashMap::new(),
            listings: vec![None; measures],
            views: RwLock::new(HashMap::new()),
        }
    }
}

#[derive(Debug)]
pub struct Tree {
    nodes: Vec<Node>,
    hashes: HashMap<u64, NodeId>,
    measures: usize,
}

impl Tree {
    pub fn new(measures: usize) -> Self {
        Self {
            nodes: vec![Node::new(None, 0, 0, measures)],
            hashes: HashMap::from([(0, 0)]),
            measures,
        }
    }

    pub fn root(&self) -> NodeId {
        0
    }

    fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes
            .get(id as usize)
            .ok_or_else(|| format!("unknown node {id}"))
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node> {
        self.nodes
            .get_mut(id as usize)
            .ok_or_else(|| format!("unknown node {id}"))
    }

    /// The child of `parent` along `branch`, created on first use.
    pub fn child(&mut self, parent: NodeId, branch: Branch) -> Result<NodeId> {
        let node = self.node(parent)?;
        if let Some(&id) = node.children.get(&branch) {
            return Ok(id);
        }
        let hash = child_hash(node.hash, branch);
        let depth = node.depth + 1;
        if self.hashes.contains_key(&hash) {
            return Err(format!("node hash {hash:#018x} collides"));
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| "too many nodes".to_string())?;
        self.nodes.push(Node::new(
            Some((parent, branch)),
            hash,
            depth,
            self.measures,
        ));
        self.hashes.insert(hash, id);
        self.node_mut(parent)?.children.insert(branch, id);
        Ok(id)
    }

    /// Follow `path` down from `node`, creating only missing nodes.
    pub fn descend(&mut self, mut node: NodeId, path: &[Branch]) -> Result<NodeId> {
        for &branch in path {
            node = self.child(node, branch)?;
        }
        Ok(node)
    }

    pub(crate) fn child_id(&self, parent: NodeId, branch: Branch) -> Result<Option<NodeId>> {
        Ok(self.node(parent)?.children.get(&branch).copied())
    }

    /// A frame only needs the expanded exceptions, not a kind record for each
    /// of the hundred thousand unexpanded vocabulary entries.
    pub(crate) fn expanded_branches(&self, id: NodeId, measure: usize) -> Result<HashSet<Branch>> {
        self.node(id)?
            .children
            .iter()
            .filter_map(|(&branch, &child)| match self.expanded(child, measure) {
                Ok(true) => Some(Ok(branch)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }

    pub fn find(&self, hash: u64) -> Option<NodeId> {
        self.hashes.get(&hash).copied()
    }

    pub fn hash(&self, id: NodeId) -> Result<u64> {
        Ok(self.node(id)?.hash)
    }

    pub fn depth(&self, id: NodeId) -> Result<usize> {
        Ok(self.node(id)?.depth)
    }

    pub fn parent(&self, id: NodeId) -> Result<Option<NodeId>> {
        Ok(self.node(id)?.edge.map(|(parent, _)| parent))
    }

    pub fn branch(&self, id: NodeId) -> Result<Option<Branch>> {
        Ok(self.node(id)?.edge.map(|(_, branch)| branch))
    }

    /// Nodes from `origin` down to `id`, inclusive.
    pub fn chain(&self, origin: NodeId, id: NodeId) -> Result<Vec<NodeId>> {
        let mut chain = vec![id];
        let mut current = id;
        while current != origin {
            current = self
                .parent(current)?
                .ok_or_else(|| format!("node {origin} is not an ancestor of node {id}"))?;
            chain.push(current);
        }
        chain.reverse();
        Ok(chain)
    }

    /// Branches from the root down to `id`.
    pub fn path(&self, id: NodeId) -> Result<Vec<Branch>> {
        let mut path = Vec::with_capacity(self.depth(id)?);
        let mut current = self.node(id)?;
        while let Some((parent, branch)) = current.edge {
            path.push(branch);
            current = self.node(parent)?;
        }
        path.reverse();
        Ok(path)
    }

    fn check_measure(&self, measure: usize) -> Result<()> {
        if measure < self.measures {
            Ok(())
        } else {
            Err(format!("unknown measure {measure}"))
        }
    }

    pub fn listing(&self, id: NodeId, measure: usize) -> Result<Option<&Listing>> {
        self.check_measure(measure)?;
        Ok(self.node(id)?.listings[measure]
            .as_deref()
            .map(|known| &known.listing))
    }

    /// Whether `measure` has said anything about this node's children.
    pub fn expanded(&self, id: NodeId, measure: usize) -> Result<bool> {
        Ok(self
            .listing(id, measure)?
            .is_some_and(|listing| !listing.branches.is_empty() || listing.complete > 0.0))
    }

    pub fn set_listing(&mut self, id: NodeId, measure: usize, listing: Listing) -> Result<()> {
        self.check_measure(measure)?;
        listing.validate()?;
        let index = Index::new(&listing.branches)?;
        let node = self.node_mut(id)?;
        node.listings[measure] = Some(Arc::new(Known { listing, index }));
        node.views.get_mut().expect("slot cache poisoned").clear();
        Ok(())
    }

    pub fn clear_listing(&mut self, id: NodeId, measure: usize) -> Result<()> {
        self.check_measure(measure)?;
        let node = self.node_mut(id)?;
        node.listings[measure] = None;
        node.views.get_mut().expect("slot cache poisoned").clear();
        Ok(())
    }

    pub fn slots(&self, id: NodeId, lens: Lens) -> Result<Arc<Slots>> {
        self.check_measure(lens.p)?;
        self.check_measure(lens.q)?;
        let node = self.node(id)?;
        if let Some(slots) = node
            .views
            .read()
            .expect("slot cache poisoned")
            .get(&lens)
            .cloned()
        {
            return Ok(slots);
        }
        let slots = Arc::new(Slots::new(
            node.listings[lens.p].as_ref(),
            node.listings[lens.q].as_ref(),
        ));
        node.views
            .write()
            .expect("slot cache poisoned")
            .insert(lens, Arc::clone(&slots));
        Ok(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listing(weights: &[f64], branches: &[u32], complete: f64) -> Listing {
        Listing {
            horizon: complete,
            branches: branches.to_vec(),
            ln_weights: weights.iter().map(|weight| weight.ln()).collect(),
            leftover: (1.0 - weights.iter().sum::<f64>()).max(0.0),
            complete,
        }
    }

    #[test]
    fn slots_keep_p_order_tail_and_q_upper_bounds() {
        let p = listing(&[0.5, 0.3], &[0, 1], 2.0);
        let mut tree = Tree::new(2);
        tree.set_listing(0, 0, p.clone()).unwrap();
        tree.set_listing(0, 1, listing(&[0.6, 0.3], &[1, 2], 2.0))
            .unwrap();
        let slots = tree.slots(0, Lens { p: 0, q: 1 }).unwrap();
        assert_eq!(slots.branches(), [0, 1]);
        assert_eq!(slots.count(), 3);
        assert_eq!(slots.branch(2), None);
        assert!((slots.p(2) - 0.2).abs() < 1e-12);
        assert!((slots.suffix(1) - 0.5).abs() < 1e-12);
        assert_eq!(slots.slot_of(1), Some(1));
        for (actual, expected) in slots.q.iter().zip([0.1, 0.6, 0.3]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(slots.q_bound, vec![true, false, true]);

        let same = tree.slots(0, Lens { p: 0, q: 0 }).unwrap();
        assert!((same.q(2) - 0.2).abs() < 1e-12);
        assert_eq!(same.ln_p(0), 0.5f64.ln());
        assert_eq!(same.nats_q(1), -0.3f64.ln());
        tree.set_listing(0, 0, Listing { complete: 4.0, ..p })
            .unwrap();
        assert_eq!(
            tree.slots(0, Lens { p: 0, q: 0 }).unwrap().q(2),
            2f64.powi(-4)
        );
    }

    #[test]
    fn listings_are_validated_at_the_boundary() {
        let mut tree = Tree::new(1);
        for (branches, weights) in [
            (vec![0, 2, 0], vec![0.5, 0.3, 0.2]),
            (vec![0, u32::MAX, 0], vec![0.5, 0.3, 0.2]),
            (vec![0, 1], vec![0.3, 0.5]),
        ] {
            assert!(tree
                .set_listing(0, 0, listing(&weights, &branches, 40.0))
                .is_err());
        }
        let mut positive = listing(&[0.5], &[0], 40.0);
        positive.ln_weights[0] = 1e-9;
        assert!(tree.set_listing(0, 0, positive).is_err());
        let sparse = listing(&[0.5, 0.3, 0.2], &[0, u32::MAX, 7], 40.0);
        tree.set_listing(0, 0, sparse).unwrap();
        let slots = tree.slots(0, Lens { p: 0, q: 0 }).unwrap();
        assert_eq!(slots.slot_of(u32::MAX), Some(1));
        assert_eq!(slots.slot_of(1), None);
    }

    #[test]
    fn nodes_have_stable_ids_paths_chains_and_shared_hashes() {
        let mut tree = Tree::new(2);
        let child = tree.child(0, 5).unwrap();
        let grandchild = tree.child(child, 2).unwrap();
        assert_eq!(tree.child(0, 5).unwrap(), child);
        assert_ne!(tree.child(0, 6).unwrap(), child);
        assert_eq!(tree.parent(0).unwrap(), None);
        assert_eq!(tree.parent(grandchild).unwrap(), Some(child));
        assert_eq!(tree.branch(0).unwrap(), None);
        assert_eq!(tree.branch(grandchild).unwrap(), Some(2));
        assert_eq!(tree.path(grandchild).unwrap(), [5, 2]);
        assert_eq!(tree.depth(grandchild).unwrap(), 2);
        assert_eq!(tree.chain(0, grandchild).unwrap(), [0, child, grandchild]);
        assert_eq!(tree.chain(child, grandchild).unwrap(), [child, grandchild]);
        assert!(tree.chain(grandchild, child).is_err());

        // Another tree names the same paths by the same hashes, whatever
        // order it learned them in.
        let mut other = Tree::new(1);
        let far = other.descend(0, &[5, 2]).unwrap();
        assert_eq!(other.hash(far).unwrap(), tree.hash(grandchild).unwrap());
        assert_eq!(tree.find(other.hash(far).unwrap()), Some(grandchild));
        assert_eq!(tree.hash(0).unwrap(), 0);
        assert_eq!(tree.find(child_hash(0, 9)), None);
    }

    #[test]
    fn slots_are_cached_per_lens_and_invalidated_by_listing_updates() {
        let mut tree = Tree::new(2);
        let lens = Lens { p: 0, q: 1 };
        tree.set_listing(0, 0, listing(&[0.5, 0.5], &[4, 5], 40.0))
            .unwrap();
        tree.set_listing(0, 1, listing(&[0.5, 0.5], &[4, 5], 40.0))
            .unwrap();
        let first = tree.slots(0, lens).unwrap();
        assert!(Arc::ptr_eq(&first, &tree.slots(0, lens).unwrap()));
        tree.set_listing(0, 1, listing(&[0.75, 0.25], &[4, 5], 40.0))
            .unwrap();
        let updated = tree.slots(0, lens).unwrap();
        assert!(!Arc::ptr_eq(&first, &updated));
        assert!((updated.q(0) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn expansion_distinguishes_an_empty_reply_from_no_reply() {
        let mut tree = Tree::new(1);
        assert!(!tree.expanded(0, 0).unwrap());
        let mut empty = listing(&[], &[], 0.0);
        tree.set_listing(0, 0, empty.clone()).unwrap();
        assert!(!tree.expanded(0, 0).unwrap());
        empty.complete = 40.0;
        tree.set_listing(0, 0, empty).unwrap();
        assert!(tree.expanded(0, 0).unwrap());
    }

    #[test]
    fn batches_cut_at_the_horizon_and_carry_exact_remainders() {
        let tiny = 1e-30f64;
        let full = Listing {
            horizon: f64::INFINITY,
            branches: vec![4, 1, 2, 3],
            ln_weights: vec![(1.0 - 3.0 * tiny).ln(), tiny.ln(), tiny.ln(), tiny.ln()],
            leftover: 0.0,
            complete: f64::INFINITY,
        };
        assert_eq!(full.batches(0, 2), [(0..0, 0.0)]);
        let one = full.batches(8, 2);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].0, 0..1);
        assert!((one[0].1 - (3.0 * tiny).ln()).abs() < 1e-12);
        let all = full.batches(200, 3);
        assert_eq!(all[0].0, 0..3);
        assert!((all[0].1 - tiny.ln()).abs() < 1e-12);
        assert_eq!(all[1], (3..4, f64::NEG_INFINITY));
    }
}
