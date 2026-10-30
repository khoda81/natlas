use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::Result;

pub type NodeId = u32;
pub type Label = u32;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Lens {
    pub p: usize,
    pub q: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Listing {
    pub horizon: f64,
    pub labels: Vec<Label>,
    pub ln_weights: Vec<f64>,
    pub leftover: f64,
    pub complete: f64,
}

impl Listing {
    fn validate(&self) -> Result<()> {
        if self.labels.len() == self.ln_weights.len() {
            Ok(())
        } else {
            Err(format!(
                "listing has {} labels but {} log weights",
                self.labels.len(),
                self.ln_weights.len()
            ))
        }
    }
}

/// Upper bound on the weight of any child `listing` leaves out.
fn unlisted_bound(listing: &Listing) -> f64 {
    let last = listing.ln_weights.last().map_or(1.0, |weight| weight.exp());
    listing.leftover.min(last).min(2f64.powf(-listing.complete))
}

/// Direct lookup for compact vocabularies, with sparse integer labels supported.
#[derive(Debug)]
enum LabelIndex {
    Dense { slots: Vec<usize>, count: usize },
    Sparse(HashMap<Label, usize>),
}

impl LabelIndex {
    fn new(labels: &[Label]) -> Self {
        let length = labels.iter().max().map_or(0, |&label| u64::from(label) + 1);
        if length <= 1_048_576 && length <= (labels.len() as u64).saturating_mul(4) {
            let mut slots = vec![usize::MAX; length as usize];
            let mut count = 0;
            for (slot, &label) in labels.iter().enumerate() {
                count += usize::from(slots[label as usize] == usize::MAX);
                slots[label as usize] = slot;
            }
            Self::Dense { slots, count }
        } else {
            Self::Sparse(
                labels
                    .iter()
                    .enumerate()
                    .map(|(slot, &label)| (label, slot))
                    .collect(),
            )
        }
    }

    fn get(&self, label: Label) -> Option<usize> {
        match self {
            Self::Dense { slots, .. } => slots
                .get(label as usize)
                .copied()
                .filter(|&slot| slot != usize::MAX),
            Self::Sparse(slots) => slots.get(&label).copied(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense { count, .. } => *count,
            Self::Sparse(slots) => slots.len(),
        }
    }
}

/// A node's slots under one lens.
#[derive(Debug)]
pub struct Slots {
    labels: Vec<Label>,
    /// Conditional p weight per slot; the tail holds p's leftover.
    p: Vec<f64>,
    /// Conditional q weight per slot, or an upper bound where `q_bound`.
    q: Vec<f64>,
    ln_p: Vec<f64>,
    nats_q: Vec<f64>,
    q_bound: Vec<bool>,
    index: LabelIndex,
    prefixes: Vec<f64>,
    suffixes: Vec<f64>,
    entropy: f64,
    equal_entropy: f64,
    cross: f64,
    bounded: bool,
}

impl Slots {
    pub fn new(p: Option<&Listing>, q: Option<&Listing>) -> Result<Self> {
        if let Some(listing) = p {
            listing.validate()?;
        }
        if let Some(listing) = q {
            listing.validate()?;
        }

        let labels = p.map_or_else(Vec::new, |listing| listing.labels.clone());
        let listed = labels.len();
        let tail = p.is_some_and(|listing| listing.leftover > 0.0);
        let count = listed + usize::from(tail);
        let mut p_weights = vec![0.0; count];
        let index = LabelIndex::new(&labels);

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
        let equal =
            matches!((p, q), (Some(p), Some(q)) if std::ptr::eq(p, q)) && index.len() == listed;
        let mut q_weights = if equal {
            p_weights.clone()
        } else {
            vec![unlisted; count]
        };
        let mut q_bound = vec![!equal; count];
        if let Some(listing) = q.filter(|_| !equal) {
            for (rank, &label) in listing.labels.iter().enumerate() {
                let weight = listing.ln_weights[rank].exp();
                match index.get(label) {
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

        // Inspector statistics depend on the listing, not the current frame
        // or pointer. Accumulate in the same order as the original UI loop.
        let ln_p: Vec<_> = p_weights.iter().map(|p| p.ln()).collect();
        let mut nats_q: Vec<_> = if equal {
            ln_p.iter().map(|ln| -ln).collect()
        } else {
            q_weights.iter().map(|q| -q.ln()).collect()
        };
        if equal && tail {
            nats_q[listed] = -q_weights[listed].ln();
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

        Ok(Self {
            labels,
            p: p_weights,
            q: q_weights,
            ln_p,
            nats_q,
            q_bound,
            index,
            prefixes,
            suffixes,
            entropy,
            equal_entropy,
            cross,
            bounded,
        })
    }

    pub fn listed(&self) -> usize {
        self.labels.len()
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

    pub fn label(&self, slot: usize) -> Option<Label> {
        (slot < self.listed()).then(|| self.labels[slot])
    }

    pub fn slot_of(&self, label: Label) -> Option<usize> {
        self.index.get(label)
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
            labels: &self.labels,
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
    labels: &'a [Label],
    p: &'a [f64],
    q: &'a [f64],
    q_bound: &'a [bool],
    listed: usize,
    count: usize,
}

#[derive(Debug)]
struct Node {
    parent: Option<NodeId>,
    /// Label of the edge from the parent; `None` only for the root.
    label: Option<Label>,
    children: HashMap<Label, NodeId>,
    listings: Vec<Option<Listing>>,
    views: RwLock<HashMap<Lens, Arc<Slots>>>,
}

impl Node {
    fn new(parent: Option<NodeId>, label: Option<Label>, measures: usize) -> Self {
        Self {
            parent,
            label,
            children: HashMap::new(),
            listings: std::iter::repeat_with(|| None).take(measures).collect(),
            views: RwLock::new(HashMap::new()),
        }
    }
}

#[derive(Debug)]
pub struct Tree {
    nodes: Vec<Node>,
    measures: usize,
}

impl Tree {
    pub fn new(measures: usize) -> Self {
        Self {
            nodes: vec![Node::new(None, None, measures)],
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

    /// The child of `parent` with `label`, created on first use.
    pub fn child(&mut self, parent: NodeId, label: Label) -> Result<NodeId> {
        if let Some(id) = self.node(parent)?.children.get(&label).copied() {
            return Ok(id);
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| "too many nodes".to_string())?;
        self.nodes
            .push(Node::new(Some(parent), Some(label), self.measures));
        self.node_mut(parent)?.children.insert(label, id);
        Ok(id)
    }

    pub(crate) fn child_id(&self, parent: NodeId, label: Label) -> Result<Option<NodeId>> {
        Ok(self.node(parent)?.children.get(&label).copied())
    }

    /// A frame only needs the expanded exceptions, not a kind record for each
    /// of the hundred thousand unexpanded vocabulary entries.
    pub(crate) fn expanded_labels(&self, id: NodeId, measure: usize) -> Result<HashSet<Label>> {
        self.node(id)?
            .children
            .iter()
            .filter_map(|(&label, &child)| match self.expanded(child, measure) {
                Ok(true) => Some(Ok(label)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }

    pub fn parent(&self, id: NodeId) -> Result<Option<NodeId>> {
        Ok(self.node(id)?.parent)
    }

    /// Resolve a global integer path, creating only missing path nodes.
    pub fn resolve(&mut self, path: &[Label]) -> Result<NodeId> {
        let mut node = self.root();
        for &label in path {
            node = self.child(node, label)?;
        }
        Ok(node)
    }

    pub fn label(&self, id: NodeId) -> Result<Option<Label>> {
        Ok(self.node(id)?.label)
    }

    /// Nodes from the root down to `id`, inclusive.
    pub fn chain(&self, id: NodeId) -> Result<Vec<NodeId>> {
        self.node(id)?;
        let mut current = Some(id);
        let mut chain = Vec::new();
        while let Some(id) = current {
            let node = self.node(id)?;
            chain.push(id);
            current = node.parent;
        }
        chain.reverse();
        Ok(chain)
    }

    /// Edge labels from the root down to `id`.
    pub fn path(&self, id: NodeId) -> Result<Vec<Label>> {
        self.chain(id)?
            .into_iter()
            .skip(1)
            .map(|id| {
                self.label(id)?
                    .ok_or_else(|| format!("non-root node {id} has no label"))
            })
            .collect()
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
        Ok(self.node(id)?.listings[measure].as_ref())
    }

    /// Whether `measure` has said anything about this node's children.
    pub fn expanded(&self, id: NodeId, measure: usize) -> Result<bool> {
        Ok(self
            .listing(id, measure)?
            .is_some_and(|listing| !listing.labels.is_empty() || listing.complete > 0.0))
    }

    pub fn set_listing(&mut self, id: NodeId, measure: usize, listing: Listing) -> Result<()> {
        self.check_measure(measure)?;
        listing.validate()?;
        let node = self.node_mut(id)?;
        node.listings[measure] = Some(listing);
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
        )?);
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

    fn listing(weights: &[f64], labels: &[u32], complete: f64) -> Listing {
        Listing {
            horizon: complete,
            labels: labels.to_vec(),
            ln_weights: weights.iter().map(|weight| weight.ln()).collect(),
            leftover: 1.0 - weights.iter().sum::<f64>(),
            complete,
        }
    }

    #[test]
    fn slots_keep_p_order_tail_and_q_upper_bounds() {
        let p = listing(&[0.5, 0.3], &[0, 1], 2.0);
        let q = listing(&[0.6, 0.3], &[1, 2], 2.0);
        let slots = Slots::new(Some(&p), Some(&q)).unwrap();
        assert_eq!(slots.labels, [0, 1]);
        assert_eq!(slots.count(), 3);
        assert_eq!(slots.label(2), None);
        assert!((slots.p(2) - 0.2).abs() < 1e-12);
        assert!((slots.suffix(1) - 0.5).abs() < 1e-12);
        assert_eq!(slots.slot_of(1), Some(1));
        for (actual, expected) in slots.q.iter().zip([0.1, 0.6, 0.3]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(slots.q_bound, vec![true, false, true]);

        let same = Slots::new(Some(&p), Some(&p)).unwrap();
        assert!((same.q(2) - 0.2).abs() < 1e-12);
        let deep_p = Listing {
            complete: 4.0,
            ..p.clone()
        };
        let deep = Slots::new(Some(&deep_p), Some(&deep_p)).unwrap();
        assert_eq!(deep.q(2), 2f64.powi(-4));
    }

    #[test]
    fn compact_and_sparse_labels_keep_last_duplicate_lookup() {
        for labels in [[0, 2, 0], [0, u32::MAX, 0]] {
            let p = listing(&[0.5, 0.3, 0.2], &labels, 40.0);
            let slots = Slots::new(Some(&p), Some(&p)).unwrap();
            assert_eq!(slots.slot_of(0), Some(2));
            assert_eq!(slots.slot_of(labels[1]), Some(1));
            assert_eq!(slots.slot_of(1), None);
            assert_eq!(slots.q(0), 0.0);
            assert_eq!(slots.q(1), slots.p(1));
            assert_eq!(slots.q(2), slots.p(2));
            assert_eq!(slots.q_bound, [true, false, false]);
        }
    }

    #[test]
    fn child_identity_paths_and_chains_are_stable() {
        let mut tree = Tree::new(2);
        let child = tree.child(0, 5).unwrap();
        let grandchild = tree.child(child, 2).unwrap();
        assert_eq!(tree.child(0, 5).unwrap(), child);
        assert_ne!(tree.child(0, 6).unwrap(), child);
        assert_eq!(tree.parent(0).unwrap(), None);
        assert_eq!(tree.parent(grandchild).unwrap(), Some(child));
        assert_eq!(tree.label(0).unwrap(), None);
        assert_eq!(tree.label(grandchild).unwrap(), Some(2));
        assert_eq!(tree.path(grandchild).unwrap(), [5, 2]);
        assert_eq!(tree.chain(grandchild).unwrap(), [0, child, grandchild]);
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
        empty.leftover = 1.0;
        tree.set_listing(0, 0, empty.clone()).unwrap();
        assert!(!tree.expanded(0, 0).unwrap());
        empty.complete = 40.0;
        tree.set_listing(0, 0, empty).unwrap();
        assert!(tree.expanded(0, 0).unwrap());
    }
}
