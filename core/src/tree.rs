//! The integer-sequence tree every client and server names alike: nodes by
//! hashed ids, and what each measure says of a node's children.

use std::collections::HashMap;

use serde::Serialize;

use crate::distribution::{Distribution, Rank, Run};
use crate::point::Point;
use crate::Result;

/// A node's id, hashed from its branches by [`child_hash`] and shared by every
/// client and server.
pub type NodeId = u64;
pub const ROOT: NodeId = 0;

/// Serde for a node id: JavaScript receives a BigInt, since a 64-bit hash is
/// no exact Number.
pub mod node_id {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(id: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u128(u128::from(*id))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        u64::deserialize(deserializer)
    }
}

/// One of a node's 2^32 branches. Every branch exists; one a measure never
/// lists has weight zero. What a branch denotes is up to the reader.
pub type Branch = u32;

const BRANCH_STEP: u64 = 0x9e37_79b9_7f4a_7c15;

/// The measure whose order and weights split heights, and the one whose
/// surprisal sets widths.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Lens {
    pub p: usize,
    pub q: usize,
}

/// A node's id shared by every client and server: hashed from its path, so
/// naming a node never needs the path itself. SplitMix64's finalizer is a
/// bijection, so siblings never collide; other collisions are left to chance
/// at 64 bits and refused when they happen.
pub fn child_hash(parent: NodeId, branch: Branch) -> NodeId {
    let mut z = parent.wrapping_add(BRANCH_STEP.wrapping_mul(u64::from(branch) + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unmix_id(child: NodeId) -> u64 {
    // Undo each right-shift XOR with two shifts, and each odd multiplication
    // with its inverse modulo 2^64, in reverse order.
    let z = (child ^ (child >> 31) ^ (child >> 62)).wrapping_mul(0x3196_42b2_d24d_8ec3);
    let z = (z ^ (z >> 27) ^ (z >> 54)).wrapping_mul(0x96de_1b17_3f11_9089);
    z ^ (z >> 30) ^ (z >> 60)
}

/// Recover the branch when `child` is a child of the given `parent` hash.
pub fn derive_branch(parent: NodeId, child: NodeId) -> Option<Branch> {
    let branch = unmix_id(child)
        .wrapping_sub(parent)
        .wrapping_mul(0xf1de_83e1_9937_733d)
        .wrapping_sub(1);
    Branch::try_from(branch).ok()
}

/// Recover the parent hash for an edge with a known child and branch.
pub fn derive_parent(child: NodeId, branch: Branch) -> NodeId {
    unmix_id(child).wrapping_sub(BRANCH_STEP.wrapping_mul(u64::from(branch) + 1))
}

fn branch_from_hash(parent: NodeId, child: NodeId) -> Branch {
    derive_branch(parent, child).expect("stored parent and child must name a valid branch")
}

#[derive(Debug)]
struct Node {
    /// Parent id. The node keyed by `ROOT` stores `ROOT` here.
    parent: NodeId,
    /// By measure: as many as the last measure that said something of it.
    distributions: Vec<Option<Distribution>>,
}

impl Node {
    fn slot(&mut self, measure: usize) -> &mut Option<Distribution> {
        if self.distributions.len() <= measure {
            self.distributions.resize_with(measure + 1, || None);
        }
        &mut self.distributions[measure]
    }
}

/// A child as a lens sees it within its parent.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub rank: u32,
    /// p mass between the parent's top and the child's.
    pub before: f64,
    pub ln_p: f64,
    /// −ln q; infinite while q is unknown.
    pub nats_q: f64,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct Statistics {
    /// Over the known children.
    pub entropy: f64,
    pub cross: f64,
    /// Whether children or their q are still unknown, so both are lower bounds.
    pub bounded: bool,
}

/// What every measure says of the nodes learned so far. Measures are
/// numbered by whoever holds the tree, as they appear.
#[derive(Debug)]
pub struct Tree {
    nodes: HashMap<NodeId, Node>,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        let mut tree = Self {
            nodes: HashMap::new(),
        };
        tree.insert_node(ROOT, ROOT);
        tree
    }

    fn insert_node(&mut self, id: NodeId, parent: NodeId) {
        self.nodes.insert(
            id,
            Node {
                parent,
                distributions: Vec::new(),
            },
        );
    }

    fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes
            .get(&id)
            .ok_or_else(|| format!("unknown node {id:#018x}"))
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node> {
        self.nodes
            .get_mut(&id)
            .ok_or_else(|| format!("unknown node {id:#018x}"))
    }

    /// The child of `parent` along `branch`, learned on first use.
    pub fn child(&mut self, parent: NodeId, branch: Branch) -> Result<NodeId> {
        self.node(parent)?;
        let id = child_hash(parent, branch);
        match self.nodes.get(&id) {
            Some(node) if id != ROOT && node.parent == parent => Ok(id),
            Some(_) => Err(format!("node id {id:#018x} collides")),
            None => {
                self.insert_node(id, parent);
                Ok(id)
            }
        }
    }

    /// Follow `path` down from `node`, learning only missing nodes.
    pub fn descend(&mut self, mut node: NodeId, path: &[Branch]) -> Result<NodeId> {
        self.node(node)?;
        for &branch in path {
            node = self.child(node, branch)?;
        }
        Ok(node)
    }

    /// Whether this tree has learned the node.
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    pub fn parent(&self, id: NodeId) -> Result<Option<NodeId>> {
        let node = self.node(id)?;
        Ok((id != ROOT).then_some(node.parent))
    }

    pub fn branch(&self, child: NodeId) -> Result<Option<Branch>> {
        Ok(self
            .parent(child)?
            .map(|parent| branch_from_hash(parent, child)))
    }

    /// Nodes from `origin` down to `id`, inclusive.
    pub fn chain(&self, origin: NodeId, id: NodeId) -> Result<Vec<NodeId>> {
        let mut chain = vec![id];
        let mut current = id;
        while current != origin {
            current = self.parent(current)?.ok_or_else(|| {
                format!("node {origin:#018x} is not an ancestor of node {id:#018x}")
            })?;
            chain.push(current);
        }
        chain.reverse();
        Ok(chain)
    }

    /// Branches from the root down to `id`.
    pub fn path(&self, id: NodeId) -> Result<Vec<Branch>> {
        let mut path = Vec::new();
        let mut current = id;
        while let Some(parent) = self.parent(current)? {
            path.push(branch_from_hash(parent, current));
            current = parent;
        }
        path.reverse();
        Ok(path)
    }

    /// What `measure` has said of the node's children; nothing for a node
    /// this tree has not learned.
    pub fn distribution(&self, id: NodeId, measure: usize) -> Result<Option<&Distribution>> {
        Ok(self
            .nodes
            .get(&id)
            .and_then(|node| node.distributions.get(measure)?.as_ref()))
    }

    /// Add one answer for the node, and report whether anything changed.
    pub fn insert(
        &mut self,
        id: NodeId,
        measure: usize,
        runs: Vec<Run>,
        count: Option<Rank>,
    ) -> Result<bool> {
        let node = self.node_mut(id)?;
        if runs.is_empty() && count.is_none() {
            return Ok(false);
        }
        if let Some(Some(distribution)) = node.distributions.get_mut(measure) {
            return distribution.insert(runs, count);
        }
        let distribution = Distribution::from_answer(runs, count)?;
        *node.slot(measure) = Some(distribution);
        Ok(true)
    }

    /// Replace everything known of the node's children under `measure`.
    pub fn set(&mut self, id: NodeId, measure: usize, distribution: Distribution) -> Result<()> {
        *self.node_mut(id)?.slot(measure) = Some(distribution);
        Ok(())
    }

    pub fn forget(&mut self, id: NodeId, measure: usize) -> Result<()> {
        if let Some(slot) = self.node_mut(id)?.distributions.get_mut(measure) {
            *slot = None;
        }
        Ok(())
    }

    /// −ln q of a child of `id`, infinite while unknown: an unknown width
    /// reaches past any view.
    pub fn nats_q(&self, id: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> Result<f64> {
        if lens.p == lens.q {
            return Ok(-ln_p);
        }
        Ok(self
            .distribution(id, lens.q)?
            .and_then(|q| q.ln_p(branch))
            .map_or(f64::INFINITY, |ln_q| -ln_q))
    }

    /// Where p places the child along `branch` in `id`, if it does.
    pub fn entry(&self, id: NodeId, lens: Lens, branch: Branch) -> Result<Option<Entry>> {
        let Some(p) = self.distribution(id, lens.p)? else {
            return Ok(None);
        };
        let Some(rank) = p.rank(branch) else {
            return Ok(None);
        };
        let top = p.edge(rank).expect("a ranked branch has a known edge");
        let ln_p = p.ln_p(branch).expect("a ranked branch has a weight");
        Ok(Some(Entry {
            rank,
            before: Point::ln_between(&Point::top(), &top).exp(),
            ln_p,
            nats_q: self.nats_q(id, lens, branch, ln_p)?,
        }))
    }

    pub fn statistics(&self, id: NodeId, lens: Lens) -> Result<Statistics> {
        let Some(p) = self.distribution(id, lens.p)? else {
            return Ok(Statistics {
                entropy: 0.0,
                cross: 0.0,
                bounded: true,
            });
        };
        let mut entropy = 0.0;
        let mut cross = 0.0;
        let mut bounded = !p.complete();
        for run in (0..p.run_count()).map(|index| p.run(index)) {
            for at in 0..run.len() {
                let ln_p = run.ln_p(at);
                // p (−ln p) is zero both at p = 0 and at p = 1.
                if ln_p < 0.0 && f64::NEG_INFINITY < ln_p {
                    entropy += -ln_p * ln_p.exp();
                }
                if lens.p != lens.q {
                    let nats = self.nats_q(id, lens, run.branch(at), ln_p)?;
                    if nats.is_finite() {
                        cross += ln_p.exp() * nats;
                    } else {
                        bounded = true;
                    }
                }
            }
        }
        if lens.p == lens.q {
            cross = entropy;
        }
        Ok(Statistics {
            entropy,
            cross,
            bounded,
        })
    }
}

/// Digits cross to JavaScript as a Uint8Array.
pub(crate) mod bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(digits: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(digits)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        Vec::deserialize(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(weights: &[f64]) -> Distribution {
        Distribution::exact(weights)
    }

    #[test]
    fn nodes_are_named_by_their_shared_hashes() {
        let mut tree = Tree::new();
        let child = tree.child(ROOT, 0).unwrap();
        let grandchild = tree.child(child, Branch::MAX).unwrap();
        assert_eq!(grandchild, child_hash(child_hash(ROOT, 0), Branch::MAX));
        assert_eq!(tree.child(ROOT, 0).unwrap(), child);
        assert_eq!(tree.parent(ROOT).unwrap(), None);
        assert_eq!(tree.branch(ROOT).unwrap(), None);
        assert!(tree.path(ROOT).unwrap().is_empty());
        assert_eq!(tree.parent(grandchild).unwrap(), Some(child));
        assert_eq!(tree.branch(grandchild).unwrap(), Some(Branch::MAX));
        assert_eq!(tree.path(grandchild).unwrap(), [0, Branch::MAX]);
        assert_eq!(
            tree.chain(ROOT, grandchild).unwrap(),
            [ROOT, child, grandchild]
        );
        assert!(tree.chain(grandchild, child).is_err());
        let mut other = Tree::new();
        assert_eq!(other.descend(ROOT, &[0, Branch::MAX]).unwrap(), grandchild);
        assert!(!tree.contains(child_hash(ROOT, 9)));
        assert!(tree.path(child_hash(ROOT, 9)).is_err());
    }

    #[test]
    fn a_colliding_child_keeps_the_existing_parent() {
        let mut tree = Tree::new();
        let parent = tree.child(ROOT, 5).unwrap();
        let child = tree.child(parent, 1).unwrap();
        // Adding one step to the parent cancels subtracting one branch.
        let other_parent = parent.wrapping_add(BRANCH_STEP);
        tree.insert_node(other_parent, ROOT);
        assert_eq!(child_hash(other_parent, 0), child);
        assert!(tree.child(other_parent, 0).is_err());
        assert_eq!(tree.path(child).unwrap(), [5, 1]);
    }

    #[test]
    fn entries_place_children_by_p_and_size_them_by_q() {
        let mut tree = Tree::new();
        tree.set(ROOT, 0, weights(&[0.25, 0.75])).unwrap();
        let lens = Lens { p: 0, q: 1 };
        let entry = tree.entry(ROOT, lens, 0).unwrap().unwrap();
        assert_eq!(entry.rank, 1);
        assert!((entry.before - 0.75).abs() < 1e-12);
        assert_eq!(entry.nats_q, f64::INFINITY);
        tree.set(ROOT, 1, weights(&[0.5, 0.5])).unwrap();
        let entry = tree.entry(ROOT, lens, 0).unwrap().unwrap();
        assert!((entry.nats_q - 2f64.ln()).abs() < 1e-12);
        assert_eq!(tree.entry(ROOT, lens, 7).unwrap(), None);
        let statistics = tree.statistics(ROOT, lens).unwrap();
        assert!(!statistics.bounded);
        assert!((statistics.cross - 2f64.ln()).abs() < 1e-12);
    }
}
