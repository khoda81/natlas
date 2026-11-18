//! The integer-sequence tree every client and server names alike: nodes by
//! hashed ids, and what each measure says of a node's children.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

use serde::Serialize;

use crate::distribution::{Distribution, Run};
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
    distributions: Vec<Option<Distribution>>,
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

#[derive(Debug)]
pub struct Tree {
    nodes: HashMap<NodeId, Node>,
    measures: usize,
    /// Ranges of branches a reading draws together: in a node, each group's
    /// known children are drawn as wide as their mean, so they read as one.
    groups: Vec<Range<Branch>>,
    /// Each group's width in a node, by (node, lens, group), once found:
    /// kept beside the distributions it is found from, and forgotten with
    /// any change to the node's.
    widths: Mutex<HashMap<(NodeId, Lens, usize), Option<f64>>>,
}

impl Tree {
    pub fn new(measures: usize) -> Self {
        let mut tree = Self {
            nodes: HashMap::new(),
            measures,
            groups: Vec::new(),
            widths: Mutex::default(),
        };
        tree.insert_node(ROOT, None);
        tree
    }

    fn insert_node(&mut self, id: NodeId, edge: Option<(NodeId, Branch)>) {
        let distributions = (0..self.measures).map(|_| None).collect();
        self.nodes.insert(
            id,
            Node {
                edge,
                distributions,
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
            Some(node) if node.edge != Some((parent, branch)) => {
                Err(format!("node id {id:#018x} collides"))
            }
            Some(_) => Ok(id),
            None => {
                self.insert_node(id, Some((parent, branch)));
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

    pub fn measures(&self) -> usize {
        self.measures
    }

    /// Whether this tree has learned the node.
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
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

    /// What `measure` has said of the node's children; nothing for a node
    /// this tree has not learned.
    pub fn distribution(&self, id: NodeId, measure: usize) -> Result<Option<&Distribution>> {
        self.check_measure(measure)?;
        Ok(self
            .nodes
            .get(&id)
            .and_then(|node| node.distributions[measure].as_ref()))
    }

    /// Add a run a source sent, and whether anything changed.
    /// Add one answer's runs for the node, and whether anything changed.
    pub fn insert(&mut self, id: NodeId, measure: usize, runs: Vec<Run>) -> Result<bool> {
        self.check_measure(measure)?;
        let slot = &mut self.node_mut(id)?.distributions[measure];
        let changed = match slot {
            Some(distribution) => distribution.insert(runs)?,
            None if runs.is_empty() => false,
            None => {
                *slot = Some(Distribution::from_answer(runs)?);
                true
            }
        };
        if changed {
            self.changed(id);
        }
        Ok(changed)
    }

    /// Replace everything known of the node's children under `measure`.
    pub fn set(&mut self, id: NodeId, measure: usize, distribution: Distribution) -> Result<()> {
        self.check_measure(measure)?;
        self.node_mut(id)?.distributions[measure] = Some(distribution);
        self.changed(id);
        Ok(())
    }

    pub fn forget(&mut self, id: NodeId, measure: usize) -> Result<()> {
        self.check_measure(measure)?;
        self.node_mut(id)?.distributions[measure] = None;
        self.changed(id);
        Ok(())
    }

    /// What is known of the node changed: its groups' widths with it.
    fn changed(&mut self, id: NodeId) {
        let widths = self
            .widths
            .get_mut()
            .expect("no width is found while poisoned");
        widths.retain(|&(node, ..), _| node != id);
    }

    /// The ranges of branches a reading draws together, in place of any
    /// before.
    pub fn set_groups(&mut self, groups: Vec<Range<Branch>>) {
        self.groups = groups;
        self.widths
            .get_mut()
            .expect("no width is found while poisoned")
            .clear();
    }

    /// The width a child of `id` is drawn at: −ln q, or for a child in a
    /// group, the group's known children's mean −ln q weighted by p, so the
    /// group keeps its area as one. Infinite while unknown.
    pub fn width(&self, id: NodeId, lens: Lens, branch: Branch, ln_p: f64) -> Result<f64> {
        let Some(group) = self.group(branch) else {
            return self.nats_q(id, lens, branch, ln_p);
        };
        let key = (id, lens, group);
        let found = self
            .widths
            .lock()
            .expect("no width is found while poisoned")
            .get(&key)
            .copied();
        let width = match found {
            Some(width) => width,
            None => {
                let width = self.group_width(id, lens, group)?;
                self.widths
                    .lock()
                    .expect("no width is found while poisoned")
                    .insert(key, width);
                width
            }
        };
        width.map_or_else(|| self.nats_q(id, lens, branch, ln_p), Ok)
    }

    /// The group a branch is in, if any.
    pub fn group(&self, branch: Branch) -> Option<usize> {
        self.groups.iter().position(|range| range.contains(&branch))
    }

    /// The mean −ln q of a group's known children of `id`, weighted by p;
    /// none where none of them has a known width.
    fn group_width(&self, id: NodeId, lens: Lens, group: usize) -> Result<Option<f64>> {
        let Some(p) = self.distribution(id, lens.p)? else {
            return Ok(None);
        };
        let range = &self.groups[group];
        let (mut mass, mut area) = (0.0, 0.0);
        for run in p.runs() {
            for at in 0..run.len() {
                let branch = run.branch(at);
                if !range.contains(&branch) {
                    continue;
                }
                let ln_p = run.ln_p(at);
                let nats = self.nats_q(id, lens, branch, ln_p)?;
                if nats.is_finite() {
                    let weight = ln_p.exp();
                    (mass, area) = (mass + weight, area + weight * nats);
                }
            }
        }
        Ok((0.0 < mass).then(|| area / mass))
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
        for run in p.runs() {
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
        let mut tree = Tree::new(2);
        let child = tree.child(ROOT, 5).unwrap();
        let grandchild = tree.child(child, 2).unwrap();
        assert_eq!(grandchild, child_hash(child_hash(ROOT, 5), 2));
        assert_eq!(tree.child(ROOT, 5).unwrap(), child);
        assert_eq!(tree.parent(ROOT).unwrap(), None);
        assert_eq!(tree.parent(grandchild).unwrap(), Some(child));
        assert_eq!(tree.branch(grandchild).unwrap(), Some(2));
        assert_eq!(tree.path(grandchild).unwrap(), [5, 2]);
        assert_eq!(
            tree.chain(ROOT, grandchild).unwrap(),
            [ROOT, child, grandchild]
        );
        assert!(tree.chain(grandchild, child).is_err());
        let mut other = Tree::new(1);
        assert_eq!(other.descend(ROOT, &[5, 2]).unwrap(), grandchild);
        assert!(!tree.contains(child_hash(ROOT, 9)));
        // Another parent claiming the same id is a collision, refused.
        tree.insert_node(child_hash(ROOT, 3), Some((ROOT, 4)));
        assert!(tree.child(ROOT, 3).is_err());
    }

    #[test]
    fn entries_place_children_by_p_and_size_them_by_q() {
        let mut tree = Tree::new(2);
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
