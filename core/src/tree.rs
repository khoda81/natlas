//! The integer-sequence tree every client and server names alike: nodes by
//! hashed ids, and what each measure says of a node's children.

use std::collections::HashMap;

use serde::Serialize;

use crate::distribution::{ln_between, Distribution, Rank, Run, Step, Window, MAX_DESCENT};
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
}

impl Tree {
    pub fn new(measures: usize) -> Self {
        let mut tree = Self {
            nodes: HashMap::new(),
            measures,
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
    pub fn insert(&mut self, id: NodeId, measure: usize, run: Run) -> Result<bool> {
        self.check_measure(measure)?;
        let slot = &mut self.node_mut(id)?.distributions[measure];
        Ok(match slot {
            Some(distribution) => distribution.insert(run),
            None => {
                *slot = Some(Distribution::new(run));
                true
            }
        })
    }

    /// Replace everything known of the node's children under `measure`.
    pub fn set(&mut self, id: NodeId, measure: usize, distribution: Distribution) -> Result<()> {
        self.check_measure(measure)?;
        self.node_mut(id)?.distributions[measure] = Some(distribution);
        Ok(())
    }

    pub fn forget(&mut self, id: NodeId, measure: usize) -> Result<()> {
        self.check_measure(measure)?;
        self.node_mut(id)?.distributions[measure] = None;
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
            before: ln_between(0.0, top).exp(),
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
            entropy += run.ln_surprisal(0, run.len()).exp();
            if lens.p != lens.q {
                for at in 0..run.len() {
                    let ln_p = run.ln_p(at);
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

/// One request of the protocol's Expand walked through a tree, read one of
/// two ways: down a path, placing each branch, then listing the node it
/// reaches; or down a window, through every child holding all of it, then
/// listing the node where it spans several children. A recording answers
/// from what it has; a server computes each missing distribution as the walk
/// asks for it.
#[derive(Debug)]
pub struct Walk {
    node: NodeId,
    path: Vec<Branch>,
    at: usize,
    window: Window,
    horizon: u32,
    descended: usize,
    done: bool,
}

#[derive(Debug, PartialEq)]
pub enum Advance {
    /// The next message: this node's run, and how many children the node
    /// has where its source knows.
    Run(NodeId, Run, Option<Rank>),
    /// The walk needs this node's distribution before it can go on.
    Missing(NodeId),
    Done,
}

impl Walk {
    /// Down `path` below `node`, then the whole node it reaches.
    pub fn path(node: NodeId, path: Vec<Branch>, horizon: u32) -> Self {
        Self::new(node, path, Window::WHOLE, horizon)
    }

    /// Down `window` within `node`.
    pub fn window(node: NodeId, window: Window, horizon: u32) -> Self {
        Self::new(node, Vec::new(), window, horizon)
    }

    fn new(node: NodeId, path: Vec<Branch>, window: Window, horizon: u32) -> Self {
        Self {
            node,
            path,
            at: 0,
            window,
            horizon,
            descended: 0,
            done: false,
        }
    }

    pub fn advance(&mut self, tree: &mut Tree, measure: usize) -> Result<Advance> {
        if self.done {
            return Ok(Advance::Done);
        }
        let node = self.node;
        let Some(distribution) = tree.distribution(node, measure)? else {
            return Ok(Advance::Missing(node));
        };
        let count = distribution.count();
        let (run, next) = if let Some(&branch) = self.path.get(self.at) {
            // A branch of weight zero has no place: the walk ends before it.
            let Some(run) = distribution.place(branch) else {
                self.done = true;
                return Ok(Advance::Done);
            };
            self.at += 1;
            (run, Some((branch, self.window)))
        } else {
            let step = if self.descended < MAX_DESCENT {
                distribution.step(self.window, self.horizon)
            } else {
                distribution.list(self.window, self.horizon).map(Step::List)
            };
            match step {
                Some(Step::Descend {
                    run,
                    branch,
                    window,
                }) => {
                    self.descended += 1;
                    (run, Some((branch, window)))
                }
                Some(Step::List(run)) => (run, None),
                // A node without children says so with its one empty run.
                None if distribution.ended() => (distribution.runs()[0].clone(), None),
                None => {
                    self.done = true;
                    return Ok(Advance::Done);
                }
            }
        };
        match next {
            Some((branch, window)) => {
                self.node = tree.child(node, branch)?;
                self.window = window;
            }
            None => self.done = true,
        }
        Ok(Advance::Run(node, run, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(weights: &[f64]) -> Distribution {
        Distribution::from_weights(&weights.iter().map(|w| w.ln()).collect::<Vec<_>>()).unwrap()
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

    #[test]
    fn a_walk_places_its_path_then_descends_and_lists_the_window() {
        let mut tree = Tree::new(1);
        let mut walk = Walk::path(ROOT, vec![2], 1);
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Missing(ROOT));
        tree.set(ROOT, 0, weights(&[0.6, 0.3, 0.1])).unwrap();
        let Advance::Run(node, run, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the root places branch 2");
        };
        assert_eq!((node, run.rank(), run.branches()), (ROOT, 2, &[2][..]));
        let child = child_hash(ROOT, 2);
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Missing(child));
        tree.set(child, 0, weights(&[0.5, 0.5])).unwrap();
        assert!(
            matches!(walk.advance(&mut tree, 0).unwrap(), Advance::Run(node, _, Some(2)) if node == child)
        );
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);

        // A window inside branch 0 of the root descends into it.
        let window = Window::new(-0.1, -0.5).unwrap();
        let mut walk = Walk::window(ROOT, window, 8);
        let Advance::Run(ROOT, run, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("the root names the child holding the window");
        };
        assert_eq!(run.branches(), [0]);
        assert_eq!(
            walk.advance(&mut tree, 0).unwrap(),
            Advance::Missing(child_hash(ROOT, 0))
        );

        // A path through a weightless branch ends before it; an ended node
        // says so.
        tree.set(child, 0, Distribution::from_weights(&[]).unwrap())
            .unwrap();
        let mut walk = Walk::path(ROOT, vec![2, 0], 8);
        assert!(matches!(
            walk.advance(&mut tree, 0).unwrap(),
            Advance::Run(ROOT, ..)
        ));
        assert_eq!(walk.advance(&mut tree, 0).unwrap(), Advance::Done);
        let mut walk = Walk::path(child, vec![], 8);
        let Advance::Run(_, run, _) = walk.advance(&mut tree, 0).unwrap() else {
            panic!("an ended node lists its empty run");
        };
        assert!(run.is_empty());
    }
}
